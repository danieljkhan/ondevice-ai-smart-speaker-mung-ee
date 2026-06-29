"""Session enumeration and fd-anchored, traversal-safe file access.

Security model (see ``Dev_Plan/2026-06-18-conversation-download-portal-plan.md`` §5):

- Session ids and file names are validated with ``re.fullmatch`` **after** URL-decoding.
- File access is resolved **component-by-component** from the canonical ``conversations``
  root using ``os.open(..., O_NOFOLLOW | O_DIRECTORY)`` per directory and ``dir_fd=``
  (openat-style) for the session dir and each file. ``fstat`` confirms a regular file;
  a symlink at **any** level (root, session dir, file) is rejected.
- Bulk zips are STORE-mode (no compression) and streamed from the opened fds; a
  ``manifest.json`` with per-file sha256 checksums is included.

``O_NOFOLLOW`` / ``O_DIRECTORY`` are absent on non-POSIX dev hosts; they are referenced
via ``getattr`` so the module imports on Windows for testing while still applying the
hardened flags on the Jetson runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Runtime default; overridable for tests / alternate hosts.
DEFAULT_CONVERSATIONS_DIR = "/var/lib/mungi/conversations"
CONVERSATIONS_DIR_ENV = "MUNGI_PORTAL_CONVERSATIONS_DIR"

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_WAV_RE = re.compile(r"^(?:input|output)_\d+\.wav$")
TRANSCRIPT_FILENAME = "conversation.jsonl"
MANIFEST_FILENAME = "manifest.json"

# POSIX openat flags (0 on platforms lacking them — Windows dev/test hosts).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

# True on POSIX (Jetson runtime + Linux CI): the hardened openat-style resolution with
# per-directory ``dir_fd=`` is TOCTOU-resistant. False on dev hosts lacking ``dir_fd``
# support (Windows), where we fall back to an ``lstat``-anchored path walk that enforces
# the SAME logical guards (per-component symlink rejection, regular-file check, no
# separators) but without kernel-level TOCTOU hardness. The Jetson and CI always take the
# ``dir_fd`` path, so the production guarantee is unchanged.
_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd

# KST is UTC+9; the device runs in Korea. Fixed offset avoids a tz database dependency.
KST_OFFSET_HOURS = 9
KST = timezone(timedelta(hours=KST_OFFSET_HOURS))

ZIP_STREAM_CHUNK = 64 * 1024


class PortalDataError(Exception):
    """Raised for path-safety violations and missing/invalid resources."""


def conversations_root() -> Path:
    """Return the canonical conversations root (env-overridable)."""
    return Path(os.environ.get(CONVERSATIONS_DIR_ENV, DEFAULT_CONVERSATIONS_DIR))


def is_safe_session_id(raw_id: str) -> bool:
    """Return ``True`` iff ``raw_id`` (already URL-decoded) is a safe session name."""
    return bool(SAFE_NAME_RE.fullmatch(raw_id))


def decode_and_validate_session_id(raw_id: str) -> str:
    """URL-decode ``raw_id`` then validate it as a safe session name.

    Args:
        raw_id: The raw (possibly percent-encoded) session id from the request path.

    Returns:
        The decoded, validated session id.

    Raises:
        PortalDataError: If the decoded id is not ``^[A-Za-z0-9_-]+$``.
    """
    decoded = unquote(raw_id)
    if not is_safe_session_id(decoded):
        raise PortalDataError(f"unsafe session id: {raw_id!r}")
    return decoded


def is_whitelisted_filename(name: str) -> bool:
    """Return ``True`` iff ``name`` is a servable file (transcript or numbered wav)."""
    return name == TRANSCRIPT_FILENAME or bool(SAFE_WAV_RE.fullmatch(name))


def _open_dir_fd(name: str, *, dir_fd: int | None) -> int:
    """Open a directory component with ``O_NOFOLLOW | O_DIRECTORY`` (openat-style).

    Args:
        name: A single path component (no separators).
        dir_fd: The parent directory fd, or ``None`` for an absolute open of the root.

    Returns:
        An open directory file descriptor.

    Raises:
        PortalDataError: If the component is unsafe, is a symlink, or is not a directory.
    """
    _reject_separators(name)
    flags = os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY | _O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise PortalDataError(f"cannot open directory component {name!r}: {exc}") from exc
    try:
        st = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        raise PortalDataError(f"fstat failed for {name!r}: {exc}") from exc
    if not stat.S_ISDIR(st.st_mode):
        os.close(fd)
        raise PortalDataError(f"component is not a directory: {name!r}")
    if stat.S_ISLNK(st.st_mode):  # defense-in-depth (O_NOFOLLOW already blocks this)
        os.close(fd)
        raise PortalDataError(f"component is a symlink: {name!r}")
    return fd


def _reject_separators(name: str) -> None:
    """Raise if ``name`` contains a path separator (defense for single components)."""
    if os.sep in name or (os.altsep and os.altsep in name):
        raise PortalDataError(f"path separator in component: {name!r}")


def _lstat_no_symlink(path: str, *, kind: str) -> os.stat_result:
    """``lstat`` ``path`` and reject symlinks (fallback-path symlink guard).

    Args:
        path: The absolute path of a single resolved component.
        kind: ``"dir"`` or ``"file"`` for the expected type check.

    Returns:
        The ``lstat`` result.

    Raises:
        PortalDataError: If the component is missing, a symlink, or the wrong type.
    """
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise PortalDataError(f"cannot lstat {path!r}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise PortalDataError(f"component is a symlink: {path!r}")
    if kind == "dir" and not stat.S_ISDIR(st.st_mode):
        raise PortalDataError(f"component is not a directory: {path!r}")
    if kind == "file" and not stat.S_ISREG(st.st_mode):
        raise PortalDataError(f"not a regular file: {path!r}")
    return st


@dataclass
class _DirAnchor:
    """A resolved directory: an fd (POSIX) or an absolute path (fallback)."""

    fd: int | None
    path: str

    def close(self) -> None:
        """Close the held fd, if any."""
        if self.fd is not None:
            os.close(self.fd)


def _open_root_anchor() -> _DirAnchor:
    """Resolve the canonical conversations root, rejecting a symlinked root."""
    root = conversations_root()
    if _SUPPORTS_DIR_FD:
        parent = str(root.parent)
        flags = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC
        try:
            parent_fd = os.open(parent, flags)
        except OSError as exc:
            raise PortalDataError(f"cannot open conversations parent: {exc}") from exc
        try:
            fd = _open_dir_fd(root.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        return _DirAnchor(fd=fd, path=str(root))
    _lstat_no_symlink(str(root), kind="dir")
    return _DirAnchor(fd=None, path=str(root))


def _open_child_dir_anchor(parent: _DirAnchor, name: str) -> _DirAnchor:
    """Resolve directory ``name`` under ``parent`` with per-component symlink rejection."""
    _reject_separators(name)
    if _SUPPORTS_DIR_FD:
        assert parent.fd is not None
        fd = _open_dir_fd(name, dir_fd=parent.fd)
        return _DirAnchor(fd=fd, path=os.path.join(parent.path, name))
    child_path = os.path.join(parent.path, name)
    _lstat_no_symlink(child_path, kind="dir")
    return _DirAnchor(fd=None, path=child_path)


def _open_regular_file_in(anchor: _DirAnchor, name: str) -> int:
    """Open whitelisted regular file ``name`` under ``anchor``, rejecting symlinks."""
    _reject_separators(name)
    flags = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    if _SUPPORTS_DIR_FD:
        assert anchor.fd is not None
        try:
            fd = os.open(name, flags, dir_fd=anchor.fd)
        except OSError as exc:
            raise PortalDataError(f"cannot open file {name!r}: {exc}") from exc
        try:
            st = os.fstat(fd)
        except OSError as exc:
            os.close(fd)
            raise PortalDataError(f"fstat failed for file {name!r}: {exc}") from exc
        if stat.S_ISLNK(st.st_mode):
            os.close(fd)
            raise PortalDataError(f"file is a symlink: {name!r}")
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise PortalDataError(f"not a regular file: {name!r}")
        return fd
    file_path = os.path.join(anchor.path, name)
    _lstat_no_symlink(file_path, kind="file")
    try:
        return os.open(file_path, os.O_RDONLY)
    except OSError as exc:
        raise PortalDataError(f"cannot open file {name!r}: {exc}") from exc


def _list_anchor_entries(anchor: _DirAnchor) -> list[str]:
    """Return the directory entries under ``anchor`` (fd-anchored on POSIX)."""
    if anchor.fd is not None:
        return os.listdir(anchor.fd)
    return os.listdir(anchor.path)


def open_session_file(session_id: str, filename: str) -> tuple[int, int]:
    """Resolve and open ``session_id/filename`` fd-anchored from the root.

    Every component is opened with ``O_NOFOLLOW`` and ``fstat``-checked (POSIX) or
    ``lstat``-checked (fallback); the final fd is guaranteed to be a regular file under the
    canonical root with no symlink traversal at any level.

    Args:
        session_id: A decoded, safe session id.
        filename: A whitelisted file name.

    Returns:
        ``(file_fd, size_bytes)``. The caller owns and must close ``file_fd``.

    Raises:
        PortalDataError: On any validation or path-safety failure.
    """
    if not is_safe_session_id(session_id):
        raise PortalDataError(f"unsafe session id: {session_id!r}")
    if not is_whitelisted_filename(filename):
        raise PortalDataError(f"non-whitelisted filename: {filename!r}")
    root = _open_root_anchor()
    try:
        session_anchor = _open_child_dir_anchor(root, session_id)
    finally:
        root.close()
    try:
        file_fd = _open_regular_file_in(session_anchor, filename)
    finally:
        session_anchor.close()
    size = os.fstat(file_fd).st_size
    return file_fd, size


def _kst_now_iso() -> str:
    """Return the current time as an ISO-8601 string in KST (UTC+9)."""
    return datetime.now(KST).isoformat()


@dataclass(frozen=True)
class SessionSummary:
    """Listing metadata for one conversation session."""

    session_id: str
    turn_count: int
    audio_count: int
    total_bytes: int
    modified_epoch: float


def _is_empty_session(summary: SessionSummary) -> bool:
    """Return True iff the session has no transcript turns and no audio."""
    return summary.turn_count == 0 and summary.audio_count == 0


def list_session_filenames(session_id: str) -> list[str]:
    """Return the sorted whitelisted file names present in ``session_id``.

    Resolution is fd-anchored: the session dir is opened with ``O_NOFOLLOW`` and entries
    are filtered through the whitelist. Non-whitelisted entries and symlinked entries are
    silently skipped (they are never served).

    Args:
        session_id: A decoded, safe session id.

    Returns:
        Sorted file names (``conversation.jsonl`` first when present, then wavs).

    Raises:
        PortalDataError: If the session id is unsafe or the dir cannot be opened.
    """
    if not is_safe_session_id(session_id):
        raise PortalDataError(f"unsafe session id: {session_id!r}")
    root = _open_root_anchor()
    try:
        session_anchor = _open_child_dir_anchor(root, session_id)
    finally:
        root.close()
    try:
        entries = _list_anchor_entries(session_anchor)
    finally:
        session_anchor.close()
    names = [name for name in entries if is_whitelisted_filename(name)]
    return sorted(names)


def summarize_session(session_id: str) -> SessionSummary:
    """Build a :class:`SessionSummary` for ``session_id`` (fd-anchored stat)."""
    if not is_safe_session_id(session_id):
        raise PortalDataError(f"unsafe session id: {session_id!r}")
    filenames = list_session_filenames(session_id)
    total_bytes = 0
    audio_count = 0
    turn_count = 0
    modified_epoch = 0.0
    for name in filenames:
        try:
            file_fd, size = open_session_file(session_id, name)
        except PortalDataError:
            continue
        try:
            st = os.fstat(file_fd)
            total_bytes += size
            modified_epoch = max(modified_epoch, st.st_mtime)
            if name == TRANSCRIPT_FILENAME:
                turn_count = _count_lines(file_fd)
            else:
                audio_count += 1
        finally:
            os.close(file_fd)
    return SessionSummary(
        session_id=session_id,
        turn_count=turn_count,
        audio_count=audio_count,
        total_bytes=total_bytes,
        modified_epoch=modified_epoch,
    )


def _count_lines(fd: int) -> int:
    """Count newline-terminated records in the open transcript ``fd``."""
    count = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, ZIP_STREAM_CHUNK)
        if not chunk:
            break
        count += chunk.count(b"\n")
    os.lseek(fd, 0, os.SEEK_SET)
    return count


def list_sessions() -> list[SessionSummary]:
    """Enumerate all safe session directories, newest first.

    Only entries matching ``^[A-Za-z0-9_-]+$`` that resolve to real directories (no
    symlinks) are listed.

    Returns:
        Session summaries sorted by modification time descending.
    """
    root = _open_root_anchor()
    try:
        entries = _list_anchor_entries(root)
    finally:
        root.close()
    summaries: list[SessionSummary] = []
    for name in entries:
        if not is_safe_session_id(name):
            continue
        try:
            summary = summarize_session(name)
        except PortalDataError:
            logger.warning("skipping unreadable session %r", name)
            continue
        if _is_empty_session(summary):
            logger.debug("Hiding empty session %r from listing", name)
            continue
        summaries.append(summary)
    summaries.sort(key=lambda s: s.modified_epoch, reverse=True)
    return summaries


def sha256_of_fd(fd: int) -> tuple[str, int]:
    """Return ``(hex_digest, byte_count)`` for the open file ``fd`` (rewinds after)."""
    digest = hashlib.sha256()
    total = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, ZIP_STREAM_CHUNK)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest(), total


@dataclass(frozen=True)
class ZipEntry:
    """One planned zip member: ``arcname`` (posix) resolved from ``session_id/filename``."""

    arcname: str
    session_id: str
    filename: str


def build_manifest(session_ids: list[str]) -> tuple[bytes, list[ZipEntry]]:
    """Build the ``manifest.json`` bytes and the ordered zip plan for ``session_ids``.

    Args:
        session_ids: The decoded, safe session ids to include.

    Returns:
        ``(manifest_bytes, entries)`` where ``entries`` is a list of :class:`ZipEntry`
        sorted by ``arcname`` (forward-slash posix paths). ``manifest_bytes`` is excluded
        from the manifest's own ``total_bytes``.
    """
    files_meta: list[dict[str, object]] = []
    entries: list[ZipEntry] = []
    total_bytes = 0
    for session_id in session_ids:
        for name in list_session_filenames(session_id):
            arcname = f"{session_id}/{name}"
            file_fd, _ = open_session_file(session_id, name)
            try:
                digest, size = sha256_of_fd(file_fd)
            finally:
                os.close(file_fd)
            files_meta.append({"path": arcname, "bytes": size, "sha256": digest})
            entries.append(ZipEntry(arcname=arcname, session_id=session_id, filename=name))
            total_bytes += size
    files_meta.sort(key=lambda meta: str(meta["path"]))
    entries.sort(key=lambda entry: entry.arcname)
    manifest = {
        "schema_version": 1,
        "session_id": session_ids[0] if len(session_ids) == 1 else "",
        "generated_at": _kst_now_iso(),
        "files": files_meta,
        "total_bytes": total_bytes,
        "file_count": len(files_meta),
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return manifest_bytes, entries


# --- Minimal streaming STORE-mode zip writer ---------------------------------
# We cannot use ``zipfile`` for streaming to a non-seekable HTTP body, so we emit the
# ZIP format directly in STORE mode (no compression) with data descriptors.

_ZIP_VERSION_NEEDED = 20
_GP_FLAG_DATA_DESCRIPTOR = 0x0008


def _dos_datetime(epoch: float) -> tuple[int, int]:
    """Return ``(dos_time, dos_date)`` for a POSIX ``epoch`` (local-naive, clamped)."""
    dt = datetime.fromtimestamp(max(epoch, 315532800.0))  # >= 1980-01-01
    dos_time = (dt.hour << 11) | (dt.minute << 5) | (dt.second // 2)
    dos_date = ((dt.year - 1980) << 9) | (dt.month << 5) | dt.day
    return dos_time, dos_date


def _zip_local_header(arcname: bytes, dos_time: int, dos_date: int) -> bytes:
    """Build a local file header with sizes deferred to a data descriptor."""
    return struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,  # local file header signature
        _ZIP_VERSION_NEEDED,
        _GP_FLAG_DATA_DESCRIPTOR,
        0,  # compression method = stored
        dos_time,
        dos_date,
        0,  # crc-32 (in data descriptor)
        0,  # compressed size (in data descriptor)
        0,  # uncompressed size (in data descriptor)
        len(arcname),
        0,  # extra field length
    )


def _zip_data_descriptor(crc: int, size: int) -> bytes:
    """Build a data descriptor (STORE: compressed size == uncompressed size)."""
    return struct.pack("<IIII", 0x08074B50, crc & 0xFFFFFFFF, size, size)


def _zip_central_header(
    arcname: bytes, crc: int, size: int, offset: int, dos_time: int, dos_date: int
) -> bytes:
    """Build a central-directory header for a stored entry."""
    return (
        struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,  # central file header signature
            _ZIP_VERSION_NEEDED,  # version made by
            _ZIP_VERSION_NEEDED,  # version needed
            _GP_FLAG_DATA_DESCRIPTOR,
            0,  # method = stored
            dos_time,
            dos_date,
            crc & 0xFFFFFFFF,
            size,  # compressed size
            size,  # uncompressed size
            len(arcname),
            0,  # extra len
            0,  # comment len
            0,  # disk number start
            0,  # internal attrs
            0,  # external attrs
            offset,
        )
        + arcname
    )


def _zip_eocd(count: int, cd_size: int, cd_offset: int) -> bytes:
    """Build the end-of-central-directory record."""
    return struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        count,
        count,
        cd_size,
        cd_offset,
        0,
    )


def stream_store_zip(
    manifest_bytes: bytes,
    plan: list[ZipEntry],
) -> Iterator[bytes]:
    """Yield a STORE-mode zip of ``plan`` + ``manifest.json`` as a byte stream.

    The stream is produced incrementally from opened fds (no whole-file buffering) and is
    suitable for HTTP/1.1 chunked transfer.

    Args:
        manifest_bytes: The serialized ``manifest.json`` content.
        plan: :class:`ZipEntry` items, sorted by arcname.

    Yields:
        Successive byte chunks comprising the complete zip archive.
    """
    central: list[bytes] = []
    offset = 0

    # Emit manifest.json first (deterministic, small).
    manifest_arc = MANIFEST_FILENAME.encode("utf-8")
    dos_time, dos_date = _dos_datetime(datetime.now().timestamp())
    crc = zlib.crc32(manifest_bytes) & 0xFFFFFFFF
    header = _zip_local_header(manifest_arc, dos_time, dos_date)
    yield header
    yield manifest_arc
    yield manifest_bytes
    yield _zip_data_descriptor(crc, len(manifest_bytes))
    central.append(
        _zip_central_header(manifest_arc, crc, len(manifest_bytes), offset, dos_time, dos_date)
    )
    offset += len(header) + len(manifest_arc) + len(manifest_bytes) + 16

    for entry in plan:
        file_fd, _ = open_session_file(entry.session_id, entry.filename)
        try:
            st = os.fstat(file_fd)
            dos_time, dos_date = _dos_datetime(st.st_mtime)
            arc_bytes = entry.arcname.encode("utf-8")
            header = _zip_local_header(arc_bytes, dos_time, dos_date)
            yield header
            yield arc_bytes
            running_crc = 0
            written = 0
            os.lseek(file_fd, 0, os.SEEK_SET)
            while True:
                chunk = os.read(file_fd, ZIP_STREAM_CHUNK)
                if not chunk:
                    break
                running_crc = zlib.crc32(chunk, running_crc)
                written += len(chunk)
                yield chunk
            running_crc &= 0xFFFFFFFF
            yield _zip_data_descriptor(running_crc, written)
            central.append(
                _zip_central_header(arc_bytes, running_crc, written, offset, dos_time, dos_date)
            )
            offset += len(header) + len(arc_bytes) + written + 16
        finally:
            os.close(file_fd)

    cd_offset = offset
    cd_size = 0
    for central_record in central:
        yield central_record
        cd_size += len(central_record)
    yield _zip_eocd(len(central), cd_size, cd_offset)


def read_transcript_bytes(session_id: str) -> bytes:
    """Read the full ``conversation.jsonl`` for ``session_id`` (fd-anchored).

    Args:
        session_id: A decoded, safe session id.

    Returns:
        The transcript file content.

    Raises:
        PortalDataError: If the session id is unsafe or the transcript is missing.
    """
    file_fd, _ = open_session_file(session_id, TRANSCRIPT_FILENAME)
    try:
        os.lseek(file_fd, 0, os.SEEK_SET)
        parts: list[bytes] = []
        while True:
            chunk = os.read(file_fd, ZIP_STREAM_CHUNK)
            if not chunk:
                break
            parts.append(chunk)
        return b"".join(parts)
    finally:
        os.close(file_fd)


def iter_file_chunks(fd: int, *, chunk_size: int = ZIP_STREAM_CHUNK) -> Iterator[bytes]:
    """Yield ``chunk_size`` byte blocks from the open file ``fd`` (caller closes ``fd``)."""
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        yield chunk


def write_stream_to(out: IO[bytes], stream: Iterator[bytes]) -> int:
    """Write ``stream`` to ``out`` (for tests that materialize a zip in memory)."""
    total = 0
    for chunk in stream:
        out.write(chunk)
        total += len(chunk)
    return total
