"""Security tests for session enumeration and fd-anchored path safety.

Covers §9 ``data``: safe-name ``fullmatch`` (post URL-decode), ``..``/abs/symlink-at-each-
level/non-regular rejection, bulk-id validation, and the streamed STORE-zip (chunked
framing parses, manifest checksums match). Symlink tests require OS symlink support; they
skip on platforms (Windows without privilege) that cannot create symlinks. The Jetson
runtime and Linux CI always exercise them.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from parental.download_portal import data


@pytest.fixture
def conversations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated conversations root with two sessions."""
    root = tmp_path / "conversations"
    root.mkdir()
    monkeypatch.setenv(data.CONVERSATIONS_DIR_ENV, str(root))

    sess_a = root / "2026-06-18_10-00-00"
    sess_a.mkdir()
    # Write bytes (not text) so newlines are not platform-translated; the byte counts
    # and checksums must be identical on Windows dev and Linux CI.
    (sess_a / "conversation.jsonl").write_bytes(b'{"turn":1}\n{"turn":2}\n{"turn":3}\n')
    (sess_a / "input_001.wav").write_bytes(b"RIFF\x00\x00input")
    (sess_a / "output_001.wav").write_bytes(b"RIFF\x00\x00output")
    # A non-whitelisted file must never be served or listed.
    (sess_a / "secret.txt").write_bytes(b"topsecret")

    sess_b = root / "qwen3_mix_007"
    sess_b.mkdir()
    (sess_b / "conversation.jsonl").write_bytes(b'{"turn":1}\n')
    return root


def _can_symlink(tmp_path: Path) -> bool:
    """Return ``True`` iff this platform/user can create symlinks."""
    probe_target = tmp_path / "_probe_target"
    probe_target.write_text("x", encoding="utf-8")
    probe_link = tmp_path / "_probe_link"
    try:
        os.symlink(probe_target, probe_link)
    except (OSError, NotImplementedError):
        return False
    finally:
        if probe_link.exists() or probe_link.is_symlink():
            probe_link.unlink()
        probe_target.unlink()
    return True


# --- Safe-name validation ----------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["2026-06-18_10-00-00", "qwen3_mix_007", "abc123", "A-B_c-9"],
)
def test_safe_session_ids_accepted(name: str) -> None:
    assert data.is_safe_session_id(name) is True


@pytest.mark.parametrize(
    "name",
    ["../etc", "a/b", "a.b", "a b", ".", "..", "a;b", "a$b", "", "café", "a\x00b"],
)
def test_unsafe_session_ids_rejected(name: str) -> None:
    assert data.is_safe_session_id(name) is False


def test_decode_then_validate_rejects_encoded_traversal() -> None:
    with pytest.raises(data.PortalDataError):
        data.decode_and_validate_session_id("..%2F..%2Fetc")
    with pytest.raises(data.PortalDataError):
        data.decode_and_validate_session_id("%2e%2e")  # ".."
    with pytest.raises(data.PortalDataError):
        data.decode_and_validate_session_id("a%2Fb")  # "a/b"


def test_decode_then_validate_accepts_safe_encoded() -> None:
    # Percent-encoding of safe characters still decodes to a safe name.
    assert data.decode_and_validate_session_id("qwen3%5Fmix") == "qwen3_mix"


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("conversation.jsonl", True),
        ("input_001.wav", True),
        ("output_123.wav", True),
        ("input_.wav", False),
        ("output.wav", False),
        ("secret.txt", False),
        ("conversation.jsonl.bak", False),
        ("../x", False),
    ],
)
def test_filename_whitelist(name: str, ok: bool) -> None:
    assert data.is_whitelisted_filename(name) is ok


# --- Enumeration -------------------------------------------------------------
def test_list_sessions_lists_both_naming_styles(conversations: Path) -> None:
    sessions = data.list_sessions()
    ids = {s.session_id for s in sessions}
    assert ids == {"2026-06-18_10-00-00", "qwen3_mix_007"}


def test_list_sessions_hides_empty_and_metadata_only_sessions(conversations: Path) -> None:
    (conversations / "empty_session").mkdir()
    metadata_only = conversations / "metadata_only"
    metadata_only.mkdir()
    (metadata_only / "session_end.json").write_bytes(b'{"ended":true}\n')

    sessions = data.list_sessions()
    ids = {s.session_id for s in sessions}

    assert "empty_session" not in ids
    assert "metadata_only" not in ids
    assert ids == {"2026-06-18_10-00-00", "qwen3_mix_007"}


def test_list_sessions_keeps_audio_only_and_transcript_only_sessions(
    conversations: Path,
) -> None:
    audio_only = conversations / "audio_only"
    audio_only.mkdir()
    (audio_only / "input_001.wav").write_bytes(b"RIFF\x00\x00input")

    transcript_only = conversations / "transcript_only"
    transcript_only.mkdir()
    (transcript_only / "conversation.jsonl").write_bytes(b'{"turn":1}\n')

    summaries = {s.session_id: s for s in data.list_sessions()}

    assert summaries["audio_only"].audio_count == 1
    assert summaries["audio_only"].turn_count == 0
    assert summaries["transcript_only"].audio_count == 0
    assert summaries["transcript_only"].turn_count == 1


def test_summary_counts_turns_and_audio_excludes_non_whitelisted(
    conversations: Path,
) -> None:
    summary = data.summarize_session("2026-06-18_10-00-00")
    assert summary.turn_count == 3
    assert summary.audio_count == 2
    # secret.txt is excluded from the byte total.
    expected = (
        len(b'{"turn":1}\n{"turn":2}\n{"turn":3}\n')
        + len(b"RIFF\x00\x00input")
        + len(b"RIFF\x00\x00output")
    )
    assert summary.total_bytes == expected


def test_list_session_filenames_excludes_non_whitelisted(conversations: Path) -> None:
    names = data.list_session_filenames("2026-06-18_10-00-00")
    assert "secret.txt" not in names
    assert set(names) == {"conversation.jsonl", "input_001.wav", "output_001.wav"}


# --- Path-safety rejections --------------------------------------------------
def test_open_rejects_traversal_session_id(conversations: Path) -> None:
    with pytest.raises(data.PortalDataError):
        data.open_session_file("..", "conversation.jsonl")


def test_open_rejects_absolute_session_id(conversations: Path) -> None:
    with pytest.raises(data.PortalDataError):
        data.open_session_file("/etc", "passwd")


def test_open_rejects_non_whitelisted_file(conversations: Path) -> None:
    with pytest.raises(data.PortalDataError):
        data.open_session_file("2026-06-18_10-00-00", "secret.txt")


def test_open_rejects_separator_in_filename(conversations: Path) -> None:
    with pytest.raises(data.PortalDataError):
        data.open_session_file("2026-06-18_10-00-00", "../secret.txt")


def test_open_valid_file_succeeds(conversations: Path) -> None:
    fd, size = data.open_session_file("2026-06-18_10-00-00", "conversation.jsonl")
    try:
        assert size > 0
    finally:
        os.close(fd)


def test_missing_session_rejected(conversations: Path) -> None:
    with pytest.raises(data.PortalDataError):
        data.open_session_file("no_such_session", "conversation.jsonl")


# --- Symlink rejection at each level (skips where symlinks are unavailable) ---
def test_symlinked_session_dir_rejected(conversations: Path, tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("platform cannot create symlinks")
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "conversation.jsonl").write_text("leak", encoding="utf-8")
    link = conversations / "linked_session"
    os.symlink(outside, link, target_is_directory=True)
    with pytest.raises(data.PortalDataError):
        data.open_session_file("linked_session", "conversation.jsonl")


def test_symlinked_file_rejected(conversations: Path, tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("platform cannot create symlinks")
    secret = tmp_path / "etc_passwd"
    secret.write_text("root:x:0:0", encoding="utf-8")
    sess = conversations / "2026-06-18_10-00-00"
    link = sess / "output_999.wav"  # whitelisted name pointing outside
    os.symlink(secret, link)
    with pytest.raises(data.PortalDataError):
        data.open_session_file("2026-06-18_10-00-00", "output_999.wav")


def test_symlinked_root_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("platform cannot create symlinks")
    real_root = tmp_path / "real_conversations"
    real_root.mkdir()
    (real_root / "s1").mkdir()
    (real_root / "s1" / "conversation.jsonl").write_text("x", encoding="utf-8")
    linked_root = tmp_path / "linked_conversations"
    os.symlink(real_root, linked_root, target_is_directory=True)
    monkeypatch.setenv(data.CONVERSATIONS_DIR_ENV, str(linked_root))
    with pytest.raises(data.PortalDataError):
        data.open_session_file("s1", "conversation.jsonl")


# --- Non-regular file rejection ----------------------------------------------
def test_directory_named_like_file_rejected(
    conversations: Path,
) -> None:
    sess = conversations / "2026-06-18_10-00-00"
    (sess / "input_555.wav").mkdir()  # a directory wearing a whitelisted name
    with pytest.raises(data.PortalDataError):
        data.open_session_file("2026-06-18_10-00-00", "input_555.wav")


# --- Manifest + streamed STORE-zip -------------------------------------------
def test_build_manifest_checksums_and_sizes(conversations: Path) -> None:
    manifest_bytes, entries = data.build_manifest(["2026-06-18_10-00-00"])
    manifest = json.loads(manifest_bytes)
    assert manifest["schema_version"] == 1
    assert manifest["session_id"] == "2026-06-18_10-00-00"
    assert manifest["file_count"] == 3
    # files sorted by path ascending.
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(paths)
    # checksums match the real file content.
    sess = conversations / "2026-06-18_10-00-00"
    for meta in manifest["files"]:
        name = meta["path"].split("/", 1)[1]
        raw = (sess / name).read_bytes()
        assert meta["bytes"] == len(raw)
        assert meta["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["total_bytes"] == sum(f["bytes"] for f in manifest["files"])


def test_streamed_store_zip_parses_and_roundtrips(conversations: Path) -> None:
    manifest_bytes, entries = data.build_manifest(["2026-06-18_10-00-00", "qwen3_mix_007"])
    buffer = io.BytesIO()
    data.write_stream_to(buffer, data.stream_store_zip(manifest_bytes, entries))
    buffer.seek(0)
    archive = zipfile.ZipFile(buffer)
    # STORE mode: no compression.
    for info in archive.infolist():
        assert info.compress_type == zipfile.ZIP_STORED
    assert archive.testzip() is None  # all CRCs valid
    names = set(archive.namelist())
    assert "manifest.json" in names
    assert "2026-06-18_10-00-00/conversation.jsonl" in names
    assert "qwen3_mix_007/conversation.jsonl" in names
    # content roundtrip.
    sess = conversations / "2026-06-18_10-00-00"
    assert (
        archive.read("2026-06-18_10-00-00/conversation.jsonl")
        == (sess / "conversation.jsonl").read_bytes()
    )
    # manifest embedded in the zip matches.
    embedded = json.loads(archive.read("manifest.json"))
    assert embedded["file_count"] == 4  # 3 + 1


def test_bulk_manifest_multi_session_id_is_blank(conversations: Path) -> None:
    manifest_bytes, _ = data.build_manifest(["2026-06-18_10-00-00", "qwen3_mix_007"])
    manifest = json.loads(manifest_bytes)
    assert manifest["session_id"] == ""  # blank for multi-session bundles


def test_read_transcript_bytes(conversations: Path) -> None:
    body = data.read_transcript_bytes("qwen3_mix_007")
    assert body == b'{"turn":1}\n'


def test_read_transcript_missing_rejected(conversations: Path) -> None:
    with pytest.raises(data.PortalDataError):
        data.read_transcript_bytes("no_such_session")
