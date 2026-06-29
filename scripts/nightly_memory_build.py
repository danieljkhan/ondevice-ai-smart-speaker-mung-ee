"""Build conversation-memory v0 artifacts from persisted conversations.

The job is flag-gated by ``MUNGI_CONV_MEMORY`` and publishes immutable
generations under ``<mutable_root>/conversation_memory``. v0 builds only the raw
turn layer and keyword index; the v1 embedding/cluster/summary stage is a
future extension point after the raw layer is written.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

from core.conversation_memory import (
    content_tokens,
    conversation_memory_root,
    normalize,
    strip_particle,
)
from core.conversation_memory_schema import (
    CONVERSATION_MEMORY_ENV_FLAG,
    GENERATION_POINTER_FILENAME,
    KST,
    RAW_TURN_RETENTION_DAYS,
    IndexReference,
    ManifestEntry,
    SchemaError,
    TurnSnippet,
    format_generation_pointer,
    is_crisis_turn,
    is_quarantined_turn,
    parse_generation_pointer,
    parse_turn_json_line,
)
from core.runtime import detect_runtime_paths

logger = logging.getLogger("mungi.scripts.nightly_memory_build")

_CONVERSATION_FILENAME: Final[str] = "conversation.jsonl"
_MANIFEST_FILENAME: Final[str] = "manifest.json"
_TURNS_FILENAME: Final[str] = "turns.jsonl"
_INDEX_FILENAME: Final[str] = "index.json"
_QUARANTINED_DAYS_FILENAME: Final[str] = "quarantined_days.json"
_QUIESCENCE_MINUTES: Final[int] = 10
_DEFAULT_DEFER_ATTEMPTS: Final[int] = 3
# Worst-case defer wall-time is (_DEFAULT_DEFER_ATTEMPTS - 1) * _DEFAULT_DEFER_SLEEP_S
# = 2 * 300 = 600s (10 min), which must stay well under the systemd
# ``TimeoutStartSec=1800`` (30 min) so the build retains >=20 min of headroom.
_DEFAULT_DEFER_SLEEP_S: Final[float] = 300.0
_KEEP_GENERATIONS: Final[int] = 2
_DIR_MODE: Final[int] = 0o700
_FILE_MODE: Final[int] = 0o600


@dataclass(frozen=True)
class SessionIngestResult:
    """Result of parsing one ``conversation.jsonl`` session file."""

    snippets: tuple[TurnSnippet, ...]
    manifest_entry: ManifestEntry
    quarantined_days: frozenset[str]
    turns_seen: int
    turns_dropped: int
    session_quarantined: bool


@dataclass(frozen=True)
class BuildResult:
    """Summary of one nightly generation build."""

    generation_id: str
    sessions_seen: int
    sessions_quarantined: int
    turns_seen: int
    turns_indexed: int
    turns_dropped: int
    skipped_active_session: bool


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the nightly memory builder."""

    enabled = os.getenv(CONVERSATION_MEMORY_ENV_FLAG, "").strip() == "1"
    _configure_logging()
    if not enabled:
        logger.info("conversation_memory_disabled flag=%s exit=0", CONVERSATION_MEMORY_ENV_FLAG)
        return 0

    args = _parse_args(argv)
    mutable_root = args.mutable_root or Path(detect_runtime_paths().mutable_root)
    result = run_build(
        mutable_root=mutable_root,
        now=_parse_now(args.now),
        defer_attempts=args.defer_attempts,
        defer_sleep_s=args.defer_sleep_s,
    )
    logger.info(
        "conversation_memory_build_complete generation=%s sessions=%d quarantined=%d "
        "turns_seen=%d turns_indexed=%d turns_dropped=%d active_skip=%s",
        result.generation_id,
        result.sessions_seen,
        result.sessions_quarantined,
        result.turns_seen,
        result.turns_indexed,
        result.turns_dropped,
        result.skipped_active_session,
    )
    return 0


def run_build(
    *,
    mutable_root: Path,
    now: datetime | None = None,
    defer_attempts: int = _DEFAULT_DEFER_ATTEMPTS,
    defer_sleep_s: float = _DEFAULT_DEFER_SLEEP_S,
) -> BuildResult:
    """Build and publish a new generation if no active session is in progress."""

    anchor = _normalize_kst(now or _clock_kst())
    conversation_root = mutable_root / "conversations"
    memory_root = conversation_memory_root(mutable_root)

    if _defer_for_active_session(
        conversation_root,
        now=anchor,
        attempts=defer_attempts,
        sleep_s=defer_sleep_s,
        clock=_clock_kst,
    ):
        logger.info("conversation_memory_build_skipped reason=active_session")
        return BuildResult(
            generation_id="",
            sessions_seen=0,
            sessions_quarantined=0,
            turns_seen=0,
            turns_indexed=0,
            turns_dropped=0,
            skipped_active_session=True,
        )

    generation_id = _unique_generation_id(memory_root, anchor)
    generation_dir = memory_root / "generations" / generation_id
    _ensure_private_dir(generation_dir)
    _fsync_directory(generation_dir.parent)
    previous_manifest = _load_current_manifest(memory_root)
    previous_snippets = _load_current_snippets(memory_root)

    snippets: list[TurnSnippet] = []
    manifest: dict[str, ManifestEntry] = {}
    quarantined_days: set[str] = set()
    sessions_seen = 0
    sessions_quarantined = 0
    turns_seen = 0
    turns_dropped = 0

    retention_start = anchor - timedelta(days=RAW_TURN_RETENTION_DAYS)
    for conversation_path in _iter_conversation_files(conversation_root):
        sessions_seen += 1
        session_dir = conversation_path.parent.name
        try:
            raw_bytes = _read_bytes_retry(conversation_path)
        except OSError as exc:
            logger.warning(
                "conversation_memory_session_skipped path=%s error=read_failed detail=%s",
                conversation_path,
                exc,
            )
            continue
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        previous_entry = previous_manifest.get(session_dir)
        reusable_snippets = previous_snippets.get((session_dir, source_hash), ())
        if (
            previous_entry is not None
            and previous_entry.sha256 == source_hash
            and reusable_snippets
        ):
            retained_reusable_snippets = tuple(
                snippet for snippet in reusable_snippets if snippet.timestamp >= retention_start
            )
            manifest[session_dir] = ManifestEntry(
                session_dir=session_dir,
                sha256=source_hash,
                processed_at=anchor,
            )
            turns_dropped += len(reusable_snippets) - len(retained_reusable_snippets)
            snippets.extend(retained_reusable_snippets)
            continue

        try:
            result = ingest_session(
                conversation_path,
                processed_at=anchor,
                raw_bytes=raw_bytes,
            )
        except UnicodeDecodeError as exc:
            logger.warning(
                "conversation_memory_session_skipped path=%s error=utf8_decode_failed detail=%s",
                conversation_path,
                exc,
            )
            continue
        manifest[session_dir] = result.manifest_entry
        quarantined_days.update(result.quarantined_days)
        turns_seen += result.turns_seen
        turns_dropped += result.turns_dropped
        if result.session_quarantined:
            sessions_quarantined += 1
        for snippet in result.snippets:
            if snippet.timestamp >= retention_start:
                snippets.append(snippet)
            else:
                turns_dropped += 1

    snippets = _deduplicate_snippets(snippets)
    index = build_index(snippets)
    _write_turns(generation_dir / _TURNS_FILENAME, snippets)
    _write_index(generation_dir / _INDEX_FILENAME, index)
    _write_manifest(generation_dir / _MANIFEST_FILENAME, manifest)
    _write_quarantined_days(generation_dir / _QUARANTINED_DAYS_FILENAME, quarantined_days)
    _publish_pointer(memory_root, generation_id)
    _gc_generations(memory_root / "generations", keep=_KEEP_GENERATIONS)

    return BuildResult(
        generation_id=generation_id,
        sessions_seen=sessions_seen,
        sessions_quarantined=sessions_quarantined,
        turns_seen=turns_seen,
        turns_indexed=len(snippets),
        turns_dropped=turns_dropped,
        skipped_active_session=False,
    )


def ingest_session(
    conversation_path: Path,
    *,
    processed_at: datetime,
    raw_bytes: bytes | None = None,
) -> SessionIngestResult:
    """Parse one conversation session and apply turn/session quarantine."""

    if raw_bytes is None:
        raw_bytes = _read_bytes_retry(conversation_path)
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    session_dir = conversation_path.parent.name
    manifest_entry = ManifestEntry(
        session_dir=session_dir,
        sha256=source_hash,
        processed_at=processed_at,
    )

    records = []
    for line in raw_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(parse_turn_json_line(line))
        except SchemaError as exc:
            logger.warning(
                "conversation_memory_row_skipped path=%s error=%s",
                conversation_path,
                exc,
            )

    turns_seen = len(records)
    crisis_days = frozenset(
        record.timestamp.date().isoformat() for record in records if is_crisis_turn(record)
    )
    if crisis_days:
        return SessionIngestResult(
            snippets=(),
            manifest_entry=manifest_entry,
            quarantined_days=crisis_days,
            turns_seen=turns_seen,
            turns_dropped=turns_seen,
            session_quarantined=True,
        )

    snippets: list[TurnSnippet] = []
    turns_dropped = 0
    for record in records:
        if is_quarantined_turn(record) or not record.user_text.strip():
            turns_dropped += 1
            continue
        snippets.append(
            TurnSnippet(
                id=_snippet_id(source_hash, record.turn),
                session_dir=session_dir,
                turn=record.turn,
                text=record.user_text.strip(),
                timestamp=record.timestamp,
                source_hash=source_hash,
            )
        )

    return SessionIngestResult(
        snippets=tuple(snippets),
        manifest_entry=manifest_entry,
        quarantined_days=frozenset(),
        turns_seen=turns_seen,
        turns_dropped=turns_dropped,
        session_quarantined=False,
    )


def build_index(snippets: Iterable[TurnSnippet]) -> dict[str, tuple[IndexReference, ...]]:
    """Build the normalized keyword index for raw-turn snippets."""

    index: dict[str, dict[IndexReference, None]] = {}
    for snippet in snippets:
        reference = IndexReference(layer="turns", id=snippet.id)
        for token in content_tokens(snippet.text):
            for keyword in _index_keywords(token):
                index.setdefault(keyword, {})[reference] = None
    return {keyword: tuple(refs) for keyword, refs in sorted(index.items())}


def _index_keywords(token: str) -> tuple[str, ...]:
    normalized = normalize(token)
    stripped = strip_particle(normalized)
    return tuple(sorted({item for item in (normalized, stripped) if len(item) >= 2}))


def _defer_for_active_session(
    conversation_root: Path,
    *,
    now: datetime,
    attempts: int,
    sleep_s: float,
    clock: Callable[[], datetime],
) -> bool:
    anchor = now
    for attempt in range(max(1, attempts)):
        active = _newest_session_is_active(conversation_root, now=anchor)
        if not active:
            return False
        logger.info(
            "conversation_memory_active_session attempt=%d attempts=%d",
            attempt + 1,
            attempts,
        )
        if attempt + 1 < attempts:
            time.sleep(max(0.0, sleep_s))
            anchor = _normalize_kst(clock())
    return True


def _newest_session_is_active(conversation_root: Path, *, now: datetime) -> bool:
    session_dirs = [path for path in conversation_root.glob("*") if path.is_dir()]
    if not session_dirs:
        return False
    newest = max(session_dirs, key=lambda item: item.stat().st_mtime)
    if (newest / "session_end.json").exists():
        return False
    conversation_path = newest / _CONVERSATION_FILENAME
    if not conversation_path.exists():
        return True
    mtime = datetime.fromtimestamp(conversation_path.stat().st_mtime, tz=KST)
    return now - mtime < timedelta(minutes=_QUIESCENCE_MINUTES)


def _iter_conversation_files(conversation_root: Path) -> tuple[Path, ...]:
    if not conversation_root.exists():
        return ()
    return tuple(sorted(conversation_root.glob(f"*/{_CONVERSATION_FILENAME}")))


def _deduplicate_snippets(snippets: Iterable[TurnSnippet]) -> list[TurnSnippet]:
    seen: dict[tuple[str, int, str], TurnSnippet] = {}
    for snippet in snippets:
        key = (snippet.source_hash, snippet.turn, normalize(snippet.text))
        seen.setdefault(key, snippet)
    return sorted(seen.values(), key=lambda item: (item.timestamp, item.session_dir, item.turn))


def _write_turns(path: Path, snippets: Sequence[TurnSnippet]) -> None:
    payload = "".join(
        json.dumps(snippet.to_json_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        for snippet in snippets
    )
    _atomic_write_text(path, payload)


def _write_index(path: Path, index: Mapping[str, tuple[IndexReference, ...]]) -> None:
    payload = {
        keyword: [reference.to_json_dict() for reference in references]
        for keyword, references in index.items()
    }
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    )


def _write_manifest(path: Path, manifest: Mapping[str, ManifestEntry]) -> None:
    payload = {
        key: {
            "sha256": entry.sha256,
            "processed_at": _format_kst(entry.processed_at),
        }
        for key, entry in sorted(manifest.items())
    }
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    )


def _write_quarantined_days(path: Path, quarantined_days: Iterable[str]) -> None:
    payload = {"quarantined_days": sorted(set(quarantined_days))}
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    )


def _publish_pointer(memory_root: Path, generation_id: str) -> None:
    _ensure_private_dir(memory_root)
    _cleanup_pointer_temps(memory_root)
    _atomic_write_text(
        memory_root / GENERATION_POINTER_FILENAME,
        format_generation_pointer(generation_id),
    )


def _gc_generations(generations_root: Path, *, keep: int) -> None:
    if not generations_root.exists():
        return
    generations = sorted(
        (path for path in generations_root.iterdir() if path.is_dir()), reverse=True
    )
    for generation in generations[keep:]:
        resolved_root = generations_root.resolve()
        resolved_generation = generation.resolve()
        if resolved_root not in resolved_generation.parents:
            logger.warning("conversation_memory_gc_skip_outside_root path=%s", generation)
            continue
        shutil.rmtree(generation)
        _fsync_directory(generations_root)


def _unique_generation_id(memory_root: Path, now: datetime) -> str:
    base = _normalize_kst(now).strftime("%Y%m%dT%H%M%SKST")
    generation_root = memory_root / "generations"
    candidate = base
    suffix = 1
    while (generation_root / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def _snippet_id(source_hash: str, turn: int) -> str:
    return f"turn-{source_hash[:16]}-{turn:04d}"


def _read_bytes_retry(path: Path, *, attempts: int = 3, sleep_s: float = 0.05) -> bytes:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(sleep_s)
    if last_error is not None:
        raise last_error
    return b""


def _atomic_write_text(path: Path, content: str) -> None:
    _ensure_private_dir(path.parent)
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(_FILE_MODE)
        temp_path.replace(path)
        with contextlib.suppress(OSError):
            path.chmod(_FILE_MODE)
        _fsync_directory(path.parent)
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(_DIR_MODE)


def _cleanup_pointer_temps(memory_root: Path) -> None:
    for temp_path in memory_root.glob(f"{GENERATION_POINTER_FILENAME}.tmp-*"):
        with contextlib.suppress(OSError):
            temp_path.unlink()
    _fsync_directory(memory_root)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutable-root", type=Path, default=None)
    parser.add_argument("--now", default=None, help="ISO KST anchor time for tests")
    parser.add_argument("--defer-attempts", type=int, default=_DEFAULT_DEFER_ATTEMPTS)
    parser.add_argument("--defer-sleep-s", type=float, default=_DEFAULT_DEFER_SLEEP_S)
    return parser.parse_args(argv)


def _parse_now(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return _normalize_kst(datetime.fromisoformat(raw))


def _clock_kst() -> datetime:
    return datetime.now(KST)


def _normalize_kst(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


def _format_kst(value: datetime) -> str:
    return _normalize_kst(value).replace(microsecond=0).isoformat(timespec="seconds")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_current_manifest(memory_root: Path) -> dict[str, ManifestEntry]:
    """Load the current manifest for diagnostics and future incremental seams."""

    try:
        generation_id = parse_generation_pointer(
            (memory_root / GENERATION_POINTER_FILENAME).read_text(encoding="utf-8")
        )
        manifest_path = memory_root / "generations" / generation_id / _MANIFEST_FILENAME
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, SchemaError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    manifest: dict[str, ManifestEntry] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, Mapping):
            try:
                manifest[key] = ManifestEntry.from_json_dict({key: dict(value)})
            except SchemaError:
                continue
    return manifest


def _load_current_snippets(memory_root: Path) -> dict[tuple[str, str], tuple[TurnSnippet, ...]]:
    """Load reusable raw snippets keyed by ``(session_dir, source_hash)``."""

    try:
        generation_id = parse_generation_pointer(
            (memory_root / GENERATION_POINTER_FILENAME).read_text(encoding="utf-8")
        )
        turns_path = memory_root / "generations" / generation_id / _TURNS_FILENAME
        lines = turns_path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError, SchemaError):
        return {}

    grouped: dict[tuple[str, str], list[TurnSnippet]] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                continue
            snippet = TurnSnippet.from_json_dict(raw)
        except (json.JSONDecodeError, SchemaError):
            continue
        grouped.setdefault((snippet.session_dir, snippet.source_hash), []).append(snippet)
    return {key: tuple(value) for key, value in grouped.items()}


if __name__ == "__main__":
    sys.exit(main())
