"""Tests for append-only JSONL event logging."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from core import event_log
from core.event_log import MAX_EVENT_LINE_BYTES, SIZE_WARNING_BYTES, EventLog


def test_append_creates_parent_and_writes_jsonl(tmp_path: Path, monkeypatch: Any) -> None:
    """append() creates parent directories, writes one line, and fsyncs."""
    fsync_calls: list[int] = []
    monkeypatch.setattr(event_log.os, "fsync", lambda fd: fsync_calls.append(fd))
    path = tmp_path / "nested" / "events.jsonl"
    log = EventLog(path)

    log.append({"event": "parent_mode_requested", "schema_version": 1})

    assert fsync_calls
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "parent_mode_requested"


def test_line_size_limit_is_enforced(tmp_path: Path) -> None:
    """Atomic event lines must remain below 4096 bytes."""
    log = EventLog(tmp_path / "events.jsonl")
    assert MAX_EVENT_LINE_BYTES == 4096
    with pytest.raises(ValueError, match="4096"):
        log.append({"event": "x", "payload": "a" * MAX_EVENT_LINE_BYTES})


def test_read_entries_skips_malformed_trailing_line(tmp_path: Path) -> None:
    """A partial final line is ignored during reads."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"event":"ok"}\n{"event":', encoding="utf-8")

    assert EventLog(path).read_entries() == [{"event": "ok"}]


def test_size_warning_at_init(tmp_path: Path, caplog: Any) -> None:
    """A log already at 100 MB emits one warning when EventLog is created."""
    path = tmp_path / "events.jsonl"
    with path.open("wb") as handle:
        handle.seek(SIZE_WARNING_BYTES)
        handle.write(b"\0")

    with caplog.at_level("WARNING", logger="core.event_log"):
        EventLog(path)

    assert SIZE_WARNING_BYTES == 100_000_000
    assert "100 MB" in caplog.text


def test_disk_full_uses_memory_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    """An OSError during append preserves events for a later retry."""
    path = tmp_path / "events.jsonl"
    log = EventLog(path)

    def fail_open(*args: Any, **kwargs: Any) -> int:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(event_log.os, "open", fail_open)

    log.append({"event": "queued"})

    assert log.pending_fallback_count == 1
    assert not path.exists()


def test_append_uses_o_append_flag(tmp_path: Path, monkeypatch: Any) -> None:
    """Writes use O_APPEND for single-process append-only behavior."""
    flags_seen: list[int] = []
    real_open = os.open

    def recording_open(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        flags_seen.append(flags)
        return real_open(path, flags, mode)

    monkeypatch.setattr(event_log.os, "open", recording_open)
    log = EventLog(tmp_path / "events.jsonl")

    log.append({"event": "ok"})

    assert flags_seen
    assert flags_seen[0] & os.O_APPEND
