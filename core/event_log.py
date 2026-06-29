"""Append-only JSONL event log with synchronous best-effort writes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_EVENT_LINE_BYTES = 4096
SIZE_WARNING_BYTES = 100_000_000


class EventLog:
    """Single-process append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        """Create an event log for ``path`` without opening it permanently."""
        self.path = path
        self._memory_fallback: list[str] = []
        self._warned_disk_fallback = False
        try:
            if path.exists() and path.stat().st_size >= SIZE_WARNING_BYTES:
                logger.warning("Event log %s is at or above 100 MB", path)
        except OSError as exc:
            logger.warning("Could not inspect event log size for %s: %s", path, exc)

    @property
    def pending_fallback_count(self) -> int:
        """Return the number of event lines waiting for disk retry."""
        return len(self._memory_fallback)

    def append(self, payload: dict[str, Any]) -> None:
        """Append one JSON payload as a line, falling back to memory on OSError."""
        line = self._serialize(payload)
        pending = [*self._memory_fallback, line]
        self._memory_fallback = []
        try:
            self._append_lines(pending)
        except OSError as exc:
            self._memory_fallback.extend(pending)
            if not self._warned_disk_fallback:
                logger.warning("Event log write failed; using memory fallback: %s", exc)
                self._warned_disk_fallback = True

    def read_entries(self) -> list[dict[str, Any]]:
        """Read valid JSONL entries, skipping a malformed trailing partial line."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        lines = text.splitlines()
        entries: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if idx == len(lines) - 1:
                    logger.warning("Skipping malformed trailing event log line in %s", self.path)
                    break
                raise
            if isinstance(value, dict):
                entries.append(value)
        return entries

    def _serialize(self, payload: dict[str, Any]) -> str:
        """Serialize an event payload and enforce the atomic-line budget."""
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        if len(line.encode("utf-8")) >= MAX_EVENT_LINE_BYTES:
            msg = f"Event log line must be smaller than {MAX_EVENT_LINE_BYTES} bytes"
            raise ValueError(msg)
        return line

    def _append_lines(self, lines: list[str]) -> None:
        """Append pre-serialized lines using O_APPEND and fsync."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            for line in lines:
                os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
