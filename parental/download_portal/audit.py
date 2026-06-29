"""Append-only audit log for downloads and failed authentications.

Security model (see ``Dev_Plan/2026-06-18-conversation-download-portal-plan.md`` §6):

- The audit log lives at ``/var/lib/mungi/logs/portal-audit.log`` with mode ``0600``.
- Initialization is **fail-closed**: if the log directory cannot be created or the file
  cannot be opened for append, :class:`AuditLog` raises and the daemon refuses to start.
- One structured line is written per download and per failed auth (timestamp, peer IP,
  session ids, file count, byte count, outcome). No conversation content is logged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from .data import KST

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/var/lib/mungi/logs"
LOG_DIR_ENV = "MUNGI_PORTAL_LOG_DIR"
AUDIT_FILENAME = "portal-audit.log"
AUDIT_FILE_MODE = 0o600


def log_dir() -> Path:
    """Return the audit log directory (env-overridable)."""
    return Path(os.environ.get(LOG_DIR_ENV, DEFAULT_LOG_DIR))


def audit_log_path() -> Path:
    """Return the absolute audit log path."""
    return log_dir() / AUDIT_FILENAME


class AuditLog:
    """Thread-safe append-only audit writer with fail-closed initialization."""

    def __init__(self, path: Path | None = None) -> None:
        """Open (creating ``0600`` if needed) the audit log for append.

        Args:
            path: Optional explicit log path (defaults to :func:`audit_log_path`).

        Raises:
            OSError: If the directory or file cannot be created/opened (fail-closed).
        """
        self._path = path if path is not None else audit_log_path()
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Create with 0600 from the outset; never world/group readable, even transiently.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(self._path, flags, AUDIT_FILE_MODE)
        os.close(fd)
        try:
            os.chmod(self._path, AUDIT_FILE_MODE)
        except OSError:
            logger.warning("could not chmod audit log %s to 0600", self._path)

    @property
    def path(self) -> Path:
        """Return the audit log path."""
        return self._path

    def _write(self, record: dict[str, object]) -> None:
        """Serialize and append one JSON record (fail-closed on write error)."""
        record["ts"] = datetime.now(KST).isoformat()
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with self._lock:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            fd = os.open(self._path, flags, AUDIT_FILE_MODE)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)

    def record_download(
        self,
        *,
        peer: str,
        session_ids: list[str],
        file_count: int,
        byte_count: int,
        route: str,
    ) -> None:
        """Append a successful-download record."""
        self._write(
            {
                "event": "download",
                "peer": peer,
                "sessions": session_ids,
                "file_count": file_count,
                "bytes": byte_count,
                "route": route,
            }
        )

    def record_failed_auth(self, *, peer: str, reason: str, route: str) -> None:
        """Append a failed-authentication / rejected-request record."""
        self._write(
            {
                "event": "failed_auth",
                "peer": peer,
                "reason": reason,
                "route": route,
            }
        )

    def record_shutdown(self, *, peer: str) -> None:
        """Append a portal-shutdown record (operator stopped the server from the UI)."""
        self._write(
            {
                "event": "shutdown",
                "peer": peer,
            }
        )
