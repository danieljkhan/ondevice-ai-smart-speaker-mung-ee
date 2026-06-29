"""Portal configuration paths and ``portal.json`` load/save helpers.

The config holds the hashed PIN record and the 32-byte session secret. It is written
``0600`` and never logged. Paths are absolute because they target the Jetson runtime
(``/var/lib/mungi``); tests override them via the environment variable below.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Runtime defaults (Jetson). Overridable for tests / alternate hosts.
DEFAULT_CONFIG_DIR = "/var/lib/mungi/config"
CONFIG_DIR_ENV = "MUNGI_PORTAL_CONFIG_DIR"
PORTAL_CONFIG_FILENAME = "portal.json"
PORTAL_CONFIG_SCHEMA_VERSION = 1
CONFIG_FILE_MODE = 0o600


def config_dir() -> Path:
    """Return the directory holding ``portal.json`` (env-overridable)."""
    return Path(os.environ.get(CONFIG_DIR_ENV, DEFAULT_CONFIG_DIR))


def portal_config_path() -> Path:
    """Return the absolute path to ``portal.json``."""
    return config_dir() / PORTAL_CONFIG_FILENAME


@dataclass(frozen=True)
class PortalConfig:
    """Loaded portal configuration: hashed PIN record + session secret."""

    pin_record: dict[str, object]
    session_secret: bytes


def is_initialized() -> bool:
    """Return ``True`` iff a readable, parseable ``portal.json`` exists."""
    try:
        load_config()
    except (FileNotFoundError, ValueError, OSError):
        return False
    return True


def load_config() -> PortalConfig:
    """Load and validate ``portal.json``.

    Returns:
        A :class:`PortalConfig`.

    Raises:
        FileNotFoundError: If the config file is absent.
        ValueError: If the file is malformed or missing required fields.
    """
    path = portal_config_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("portal.json must be a JSON object")
    pin_record = raw.get("pin")
    secret_b64 = raw.get("session_secret")
    if not isinstance(pin_record, dict) or not isinstance(secret_b64, str):
        raise ValueError("portal.json missing 'pin' or 'session_secret'")
    try:
        session_secret = base64.b64decode(secret_b64)
    except (ValueError, TypeError) as exc:
        raise ValueError("portal.json 'session_secret' is not valid base64") from exc
    if len(session_secret) < 32:
        raise ValueError("portal.json 'session_secret' must be at least 32 bytes")
    return PortalConfig(pin_record=pin_record, session_secret=session_secret)


def save_config(pin_record: dict[str, object], session_secret: bytes) -> None:
    """Atomically write ``portal.json`` with mode ``0600``.

    The file is created with restrictive permissions from the outset (never world- or
    group-readable, even transiently) and written via a temp file + ``os.replace``.

    Args:
        pin_record: A PBKDF2 record from ``auth.hash_pin`` (no plaintext PIN).
        session_secret: The 32-byte HMAC session secret.
    """
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PORTAL_CONFIG_SCHEMA_VERSION,
        "pin": pin_record,
        "session_secret": base64.b64encode(session_secret).decode("ascii"),
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path = portal_config_path()
    tmp_path = path.with_name(path.name + ".tmp")
    # Open with O_CREAT|O_EXCL semantics replaced by explicit 0600 mode; remove any
    # stale temp first so the exclusive-create cannot fail on a leftover file.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp_path, flags, CONFIG_FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, path)
    # Re-assert mode in case the umask widened the temp file's permissions.
    os.chmod(path, CONFIG_FILE_MODE)
    logger.info("portal config written to %s (mode 0600)", path)


def assert_secure_mode(path: Path) -> None:
    """Log a warning if ``path`` is group/world readable (best-effort on POSIX)."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning("portal config %s has loose permissions: %o", path, mode)
