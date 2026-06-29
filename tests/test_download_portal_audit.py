"""Security tests for the audit log + config + set-pin CLI.

Covers §9 ``audit``: the log is written on download + failed auth (download/failed-auth
content asserted in the server tests; here we assert structure + mode + fail-closed init),
the file mode is ``0600``, and initialization fails closed when the log directory cannot be
created. Also covers the ``set-pin`` CLI / config: ``portal.json`` is ``0600``, the PIN is
never echoed/logged, and a fresh session secret is generated on each set.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from parental.download_portal import audit, auth, cli, config


# --- Audit log ---------------------------------------------------------------
def test_audit_init_creates_0600_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("MUNGI_PORTAL_LOG_DIR", str(log_dir))
    log = audit.AuditLog()
    assert log.path.exists()
    if os.name == "posix":
        mode = stat.S_IMODE(os.stat(log.path).st_mode)
        assert mode == 0o600


def test_audit_records_download_and_failed_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_LOG_DIR", str(tmp_path / "logs"))
    log = audit.AuditLog()
    log.record_download(
        peer="100.64.0.1",
        session_ids=["s1", "s2"],
        file_count=4,
        byte_count=2048,
        route="/download",
    )
    log.record_failed_auth(peer="100.64.0.9", reason="no_session", route="/download/s1")
    import json

    events = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["event"] == "download"
    assert events[0]["file_count"] == 4
    assert events[0]["bytes"] == 2048
    assert events[0]["sessions"] == ["s1", "s2"]
    assert "ts" in events[0]
    assert events[1]["event"] == "failed_auth"
    assert events[1]["reason"] == "no_session"


def test_audit_does_not_log_conversation_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_LOG_DIR", str(tmp_path / "logs"))
    log = audit.AuditLog()
    log.record_download(
        peer="100.64.0.1", session_ids=["s1"], file_count=1, byte_count=10, route="/x"
    )
    text = log.path.read_text(encoding="utf-8")
    # Only metadata fields are present; no message/text payloads.
    assert "user_text" not in text
    assert "response_text" not in text


def test_audit_fail_closed_when_dir_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Init must raise (fail-closed) when the log path cannot be created."""
    # Point the log dir at a path whose parent is a regular file -> mkdir fails.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MUNGI_PORTAL_LOG_DIR", str(blocker / "logs"))
    with pytest.raises(OSError):
        audit.AuditLog()


# --- Config / set-pin --------------------------------------------------------
def test_save_and_load_config_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    record = auth.hash_pin("12345678")
    secret = os.urandom(32)
    config.save_config(record, secret)
    loaded = config.load_config()
    assert loaded.session_secret == secret
    assert auth.verify_pin("12345678", loaded.pin_record) is True


def test_config_file_is_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    config.save_config(auth.hash_pin("12345678"), os.urandom(32))
    if os.name == "posix":
        mode = stat.S_IMODE(os.stat(config.portal_config_path()).st_mode)
        assert mode == 0o600


def test_is_initialized_false_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "nope"))
    assert config.is_initialized() is False


def test_load_config_rejects_short_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import base64
    import json

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(cfg_dir))
    (cfg_dir / "portal.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pin": auth.hash_pin("12345678"),
                "session_secret": base64.b64encode(b"short").decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        config.load_config()


def test_set_pin_writes_config_and_never_logs_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    secret_pin = "13572468"
    # Feed the PIN via stdin (non-TTY single-line path); not via argv.
    monkeypatch.setattr(sys, "stdin", _FakeStdin(f"{secret_pin}\n"))
    rc = cli.set_pin([])
    assert rc == 0
    assert config.is_initialized() is True
    loaded = config.load_config()
    assert auth.verify_pin(secret_pin, loaded.pin_record) is True
    captured = capsys.readouterr()
    # The PIN must never appear in stdout/stderr.
    assert secret_pin not in captured.out
    assert secret_pin not in captured.err


def test_set_pin_rejects_short_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(sys, "stdin", _FakeStdin("1234\n"))
    rc = cli.set_pin([])
    assert rc != 0
    assert config.is_initialized() is False


def test_set_pin_generates_fresh_secret_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(sys, "stdin", _FakeStdin("12345678\n"))
    assert cli.set_pin([]) == 0
    first_secret = config.load_config().session_secret
    monkeypatch.setattr(sys, "stdin", _FakeStdin("12345678\n"))
    assert cli.set_pin([]) == 0
    second_secret = config.load_config().session_secret
    assert first_secret != second_secret  # rotation on PIN reset


def test_cli_main_unknown_command(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli.main(["bogus"]) != 0


class _FakeStdin:
    """A minimal non-TTY stdin stub for piping a single PIN line."""

    def __init__(self, text: str) -> None:
        self._text = text

    def isatty(self) -> bool:
        return False

    def readline(self) -> str:
        line, _, self._text = self._text.partition("\n")
        return line + "\n"
