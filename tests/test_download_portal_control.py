"""Tests for the portal on-demand control helper (subprocess mocked)."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from parental.download_portal import control


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_start_service_success_runs_narrow_sudo_command() -> None:
    with patch("subprocess.run", return_value=_completed(0)) as run:
        result = control.start_service()
    assert result.ok is True
    assert result.action == "start"
    assert run.call_args.args[0] == [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/systemctl",
        "start",
        control.PORTAL_SERVICE,
    ]


def test_start_service_failure_reports_stderr() -> None:
    with patch("subprocess.run", return_value=_completed(1, stderr="a password is required")):
        result = control.start_service()
    assert result.ok is False
    assert "password is required" in result.detail


def test_start_service_timeout_is_swallowed() -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
        result = control.start_service()
    assert result.ok is False
    assert result.detail == "timeout"


def test_start_service_missing_binary_is_swallowed() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("no sudo")):
        result = control.start_service()
    assert result.ok is False


def test_stop_service_runs_stop_action() -> None:
    with patch("subprocess.run", return_value=_completed(0)) as run:
        result = control.stop_service()
    assert result.ok is True
    assert run.call_args.args[0][3] == "stop"
