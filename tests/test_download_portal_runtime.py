"""Static runtime tests: systemd unit assertions + flag/config gating.

Covers §9 ``runtime/service``: the unit file's ``ExecCondition`` starts only when the flag
is ``1`` and stays inactive otherwise; the sandbox directives and resource bounds are
present. Also asserts the daemon entry point's fail-closed flag/config gating (flag unset
=> clean exit 0; uninitialized config => refuse with non-zero).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parental.download_portal import FEATURE_FLAG_ENV
from parental.download_portal import __main__ as daemon

_UNIT_PATH = Path(__file__).resolve().parents[1] / "systemd" / "mungi-download-portal.service"


@pytest.fixture(scope="module")
def unit_text() -> str:
    """Return the systemd unit file contents."""
    return _UNIT_PATH.read_text(encoding="utf-8")


def test_unit_file_exists(unit_text: str) -> None:
    assert "[Service]" in unit_text


def test_exec_condition_starts_only_when_flag_is_one(unit_text: str) -> None:
    # ExecCondition exits 0 (start) only when MUNGI_DOWNLOAD_PORTAL == "1".
    assert 'ExecCondition=/usr/bin/test "${MUNGI_DOWNLOAD_PORTAL}" = "1"' in unit_text


def test_exec_start_runs_module(unit_text: str) -> None:
    assert "ExecStart=/opt/mungi-repo/.venv/bin/python -m parental.download_portal" in unit_text


@pytest.mark.parametrize(
    "directive",
    [
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "PrivateTmp=yes",
        "ReadOnlyPaths=/var/lib/mungi/conversations",
        "ReadWritePaths=/var/lib/mungi/logs",
        "UMask=0077",
        "ProtectKernelTunables=yes",
        "ProtectKernelModules=yes",
        "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX",
    ],
)
def test_sandbox_directives_present(unit_text: str, directive: str) -> None:
    assert directive in unit_text


@pytest.mark.parametrize(
    "directive",
    ["MemoryMax=128M", "TasksMax=32", "LimitNOFILE=256", "Nice=10"],
)
def test_resource_bounds_present(unit_text: str, directive: str) -> None:
    assert directive in unit_text


def test_ordering_after_tailscaled(unit_text: str) -> None:
    assert "After=tailscaled.service network-online.target" in unit_text
    assert "Restart=on-failure" in unit_text


# --- Daemon flag / config gating ---------------------------------------------
def test_daemon_noops_when_flag_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
    assert daemon.run_daemon() == 0


def test_daemon_noops_when_flag_not_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV, "0")
    assert daemon.run_daemon() == 0


def test_daemon_refuses_when_uninitialized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
    monkeypatch.setenv("MUNGI_PORTAL_CONFIG_DIR", str(tmp_path / "empty_config"))
    # No portal.json present -> fail-closed non-zero.
    assert daemon.run_daemon() == 2
