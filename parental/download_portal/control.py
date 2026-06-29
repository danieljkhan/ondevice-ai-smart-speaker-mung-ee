"""Privileged on-demand start/stop control for the download-portal service.

The kiosk (user ``mungi``) activates the portal on demand from the touchscreen,
which means starting a systemd service — a root action. Rather than run the
portal outside systemd (which would forfeit its sandbox hardening), the kiosk
shells out to ``sudo -n systemctl start/stop`` guarded by a NARROW passwordless
sudoers rule that permits only these two exact commands. See
``systemd/mungi-portal-control.sudoers`` and the runbook.

Starting the portal does not expose data on its own: the portal is still
Tailscale-only + PIN-gated. This module only flips the service on/off.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PORTAL_SERVICE = "mungi-download-portal.service"
_SUDO = "/usr/bin/sudo"
_SYSTEMCTL = "/usr/bin/systemctl"
_DEFAULT_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class ControlResult:
    """Outcome of a portal service control action."""

    ok: bool
    action: str
    detail: str


def _run(action: str, timeout: float) -> ControlResult:
    """Run ``sudo -n systemctl <action> <service>`` and classify the result.

    Never raises: subprocess failures are folded into a ``ControlResult`` so the
    caller (the kiosk session loop) degrades gracefully instead of crashing.
    """
    cmd = [_SUDO, "-n", _SYSTEMCTL, action, PORTAL_SERVICE]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        logger.error("Portal %s timed out after %.0fs", action, timeout)
        return ControlResult(ok=False, action=action, detail="timeout")
    except OSError as exc:  # sudo/systemctl missing, etc.
        logger.error("Portal %s could not be launched: %s", action, exc)
        return ControlResult(ok=False, action=action, detail=str(exc))
    if completed.returncode == 0:
        logger.info("Portal %s succeeded", action)
        return ControlResult(ok=True, action=action, detail="ok")
    detail = (completed.stderr or completed.stdout or "").strip() or f"exit {completed.returncode}"
    logger.error("Portal %s failed (rc=%d): %s", action, completed.returncode, detail)
    return ControlResult(ok=False, action=action, detail=detail)


def start_service(*, timeout: float = _DEFAULT_TIMEOUT_S) -> ControlResult:
    """Start the download-portal service via the narrow passwordless sudo rule."""
    return _run("start", timeout)


def stop_service(*, timeout: float = _DEFAULT_TIMEOUT_S) -> ControlResult:
    """Stop the download-portal service via the narrow passwordless sudo rule."""
    return _run("stop", timeout)


__all__ = ["ControlResult", "PORTAL_SERVICE", "start_service", "stop_service"]
