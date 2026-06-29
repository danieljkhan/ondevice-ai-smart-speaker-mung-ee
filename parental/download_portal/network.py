"""Tailscale-only address resolution and peer validation.

Security model (see ``Dev_Plan/2026-06-18-conversation-download-portal-plan.md`` §3):

- The server binds to the ``tailscale0`` IPv4 address ONLY. It is resolved at startup via
  ``tailscale ip -4`` with a fallback that reads the ``tailscale0`` interface address.
- If no ``100.64.0.0/10`` (CGNAT / Tailscale) address is available the caller MUST fail
  closed (log + exit non-zero). There is **no** LAN/public fallback.
- Identity is the TCP peer address only; forwarded headers are ignored. A peer must be in
  ``100.64.0.0/10`` to be served.

No ``tailscale`` subprocess is invoked in tests — the resolver is injectable and the
peer check is a pure function over an address string.
"""

from __future__ import annotations

import ipaddress
import logging
import subprocess

logger = logging.getLogger(__name__)

# Tailscale uses the 100.64.0.0/10 CGNAT range for tailnet addresses.
TAILSCALE_CGNAT_CIDR = "100.64.0.0/10"
_TAILSCALE_NET = ipaddress.ip_network(TAILSCALE_CGNAT_CIDR)
TAILSCALE_INTERFACE = "tailscale0"
TAILSCALE_RESOLVE_TIMEOUT_S = 5.0


class TailscaleUnavailableError(RuntimeError):
    """Raised when no usable ``100.64.0.0/10`` address can be resolved."""


def is_tailscale_address(addr: str) -> bool:
    """Return ``True`` iff ``addr`` is a valid IPv4 in ``100.64.0.0/10``.

    Args:
        addr: A bare IPv4 address string (no port, no scope id).

    Returns:
        ``True`` iff the address parses and falls inside the CGNAT range.
    """
    try:
        parsed = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if parsed.version != 4:
        return False
    return parsed in _TAILSCALE_NET


def is_loopback_or_public(addr: str) -> bool:
    """Return ``True`` iff ``addr`` is loopback, private (LAN), or public (non-tailnet)."""
    return not is_tailscale_address(addr)


def _resolve_via_cli() -> str | None:
    """Return the first ``100.64/10`` address from ``tailscale ip -4`` (or ``None``)."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=TAILSCALE_RESOLVE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("tailscale ip -4 failed: %s", exc)
        return None
    if completed.returncode != 0:
        logger.warning("tailscale ip -4 returned %d", completed.returncode)
        return None
    for line in completed.stdout.splitlines():
        candidate = line.strip()
        if is_tailscale_address(candidate):
            return candidate
    return None


def _resolve_via_interface() -> str | None:
    """Return the ``tailscale0`` IPv4 from ``ip -4 addr show`` (or ``None``)."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["ip", "-4", "-o", "addr", "show", "dev", TAILSCALE_INTERFACE],
            capture_output=True,
            text=True,
            timeout=TAILSCALE_RESOLVE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ip addr show %s failed: %s", TAILSCALE_INTERFACE, exc)
        return None
    if completed.returncode != 0:
        return None
    for token in completed.stdout.split():
        candidate = token.split("/", 1)[0]
        if is_tailscale_address(candidate):
            return candidate
    return None


def resolve_tailscale_ipv4() -> str:
    """Resolve the local ``tailscale0`` IPv4, preferring the CLI then the interface.

    Returns:
        A ``100.64.0.0/10`` IPv4 address string.

    Raises:
        TailscaleUnavailableError: If no tailnet address is available (fail-closed).
    """
    for resolver in (_resolve_via_cli, _resolve_via_interface):
        addr = resolver()
        if addr is not None:
            logger.info("resolved tailscale bind address %s", addr)
            return addr
    raise TailscaleUnavailableError(
        "no 100.64.0.0/10 address available — Tailscale appears down; refusing to bind"
    )
