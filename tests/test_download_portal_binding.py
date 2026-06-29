"""Security tests for Tailscale-only binding and peer validation.

Covers §9 ``binding``: the server refuses to bind ``0.0.0.0``/LAN/public (only a
``100.64.0.0/10`` address is accepted), the resolver fails closed when no tailnet address
exists, and the peer check rejects LAN/private/public peers plus forwarded-header spoofing
while accepting only a ``100.64.0.0/10`` TCP peer.

No real socket bind and no ``tailscale`` subprocess are exercised — the resolver is
injected and binding is asserted to raise before any OS bind.
"""

from __future__ import annotations

import pytest

from parental.download_portal import network, server
from parental.download_portal.audit import AuditLog
from parental.download_portal.config import PortalConfig


@pytest.fixture
def portal_state(tmp_path, monkeypatch) -> server.PortalState:
    """A minimal PortalState with an isolated audit log (no real network)."""
    monkeypatch.setenv("MUNGI_PORTAL_LOG_DIR", str(tmp_path / "logs"))
    from parental.download_portal import auth

    config = PortalConfig(pin_record=auth.hash_pin("12345678"), session_secret=b"x" * 32)
    return server.PortalState(config, AuditLog())


# --- Address classification --------------------------------------------------
@pytest.mark.parametrize(
    "addr",
    ["100.64.0.1", "100.64.255.254", "100.100.50.7", "100.127.255.255"],
)
def test_tailscale_addresses_accepted(addr: str) -> None:
    assert network.is_tailscale_address(addr) is True


@pytest.mark.parametrize(
    "addr",
    [
        "0.0.0.0",  # wildcard bind
        "127.0.0.1",  # loopback
        "192.168.1.10",  # LAN (private)
        "10.0.0.5",  # LAN (private)
        "172.16.0.1",  # LAN (private)
        "8.8.8.8",  # public
        "100.63.255.255",  # just below CGNAT range
        "100.128.0.0",  # just above CGNAT range
        "::1",  # IPv6 loopback (IPv4-only policy)
        "fd7a:115c:a1e0::1",  # Tailscale IPv6 (out of scope; IPv4-only)
        "not-an-ip",
    ],
)
def test_non_tailscale_addresses_rejected(addr: str) -> None:
    assert network.is_tailscale_address(addr) is False
    assert network.is_loopback_or_public(addr) is True


# --- Resolver fail-closed ----------------------------------------------------
def test_resolver_fails_closed_when_no_tailscale(monkeypatch) -> None:
    monkeypatch.setattr(network, "_resolve_via_cli", lambda: None)
    monkeypatch.setattr(network, "_resolve_via_interface", lambda: None)
    with pytest.raises(network.TailscaleUnavailableError):
        network.resolve_tailscale_ipv4()


def test_resolver_prefers_cli_address(monkeypatch) -> None:
    monkeypatch.setattr(network, "_resolve_via_cli", lambda: "100.64.7.7")
    monkeypatch.setattr(network, "_resolve_via_interface", lambda: "100.64.9.9")
    assert network.resolve_tailscale_ipv4() == "100.64.7.7"


def test_resolver_falls_back_to_interface(monkeypatch) -> None:
    monkeypatch.setattr(network, "_resolve_via_cli", lambda: None)
    monkeypatch.setattr(network, "_resolve_via_interface", lambda: "100.64.9.9")
    assert network.resolve_tailscale_ipv4() == "100.64.9.9"


# --- Bind refusal ------------------------------------------------------------
@pytest.mark.parametrize(
    "addr",
    ["0.0.0.0", "127.0.0.1", "192.168.1.10", "10.0.0.5", "8.8.8.8"],
)
def test_server_refuses_non_tailscale_bind(addr: str, portal_state: server.PortalState) -> None:
    """Constructing the server with a non-tailnet address raises before any OS bind."""
    with pytest.raises(ValueError):
        server.PortalServer(portal_state, addr, resolve_ip=lambda: addr)


def test_serve_returns_nonzero_when_tailscale_down(
    portal_state: server.PortalState, monkeypatch
) -> None:
    """serve() fails closed (non-zero exit) when no tailnet address resolves."""

    def _raise() -> str:
        raise network.TailscaleUnavailableError("down")

    code = server.serve(portal_state.config, portal_state.audit, resolve_ip=_raise)
    assert code != 0


# --- Peer check (TCP peer only; ignore forwarded headers) --------------------
def _make_handler(portal_state: server.PortalState, peer_ip: str, headers: dict[str, str]):
    """Build a handler bound to ``portal_state`` with a fake peer + headers (no socket)."""
    from tests._portal_handler_harness import build_fake_handler

    return build_fake_handler(
        portal_state, peer_ip=peer_ip, method="GET", path="/", headers=headers
    )


def test_peer_check_accepts_tailscale_peer(portal_state: server.PortalState) -> None:
    handler = _make_handler(portal_state, "100.64.0.10", {})
    assert handler._peer_is_tailscale() is True


@pytest.mark.parametrize(
    "peer_ip",
    ["127.0.0.1", "192.168.1.50", "10.1.2.3", "8.8.8.8", "100.63.0.1"],
)
def test_peer_check_rejects_non_tailscale_peer(
    peer_ip: str, portal_state: server.PortalState
) -> None:
    handler = _make_handler(portal_state, peer_ip, {})
    assert handler._peer_is_tailscale() is False


def test_forwarded_header_spoof_is_ignored(portal_state: server.PortalState) -> None:
    """A LAN peer cannot fake a tailnet identity via X-Forwarded-For."""
    handler = _make_handler(
        portal_state,
        "192.168.1.50",
        {"X-Forwarded-For": "100.64.0.1", "X-Real-IP": "100.64.0.1"},
    )
    # The handler trusts only the TCP peer address.
    assert handler._peer_ip() == "192.168.1.50"
    assert handler._peer_is_tailscale() is False
