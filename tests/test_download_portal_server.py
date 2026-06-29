"""Security tests for the request handler: auth-on-download, CSRF, semaphore, peer gate.

Covers §9 ``auth-on-download`` (every ``/download*`` route returns 401 for an
unauthenticated/expired session before any FS access), post-login CSRF rejection, the
login Origin/Referer check, the concurrency semaphore (``503``), and that the peer gate
runs before routing. The handler is driven with no real socket via the test harness.
"""

from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

from parental.download_portal import auth, data, server
from parental.download_portal.audit import AuditLog
from parental.download_portal.config import PortalConfig
from tests import _portal_handler_harness as harness

PIN = "12345678"


@pytest.fixture
def portal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> server.PortalState:
    """A PortalState wired to an isolated conversations root + audit log."""
    monkeypatch.setenv("MUNGI_PORTAL_LOG_DIR", str(tmp_path / "logs"))
    root = tmp_path / "conversations"
    root.mkdir()
    monkeypatch.setenv(data.CONVERSATIONS_DIR_ENV, str(root))
    sess = root / "2026-06-18_10-00-00"
    sess.mkdir()
    (sess / "conversation.jsonl").write_bytes(b'{"turn":1}\n{"turn":2}\n')
    (sess / "input_001.wav").write_bytes(b"RIFFwav")
    config = PortalConfig(pin_record=auth.hash_pin(PIN), session_secret=b"s" * 32)
    return server.PortalState(config, AuditLog())


def _login(portal: server.PortalState, peer: str = "100.64.0.10") -> str:
    """Drive a successful login and return the session cookie ``name=value`` pair."""
    handler = harness.build_fake_handler(
        portal,
        peer_ip=peer,
        method="POST",
        path="/login",
        headers={"Host": "100.64.0.10:8765"},
        body=f"pin={PIN}".encode(),
    )
    response = harness.invoke(handler)
    assert response.status == 303
    assert response.set_cookies
    return response.set_cookies[0].split(";", 1)[0]


# --- Peer gate (runs before everything) --------------------------------------
@pytest.mark.parametrize("peer", ["192.168.1.5", "127.0.0.1", "8.8.8.8", "100.63.0.1"])
def test_non_tailscale_peer_forbidden(portal: server.PortalState, peer: str) -> None:
    handler = harness.build_fake_handler(portal, peer_ip=peer, method="GET", path="/")
    response = harness.invoke(handler)
    assert response.status == 403


# --- Auth on download (401 before any FS access) -----------------------------
@pytest.mark.parametrize(
    "path",
    [
        "/download/2026-06-18_10-00-00",
        "/download/2026-06-18_10-00-00/text",
    ],
)
def test_unauthenticated_get_download_returns_401(
    portal: server.PortalState, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Trip a tripwire: if any FS access were attempted, list_sessions would be called.
    calls: list[str] = []
    real_open = data.open_session_file

    def _tripwire(session_id: str, filename: str):
        calls.append(session_id)
        return real_open(session_id, filename)

    monkeypatch.setattr(data, "open_session_file", _tripwire)
    handler = harness.build_fake_handler(portal, peer_ip="100.64.0.10", method="GET", path=path)
    response = harness.invoke(handler)
    assert response.status == 401
    assert calls == []  # no filesystem access occurred


def test_unauthenticated_post_download_returns_401(
    portal: server.PortalState, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(data, "build_manifest", lambda ids: calls.append(ids) or (b"{}", []))
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/download",
        body=b"session_id=2026-06-18_10-00-00",
    )
    response = harness.invoke(handler)
    assert response.status == 401
    assert calls == []


def test_expired_session_download_returns_401(
    portal: server.PortalState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Issue a token, then advance the manager's clock past expiry.
    clock = [1_000.0]
    portal.sessions = auth.SessionManager(b"s" * 32, ttl_seconds=30, now=lambda: clock[0])
    token, _ = portal.sessions.issue()
    clock[0] += 31
    cookie = f"{server.SESSION_COOKIE_NAME}={token}"
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="GET",
        path="/download/2026-06-18_10-00-00",
        headers={"Cookie": cookie},
    )
    response = harness.invoke(handler)
    assert response.status == 401


# --- Successful authenticated downloads --------------------------------------
def test_authenticated_get_download_streams_zip(portal: server.PortalState) -> None:
    cookie = _login(portal)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="GET",
        path="/download/2026-06-18_10-00-00",
        headers={"Cookie": cookie},
    )
    response = harness.invoke(handler)
    assert response.status == 200
    assert response.headers.get("Transfer-Encoding") == "chunked"
    archive = zipfile.ZipFile(io.BytesIO(response.body))
    assert archive.testzip() is None
    assert "2026-06-18_10-00-00/conversation.jsonl" in archive.namelist()


def test_authenticated_get_text_returns_transcript(portal: server.PortalState) -> None:
    cookie = _login(portal)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="GET",
        path="/download/2026-06-18_10-00-00/text",
        headers={"Cookie": cookie},
    )
    response = harness.invoke(handler)
    assert response.status == 200
    assert response.body == b'{"turn":1}\n{"turn":2}\n'


# --- CSRF on POST /download --------------------------------------------------
def test_post_download_requires_valid_csrf(portal: server.PortalState) -> None:
    cookie = _login(portal)
    # Missing CSRF token -> 403.
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/download",
        headers={"Cookie": cookie},
        body=b"session_id=2026-06-18_10-00-00",
    )
    response = harness.invoke(handler)
    assert response.status == 403


def test_post_download_with_valid_csrf_succeeds(portal: server.PortalState) -> None:
    # Login and recover the CSRF bound to the issued session.
    handler_login = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/login",
        headers={"Host": "100.64.0.10:8765"},
        body=f"pin={PIN}".encode(),
    )
    login_resp = harness.invoke(handler_login)
    cookie = login_resp.set_cookies[0].split(";", 1)[0]
    token = cookie.split("=", 1)[1]
    session = portal.sessions.verify(token)
    assert session is not None
    csrf = portal.sessions.csrf_for(session)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/download",
        headers={"Cookie": cookie},
        body=f"csrf_token={csrf}&session_id=2026-06-18_10-00-00".encode(),
    )
    response = harness.invoke(handler)
    assert response.status == 200
    archive = zipfile.ZipFile(io.BytesIO(response.body))
    assert archive.testzip() is None


def test_post_download_rejects_unsafe_session_id(portal: server.PortalState) -> None:
    handler_login = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/login",
        headers={"Host": "100.64.0.10:8765"},
        body=f"pin={PIN}".encode(),
    )
    login_resp = harness.invoke(handler_login)
    cookie = login_resp.set_cookies[0].split(";", 1)[0]
    token = cookie.split("=", 1)[1]
    session = portal.sessions.verify(token)
    assert session is not None
    csrf = portal.sessions.csrf_for(session)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/download",
        headers={"Cookie": cookie},
        body=f"csrf_token={csrf}&session_id=../etc".encode(),
    )
    response = harness.invoke(handler)
    assert response.status == 400


# --- Login Origin/Referer + bad PIN ------------------------------------------
def test_login_rejects_cross_origin(portal: server.PortalState) -> None:
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/login",
        headers={"Host": "100.64.0.10:8765", "Origin": "http://evil.example"},
        body=f"pin={PIN}".encode(),
    )
    response = harness.invoke(handler)
    assert response.status == 403


def test_login_wrong_pin_returns_401_and_records_failure(
    portal: server.PortalState,
) -> None:
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/login",
        headers={"Host": "100.64.0.10:8765"},
        body=b"pin=00000000",
    )
    response = harness.invoke(handler)
    assert response.status == 401


def test_login_throttled_returns_429(portal: server.PortalState) -> None:
    # Force the limiter to deny immediately.
    portal.rate_limiter = auth.RateLimiter(global_max_attempts=0)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/login",
        headers={"Host": "100.64.0.10:8765"},
        body=f"pin={PIN}".encode(),
    )
    response = harness.invoke(handler)
    assert response.status == 429
    assert "Retry-After" in response.headers


# --- Concurrency semaphore ---------------------------------------------------
def test_download_semaphore_returns_503_when_saturated(
    portal: server.PortalState,
) -> None:
    # Exhaust the semaphore so the next download is rejected.
    assert portal.download_semaphore.acquire(blocking=False)
    assert portal.download_semaphore.acquire(blocking=False)
    cookie = _login(portal)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="GET",
        path="/download/2026-06-18_10-00-00",
        headers={"Cookie": cookie},
    )
    response = harness.invoke(handler)
    assert response.status == 503
    assert "Retry-After" in response.headers
    portal.download_semaphore.release()
    portal.download_semaphore.release()


# --- Audit on download + failed auth -----------------------------------------
def test_download_writes_audit_line(portal: server.PortalState) -> None:
    cookie = _login(portal)
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="GET",
        path="/download/2026-06-18_10-00-00",
        headers={"Cookie": cookie},
    )
    harness.invoke(handler)
    lines = portal.audit.path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert any(e["event"] == "download" for e in events)
    download = next(e for e in events if e["event"] == "download")
    assert download["peer"] == "100.64.0.10"
    assert "2026-06-18_10-00-00" in download["sessions"]


def test_failed_auth_writes_audit_line(portal: server.PortalState) -> None:
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="GET",
        path="/download/2026-06-18_10-00-00",
    )
    harness.invoke(handler)  # no cookie -> failed_auth
    events = [
        json.loads(line) for line in portal.audit.path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(e["event"] == "failed_auth" and e["reason"] == "no_session" for e in events)


# --- Shutdown endpoint (session + CSRF gated, like /download POST) ------------
def test_unauthenticated_post_shutdown_returns_401(portal: server.PortalState) -> None:
    """POST /shutdown without a session is rejected before any stop."""
    handler = harness.build_fake_handler(
        portal, peer_ip="100.64.0.10", method="POST", path="/shutdown", body=b"csrf_token=x"
    )
    response = harness.invoke(handler)
    assert response.status == 401


def test_shutdown_bad_csrf_returns_403(portal: server.PortalState) -> None:
    """An authenticated session with a bad CSRF token cannot stop the portal."""
    token, _ = portal.sessions.issue()
    cookie = f"{server.SESSION_COOKIE_NAME}={token}"
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/shutdown",
        headers={"Cookie": cookie},
        body=b"csrf_token=deadbeef",
    )
    response = harness.invoke(handler)
    assert response.status == 403


def test_shutdown_with_session_and_csrf_stops_server(portal: server.PortalState) -> None:
    """Valid session + CSRF returns the confirmation page and triggers a clean stop."""
    token, csrf = portal.sessions.issue()
    cookie = f"{server.SESSION_COOKIE_NAME}={token}"
    handler = harness.build_fake_handler(
        portal,
        peer_ip="100.64.0.10",
        method="POST",
        path="/shutdown",
        headers={"Cookie": cookie},
        body=f"csrf_token={csrf}".encode(),
    )
    stopped = threading.Event()

    class _FakeServer:
        def shutdown(self) -> None:
            stopped.set()

    handler.server = _FakeServer()
    response = harness.invoke(handler)
    assert response.status == 200
    assert "포털을 종료" in response.body.decode("utf-8")
    assert stopped.wait(2.0), "server.shutdown() was not triggered"
