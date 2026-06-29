"""HTTP server: Tailscale-only binding, session-gated routes, bounded downloads.

Security model (see ``Dev_Plan/2026-06-18-conversation-download-portal-plan.md`` §3-§6):

- Binds to the ``tailscale0`` IPv4 ONLY (never ``0.0.0.0``/LAN/public). Fail-closed if no
  ``100.64.0.0/10`` address is available. An IP-change watcher exits non-zero so systemd
  rebinds cleanly.
- The TCP peer address (only) must be in ``100.64.0.0/10``; forwarded headers are ignored.
- Every ``/download*`` route requires a valid session **before** any filesystem access.
- A bounded concurrency semaphore + socket timeouts protect against resource exhaustion;
  excess downloads get ``503``. Downloads stream a STORE-mode zip via HTTP/1.1 chunked
  framing from opened fds (no buffering).
"""

from __future__ import annotations

import html
import http.cookies
import logging
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import auth, data, network
from .audit import AuditLog
from .config import PortalConfig

logger = logging.getLogger(__name__)

PORTAL_PORT = 8765
SESSION_COOKIE_NAME = "mungi_portal_session"
COOKIE_MAX_AGE = 1800
MAX_LOGIN_BODY_BYTES = 4096
MAX_DOWNLOAD_BODY_BYTES = 64 * 1024
MAX_SHUTDOWN_BODY_BYTES = 4096
MAX_CONCURRENT_DOWNLOADS = 2
SOCKET_TIMEOUT_S = 30.0
IP_WATCH_INTERVAL_S = 30.0
CHUNK_TERMINATOR = b"0\r\n\r\n"

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _load_asset(name: str) -> str:
    """Read a bundled text asset (HTML/CSS) from the package ``assets/`` dir."""
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def _human_bytes(num: int) -> str:
    """Return a compact human-readable size string for ``num`` bytes."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


class PortalState:
    """Shared, thread-safe runtime state injected into every request handler."""

    def __init__(
        self,
        config: PortalConfig,
        audit: AuditLog,
        *,
        sessions: auth.SessionManager | None = None,
        rate_limiter: auth.RateLimiter | None = None,
        allowed_origins: frozenset[str] | None = None,
    ) -> None:
        """Bundle the config, audit log, session manager, and rate limiter.

        Args:
            config: Loaded portal config (PIN record + session secret).
            audit: The fail-closed audit log.
            sessions: Optional session manager (constructed from the secret otherwise).
            rate_limiter: Optional rate limiter (default thresholds otherwise).
            allowed_origins: Optional explicit Origin allow-list (host:port). When None,
                the bind address is added at serve time.
        """
        self.config = config
        self.audit = audit
        self.sessions = sessions or auth.SessionManager(config.session_secret)
        self.rate_limiter = rate_limiter or auth.RateLimiter()
        self.download_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_DOWNLOADS)
        self.allowed_origins: set[str] = set(allowed_origins or ())

    def verify_pin(self, pin: str) -> bool:
        """Constant-time PIN check against the stored record."""
        return auth.verify_pin(pin, self.config.pin_record)


class PortalRequestHandler(BaseHTTPRequestHandler):
    """Stdlib request handler enforcing peer checks, session gating, and bounds."""

    protocol_version = "HTTP/1.1"
    server_version = "MungiPortal/1.0"
    state: PortalState  # injected via the server instance

    # -- Logging: route stdlib access logs through structured logging (no stderr). --
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Send access logs through the module logger instead of stderr."""
        logger.info("%s - %s", self.address_string(), format % args)

    # -- Peer identity ---------------------------------------------------------
    def _peer_ip(self) -> str:
        """Return the TCP peer IP (ignores any forwarded headers)."""
        return self.client_address[0]

    def _peer_is_tailscale(self) -> bool:
        """Return ``True`` iff the TCP peer is within ``100.64.0.0/10``."""
        return network.is_tailscale_address(self._peer_ip())

    # -- Session helpers -------------------------------------------------------
    def _current_session(self) -> auth.SessionToken | None:
        """Return the verified session from the request cookie, or ``None``."""
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        jar: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
        try:
            jar.load(cookie_header)
        except http.cookies.CookieError:
            return None
        morsel = jar.get(SESSION_COOKIE_NAME)
        if morsel is None:
            return None
        return self.state.sessions.verify(morsel.value)

    # -- Response helpers ------------------------------------------------------
    def _send_html(self, status: HTTPStatus, body: str) -> None:
        """Send an HTML response with a fixed Content-Length and security headers."""
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(
        self, status: HTTPStatus, body: bytes, content_type: str, *, filename: str | None = None
    ) -> None:
        """Send a raw byte body with a fixed Content-Length."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_security_headers(self) -> None:
        """Emit defense-in-depth response headers."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")

    def _send_status(self, status: HTTPStatus, message: str = "") -> None:
        """Send a tiny text/plain status response."""
        body = (message or status.phrase).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _redirect_to_login(self, status: HTTPStatus = HTTPStatus.UNAUTHORIZED) -> None:
        """Send a 401 (or given status) that points the client at the login page."""
        body = b"Authentication required. Open / to log in.\n"
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Location", "/")
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    # -- Routing ---------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        """Dispatch GET requests after the peer check."""
        if not self._guard_peer():
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._handle_index()
        elif path == "/assets/style.css":
            self._handle_style()
        elif path.startswith("/download/") and path.endswith("/text"):
            self._handle_download_text(path)
        elif path.startswith("/download/"):
            self._handle_download_get(path)
        else:
            self._send_status(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        """Dispatch POST requests after the peer check."""
        if not self._guard_peer():
            return
        path = urlsplit(self.path).path
        if path == "/login":
            self._handle_login()
        elif path == "/download":
            self._handle_download_post()
        elif path == "/shutdown":
            self._handle_shutdown()
        else:
            self._send_status(HTTPStatus.NOT_FOUND)

    def _guard_peer(self) -> bool:
        """Reject non-Tailscale peers with 403 (audited). Returns admit decision."""
        if not self._peer_is_tailscale():
            self.state.audit.record_failed_auth(
                peer=self._peer_ip(), reason="non_tailscale_peer", route=self.path
            )
            self._send_status(HTTPStatus.FORBIDDEN, "Forbidden")
            return False
        return True

    # -- Handlers --------------------------------------------------------------
    def _handle_index(self) -> None:
        """Render the login page (no session) or the session list (valid session)."""
        session = self._current_session()
        if session is None:
            self._send_html(HTTPStatus.OK, self._render_login())
            return
        self._send_html(HTTPStatus.OK, self._render_list(session))

    def _handle_style(self) -> None:
        """Serve the bundled stylesheet."""
        try:
            css = _load_asset("style.css")
        except OSError:
            self._send_status(HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(HTTPStatus.OK, css.encode("utf-8"), "text/css; charset=utf-8")

    def _render_login(self, error: str = "") -> str:
        """Return the login HTML with an optional error block."""
        template = _load_asset("login.html")
        error_block = f'<p class="error">{html.escape(error)}</p>' if error else ""
        return template.replace("{error_block}", error_block)

    def _render_list(self, session: auth.SessionToken) -> str:
        """Return the session-list HTML for an authenticated session."""
        template = _load_asset("list.html")
        sessions = data.list_sessions()
        csrf = self.state.sessions.csrf_for(session) or ""
        rows: list[str] = []
        for summary in sessions:
            sid = html.escape(summary.session_id)
            rows.append(
                "<tr>"
                f'<td><input type="checkbox" name="session_id" value="{sid}"></td>'
                f"<td>{sid}</td>"
                f'<td class="num">{summary.turn_count}</td>'
                f'<td class="num">{summary.audio_count}</td>'
                f'<td class="num">{html.escape(_human_bytes(summary.total_bytes))}</td>'
                f'<td><a class="row-link" href="/download/{sid}">ZIP</a> '
                f'<a class="row-link" href="/download/{sid}/text">텍스트</a></td>'
                "</tr>"
            )
        return (
            template.replace("{session_count}", str(len(sessions)))
            .replace("{csrf_token}", html.escape(csrf))
            .replace("{rows}", "\n".join(rows))
        )

    def _read_body(self, max_bytes: int) -> bytes | None:
        """Read the request body up to ``max_bytes``; return ``None`` if oversized."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length < 0 or length > max_bytes:
            return None
        return self.rfile.read(length)

    def _origin_allowed(self) -> bool:
        """Return ``True`` iff the Origin/Referer is same-origin (login CSRF defense).

        With no per-session token available pre-login, ``POST /login`` relies on
        ``SameSite=Strict`` plus this Origin/Referer check. The Host header is trusted only
        to derive the expected origin; the peer-IP gate already restricts callers.
        """
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        referer = self.headers.get("Referer")
        expected = {f"http://{host}", f"https://{host}"} | self.state.allowed_origins
        if origin is not None:
            return origin in expected
        if referer is not None:
            split = urlsplit(referer)
            return f"{split.scheme}://{split.netloc}" in expected
        # No Origin and no Referer: allow (native form posts may omit both), the
        # SameSite=Strict cookie + peer gate remain in force.
        return True

    def _handle_login(self) -> None:
        """Authenticate a PIN: rate-limit FIRST, then PBKDF2; issue a session on success."""
        peer = self._peer_ip()
        # --- PBKDF2-DoS guard: throttle/lockout BEFORE any hashing. ---
        decision = self.state.rate_limiter.check_allowed(peer)
        if not decision.allowed:
            self.state.audit.record_failed_auth(peer=peer, reason=decision.reason, route="/login")
            self.send_response(HTTPStatus.TOO_MANY_REQUESTS)
            self.send_header("Retry-After", str(decision.retry_after_seconds))
            self.send_header("Content-Length", "0")
            self._send_security_headers()
            self.end_headers()
            return
        if not self._origin_allowed():
            self.state.audit.record_failed_auth(peer=peer, reason="bad_origin", route="/login")
            self._send_status(HTTPStatus.FORBIDDEN, "Bad Origin")
            return
        body = self._read_body(MAX_LOGIN_BODY_BYTES)
        if body is None:
            self._send_status(HTTPStatus.BAD_REQUEST, "Bad Request")
            return
        fields = parse_qs(body.decode("utf-8", errors="replace"))
        pin_values = fields.get("pin", [])
        pin = pin_values[0] if pin_values else ""
        if not pin or not self.state.verify_pin(pin):
            self.state.rate_limiter.record_failure(peer)
            self.state.audit.record_failed_auth(peer=peer, reason="bad_pin", route="/login")
            self._send_html(HTTPStatus.UNAUTHORIZED, self._render_login("PIN이 올바르지 않습니다."))
            return
        # Success: reset the limiter, mint a session, set the cookie, redirect to "/".
        self.state.rate_limiter.record_success(peer)
        token, _csrf = self.state.sessions.issue()
        cookie = (
            f"{SESSION_COOKIE_NAME}={token}; HttpOnly; SameSite=Strict; "
            f"Path=/; Max-Age={COOKIE_MAX_AGE}"
        )
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self._send_security_headers()
        self.end_headers()

    def _require_session(self, route: str) -> auth.SessionToken | None:
        """Return a valid session or send 401 (audited) — call BEFORE any FS access."""
        session = self._current_session()
        if session is None:
            self.state.audit.record_failed_auth(
                peer=self._peer_ip(), reason="no_session", route=route
            )
            self._redirect_to_login()
            return None
        return session

    def _handle_download_get(self, path: str) -> None:
        """Serve a single-session zip (session-gated before any FS access)."""
        if self._require_session(path) is None:
            return
        raw_id = path[len("/download/") :]
        try:
            session_id = data.decode_and_validate_session_id(raw_id)
        except data.PortalDataError:
            self._send_status(HTTPStatus.NOT_FOUND)
            return
        self._stream_zip([session_id], path)

    def _handle_download_text(self, path: str) -> None:
        """Serve a single session's ``conversation.jsonl`` (session-gated)."""
        if self._require_session(path) is None:
            return
        raw_id = path[len("/download/") : -len("/text")]
        try:
            session_id = data.decode_and_validate_session_id(raw_id)
        except data.PortalDataError:
            self._send_status(HTTPStatus.NOT_FOUND)
            return
        try:
            body = data.read_transcript_bytes(session_id)
        except data.PortalDataError:
            self._send_status(HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(
            HTTPStatus.OK,
            body,
            "application/json; charset=utf-8",
            filename=f"{session_id}_conversation.jsonl",
        )
        self.state.audit.record_download(
            peer=self._peer_ip(),
            session_ids=[session_id],
            file_count=1,
            byte_count=len(body),
            route=path,
        )

    def _handle_download_post(self) -> None:
        """Serve a multi-session zip from posted ids (session-gated + CSRF)."""
        session = self._require_session("/download")
        if session is None:
            return
        body = self._read_body(MAX_DOWNLOAD_BODY_BYTES)
        if body is None:
            self._send_status(HTTPStatus.BAD_REQUEST, "Bad Request")
            return
        fields = parse_qs(body.decode("utf-8", errors="replace"))
        csrf = (fields.get("csrf_token", [""]) or [""])[0]
        if not self.state.sessions.check_csrf(session, csrf):
            self.state.audit.record_failed_auth(
                peer=self._peer_ip(), reason="bad_csrf", route="/download"
            )
            self._send_status(HTTPStatus.FORBIDDEN, "Bad CSRF token")
            return
        raw_ids = fields.get("session_id", [])
        session_ids: list[str] = []
        for raw in raw_ids:
            decoded = unquote(raw)
            if not data.is_safe_session_id(decoded):
                self._send_status(HTTPStatus.BAD_REQUEST, "Invalid session id")
                return
            session_ids.append(decoded)
        if not session_ids:
            self._send_status(HTTPStatus.BAD_REQUEST, "No sessions selected")
            return
        self._stream_zip(session_ids, "/download")

    def _handle_shutdown(self) -> None:
        """Stop the portal server (session-gated + CSRF).

        Mirrors the download-POST gating exactly — Tailscale peer (do_POST),
        valid session, valid CSRF. Triggers a graceful stop on a separate
        thread so the response flushes first; the clean exit means the
        ``Restart=on-failure`` systemd unit does NOT respawn the portal.
        """
        session = self._require_session("/shutdown")
        if session is None:
            return
        body = self._read_body(MAX_SHUTDOWN_BODY_BYTES)
        if body is None:
            self._send_status(HTTPStatus.BAD_REQUEST, "Bad Request")
            return
        fields = parse_qs(body.decode("utf-8", errors="replace"))
        csrf = (fields.get("csrf_token", [""]) or [""])[0]
        if not self.state.sessions.check_csrf(session, csrf):
            self.state.audit.record_failed_auth(
                peer=self._peer_ip(), reason="bad_csrf", route="/shutdown"
            )
            self._send_status(HTTPStatus.FORBIDDEN, "Bad CSRF token")
            return
        shutdown_page = (
            '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>포털 종료</title></head>"
            '<body style="font-family:sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem">'
            "<h1>포털을 종료했습니다</h1>"
            "<p>다운로드 포털 서버가 안전하게 멈췄습니다 — 메모리가 확보됩니다.</p>"
            "<p>다시 켜려면 젯슨에서 "
            "<code>sudo systemctl start mungi-download-portal.service</code> 를 실행하거나 "
            "기기를 재부팅하세요.</p></body></html>"
        )
        self.state.audit.record_shutdown(peer=self._peer_ip())
        logger.info("Portal shutdown requested by an authenticated session")
        self._send_html(HTTPStatus.OK, shutdown_page)
        threading.Thread(target=self.server.shutdown, name="portal-shutdown", daemon=True).start()

    def _stream_zip(self, session_ids: list[str], route: str) -> None:
        """Stream a STORE-mode zip via HTTP/1.1 chunked framing (bounded concurrency).

        Acquires the download semaphore (``503`` if saturated), builds the manifest, then
        emits ``Transfer-Encoding: chunked`` framed from opened fds with no buffering.
        """
        acquired = self.state.download_semaphore.acquire(blocking=False)
        if not acquired:
            self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Retry-After", "5")
            self.send_header("Content-Length", "0")
            self._send_security_headers()
            self.end_headers()
            return
        try:
            try:
                manifest_bytes, plan = data.build_manifest(session_ids)
            except data.PortalDataError:
                self._send_status(HTTPStatus.NOT_FOUND)
                return
            download_name = (
                f"{session_ids[0]}.zip" if len(session_ids) == 1 else "mungi_sessions.zip"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.send_header("Transfer-Encoding", "chunked")
            self._send_security_headers()
            self.end_headers()
            total = 0
            for chunk in data.stream_store_zip(manifest_bytes, plan):
                if not chunk:
                    continue
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                total += len(chunk)
            self.wfile.write(CHUNK_TERMINATOR)
            self.state.audit.record_download(
                peer=self._peer_ip(),
                session_ids=session_ids,
                file_count=len(plan),
                byte_count=total,
                route=route,
            )
        finally:
            self.state.download_semaphore.release()


def build_handler_class(state: PortalState) -> type[PortalRequestHandler]:
    """Return a handler subclass with ``state`` bound as a class attribute."""

    class _BoundHandler(PortalRequestHandler):
        pass

    _BoundHandler.state = state
    return _BoundHandler


class PortalServer:
    """Owns the bound :class:`ThreadingHTTPServer` and the IP-change watcher."""

    def __init__(
        self,
        state: PortalState,
        bind_address: str,
        *,
        port: int = PORTAL_PORT,
        resolve_ip: Callable[[], str] = network.resolve_tailscale_ipv4,
        on_ip_change: Callable[[], None] | None = None,
    ) -> None:
        """Construct a server bound to a verified Tailscale ``bind_address``.

        Args:
            state: Shared portal state.
            bind_address: The resolved ``100.64.0.0/10`` IPv4 to bind.
            port: TCP port (default 8765).
            resolve_ip: Tailscale IP resolver (injectable for tests).
            on_ip_change: Callback invoked when the tailnet IP changes (default: exit).

        Raises:
            ValueError: If ``bind_address`` is not a ``100.64.0.0/10`` address
                (refuses ``0.0.0.0``/LAN/public binds — fail-closed).
        """
        if not network.is_tailscale_address(bind_address):
            raise ValueError(
                f"refusing to bind non-Tailscale address {bind_address!r} "
                "(only 100.64.0.0/10 is permitted)"
            )
        self._state = state
        self._bind_address = bind_address
        self._port = port
        self._resolve_ip = resolve_ip
        self._on_ip_change = on_ip_change or _default_ip_change_exit
        state.allowed_origins.add(f"{bind_address}:{port}")
        handler_cls = build_handler_class(state)
        self._httpd = ThreadingHTTPServer((bind_address, port), handler_cls)
        self._httpd.timeout = SOCKET_TIMEOUT_S
        self._httpd.socket.settimeout(SOCKET_TIMEOUT_S)
        self._watch_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None

    @property
    def bind_address(self) -> str:
        """Return the bound IPv4 address."""
        return self._bind_address

    def start_ip_watcher(self) -> None:
        """Start the background thread that exits if the tailnet IP changes."""
        thread = threading.Thread(target=self._watch_ip, name="portal-ip-watch", daemon=True)
        self._watch_thread = thread
        thread.start()

    def _watch_ip(self) -> None:
        """Poll the tailnet IP; trigger the change callback on mismatch/unavailability."""
        while not self._watch_stop.wait(IP_WATCH_INTERVAL_S):
            try:
                current = self._resolve_ip()
            except network.TailscaleUnavailableError:
                logger.error("Tailscale IP unavailable during watch — triggering rebind")
                self._on_ip_change()
                return
            if current != self._bind_address:
                logger.warning(
                    "tailnet IP changed %s -> %s — triggering rebind",
                    self._bind_address,
                    current,
                )
                self._on_ip_change()
                return

    def serve_forever(self) -> None:
        """Serve requests until shutdown (starts the IP watcher first)."""
        self.start_ip_watcher()
        try:
            self._httpd.serve_forever()
        finally:
            self._watch_stop.set()
            self._httpd.server_close()

    def shutdown(self) -> None:
        """Stop serving and join the server socket."""
        self._watch_stop.set()
        self._httpd.shutdown()
        self._httpd.server_close()


def _default_ip_change_exit() -> None:
    """Exit non-zero so systemd restarts the unit with a fresh bind."""
    logger.error("exiting for clean rebind (systemd will restart)")
    # Use os._exit-free hard stop: raise SystemExit on the main loop via interrupt is
    # unreliable from a thread, so exit the process directly.
    import os

    os._exit(3)


def serve(
    config: PortalConfig,
    audit: AuditLog,
    *,
    resolve_ip: Callable[[], str] = network.resolve_tailscale_ipv4,
) -> int:
    """Resolve the bind address, construct the server, and serve forever.

    Args:
        config: Loaded portal config.
        audit: Fail-closed audit log.
        resolve_ip: Tailscale IP resolver (injectable for tests).

    Returns:
        A process exit code (non-zero on fail-closed bind failure).
    """
    try:
        bind_address = resolve_ip()
    except network.TailscaleUnavailableError as exc:
        logger.error("%s", exc)
        return 2
    state = PortalState(config, audit)
    try:
        server = PortalServer(state, bind_address, resolve_ip=resolve_ip)
    except (ValueError, OSError) as exc:
        logger.error("failed to bind portal server: %s", exc)
        return 2
    logger.info("portal serving on %s:%d (Tailscale-only)", bind_address, PORTAL_PORT)
    server.serve_forever()
    return 0


def _now_monotonic() -> float:
    """Return a monotonic timestamp (indirection kept for symmetry with auth/tests)."""
    return time.monotonic()
