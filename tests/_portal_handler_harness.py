"""Test harness: drive :class:`PortalRequestHandler` with no real socket.

The handler is a stdlib ``BaseHTTPRequestHandler`` whose ``__init__`` normally calls
``handle()`` on a live socket. For tests we bypass that constructor and wire ``rfile`` /
``wfile`` to ``BytesIO`` buffers, set ``client_address`` / ``headers`` directly, then invoke
the routing method. A small response parser turns the raw ``wfile`` bytes into a
status/headers/body triple (de-chunking ``Transfer-Encoding: chunked`` bodies).
"""

from __future__ import annotations

import email.parser
import io
from dataclasses import dataclass, field

from parental.download_portal import server


@dataclass
class FakeResponse:
    """A parsed HTTP response captured from the handler's ``wfile``."""

    status: int
    reason: str
    headers: dict[str, str]
    body: bytes
    set_cookies: list[str] = field(default_factory=list)


class _RecordingHandler(server.PortalRequestHandler):
    """A handler whose ``__init__`` is bypassed; wired to in-memory buffers."""

    def __init__(self) -> None:  # noqa: D401 - intentionally does not call super().__init__
        """Do nothing (the harness wires attributes after construction)."""


def build_fake_handler(
    state: server.PortalState,
    *,
    peer_ip: str,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> server.PortalRequestHandler:
    """Construct a handler bound to ``state`` with a fake request (no socket).

    Args:
        state: The PortalState to bind.
        peer_ip: The simulated TCP peer IP.
        method: HTTP method.
        path: Request path.
        headers: Request headers.
        body: Request body bytes.

    Returns:
        A ready-to-invoke handler whose ``wfile`` captures the response.
    """
    handler = _RecordingHandler()
    handler.state = state
    handler.client_address = (peer_ip, 54321)
    handler.command = method
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    raw_headers = dict(headers or {})
    if body and "Content-Length" not in raw_headers:
        raw_headers["Content-Length"] = str(len(body))
    header_text = "".join(f"{k}: {v}\r\n" for k, v in raw_headers.items())
    handler.headers = email.parser.Parser().parsestr(header_text)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    return handler


def invoke(handler: server.PortalRequestHandler) -> FakeResponse:
    """Dispatch the handler's method and parse the captured response."""
    if handler.command == "GET":
        handler.do_GET()
    elif handler.command == "POST":
        handler.do_POST()
    else:  # pragma: no cover - tests only use GET/POST
        raise ValueError(f"unsupported method {handler.command}")
    return parse_response(handler.wfile.getvalue())


def parse_response(raw: bytes) -> FakeResponse:
    """Parse raw HTTP/1.1 response bytes into a :class:`FakeResponse`."""
    head, _, rest = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status_line = lines[0].decode("latin-1")
    parts = status_line.split(" ", 2)
    status = int(parts[1])
    reason = parts[2] if len(parts) > 2 else ""
    headers: dict[str, str] = {}
    set_cookies: list[str] = []
    for line in lines[1:]:
        key, _, value = line.decode("latin-1").partition(":")
        key = key.strip()
        value = value.strip()
        if key.lower() == "set-cookie":
            set_cookies.append(value)
        else:
            headers[key] = value
    body = rest
    if headers.get("Transfer-Encoding", "").lower() == "chunked":
        body = _dechunk(rest)
    return FakeResponse(
        status=status, reason=reason, headers=headers, body=body, set_cookies=set_cookies
    )


def _dechunk(raw: bytes) -> bytes:
    """Decode an HTTP/1.1 chunked body into the concatenated payload."""
    out = bytearray()
    pos = 0
    while pos < len(raw):
        eol = raw.find(b"\r\n", pos)
        if eol == -1:
            break
        size = int(raw[pos:eol].split(b";", 1)[0], 16)
        pos = eol + 2
        if size == 0:
            break
        out += raw[pos : pos + size]
        pos += size + 2  # skip the chunk data and its trailing CRLF
    return bytes(out)
