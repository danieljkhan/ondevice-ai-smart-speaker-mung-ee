"""Authentication, session tokens, and rate limiting for the download portal.

Security model (see ``Dev_Plan/2026-06-18-conversation-download-portal-plan.md`` §4):

- PIN is stored only as a PBKDF2-HMAC-SHA256 hash (>=200k iterations, 16-byte random
  salt, >=8 digits). Verification is constant-time.
- A successful login issues an HMAC-SHA256 session token over ``{v, iat, exp, nonce}``
  signed with a 32-byte server secret. Tokens are stateless but bound to a per-session
  CSRF token tracked in memory.
- The global throttle and the per-peer lockout are evaluated **before** any PBKDF2 work
  so that a flood of login attempts cannot force expensive hashing (PBKDF2-DoS guard).
- All rate-limit bookkeeping uses a monotonic clock; counters reset on a successful
  login and expire after a 15-minute idle TTL.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- PIN / hashing parameters -------------------------------------------------
PIN_MIN_DIGITS = 8
PBKDF2_MIN_ITERATIONS = 200_000
PBKDF2_DEFAULT_ITERATIONS = 200_000
PBKDF2_SALT_BYTES = 16
PBKDF2_DKLEN = 32
PBKDF2_HASH_NAME = "sha256"

# --- Session token parameters -------------------------------------------------
SESSION_SECRET_BYTES = 32
SESSION_TOKEN_VERSION = 1
SESSION_TTL_SECONDS = 1800  # 30 minutes (matches cookie Max-Age)
SESSION_NONCE_BYTES = 16
CSRF_TOKEN_BYTES = 32

# --- Rate-limit parameters ----------------------------------------------------
PEER_LOCKOUT_THRESHOLD = 5  # failures before a peer is locked
PEER_LOCKOUT_SECONDS = 15 * 60  # 15-minute lockout
PEER_IDLE_TTL_SECONDS = 15 * 60  # forget a peer after 15 min idle
GLOBAL_THROTTLE_MAX_ATTEMPTS = 30  # max login attempts within the window
GLOBAL_THROTTLE_WINDOW_SECONDS = 60


def hash_pin(
    pin: str,
    *,
    iterations: int = PBKDF2_DEFAULT_ITERATIONS,
    salt: bytes | None = None,
) -> dict[str, object]:
    """Return a schema-versioned PBKDF2-HMAC-SHA256 record for ``pin``.

    The returned mapping is JSON-serializable and never contains the plaintext PIN.

    Args:
        pin: The plaintext PIN (digits only, length validated by the caller).
        iterations: PBKDF2 iteration count; clamped to ``PBKDF2_MIN_ITERATIONS``.
        salt: Optional explicit salt (for tests); a random 16-byte salt otherwise.

    Returns:
        A dict with ``algorithm``, ``iterations``, ``salt`` (b64), and ``hash`` (b64).
    """
    if iterations < PBKDF2_MIN_ITERATIONS:
        iterations = PBKDF2_MIN_ITERATIONS
    if salt is None:
        salt = os.urandom(PBKDF2_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME, pin.encode("utf-8"), salt, iterations, dklen=PBKDF2_DKLEN
    )
    return {
        "algorithm": "pbkdf2_hmac_sha256",
        "iterations": iterations,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(derived).decode("ascii"),
    }


def verify_pin(pin: str, record: dict[str, object]) -> bool:
    """Constant-time check of ``pin`` against a stored PBKDF2 ``record``.

    Args:
        pin: The candidate plaintext PIN.
        record: A record previously produced by :func:`hash_pin`.

    Returns:
        ``True`` iff the PIN matches; ``False`` on mismatch or malformed record.
    """
    try:
        salt = base64.b64decode(str(record["salt"]))
        expected = base64.b64decode(str(record["hash"]))
        iterations = int(str(record["iterations"]))
    except (KeyError, ValueError, TypeError):
        return False
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME, pin.encode("utf-8"), salt, iterations, dklen=len(expected)
    )
    return hmac.compare_digest(derived, expected)


def is_valid_pin_format(pin: str) -> bool:
    """Return ``True`` iff ``pin`` is all digits and at least ``PIN_MIN_DIGITS`` long."""
    return pin.isdigit() and len(pin) >= PIN_MIN_DIGITS


def _b64url_encode(raw: bytes) -> str:
    """Return URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """Decode URL-safe base64 that may be missing padding."""
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


@dataclass(frozen=True)
class SessionToken:
    """A decoded, verified session token payload."""

    version: int
    issued_at: int
    expires_at: int
    nonce: str


class SessionManager:
    """Issue and verify HMAC-signed stateless session tokens + per-session CSRF.

    The signing secret rotates on PIN reset (a new manager is constructed with the new
    secret). CSRF tokens are tracked in memory keyed by the session nonce so that a
    daemon restart invalidates every outstanding session.
    """

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the manager.

        Args:
            secret: The 32-byte HMAC signing secret.
            ttl_seconds: Session lifetime in seconds.
            now: Wall-clock source (injectable for tests). Used only for ``iat``/``exp``;
                token expiry is an absolute wall-clock timestamp.
        """
        if len(secret) < SESSION_SECRET_BYTES:
            raise ValueError("session secret must be at least 32 bytes")
        self._secret = secret
        self._ttl = ttl_seconds
        self._now = now
        self._csrf_by_nonce: dict[str, str] = {}
        self._lock = threading.Lock()

    def issue(self) -> tuple[str, str]:
        """Mint a new session token and its bound CSRF token.

        Returns:
            ``(session_token, csrf_token)``. The session token is opaque
            (``<payload_b64url>.<sig_b64url>``); the CSRF token is a random hex string.
        """
        issued_at = int(self._now())
        nonce = _b64url_encode(os.urandom(SESSION_NONCE_BYTES))
        payload: dict[str, object] = {
            "v": SESSION_TOKEN_VERSION,
            "iat": issued_at,
            "exp": issued_at + self._ttl,
            "nonce": nonce,
        }
        token = self._sign(payload)
        csrf = secrets.token_hex(CSRF_TOKEN_BYTES)
        with self._lock:
            self._csrf_by_nonce[nonce] = csrf
        return token, csrf

    def _sign(self, payload: dict[str, object]) -> str:
        """Return ``<payload_b64url>.<sig_b64url>`` for ``payload``."""
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        body_b64 = _b64url_encode(body)
        sig = hmac.new(self._secret, body_b64.encode("ascii"), hashlib.sha256).digest()
        return f"{body_b64}.{_b64url_encode(sig)}"

    def verify(self, token: str | None) -> SessionToken | None:
        """Verify a session token's signature and expiry.

        Args:
            token: The opaque token string, or ``None``.

        Returns:
            A :class:`SessionToken` if valid and unexpired, else ``None``.
        """
        if not token or "." not in token:
            return None
        body_b64, _, sig_b64 = token.partition(".")
        expected_sig = hmac.new(self._secret, body_b64.encode("ascii"), hashlib.sha256).digest()
        try:
            provided_sig = _b64url_decode(sig_b64)
        except (ValueError, TypeError):
            return None
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None
        try:
            payload = json.loads(_b64url_decode(body_b64))
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            version = int(payload["v"])
            issued_at = int(payload["iat"])
            expires_at = int(payload["exp"])
            nonce = str(payload["nonce"])
        except (KeyError, ValueError, TypeError):
            return None
        if version != SESSION_TOKEN_VERSION:
            return None
        if int(self._now()) >= expires_at:
            return None
        return SessionToken(
            version=version, issued_at=issued_at, expires_at=expires_at, nonce=nonce
        )

    def csrf_for(self, session: SessionToken) -> str | None:
        """Return the CSRF token bound to ``session`` (or ``None`` if unknown)."""
        with self._lock:
            return self._csrf_by_nonce.get(session.nonce)

    def check_csrf(self, session: SessionToken, candidate: str | None) -> bool:
        """Constant-time check of ``candidate`` against the session's CSRF token."""
        expected = self.csrf_for(session)
        if expected is None or not candidate:
            return False
        return hmac.compare_digest(expected, candidate)

    def invalidate_all(self) -> None:
        """Drop all tracked CSRF tokens (called on PIN reset / shutdown)."""
        with self._lock:
            self._csrf_by_nonce.clear()


@dataclass
class _PeerState:
    """Per-peer failed-login bookkeeping (monotonic timestamps)."""

    failures: int = 0
    locked_until: float = 0.0
    last_seen: float = 0.0


@dataclass
class RateLimitDecision:
    """The outcome of a pre-PBKDF2 rate-limit check."""

    allowed: bool
    reason: str = ""
    retry_after_seconds: int = 0


class RateLimiter:
    """Global throttle + per-peer lockout, evaluated BEFORE any PBKDF2 work.

    ``check_allowed`` is the PBKDF2-DoS guard: the caller MUST invoke it and only run
    PBKDF2 when the decision is ``allowed``. All timestamps use a monotonic clock so the
    limiter is immune to wall-clock changes.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        lockout_threshold: int = PEER_LOCKOUT_THRESHOLD,
        lockout_seconds: int = PEER_LOCKOUT_SECONDS,
        idle_ttl_seconds: int = PEER_IDLE_TTL_SECONDS,
        global_max_attempts: int = GLOBAL_THROTTLE_MAX_ATTEMPTS,
        global_window_seconds: int = GLOBAL_THROTTLE_WINDOW_SECONDS,
    ) -> None:
        """Initialize the limiter with injectable clock + thresholds (for tests)."""
        self._monotonic = monotonic
        self._lockout_threshold = lockout_threshold
        self._lockout_seconds = lockout_seconds
        self._idle_ttl = idle_ttl_seconds
        self._global_max = global_max_attempts
        self._global_window = global_window_seconds
        self._peers: dict[str, _PeerState] = {}
        self._global_attempts: deque[float] = deque()
        self._lock = threading.Lock()

    def _expire_locked(self, now: float) -> None:
        """Drop peers idle longer than the TTL. Caller must hold the lock."""
        stale = [
            peer
            for peer, state in self._peers.items()
            if now - state.last_seen > self._idle_ttl and now >= state.locked_until
        ]
        for peer in stale:
            del self._peers[peer]

    def check_allowed(self, peer: str) -> RateLimitDecision:
        """Decide whether a login attempt from ``peer`` may proceed to PBKDF2.

        This records the attempt against the global window and refreshes the peer's
        ``last_seen``. It does NOT increment the failure counter — call
        :meth:`record_failure` after a failed verify, or :meth:`record_success`
        after a successful one.

        Args:
            peer: The TCP peer IP address.

        Returns:
            A :class:`RateLimitDecision`; ``allowed`` is ``False`` if the peer is locked
            or the global throttle is saturated.
        """
        now = self._monotonic()
        with self._lock:
            self._expire_locked(now)

            # Global throttle: drop timestamps outside the window, then cap.
            while self._global_attempts and now - self._global_attempts[0] > self._global_window:
                self._global_attempts.popleft()
            if len(self._global_attempts) >= self._global_max:
                if self._global_attempts:
                    retry = int(self._global_window - (now - self._global_attempts[0])) + 1
                else:  # max == 0: nothing is ever admitted
                    retry = self._global_window
                return RateLimitDecision(False, "global_throttle", max(retry, 1))

            # Per-peer lockout.
            state = self._peers.get(peer)
            if state is not None and now < state.locked_until:
                retry = int(state.locked_until - now) + 1
                return RateLimitDecision(False, "peer_locked", max(retry, 1))

            # Admit the attempt: record it globally and refresh the peer.
            self._global_attempts.append(now)
            if state is None:
                state = _PeerState()
                self._peers[peer] = state
            state.last_seen = now
            return RateLimitDecision(True)

    def record_failure(self, peer: str) -> None:
        """Increment ``peer``'s failure counter; lock it past the threshold."""
        now = self._monotonic()
        with self._lock:
            state = self._peers.get(peer)
            if state is None:
                state = _PeerState()
                self._peers[peer] = state
            state.failures += 1
            state.last_seen = now
            if state.failures >= self._lockout_threshold:
                state.locked_until = now + self._lockout_seconds

    def record_success(self, peer: str) -> None:
        """Reset ``peer``'s failure counter and clear any lockout."""
        now = self._monotonic()
        with self._lock:
            self._peers[peer] = _PeerState(last_seen=now)
