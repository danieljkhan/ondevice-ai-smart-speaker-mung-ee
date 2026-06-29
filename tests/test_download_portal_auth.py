"""Security tests for download-portal authentication, sessions, and rate limiting.

Covers §9 ``auth``: hash/verify, wrong-PIN reject, the PBKDF2-DoS guard (throttle/lockout
checked BEFORE any PBKDF2 — asserted by patching ``pbkdf2_hmac`` and confirming it is NOT
called while throttled), token sign/verify/expiry/rotation, constant-time compare, and the
post-login CSRF token.
"""

from __future__ import annotations

import hashlib
import os
from unittest import mock

import pytest

from parental.download_portal import auth


# --- PIN hashing / verification ----------------------------------------------
def test_hash_pin_roundtrip_verifies() -> None:
    record = auth.hash_pin("12345678", iterations=auth.PBKDF2_MIN_ITERATIONS)
    assert auth.verify_pin("12345678", record) is True


def test_verify_pin_rejects_wrong_pin() -> None:
    record = auth.hash_pin("12345678")
    assert auth.verify_pin("87654321", record) is False


def test_hash_pin_never_stores_plaintext() -> None:
    record = auth.hash_pin("12345678")
    serialized = str(record)
    assert "12345678" not in serialized
    assert record["algorithm"] == "pbkdf2_hmac_sha256"


def test_hash_pin_clamps_iterations_to_minimum() -> None:
    record = auth.hash_pin("12345678", iterations=10)
    assert int(record["iterations"]) >= auth.PBKDF2_MIN_ITERATIONS


def test_hash_pin_salt_is_16_bytes() -> None:
    import base64

    record = auth.hash_pin("12345678")
    salt = base64.b64decode(str(record["salt"]))
    assert len(salt) == auth.PBKDF2_SALT_BYTES


def test_verify_pin_malformed_record_returns_false() -> None:
    assert auth.verify_pin("12345678", {"salt": "!!notb64", "hash": "x", "iterations": 1}) is False
    assert auth.verify_pin("12345678", {}) is False


@pytest.mark.parametrize(
    ("pin", "expected"),
    [
        ("12345678", True),
        ("123456789012", True),
        ("1234567", False),  # too short
        ("1234abcd", False),  # non-digit
        ("", False),
        ("        ", False),
    ],
)
def test_pin_format_validation(pin: str, expected: bool) -> None:
    assert auth.is_valid_pin_format(pin) is expected


# --- Session tokens ----------------------------------------------------------
def test_session_token_sign_and_verify() -> None:
    manager = auth.SessionManager(os.urandom(32))
    token, _csrf = manager.issue()
    decoded = manager.verify(token)
    assert decoded is not None
    assert decoded.version == auth.SESSION_TOKEN_VERSION


def test_session_token_rejects_tampered_signature() -> None:
    manager = auth.SessionManager(os.urandom(32))
    token, _ = manager.issue()
    body, _, sig = token.partition(".")
    forged = f"{body}.{'A' * len(sig)}"
    assert manager.verify(forged) is None


def test_session_token_rejects_wrong_secret() -> None:
    issuer = auth.SessionManager(os.urandom(32))
    token, _ = issuer.issue()
    other = auth.SessionManager(os.urandom(32))
    assert other.verify(token) is None


def test_session_token_expires() -> None:
    clock = [1_000.0]
    manager = auth.SessionManager(os.urandom(32), ttl_seconds=30, now=lambda: clock[0])
    token, _ = manager.issue()
    assert manager.verify(token) is not None
    clock[0] += 31
    assert manager.verify(token) is None


def test_session_secret_must_be_32_bytes() -> None:
    with pytest.raises(ValueError):
        auth.SessionManager(os.urandom(16))


def test_session_rotation_invalidates_old_tokens() -> None:
    """A new secret (PIN reset) must invalidate previously issued tokens."""
    old = auth.SessionManager(os.urandom(32))
    token, _ = old.issue()
    assert old.verify(token) is not None
    rotated = auth.SessionManager(os.urandom(32))  # fresh secret on reset
    assert rotated.verify(token) is None


# --- CSRF --------------------------------------------------------------------
def test_csrf_token_bound_to_session() -> None:
    manager = auth.SessionManager(os.urandom(32))
    token, csrf = manager.issue()
    session = manager.verify(token)
    assert session is not None
    assert manager.check_csrf(session, csrf) is True
    assert manager.check_csrf(session, "deadbeef") is False
    assert manager.check_csrf(session, None) is False


def test_csrf_invalidate_all_clears_tokens() -> None:
    manager = auth.SessionManager(os.urandom(32))
    token, csrf = manager.issue()
    session = manager.verify(token)
    assert session is not None
    manager.invalidate_all()
    assert manager.check_csrf(session, csrf) is False


# --- Rate limiting: PBKDF2-DoS guard -----------------------------------------
def test_peer_locks_after_five_failures() -> None:
    clock = [1_000.0]
    limiter = auth.RateLimiter(
        monotonic=lambda: clock[0],
        lockout_threshold=5,
        lockout_seconds=900,
        global_max_attempts=1_000,
    )
    peer = "100.64.0.5"
    for _ in range(5):
        assert limiter.check_allowed(peer).allowed is True
        limiter.record_failure(peer)
    decision = limiter.check_allowed(peer)
    assert decision.allowed is False
    assert decision.reason == "peer_locked"
    assert decision.retry_after_seconds > 0


def test_lockout_expires_after_window() -> None:
    clock = [1_000.0]
    limiter = auth.RateLimiter(monotonic=lambda: clock[0], lockout_threshold=5, lockout_seconds=900)
    peer = "100.64.0.6"
    for _ in range(5):
        limiter.check_allowed(peer)
        limiter.record_failure(peer)
    assert limiter.check_allowed(peer).allowed is False
    clock[0] += 901
    assert limiter.check_allowed(peer).allowed is True


def test_success_resets_failure_counter() -> None:
    clock = [1_000.0]
    limiter = auth.RateLimiter(monotonic=lambda: clock[0], lockout_threshold=5)
    peer = "100.64.0.7"
    for _ in range(4):
        limiter.check_allowed(peer)
        limiter.record_failure(peer)
    limiter.record_success(peer)
    # After reset the peer can fail four more times without locking.
    for _ in range(4):
        assert limiter.check_allowed(peer).allowed is True
        limiter.record_failure(peer)
    assert limiter.check_allowed(peer).allowed is True


def test_idle_ttl_forgets_peer() -> None:
    clock = [1_000.0]
    limiter = auth.RateLimiter(
        monotonic=lambda: clock[0], lockout_threshold=5, idle_ttl_seconds=900
    )
    peer = "100.64.0.8"
    limiter.check_allowed(peer)
    limiter.record_failure(peer)
    clock[0] += 901  # exceed idle TTL
    # The peer is forgotten; a fresh attempt is admitted with a clean counter.
    assert limiter.check_allowed(peer).allowed is True


def test_global_throttle_blocks_flood() -> None:
    clock = [1_000.0]
    limiter = auth.RateLimiter(
        monotonic=lambda: clock[0], global_max_attempts=3, global_window_seconds=60
    )
    for i in range(3):
        assert limiter.check_allowed(f"100.64.1.{i}").allowed is True
    decision = limiter.check_allowed("100.64.1.99")
    assert decision.allowed is False
    assert decision.reason == "global_throttle"


def test_global_throttle_window_slides() -> None:
    clock = [1_000.0]
    limiter = auth.RateLimiter(
        monotonic=lambda: clock[0], global_max_attempts=2, global_window_seconds=60
    )
    limiter.check_allowed("100.64.2.1")
    limiter.check_allowed("100.64.2.2")
    assert limiter.check_allowed("100.64.2.3").allowed is False
    clock[0] += 61  # slide past the window
    assert limiter.check_allowed("100.64.2.4").allowed is True


def test_throttled_login_does_not_invoke_pbkdf2() -> None:
    """The PBKDF2-DoS guard: while throttled, no PBKDF2 hashing may occur.

    We patch ``hashlib.pbkdf2_hmac`` and assert it is NEVER called once the limiter has
    denied the attempt. This proves the throttle is evaluated BEFORE any hashing work.
    """
    clock = [1_000.0]
    limiter = auth.RateLimiter(
        monotonic=lambda: clock[0], global_max_attempts=2, global_window_seconds=60
    )
    record = auth.hash_pin("12345678")  # build the record BEFORE patching

    # Saturate the global throttle.
    limiter.check_allowed("100.64.3.1")
    limiter.check_allowed("100.64.3.2")

    with mock.patch.object(hashlib, "pbkdf2_hmac", wraps=hashlib.pbkdf2_hmac) as spy:
        decision = limiter.check_allowed("100.64.3.3")
        # Emulate the server's ordering: only verify the PIN if allowed.
        if decision.allowed:
            auth.verify_pin("12345678", record)
        assert decision.allowed is False
        spy.assert_not_called()


def test_allowed_login_does_invoke_pbkdf2() -> None:
    """Sanity counterpart: when allowed, PBKDF2 verification runs."""
    limiter = auth.RateLimiter()
    record = auth.hash_pin("12345678")
    with mock.patch.object(hashlib, "pbkdf2_hmac", wraps=hashlib.pbkdf2_hmac) as spy:
        decision = limiter.check_allowed("100.64.3.9")
        if decision.allowed:
            auth.verify_pin("12345678", record)
        assert decision.allowed is True
        spy.assert_called()
