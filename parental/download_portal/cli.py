"""Command-line entry points for the download portal (``set-pin``).

``set-pin`` reads the PIN from stdin (never an argv argument, never echoed, never
logged), hashes it with PBKDF2-HMAC-SHA256, generates a fresh 32-byte session secret,
and writes ``portal.json`` with mode ``0600``. Generating a new secret on every
``set-pin`` invalidates all outstanding sessions (rotation on PIN reset).
"""

from __future__ import annotations

import getpass
import logging
import os
import sys
from collections.abc import Sequence

from . import auth, config

logger = logging.getLogger(__name__)


def _read_pin_from_stdin(prompt: str) -> str:
    """Read a single PIN value without echoing it to the terminal.

    Uses ``getpass`` when attached to a TTY; otherwise reads one line from stdin (so the
    PIN can be piped in for automation without ever touching argv).
    """
    if sys.stdin is not None and sys.stdin.isatty():
        return getpass.getpass(prompt)
    line = sys.stdin.readline()
    return line.rstrip("\n").rstrip("\r")


def set_pin(argv: Sequence[str] | None = None) -> int:
    """Interactively set the portal PIN, writing ``portal.json`` (``0600``).

    Reads the PIN twice (confirmation) when on a TTY; from a single stdin line otherwise.
    Validates the format (>=8 digits) and never echoes or logs the value.

    Args:
        argv: Unused positional args (accepted for a uniform CLI signature).

    Returns:
        Process exit code: ``0`` on success, non-zero on validation/IO failure.
    """
    interactive = sys.stdin is not None and sys.stdin.isatty()
    pin = _read_pin_from_stdin("Enter portal PIN (>=8 digits): ")
    if interactive:
        confirm = _read_pin_from_stdin("Confirm portal PIN: ")
        if pin != confirm:
            sys.stderr.write("error: PINs did not match\n")
            return 2
    if not auth.is_valid_pin_format(pin):
        sys.stderr.write(
            f"error: PIN must be at least {auth.PIN_MIN_DIGITS} digits (digits only)\n"
        )
        return 2
    record = auth.hash_pin(pin)
    secret = os.urandom(auth.SESSION_SECRET_BYTES)
    config.save_config(record, secret)
    # Best-effort scrub of the local references; CPython strings are immutable so this
    # only drops the names, but we avoid keeping the PIN around any longer than needed.
    del pin
    sys.stdout.write(f"PIN set. Config written to {config.portal_config_path()}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a portal subcommand.

    Args:
        argv: Argument vector excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stderr.write("usage: python -m parental.download_portal set-pin\n")
        return 2
    command = args[0]
    if command == "set-pin":
        return set_pin(args[1:])
    sys.stderr.write(f"error: unknown command {command!r}\n")
    return 2
