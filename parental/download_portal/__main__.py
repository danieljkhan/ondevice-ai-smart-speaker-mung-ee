"""Daemon entry point with fail-closed flag + config gating.

Behavior:

- ``python -m parental.download_portal set-pin`` dispatches to the CLI.
- ``python -m parental.download_portal`` (daemon): exits ``0`` immediately if the feature
  flag ``MUNGI_DOWNLOAD_PORTAL`` is not ``"1"`` (belt-and-suspenders with the systemd
  ``ExecCondition``); refuses to start (non-zero) if ``portal.json`` is uninitialized; and
  refuses to start (non-zero) if the audit log cannot be initialized (fail-closed). Only on
  a fully valid configuration does it bind the Tailscale-only server and serve.
"""

from __future__ import annotations

import logging
import os
import sys

from . import FEATURE_FLAG_ENV, cli, config, server
from .audit import AuditLog

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure structured logging to stderr (journald captures it under systemd)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def run_daemon() -> int:
    """Run the portal daemon after enforcing flag + config + audit gates.

    Returns:
        A process exit code: ``0`` for a clean flag-disabled no-op or normal shutdown;
        non-zero when fail-closed (uninitialized config, audit init failure, bind failure).
    """
    if os.environ.get(FEATURE_FLAG_ENV) != "1":
        logger.info("%s != '1' — portal disabled; exiting cleanly (no-op)", FEATURE_FLAG_ENV)
        return 0
    if not config.is_initialized():
        logger.error(
            "portal.json is uninitialized — run 'python -m parental.download_portal "
            "set-pin' first; refusing to start"
        )
        return 2
    try:
        portal_config = config.load_config()
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("failed to load portal config: %s", exc)
        return 2
    try:
        audit = AuditLog()
    except OSError as exc:
        logger.error("audit log init failed (fail-closed): %s", exc)
        return 2
    return server.serve(portal_config, audit)


def main() -> int:
    """Top-level dispatch between the ``set-pin`` CLI and the daemon."""
    _configure_logging()
    args = sys.argv[1:]
    if args:
        return cli.main(args)
    return run_daemon()


if __name__ == "__main__":
    sys.exit(main())
