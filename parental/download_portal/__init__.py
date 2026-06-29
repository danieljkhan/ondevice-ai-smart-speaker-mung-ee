"""Conversation Data Download Portal — Tailscale-only, PIN-gated, read-only.

This package serves child-뭉이 conversation data (transcripts + audio) from the Jetson to
a parent's PC over Tailscale ONLY, behind a hashed PIN. It is disabled by default and is
enforced fail-closed at every boundary (binding, auth, path resolution, downloads, audit).
See ``Dev_Plan/2026-06-18-conversation-download-portal-plan.md`` for the authoritative
security specification.
"""

from __future__ import annotations

FEATURE_FLAG_ENV = "MUNGI_DOWNLOAD_PORTAL"
