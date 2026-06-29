"""Shared pytest fixtures for the Mungi test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_dummy_sdl_video_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run tests against Pygame's headless SDL video backend."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
