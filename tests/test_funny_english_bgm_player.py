"""Tests for the Funny English BGM player."""

from __future__ import annotations

from typing import Any

import numpy as np

from core import bgm_player
from core.bgm_player import BgmPlayer


class FakeStream:
    """Fake sounddevice OutputStream."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        """Record stream start."""
        self.started = True

    def stop(self) -> None:
        """Record stream stop."""
        self.stopped = True

    def close(self) -> None:
        """Record stream close."""
        self.closed = True


class FakeSoundDevice:
    """Fake sounddevice module."""

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []

    def query_devices(self) -> list[dict[str, Any]]:
        """Return one output-capable fake device."""
        return [{"name": "USB Audio", "max_output_channels": 2, "default_samplerate": 16000}]

    def OutputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
        """Create a fake output stream."""
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


def test_bgm_player_starts_ducks_and_stops(monkeypatch: Any) -> None:
    """BGM playback owns a separate non-blocking loop stream."""
    fake_sd = FakeSoundDevice()
    monkeypatch.setattr(bgm_player, "_sd", fake_sd)
    player = BgmPlayer()

    player.start_loop(np.ones(16, dtype=np.float32), 16_000, volume=0.2)
    player.duck(volume=0.0)
    player.stop()

    assert fake_sd.streams[0].started is True
    assert player.volume == 0.0
    assert fake_sd.streams[0].stopped is True
    assert fake_sd.streams[0].closed is True
