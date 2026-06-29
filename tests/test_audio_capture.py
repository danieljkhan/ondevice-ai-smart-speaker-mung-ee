"""Tests for the AudioCapture sounddevice wrapper."""

from __future__ import annotations

import logging
import queue
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.audio_capture import AUDIO_QUEUE_MAXSIZE, CALLBACK_WARNING_INTERVAL_S, AudioCapture


class FakeInputStream:
    """Minimal sounddevice InputStream fake."""

    instances: list[FakeInputStream] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self) -> None:
        """Record stream start."""
        self.started = True

    def stop(self) -> None:
        """Record stream stop."""
        self.stopped = True

    def close(self) -> None:
        """Record stream close."""
        self.closed = True


def install_fake_sounddevice(monkeypatch: Any) -> None:
    """Install the fake sounddevice module."""
    FakeInputStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeInputStream))


def test_start_stop_lifecycle_is_idempotent(monkeypatch: Any) -> None:
    """start() opens one stream and stop() closes it once."""
    install_fake_sounddevice(monkeypatch)
    capture = AudioCapture(device="usb-mic")

    capture.start()
    capture.start()
    assert len(FakeInputStream.instances) == 1
    stream = FakeInputStream.instances[0]
    assert stream.started is True
    assert stream.kwargs["samplerate"] == 48_000
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["dtype"] == "float32"
    assert stream.kwargs["device"] == "usb-mic"

    capture.stop()
    capture.stop()
    assert stream.stopped is True
    assert stream.closed is True


def test_stop_event_tracks_capture_lifecycle(monkeypatch: Any) -> None:
    """start() clears the stop signal; stop() and close() set it."""
    install_fake_sounddevice(monkeypatch)
    capture = AudioCapture()
    capture.stop_event.set()

    capture.start()
    assert capture.stop_event.is_set() is False

    capture.stop()
    assert capture.stop_event.is_set() is True

    capture.start()
    assert capture.stop_event.is_set() is False

    capture.close()
    assert capture.stop_event.is_set() is True


def test_callback_drops_oldest_when_queue_is_full() -> None:
    """Queue overflow drops the oldest frame and keeps the newest."""
    capture = AudioCapture(queue_maxsize=2)
    capture.audio_queue.put_nowait(np.array([1.0], dtype=np.float32))
    capture.audio_queue.put_nowait(np.array([2.0], dtype=np.float32))

    capture._callback(np.array([3.0], dtype=np.float32), 1, None, None)

    first = capture.audio_queue.get_nowait()
    second = capture.audio_queue.get_nowait()
    assert first.tolist() == [2.0]
    assert second.tolist() == [3.0]


def test_callback_queue_overflow_warnings_are_rate_limited(caplog: Any) -> None:
    """Repeated queue overflow emits a compact warning summary about once per second."""
    now = 10.0
    capture = AudioCapture(queue_maxsize=1, monotonic_clock=lambda: now)
    caplog.set_level(logging.WARNING, logger="core.audio_capture")
    capture.audio_queue.put_nowait(np.array([0.0], dtype=np.float32))

    for value in range(5):
        capture._callback(np.array([float(value)], dtype=np.float32), 1, None, None)

    records = [record for record in caplog.records if record.name == "core.audio_capture"]
    assert len(records) == 1
    assert "dropped_oldest=1" in records[0].message

    now += CALLBACK_WARNING_INTERVAL_S
    capture._callback(np.array([5.0], dtype=np.float32), 1, None, None)

    records = [record for record in caplog.records if record.name == "core.audio_capture"]
    assert len(records) == 2
    assert "dropped_oldest=5" in records[1].message


def test_callback_status_warnings_are_rate_limited(caplog: Any) -> None:
    """Repeated sounddevice status warnings are rate-limited in the callback."""
    now = 20.0
    capture = AudioCapture(monotonic_clock=lambda: now)
    caplog.set_level(logging.WARNING, logger="core.audio_capture")

    for _ in range(3):
        capture._callback(np.array([1.0], dtype=np.float32), 1, None, "overflow")

    records = [record for record in caplog.records if record.name == "core.audio_capture"]
    assert len(records) == 1
    assert "status_count=1" in records[0].message
    assert "last_status=overflow" in records[0].message

    now += CALLBACK_WARNING_INTERVAL_S
    capture._callback(np.array([1.0], dtype=np.float32), 1, None, "overflow")

    records = [record for record in caplog.records if record.name == "core.audio_capture"]
    assert len(records) == 2
    assert "status_count=3" in records[1].message


def test_mute_then_unmute_toggles_is_muted() -> None:
    """mute() and unmute() toggle the public muted state."""
    capture = AudioCapture()

    assert capture.is_muted() is False
    capture.mute()
    assert capture.is_muted() is True
    capture.unmute()
    assert capture.is_muted() is False


def test_drain_empties_public_queue() -> None:
    """drain() removes all queued capture frames."""
    capture = AudioCapture()
    for value in range(3):
        capture.audio_queue.put_nowait(np.array([value], dtype=np.float32))

    capture.drain()

    assert capture.audio_queue.empty()


def test_pause_then_resume_is_idempotent_when_no_stream() -> None:
    """pause() and resume() are no-ops before the stream starts."""
    capture = AudioCapture()

    capture.pause()
    capture.resume()

    assert capture.stop_event.is_set() is False


def test_pause_drains_queue_after_stopping_active_stream(monkeypatch: Any) -> None:
    """pause() clears stale queued frames after stopping an active stream."""
    install_fake_sounddevice(monkeypatch)
    capture = AudioCapture()
    capture.start()
    capture.audio_queue.put_nowait(np.zeros(1, dtype=np.float32))

    capture.pause()

    assert FakeInputStream.instances[0].stopped is True
    assert capture.audio_queue.empty()


def test_pause_drains_queue_when_stream_is_already_stopped(monkeypatch: Any) -> None:
    """pause() also clears queued frames when the stream was already stopped."""
    install_fake_sounddevice(monkeypatch)
    capture = AudioCapture()
    capture.start()
    stream = FakeInputStream.instances[0]
    stream.stopped = True
    capture.audio_queue.put_nowait(np.zeros(1, dtype=np.float32))

    capture.pause()

    assert capture.audio_queue.empty()


def test_callback_drops_frames_when_muted() -> None:
    """Muted capture drops callback frames before they enter the public queue."""
    capture = AudioCapture()
    capture.mute()

    capture._callback(np.array([1.0], dtype=np.float32), 1, None, None)

    assert capture.audio_queue.empty()


def test_close_drains_queue_and_prevents_restart(monkeypatch: Any) -> None:
    """close() drains queued frames and disallows later start()."""
    install_fake_sounddevice(monkeypatch)
    capture = AudioCapture()
    capture.start()
    capture.audio_queue.put_nowait(np.zeros(1, dtype=np.float32))

    capture.close()

    assert capture.audio_queue.empty()
    try:
        capture.audio_queue.get_nowait()
    except queue.Empty:
        pass
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("queue should be empty")

    try:
        capture.start()
    except RuntimeError as exc:
        assert "cannot be restarted" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("restart after close should fail")


def test_audio_queue_default_maxsize_constant() -> None:
    """The capture queue stays capped at 100 frames."""
    capture = AudioCapture()
    assert AUDIO_QUEUE_MAXSIZE == 100
    assert capture.audio_queue.maxsize == 100
