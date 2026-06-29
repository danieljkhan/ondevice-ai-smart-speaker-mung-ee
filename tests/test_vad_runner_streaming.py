"""Tests for the streaming VAD wrapper."""

from __future__ import annotations

import queue
import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from models import vad_runner


class _FakeProb:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        """Return the fake speech probability."""
        return self._value


class _SequenceVadModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self.reset_calls = 0
        self.calls: list[Any] = []

    def reset_states(self) -> None:
        """Record reset calls."""
        self.reset_calls += 1

    def __call__(self, tensor: Any, sample_rate: int) -> _FakeProb:
        """Return the next configured speech probability."""
        self.calls.append((tensor, sample_rate))
        if not self._probabilities:
            return _FakeProb(0.0)
        return _FakeProb(self._probabilities.pop(0))


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(FloatTensor=lambda chunk: chunk))


def _queue_with_attrs(
    *,
    model: _SequenceVadModel,
    sample_rate: int = vad_runner.SAMPLE_RATE,
) -> queue.Queue[np.ndarray]:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    audio_queue.vad_model = model
    audio_queue.sample_rate = sample_rate
    return audio_queue


def test_iter_utterances_segments_after_800ms_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One speech window followed by 25 silence windows yields one utterance."""
    _install_fake_torch(monkeypatch)
    model = _SequenceVadModel([0.9, *([0.0] * vad_runner.DEFAULT_STREAMING_SILENCE_FRAMES)])
    audio_queue = _queue_with_attrs(model=model)
    audio_queue.put(np.ones(vad_runner.WINDOW_SIZE_SAMPLES, dtype=np.float32))
    for _ in range(vad_runner.DEFAULT_STREAMING_SILENCE_FRAMES):
        audio_queue.put(np.zeros(vad_runner.WINDOW_SIZE_SAMPLES, dtype=np.float32))

    iterator = vad_runner.iter_utterances(audio_queue, 1.0, stop_event=threading.Event())
    try:
        utterance = next(iterator)
    finally:
        iterator.close()  # type: ignore[attr-defined]

    assert utterance.sample_rate == vad_runner.SAMPLE_RATE
    assert utterance.audio.shape == (vad_runner.WINDOW_SIZE_SAMPLES,)
    assert np.allclose(utterance.audio, 1.0)
    assert model.reset_calls == 1
    assert len(model.calls) == 1 + vad_runner.DEFAULT_STREAMING_SILENCE_FRAMES
    assert {sample_rate for _, sample_rate in model.calls} == {vad_runner.SAMPLE_RATE}


def test_iter_utterances_finalizes_active_speech_after_frame_starvation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active speech with no more frames yields buffered speech instead of spinning."""
    _install_fake_torch(monkeypatch)
    monkeypatch.setenv("MUNGI_VAD_ACTIVE_NO_FRAME_TIMEOUT_S", "0.001")
    monkeypatch.setenv("MUNGI_VAD_SILENCE_FRAMES", "1")
    monkeypatch.setattr(vad_runner, "STREAMING_QUEUE_POLL_S", 0.001)
    model = _SequenceVadModel([0.9])
    audio_queue = _queue_with_attrs(model=model)
    audio_queue.put(np.ones(vad_runner.WINDOW_SIZE_SAMPLES, dtype=np.float32))

    iterator = vad_runner.iter_utterances(audio_queue, 1.0, stop_event=threading.Event())
    try:
        utterance = next(iterator)
    finally:
        iterator.close()  # type: ignore[attr-defined]

    assert utterance.sample_rate == vad_runner.SAMPLE_RATE
    assert utterance.audio.shape == (vad_runner.WINDOW_SIZE_SAMPLES,)
    assert np.allclose(utterance.audio, 1.0)
    assert len(model.calls) == 1


def test_iter_utterances_resamples_stereo_48k_to_mono_16k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-channel device-rate frames are downmixed and resampled before VAD."""
    _install_fake_torch(monkeypatch)

    def fake_resample_poly(samples: np.ndarray, up: int, down: int) -> np.ndarray:
        assert up == 1
        assert down == 3
        return samples[::3]

    monkeypatch.setitem(sys.modules, "scipy", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "scipy.signal",
        SimpleNamespace(resample_poly=fake_resample_poly),
    )
    model = _SequenceVadModel([0.9, *([0.0] * vad_runner.DEFAULT_STREAMING_SILENCE_FRAMES)])
    audio_queue = _queue_with_attrs(model=model, sample_rate=48_000)
    stereo_speech = np.column_stack(
        (
            np.ones(1536, dtype=np.float32),
            np.full(1536, 3.0, dtype=np.float32),
        )
    )
    audio_queue.put(stereo_speech)
    for _ in range(vad_runner.DEFAULT_STREAMING_SILENCE_FRAMES):
        audio_queue.put(np.zeros((1536, 2), dtype=np.float32))

    iterator = vad_runner.iter_utterances(audio_queue, 1.0, stop_event=threading.Event())
    try:
        utterance = next(iterator)
    finally:
        iterator.close()  # type: ignore[attr-defined]

    assert utterance.sample_rate == vad_runner.SAMPLE_RATE
    assert utterance.audio.shape == (vad_runner.WINDOW_SIZE_SAMPLES,)
    assert np.allclose(utterance.audio, 2.0)


def test_iter_utterances_enforces_single_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second streaming wrapper cannot be entered while one is active."""
    _install_fake_torch(monkeypatch)
    first_queue = _queue_with_attrs(model=_SequenceVadModel([0.0]))
    second_queue = _queue_with_attrs(model=_SequenceVadModel([0.0]))
    first = vad_runner.iter_utterances(first_queue, 1.0, stop_event=threading.Event())

    try:
        with pytest.raises(RuntimeError, match="already active"):
            vad_runner.iter_utterances(second_queue, 1.0, stop_event=threading.Event())
    finally:
        first.close()  # type: ignore[attr-defined]


def test_iter_utterances_drains_queue_on_stop_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set stop event drains queued frames without yielding a partial utterance."""
    _install_fake_torch(monkeypatch)
    stop_event = threading.Event()
    stop_event.set()
    audio_queue = _queue_with_attrs(model=_SequenceVadModel([0.0]))
    audio_queue.put(np.ones(vad_runner.WINDOW_SIZE_SAMPLES, dtype=np.float32))

    utterances = list(vad_runner.iter_utterances(audio_queue, 1.0, stop_event=stop_event))

    assert utterances == []
    assert audio_queue.empty()


def test_iter_utterances_times_out_before_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    """No speech before timeout returns without yielding."""
    _install_fake_torch(monkeypatch)
    audio_queue = _queue_with_attrs(model=_SequenceVadModel([]))

    utterances = list(vad_runner.iter_utterances(audio_queue, 0.0, stop_event=threading.Event()))

    assert utterances == []
