"""Unit tests for models.vad_runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class _FakeProb:
    value: float

    def item(self) -> float:
        return self.value


class _FakeVadModel:
    def __init__(self, probability: float = 0.0) -> None:
        self._probability = probability
        self.reset_calls = 0
        self.call_count = 0

    def reset_states(self) -> None:
        self.reset_calls += 1

    def __call__(self, _tensor: object, _sample_rate: int) -> _FakeProb:
        self.call_count += 1
        return _FakeProb(self._probability)


def test_run_vad_accepts_numpy_audio_with_short_final_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models.vad_runner import run_vad

    audio = np.zeros(417, dtype=np.float32)
    model = _FakeVadModel(probability=0.0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(FloatTensor=lambda chunk: chunk))

    segments = run_vad(audio, model)

    assert segments == []
    assert model.reset_calls == 1
    assert model.call_count == 1
