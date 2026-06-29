"""Tests for bilingual TTS language plumbing."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from models.tts_runner import SupertonicEngine


def test_supertonic_synthesize_forwards_language_to_model() -> None:
    """Supertonic should pass the requested language to the model."""
    engine = SupertonicEngine("unused")
    engine._model = MagicMock()
    engine._model.synthesize.return_value = np.array([0.1, 0.2], dtype=np.float32)

    audio, sample_rate = engine.synthesize("hello", language="en")

    assert np.array_equal(audio, np.array([0.1, 0.2], dtype=np.float32))
    assert sample_rate == engine._sample_rate
    engine._model.synthesize.assert_called_once_with("hello", lang="en", speed=0.95, total_steps=10)


def test_supertonic_synthesize_defaults_language_to_ko() -> None:
    """Supertonic should default the language hint to Korean."""
    engine = SupertonicEngine("unused")
    engine._model = MagicMock()
    engine._model.synthesize.return_value = np.array([0.1], dtype=np.float32)

    engine.synthesize("hello")

    engine._model.synthesize.assert_called_once_with("hello", lang="ko", speed=0.95, total_steps=10)
