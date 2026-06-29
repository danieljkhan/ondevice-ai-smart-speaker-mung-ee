"""Tests for sentence-level TTS chunking helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from core import model_manager
from models import tts_runner


def _cleanup_temp_wav(result: tts_runner.SentenceSynthesisResult) -> None:
    """Remove helper-generated temporary WAV artifacts."""
    if result.full_wav_path is None:
        return
    Path(result.full_wav_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\uace0(\u53e4)", "\uace0"),
        ("\uc785\uc11d(\u7acb\u77f3)\ub9c8\uc744", "\uc785\uc11d\ub9c8\uc744"),
        ("\uace0\ub824(\u9ad8\u9e97)", "\uace0\ub824"),
        (
            "\uc2dc\ud150\ub178\uc0ac\u3014\u56db\u5929\u738b\u5bfa\u3015",
            "\uc2dc\ud150\ub178\uc0ac",
        ),
        ("\uc774\uc57c\uae30\u300a\u53e4\u300b", "\uc774\uc57c\uae30"),
        ("\uc774\uc57c\uae30\u300c\u53e4\u300d", "\uc774\uc57c\uae30"),
        ("\ub9d0\uff08\u53e4\uff09", "\ub9d0"),
        ("\ub9d0[\u53e4]", "\ub9d0"),
    ],
)
def test_normalize_tts_text_removes_bracketed_cjk_groups(
    raw: str,
    expected: str,
) -> None:
    """Bracketed CJK glosses should be removed before TTS synthesis."""
    assert tts_runner.normalize_tts_text(raw) == expected


def test_normalize_tts_text_preserves_non_cjk_brackets() -> None:
    """Bracketed groups without CJK ideographs should stay intact."""
    assert tts_runner.normalize_tts_text("(\uc77c)") == "(\uc77c)"
    assert tts_runner.normalize_tts_text("(\u53e4)") == ""
    assert tts_runner.normalize_tts_text("(\uc815)") == "(\uc815)"
    # CJK normalization keeps a non-CJK numeric gloss bracket; number expansion still applies.
    assert tts_runner._normalize_cjk("(391\ub144)") == "(391\ub144)"


def test_normalize_tts_text_strips_residual_bare_cjk() -> None:
    """Bare CJK ideographs should be stripped without leading whitespace."""
    assert tts_runner.normalize_tts_text("\u4e01(\uc815)") == "(\uc815)"
    assert tts_runner.normalize_tts_text("\u800c\u502d\u4ee5\u8f9b\u536f\u5e74 \ud55c\uc790") == (
        "\ud55c\uc790"
    )


def test_normalize_tts_text_cjk_normalization_is_idempotent() -> None:
    """CJK normalization should be stable after the first normalization pass."""
    raw = "\uc785\uc11d(\u7acb\u77f3)\ub9c8\uc744 \u4e01(\uc815)"
    normalized = tts_runner.normalize_tts_text(raw)

    assert tts_runner.normalize_tts_text(normalized) == normalized


def test_normalize_tts_text_leaves_non_cjk_text_unchanged() -> None:
    """Text without CJK ideographs should retain Hangul, Latin, and digits."""
    text = "\ud55c\uae00 ABC 123."

    assert tts_runner.normalize_tts_text(text) == text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\ubc31\uc81c\u318d\uac00\uc57c", "\ubc31\uc81c \uac00\uc57c"),
        ("\ubc31\uc81c\u00b7\uac00\uc57c", "\ubc31\uc81c \uac00\uc57c"),
        ("6\u202225", "6 25"),
        ("4\u30fb3", "4 3"),
        ("1\u223c2", "1 2"),
        ("\u25b2 \ubd09\uc218\ud615", "\ubd09\uc218\ud615"),
    ],
)
def test_normalize_tts_text_replaces_unsupported_separators(
    raw: str,
    expected: str,
) -> None:
    """Supertonic-unsupported separators should become clean word breaks."""
    assert tts_runner.normalize_tts_text(raw) == expected


def test_normalize_tts_text_replaces_celsius_symbol() -> None:
    """The Celsius symbol should use the Korean degree reading (with number expansion)."""
    assert tts_runner.normalize_tts_text("900\u2103") == "\uad6c\ubc31\ub3c4"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("112에 전화하자", "일일이에 전화하자"),
        ("119에 전화해", "일일구에 전화해"),
        ("119", "일일구"),
        ("119!", "일일구!"),
        ("112.", "일일이."),
        ("119 ", "일일구 "),
        ("112번지", "112번지"),
        ("119번", "백십구번"),
        ("112년", "백십이년"),
        ("AB119", "AB119"),
        ("119-1234", "119-1234"),
        ("version 119", "version 119"),
        ("사과 3개", "사과 세 개"),
    ],
)
def test_expand_ko_number_tokens_handles_emergency_numbers(
    raw: str,
    expected: str,
) -> None:
    """Emergency numbers should use spoken-only digit readings in Korean contexts."""
    assert tts_runner._expand_ko_number_tokens(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Korean-history middle dot, e.g. "3\u00b71\uc6b4\ub3d9" / "\uac00\u00b7\ub098".
        ("\uac00\u00b7\ub098", "\uac00 \ub098"),
        # Fullwidth comma (U+FF0C) and ideographic full stop (U+3002).
        ("\uac00\uff0c\ub098\u3002", "\uac00,\ub098."),
        # Fullwidth full stop (U+FF0E), exclamation (U+FF01), question (U+FF1F).
        ("\uc88b\uc544\uff01 \uc798\uac00\uff1f", "\uc88b\uc544! \uc798\uac00?"),
        # Bare CJK ideograph (U+53E4 '\u53e4') is stripped to a clean break.
        (
            "\uace0\ub300 \uc5ed\uc0ac \u53e4 \uc774\uc57c\uae30",
            "\uace0\ub300 \uc5ed\uc0ac \uc774\uc57c\uae30",
        ),
    ],
)
def test_normalize_tts_text_handles_history_unsupported_chars(
    raw: str,
    expected: str,
) -> None:
    """Hanja, middle dots, and fullwidth punctuation must become Supertonic-safe text."""
    result = tts_runner.normalize_tts_text(raw)
    assert result == expected
    # No CJK ideograph, middle dot, or fullwidth punctuation may survive.
    for char in result:
        codepoint = ord(char)
        assert char not in ("\u00b7", "\uff0c", "\u3002", "\uff0e", "\uff01", "\uff1f")
        assert not (0x3400 <= codepoint <= 0x9FFF)
        assert not (0xF900 <= codepoint <= 0xFAFF)


def test_normalize_tts_text_strips_unsupported_scripts_and_placeholders() -> None:
    """Unsupported redundant scripts, placeholders, and PUA artifacts should be stripped."""
    assert tts_runner.normalize_tts_text("\uc2dc\ud150\ub178\uc0ac(\u30ef\u30c3\u30bd)") == (
        "\uc2dc\ud150\ub178\uc0ac"
    )
    assert tts_runner.normalize_tts_text("\uae40\u25cb\u25cb") == "\uae40"
    assert tts_runner.normalize_tts_text("\u25a1\u25a1\u25a1 \ud55c\uc790") == "\ud55c\uc790"
    assert tts_runner.normalize_tts_text("\uc120\ubb3c\ue0c2\ue0c3") == "\uc120\ubb3c"


def test_normalize_tts_text_unsupported_character_normalization_is_idempotent() -> None:
    """Unsupported-character normalization should be stable after one pass."""
    raw = "\ubc31\uc81c\u318d\uac00\uc57c \uc2dc\ud150\ub178\uc0ac(\u30ef) \uc120\ubb3c\ue0c2"
    normalized = tts_runner.normalize_tts_text(raw)

    assert tts_runner.normalize_tts_text(normalized) == normalized


def test_synthesize_by_sentence_splits_ko_text() -> None:
    """Korean text should split into three non-empty sentence chunks."""
    text = (
        "\uc548\ub155\ud558\uc138\uc694 \uc5ec\ub7ec\ubd84. "
        "\uc624\ub298\uc740 \uc5b4\ub54c\uc694? "
        "\uc815\ub9d0 \uc7ac\ubbf8\uc788\uc5c8\uc5b4\uc694!"
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert len(sentences) == 3
    assert all(sentence for sentence in sentences)


def test_synthesize_by_sentence_splits_en_text() -> None:
    """English text should split into three non-empty sentence chunks."""
    sentences = tts_runner._split_text_into_sentences("Hello there. How are you? This is fun!")

    assert len(sentences) == 3
    assert all(sentence for sentence in sentences)


def test_synthesize_by_sentence_keeps_abbreviations() -> None:
    """Known abbreviations should not create extra sentence boundaries."""
    sentences = tts_runner._split_text_into_sentences("Hi Mr. Smith. Nice day.")

    assert sentences == ["Hi Mr. Smith.", "Nice day."]


def test_synthesize_by_sentence_merges_short_sentence() -> None:
    """An undersized first sentence should merge into the next chunk."""
    text = (
        "\uc548\ub155. \uae38\uace0 \uc790\uc5f0\uc2a4\ub7ec\uc6b4 \ubb38\uc7a5\uc785\ub2c8\ub2e4."
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "\uc548\ub155. \uae38\uace0 \uc790\uc5f0\uc2a4\ub7ec\uc6b4 \ubb38\uc7a5\uc785\ub2c8\ub2e4."
    ]


def test_split_text_into_sentences_splits_after_closing_quote() -> None:
    """Sentence-ending punctuation before a closing quote should split."""
    text = (
        "\uadf8\uac00 \ub9d0\ud588\uc5b4\uc694. "
        "\u201c\uc548\ub155!\u201d "
        "\uadf8\ub9ac\uace0 \uac14\uc5b4\uc694."
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "\uadf8\uac00 \ub9d0\ud588\uc5b4\uc694.",
        "\u201c\uc548\ub155!\u201d",
        "\uadf8\ub9ac\uace0 \uac14\uc5b4\uc694.",
    ]


def test_split_text_into_sentences_handles_stacked_closers() -> None:
    """Stacked sentence enders and closing quotes should remain with the turn."""
    text = (
        "\ub204\uad70\uac00 \ubb3c\uc5c8\uc5b4\uc694?!\u201d\u2019 "
        "\uadf8\ub140\uac00 \ub300\ub2f5\ud588\uc5b4\uc694."
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "\ub204\uad70\uac00 \ubb3c\uc5c8\uc5b4\uc694?!\u201d\u2019",
        "\uadf8\ub140\uac00 \ub300\ub2f5\ud588\uc5b4\uc694.",
    ]


def test_split_text_into_sentences_splits_before_opening_quote_without_space() -> None:
    """A sentence ender followed directly by an opening quote should split."""
    text = (
        "\uc7a0\uc2dc \uba48\ucdc4\ub2e4\u2026\ud588\ub2e4."
        "\u201c\ub2e4\uc74c \uc774\uc57c\uae30\ub97c \uc2dc\uc791\ud588\ub2e4.\u201d"
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "\uc7a0\uc2dc \uba48\ucdc4\ub2e4\u2026\ud588\ub2e4.",
        "\u201c\ub2e4\uc74c \uc774\uc57c\uae30\ub97c \uc2dc\uc791\ud588\ub2e4.\u201d",
    ]


def test_split_text_into_sentences_keeps_decimals_and_abbreviations() -> None:
    """Decimals and known abbreviations should not create false boundaries."""
    text = "It was 3.5 degrees. He wrote e.g.\u201d then continued. Mr. Smith agreed."

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "It was 3.5 degrees.",
        "He wrote e.g.\u201d then continued.",
        "Mr. Smith agreed.",
    ]


def test_split_text_into_sentences_splits_real_census_dialogue_turns() -> None:
    """The Yu Seong-ryong dialogue block should not become one run-on segment."""
    text = (
        "\uc720\uc131\ub8e1, \uc784\uc9c4\uc65c\ub780\uc744 \uae30\ub85d\ud558\ub2e4\n"
        "\u201c\uc544\uc774\uace0 \uc774\ub7f0, "
        "\ub098\ub77c\uc758 \ud070 \uc5b4\ub978\uc774 "
        "\ub3cc\uc544\uac00\uc2dc\ub2e4\ub2c8!\u201d\n"
        "\u201c\uc774 \uc5b4\ub978\uc774 \uacc4\uc154\uc11c "
        "\uadf8\ub798\ub3c4 \uc774\ub9ac \uc0b4 \uae38\uc774 \uc5f4\ub838\ub294\ub370.\u201d\n"
        "\u201c\ub2f9\uc2dc \uc0c1\ud669\uc744 \uc544\ud504\uc9c0\ub9cc "
        "\uae30\ub85d\uc73c\ub85c \ub0a8\uae34 \uadf8 \ubd84\uc758 "
        "\ub73b\uc744 \uc0dd\uac01\ud574\ubcf4\uba74 \uc88b\uaca0\ub124.\u201d\n"
        "\ub098\ub77c\uc5d0\uc11c \uc874\uacbd\ubc1b\ub294 "
        "\uc5b4\ub978\uc774 \ub3cc\uc544\uac00\uc168\uc5b4\uc694."
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "\uc720\uc131\ub8e1, \uc784\uc9c4\uc65c\ub780\uc744 \uae30\ub85d\ud558\ub2e4 "
        "\u201c\uc544\uc774\uace0 \uc774\ub7f0, "
        "\ub098\ub77c\uc758 \ud070 \uc5b4\ub978\uc774 "
        "\ub3cc\uc544\uac00\uc2dc\ub2e4\ub2c8!\u201d",
        "\u201c\uc774 \uc5b4\ub978\uc774 \uacc4\uc154\uc11c "
        "\uadf8\ub798\ub3c4 \uc774\ub9ac \uc0b4 \uae38\uc774 \uc5f4\ub838\ub294\ub370.\u201d",
        "\u201c\ub2f9\uc2dc \uc0c1\ud669\uc744 \uc544\ud504\uc9c0\ub9cc "
        "\uae30\ub85d\uc73c\ub85c \ub0a8\uae34 \uadf8 \ubd84\uc758 "
        "\ub73b\uc744 \uc0dd\uac01\ud574\ubcf4\uba74 \uc88b\uaca0\ub124.\u201d",
        "\ub098\ub77c\uc5d0\uc11c \uc874\uacbd\ubc1b\ub294 "
        "\uc5b4\ub978\uc774 \ub3cc\uc544\uac00\uc168\uc5b4\uc694.",
    ]


def test_merge_short_sentences_keeps_short_quoted_reply_separate() -> None:
    """A short complete quoted reply should not merge into the prior sentence."""
    text = (
        "\uc120\uc0dd\ub2d8\uc774 \ubb3c\uc5c8\uc5b4\uc694. "
        "\u201c\uc608.\u201d "
        "\ub2e4\uc2dc \uc2dc\uc791\ud588\uc5b4\uc694."
    )

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == [
        "\uc120\uc0dd\ub2d8\uc774 \ubb3c\uc5c8\uc5b4\uc694.",
        "\u201c\uc608.\u201d",
        "\ub2e4\uc2dc \uc2dc\uc791\ud588\uc5b4\uc694.",
    ]


def test_merge_short_sentences_still_merges_genuine_fragment() -> None:
    """A short tail without a sentence ender should still merge back."""
    text = "\uae34 \ubb38\uc7a5\uc785\ub2c8\ub2e4. \uc870\uac01"

    sentences = tts_runner._split_text_into_sentences(text)

    assert sentences == ["\uae34 \ubb38\uc7a5\uc785\ub2c8\ub2e4. \uc870\uac01"]


def test_synthesize_by_sentence_single_sentence_falls_back(monkeypatch: Any) -> None:
    """Single-sentence input should use the existing full-waveform synth path once."""

    class FakeEngine:
        """Minimal TTS engine stub for helper fallback tests."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def synthesize(self, text: str, language: str = "ko") -> tuple[np.ndarray, int]:
            del language
            self.calls.append(text)
            return np.ones(2205, dtype=np.float32), 22050

    engine = FakeEngine()
    play_calls: list[tuple[int, int, str | None]] = []
    monkeypatch.setattr(
        tts_runner,
        "_resolve_sentence_engine",
        lambda voice_style, model_dir=None: engine,
    )
    monkeypatch.setattr(
        tts_runner,
        "_play_sentence_chunk",
        lambda audio, sample_rate, output_device: play_calls.append(
            (len(np.asarray(audio)), sample_rate, output_device),
        ),
    )

    result = tts_runner.synthesize_to_speaker_by_sentence("Hello there.", "F2")

    assert engine.calls == ["Hello there."]
    assert result.sentence_count == 1
    assert result.first_chunk_ms == result.total_duration_ms
    assert play_calls == [(2205, 22050, None)]
    _cleanup_temp_wav(result)


def test_synthesize_by_sentence_first_chunk_timing_recorded(monkeypatch: Any) -> None:
    """First-chunk timing should be lower than total synthesis time for multi-sentence text."""

    current_time = 100.0

    def _advance_clock(seconds: float) -> None:
        """Advance the fake monotonic clock by a deterministic duration."""
        nonlocal current_time
        current_time += seconds

    def _fake_monotonic() -> float:
        """Return the current fake monotonic timestamp."""
        return current_time

    class FakeEngine:
        """Deterministic synth stub with staggered per-sentence latencies."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def synthesize(self, text: str, language: str = "ko") -> tuple[np.ndarray, int]:
            del language
            self.calls.append(text)
            if text.startswith("First"):
                _advance_clock(0.01)
                return np.ones(2205, dtype=np.float32), 22050
            _advance_clock(0.03)
            return np.ones(4410, dtype=np.float32), 22050

    engine = FakeEngine()
    monkeypatch.setattr(tts_runner.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(
        tts_runner,
        "_resolve_sentence_engine",
        lambda voice_style, model_dir=None: engine,
    )
    monkeypatch.setattr(
        tts_runner, "_play_sentence_chunk", lambda audio, sample_rate, output_device: None
    )

    result = tts_runner.synthesize_to_speaker_by_sentence(
        "First sentence. Second sentence. Third sentence.",
        "F2",
    )

    assert engine.calls == ["First sentence.", "Second sentence.", "Third sentence."]
    assert result.sentence_count == 3
    assert 0.0 < result.first_chunk_ms < result.total_duration_ms
    _cleanup_temp_wav(result)


def test_synthesize_by_sentence_uses_active_engine_when_registered(monkeypatch: Any) -> None:
    """A registered matching engine should be reused without fallback resolution."""

    class FakeEngine:
        """Minimal loaded engine stub for active-engine reuse tests."""

        def __init__(self) -> None:
            self._model = object()
            self._voice_style_name = "F2"
            self.calls: list[str] = []

        def synthesize(self, text: str, language: str = "ko") -> tuple[np.ndarray, int]:
            del language
            self.calls.append(text)
            return np.ones(2205, dtype=np.float32), 22050

    def _fail_resolve_streaming_model_dir() -> str:
        msg = "fallback model directory resolution should not run"
        raise AssertionError(msg)

    def _fail_supertonic_engine(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "a new Supertonic engine should not be constructed"
        raise AssertionError(msg)

    engine = FakeEngine()
    result: tts_runner.SentenceSynthesisResult | None = None
    tts_runner._STREAMING_ENGINE_CACHE.clear()
    tts_runner._set_active_supertonic_engine(None)
    tts_runner._set_active_supertonic_engine(cast(tts_runner.SupertonicEngine, engine))

    monkeypatch.setattr(
        tts_runner, "_resolve_streaming_model_dir", _fail_resolve_streaming_model_dir
    )
    monkeypatch.setattr(tts_runner, "SupertonicEngine", _fail_supertonic_engine)
    monkeypatch.setattr(
        tts_runner, "_play_sentence_chunk", lambda audio, sample_rate, output_device: None
    )

    try:
        result = tts_runner.synthesize_to_speaker_by_sentence("short text", "F2")
        assert engine.calls == ["short text"]
        assert result.sentence_count == 1
    finally:
        if result is not None:
            _cleanup_temp_wav(result)
        tts_runner._set_active_supertonic_engine(None)
        tts_runner._STREAMING_ENGINE_CACHE.clear()


def test_synthesize_by_sentence_uses_explicit_model_dir(monkeypatch: Any) -> None:
    """An explicit model_dir should bypass fallback resolution and use that path directly."""
    init_calls: list[tuple[str, str]] = []
    result: tts_runner.SentenceSynthesisResult | None = None
    tts_runner._STREAMING_ENGINE_CACHE.clear()
    tts_runner._set_active_supertonic_engine(None)

    def fake_init(self: Any, model_dir: str, voice_style: str = "F1") -> None:
        """Capture constructor args without touching the real model loader."""
        init_calls.append((model_dir, voice_style))
        self._model_dir = model_dir
        self._voice_style_name = voice_style
        self._model = object()
        self._voice_style = object()
        self._sample_rate = 22050

    def fake_load(self: Any) -> None:
        """Skip real model loading for the explicit-path test."""
        return

    def fake_synthesize(self: Any, text: str, language: str = "ko") -> tuple[np.ndarray, int]:
        """Return a deterministic single-sentence waveform."""
        del self, text, language
        return np.ones(2205, dtype=np.float32), 22050

    def _fail_resolve_streaming_model_dir() -> str:
        msg = "fallback model directory resolution should not run"
        raise AssertionError(msg)

    monkeypatch.setattr(tts_runner.SupertonicEngine, "__init__", fake_init)
    monkeypatch.setattr(tts_runner.SupertonicEngine, "load", fake_load)
    monkeypatch.setattr(tts_runner.SupertonicEngine, "synthesize", fake_synthesize)
    monkeypatch.setattr(
        tts_runner, "_resolve_streaming_model_dir", _fail_resolve_streaming_model_dir
    )
    monkeypatch.setattr(
        tts_runner, "_play_sentence_chunk", lambda audio, sample_rate, output_device: None
    )

    try:
        result = tts_runner.synthesize_to_speaker_by_sentence(
            "short text",
            "F2",
            model_dir="/fake/model/dir",
        )
        assert init_calls == [("/fake/model/dir", "F2")]
        assert result.sentence_count == 1
    finally:
        if result is not None:
            _cleanup_temp_wav(result)
        tts_runner._set_active_supertonic_engine(None)
        tts_runner._STREAMING_ENGINE_CACHE.clear()


def test_resolve_streaming_model_dir_uses_absolute_default(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Missing env overrides should use the Jetson runtime TTS model directory."""
    monkeypatch.delenv("MUNGI_TTS_MODEL_DIR", raising=False)
    monkeypatch.delenv("MUNGI_MODEL_DIR", raising=False)

    with caplog.at_level(logging.WARNING):
        model_dir = tts_runner._resolve_streaming_model_dir()

    assert model_dir == "/opt/mungi/ai_models/supertonic-2"
    assert "Jetson runtime default" in caplog.text


def test_supertonic_load_with_preset_name_calls_get_voice_style(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Preset voice names should use the bundled Supertonic style loader."""
    fake_model: Any = MagicMock()
    fake_tts: Any = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "supertonic", SimpleNamespace(TTS=fake_tts))

    engine = tts_runner.SupertonicEngine(str(tmp_path), voice_style="F2")

    try:
        engine.load()

        fake_tts.assert_called_once_with(model_dir=str(tmp_path))
        fake_model.get_voice_style.assert_called_once_with("F2")
        fake_model.get_voice_style_from_path.assert_not_called()
    finally:
        tts_runner._set_active_supertonic_engine(None)


def test_supertonic_load_with_json_path_calls_get_voice_style_from_path(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Custom voice JSON paths should use the Supertonic path style loader."""
    voice_path = "/var/lib/mungi/voices/tobi.json"
    fake_model: Any = MagicMock()
    fake_tts: Any = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "supertonic", SimpleNamespace(TTS=fake_tts))

    engine = tts_runner.SupertonicEngine(str(tmp_path), voice_style=voice_path)

    try:
        engine.load()

        fake_tts.assert_called_once_with(model_dir=str(tmp_path))
        fake_model.get_voice_style_from_path.assert_called_once_with(voice_path)
        fake_model.get_voice_style.assert_not_called()
    finally:
        tts_runner._set_active_supertonic_engine(None)


def test_supertonic_load_with_bilingual_voice_styles_loads_both(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bilingual Supertonic mode should resolve KO path and EN preset voices."""
    voice_path = "/var/lib/mungi/voices/tobi.json"
    ko_voice = object()
    en_voice = object()
    fake_model = SimpleNamespace(
        sample_rate=22050,
        get_voice_style=MagicMock(return_value=en_voice),
        get_voice_style_from_path=MagicMock(return_value=ko_voice),
    )
    fake_tts: Any = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "supertonic", SimpleNamespace(TTS=fake_tts))

    engine = tts_runner.SupertonicEngine(
        str(tmp_path),
        voice_style_ko=voice_path,
        voice_style_en="F2",
    )

    try:
        engine.load()

        fake_tts.assert_called_once_with(model_dir=str(tmp_path))
        fake_model.get_voice_style_from_path.assert_called_once_with(voice_path)
        fake_model.get_voice_style.assert_called_once_with("F2")
        assert engine._voice_style_ko is ko_voice
        assert engine._voice_style_en is en_voice
        assert engine._voice_style_name == f"<bilingual:{voice_path}|F2>"
    finally:
        tts_runner._set_active_supertonic_engine(None)


def test_supertonic_synthesize_picks_voice_by_language(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bilingual synthesis should choose the voice style from normalized language."""
    ko_voice = object()
    en_voice = object()
    engine = tts_runner.SupertonicEngine(
        "unused",
        voice_style_ko="/var/lib/mungi/voices/tobi.json",
        voice_style_en="F2",
    )
    engine._model = MagicMock()
    engine._model.synthesize.return_value = np.array([0.1], dtype=np.float32)
    engine._voice_style_ko = ko_voice
    engine._voice_style_en = en_voice

    with caplog.at_level(logging.WARNING, logger="mungi.models.tts_runner"):
        engine.synthesize("ko text", language="ko")
        engine.synthesize("en text", language=" EN ")
        engine.synthesize("invalid", language="fr")

    assert engine._model.synthesize.call_args_list == [
        call("ko text", voice_style=ko_voice, lang="ko", speed=0.95, total_steps=10),
        call("en text", voice_style=en_voice, lang="en", speed=0.95, total_steps=10),
        call("invalid", voice_style=en_voice, lang="en", speed=0.95, total_steps=10),
    ]
    warning_messages = [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert any("fr" in message and "en" in message for message in warning_messages)


def test_supertonic_synthesize_strips_remaining_unsupported_chars(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Loaded Supertonic supported_chars should defensively clean novel unsupported chars."""
    fake_model = SimpleNamespace(
        text_processor=SimpleNamespace(supported_chars=frozenset("ok text")),
        synthesize=MagicMock(return_value=np.array([0.1], dtype=np.float32)),
    )
    engine = tts_runner.SupertonicEngine("unused", voice_style="F1")
    engine._model = fake_model

    with caplog.at_level(logging.WARNING, logger="mungi.models.tts_runner"):
        engine.synthesize("ok\u2605 text", language="ko")

    fake_model.synthesize.assert_called_once_with(
        "ok text",
        lang="ko",
        speed=0.95,
        total_steps=10,
    )
    assert any("U+2605" in record.getMessage() for record in caplog.records)


def test_extract_unsupported_chars_parses_supertonic_message() -> None:
    """The offending characters should be parsed out of the Supertonic error text."""
    message = "Found 2 unsupported character(s): ['古', '·']"
    assert tts_runner._extract_unsupported_chars(message) == ("古", "·")
    assert tts_runner._is_unsupported_char_error(ValueError(message)) is True
    assert tts_runner._is_unsupported_char_error(ValueError("disk full")) is False


def test_supertonic_synthesize_retries_after_unsupported_char_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unsupported-character failure should strip the char and retry once."""
    calls: list[str] = []

    def fake_synthesize(text: str, **_kwargs: Any) -> np.ndarray:
        calls.append(text)
        offending = [char for char in text if char == "Q"]
        if offending:
            raise ValueError(f"Found {len(offending)} unsupported character(s): {offending}")
        return np.array([0.42], dtype=np.float32)

    # No ``text_processor`` so the proactive strip is skipped and the model's
    # own validation is what rejects the character (mirrors the live crash).
    fake_model = SimpleNamespace(synthesize=fake_synthesize)
    engine = tts_runner.SupertonicEngine("unused", voice_style="F1")
    engine._model = fake_model

    with caplog.at_level(logging.WARNING, logger="mungi.models.tts_runner"):
        audio, _sample_rate = engine.synthesize("hi Q there", language="en")

    assert calls == ["hi Q there", "hi there"]
    assert audio.size == 1
    assert any("retry" in record.getMessage().lower() for record in caplog.records)


def test_supertonic_synthesize_skips_segment_when_retry_still_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A persistently unsupported character degrades to an empty (skipped) segment."""

    def always_reject(text: str, **_kwargs: Any) -> np.ndarray:
        char = next((candidate for candidate in text if not candidate.isspace()), "x")
        raise ValueError(f"Found 1 unsupported character(s): ['{char}']")

    fake_model = SimpleNamespace(synthesize=always_reject)
    engine = tts_runner.SupertonicEngine("unused", voice_style="F1")
    engine._model = fake_model

    with caplog.at_level(logging.WARNING, logger="mungi.models.tts_runner"):
        audio, sample_rate = engine.synthesize("abc", language="en")

    # No exception is raised; the segment yields empty audio instead of crashing.
    assert audio.shape == (0,)
    assert sample_rate == engine._sample_rate
    assert any("skipped" in record.getMessage().lower() for record in caplog.records)


def test_supertonic_synthesize_propagates_non_unsupported_errors() -> None:
    """Failures unrelated to unsupported characters must still raise (no silent swallow)."""

    def boom(_text: str, **_kwargs: Any) -> np.ndarray:
        raise ValueError("a completely different synthesis failure")

    fake_model = SimpleNamespace(synthesize=boom)
    engine = tts_runner.SupertonicEngine("unused", voice_style="F1")
    engine._model = fake_model

    with pytest.raises(RuntimeError, match="Supertonic synthesis failed"):
        engine.synthesize("hello", language="en")


def test_supertonic_init_rejects_partial_bilingual_args(tmp_path: Path) -> None:
    """Direct callers must provide both bilingual voice styles or neither."""
    with pytest.raises(ValueError):
        tts_runner.SupertonicEngine(str(tmp_path), voice_style_ko="ko-only")

    with pytest.raises(ValueError):
        tts_runner.SupertonicEngine(str(tmp_path), voice_style_en="en-only")


def _base_manager_config(**overrides: Any) -> model_manager.ManagerConfig:
    """Build ManagerConfig without relying on machine-specific runtime paths."""
    defaults: dict[str, Any] = {
        "model_dir": "/models",
        "tts_model_dir": "/models/supertonic-2",
    }
    defaults.update(overrides)
    return model_manager.ManagerConfig(**defaults)


def _clear_tts_voice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear TTS voice environment overrides for isolated config subcases."""
    monkeypatch.delenv("MUNGI_TTS_VOICE_STYLE", raising=False)
    monkeypatch.delenv("MUNGI_TTS_VOICE_STYLE_KO", raising=False)
    monkeypatch.delenv("MUNGI_TTS_VOICE_STYLE_EN", raising=False)


def test_manager_config_per_lang_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-language TTS env vars should override legacy fallback per lane."""
    _clear_tts_voice_env(monkeypatch)
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE", "legacy")
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE_KO", "ko-specific")
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE_EN", "en-specific")
    cfg = _base_manager_config()
    assert cfg.tts_voice_style == "legacy"
    assert cfg.tts_voice_style_ko == "ko-specific"
    assert cfg.tts_voice_style_en == "en-specific"

    _clear_tts_voice_env(monkeypatch)
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE", "legacy-only")
    cfg = _base_manager_config()
    assert cfg.tts_voice_style == "legacy-only"
    assert cfg.tts_voice_style_ko == "legacy-only"
    assert cfg.tts_voice_style_en == "legacy-only"

    _clear_tts_voice_env(monkeypatch)
    monkeypatch.setenv("MUNGI_TTS_VOICE_STYLE_KO", "ko-only")
    cfg = _base_manager_config()
    assert cfg.tts_voice_style == "F2"
    assert cfg.tts_voice_style_ko == "ko-only"
    assert cfg.tts_voice_style_en is None

    _clear_tts_voice_env(monkeypatch)
    cfg = _base_manager_config()
    assert cfg.tts_voice_style == "F2"
    assert cfg.tts_voice_style_ko is None
    assert cfg.tts_voice_style_en is None


def _capture_tts_construction(
    monkeypatch: pytest.MonkeyPatch,
    cfg: model_manager.ManagerConfig,
) -> list[dict[str, str | None]]:
    """Run load_tts with a fake SupertonicEngine and capture constructor args."""
    constructions: list[dict[str, str | None]] = []

    class FakeSupertonicEngine:
        """Fake TTS engine that records constructor inputs and load calls."""

        def __init__(
            self,
            model_dir: str,
            voice_style: str = "F1",
            *,
            voice_style_ko: str | None = None,
            voice_style_en: str | None = None,
        ) -> None:
            constructions.append(
                {
                    "model_dir": model_dir,
                    "voice_style": voice_style,
                    "voice_style_ko": voice_style_ko,
                    "voice_style_en": voice_style_en,
                }
            )

        def load(self) -> None:
            """Pretend to load the TTS model."""
            return

    monkeypatch.setattr(tts_runner, "SupertonicEngine", FakeSupertonicEngine)
    try:
        model_manager.ModelManager(cfg).load_tts()
    finally:
        tts_runner._set_active_supertonic_engine(None)
    return constructions


def test_load_tts_constructs_bilingual_engine_when_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelManager should pass both per-language styles to SupertonicEngine."""
    cfg = _base_manager_config(
        tts_voice_style_ko="/var/lib/mungi/voices/tobi.json",
        tts_voice_style_en="F2",
    )

    constructions = _capture_tts_construction(monkeypatch, cfg)

    assert len(constructions) == 1
    assert constructions[0]["model_dir"] == "/models/supertonic-2"
    assert constructions[0]["voice_style_ko"] == "/var/lib/mungi/voices/tobi.json"
    assert constructions[0]["voice_style_en"] == "F2"


def test_load_tts_fills_partial_config_from_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelManager should XOR-fill partial bilingual config from legacy style."""
    constructions = _capture_tts_construction(
        monkeypatch,
        _base_manager_config(
            tts_voice_style="/legacy/fallback.json",
            tts_voice_style_ko="/var/lib/mungi/voices/tobi.json",
        ),
    )
    assert constructions[-1]["model_dir"] == "/models/supertonic-2"
    assert constructions[-1]["voice_style_ko"] == "/var/lib/mungi/voices/tobi.json"
    assert constructions[-1]["voice_style_en"] == "/legacy/fallback.json"

    constructions = _capture_tts_construction(
        monkeypatch,
        _base_manager_config(tts_voice_style_en="en-only"),
    )
    assert constructions[-1]["model_dir"] == "/models/supertonic-2"
    assert constructions[-1]["voice_style_ko"] == "F2"
    assert constructions[-1]["voice_style_en"] == "en-only"

    constructions = _capture_tts_construction(
        monkeypatch,
        _base_manager_config(tts_voice_style="/legacy/single.json"),
    )
    assert constructions[-1] == {
        "model_dir": "/models/supertonic-2",
        "voice_style": "/legacy/single.json",
        "voice_style_ko": None,
        "voice_style_en": None,
    }


def test_supertonic_legacy_single_voice_synthesize_still_works() -> None:
    """Legacy single-voice mode should keep forwarding one voice style."""
    voice_style = object()
    engine = tts_runner.SupertonicEngine("unused", voice_style="F1")
    engine._model = MagicMock()
    engine._model.synthesize.return_value = np.array([0.1, 0.2], dtype=np.float32)
    engine._voice_style = voice_style

    engine.synthesize("ko text", language="ko")
    engine.synthesize("en text", language="en")

    assert engine._voice_style_name == "F1"
    assert engine._model.synthesize.call_args_list == [
        call("ko text", voice_style=voice_style, lang="ko", speed=0.95, total_steps=10),
        call("en text", voice_style=voice_style, lang="en", speed=0.95, total_steps=10),
    ]
