"""Pipeline tests for Funny English entry and STT-only attempts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.character_expression import CharacterExpression
from core.model_manager import ModelType
from core.pipeline import ConversationPipeline, PipelineConfig, Utterance


def test_funny_english_entry_bypasses_llm_and_suppresses_preload() -> None:
    """Entry confirmation uses fixed TTS and does not schedule unprompted STT preload."""
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, PipelineConfig(enable_stt_preload=True))
    funny_sink = MagicMock()
    pipeline.set_funny_english_sink(funny_sink)

    with (
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out") as mock_play_audio,
    ):
        result = pipeline.run_text_turn("Funny English")

    assert pipeline.session_language == "en"
    assert result.metrics.funny_english_matched is True
    assert result.metrics.language_switch_matched is False
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(result.response_text, language="ko")
    assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.EXCITED
    mm.preload_stt.assert_not_called()
    funny_sink.assert_called_once_with()


def test_funny_english_attempt_is_stt_only_and_uses_card_hotwords() -> None:
    """Attempt scoring never calls filter/crisis/template/LLM paths."""
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, PipelineConfig())
    utterance = Utterance(audio=np.zeros(160, dtype=np.float32), sample_rate=16_000)
    card = SimpleNamespace(tokens=("cat",), hotwords=("cat",))

    with (
        patch.object(pipeline, "_run_stt", return_value="cat") as mock_stt,
        patch.object(pipeline, "_filter_text") as mock_filter,
        patch("core.pipeline.match_crisis_disclosure") as mock_crisis,
        patch("core.pipeline.match_parent_disclosure") as mock_parent,
        patch("core.pipeline.check_approved_template") as mock_template,
        patch.object(pipeline, "_run_llm") as mock_llm,
    ):
        result = pipeline.run_funny_english_attempt(utterance, card)

    assert result.band == "pass"
    mm.load.assert_called_once_with(ModelType.STT, stt_hotwords_csv="cat,뭉이,뭉이야")
    mock_stt.assert_called_once()
    mock_filter.assert_not_called()
    mock_crisis.assert_not_called()
    mock_parent.assert_not_called()
    mock_template.assert_not_called()
    mock_llm.assert_not_called()


def test_funny_english_attempt_uses_explicit_stt_hotwords_csv_verbatim() -> None:
    """Explicit threaded hotword CSV is passed to the STT load unchanged."""
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, PipelineConfig())
    utterance = Utterance(audio=np.zeros(160, dtype=np.float32), sample_rate=16_000)
    card = SimpleNamespace(tokens=("cat",), hotwords=("cat",))
    stt_hotwords_csv = "cat,dog,뭉이,뭉이야"

    with patch.object(pipeline, "_run_stt", return_value="cat"):
        result = pipeline.run_funny_english_attempt(
            utterance,
            card,
            stt_hotwords_csv=stt_hotwords_csv,
        )

    assert result.band == "pass"
    mm.load.assert_called_once_with(ModelType.STT, stt_hotwords_csv=stt_hotwords_csv)


def test_funny_english_attempt_uses_baked_default_thresholds() -> None:
    """Unset env thresholds should use the baked 0.3 pct / 0.4 similarity defaults."""
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, PipelineConfig())
    utterance = Utterance(audio=np.zeros(160, dtype=np.float32), sample_rate=16_000)
    card = SimpleNamespace(tokens=("i", "see", "a", "cat"), hotwords=("cat",))

    with patch.object(pipeline, "_run_stt", return_value="see"):
        result = pipeline.run_funny_english_attempt(utterance, card)

    assert result.matched_pct < 0.3
    assert result.similarity >= 0.4
    assert result.band == "pass"
