"""Pipeline integration test for the deterministic datetime intercept."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.character_expression import CharacterExpression
from core.pipeline import ConversationPipeline, PipelineConfig


def test_datetime_query_bypasses_llm_and_answers_in_korean() -> None:
    """A spoken time query answers from the local clock without loading the LLM."""
    mm = MagicMock()
    pipeline = ConversationPipeline(mm, PipelineConfig())

    with (
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out") as mock_play_audio,
    ):
        result = pipeline.run_text_turn("지금 몇 시야?")

    assert result.metrics.datetime_query_matched is True
    assert result.response_text.startswith("지금")
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(result.response_text, language="ko")
    assert mock_play_audio.call_args.kwargs["expression"] is CharacterExpression.HAPPY
