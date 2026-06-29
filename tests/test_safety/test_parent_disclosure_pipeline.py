"""Pipeline reachability tests for deterministic parent-disclosure routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_backend_config import LLMBackendConfig
from core.model_manager import ModelType
from core.pipeline import ConversationPipeline, PipelineConfig
from core.safety_rules import (
    BELIEF_RESPONSE_EN,
    BELIEF_RESPONSE_KO,
    CRISIS_RESPONSE_KO,
    PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
)


def _legacy_backend_config() -> LLMBackendConfig:
    """Return a lightweight backend config that avoids live model loading."""
    return LLMBackendConfig(
        backend="qwen3_legacy",
        model_path=None,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )


def _build_pipeline(tmp_path: Path) -> ConversationPipeline:
    """Build a deterministic text-turn pipeline for parent-disclosure tests."""
    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        pipeline = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
        )
    pipeline._conversation_dir = tmp_path
    return pipeline


def test_parent_disclosure_reaches_pre_filter_router_and_writes_history(
    tmp_path: Path,
) -> None:
    """A parent-disclosure match should bypass filtering and the LLM."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch("core.pipeline.check_approved_template") as mock_template_router,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("엄마한테 말하면 안 돼, 약속해.")

    load_calls = pipeline._mm.load.call_args_list
    load_args = [call.args[0] for call in load_calls]
    assert ModelType.LLM not in load_args
    assert ModelType.TTS in load_args
    assert result.success is True
    assert result.response_text == PARENT_DISCLOSURE_KO_PROBE_RESPONSE
    assert result.metrics.parent_disclosure_matched is True
    assert result.metrics.parent_disclosure_kind == "probe"
    assert result.metrics.belief_matched is False
    assert result.metrics.crisis_matched is False
    assert result.metrics.content_filter_blocked is False
    assert result.metrics.llm_time_s == 0.0
    assert result.metrics.llm_tokens == 0
    assert pipeline.conversation_history == [
        {"role": "user", "text": "엄마한테 말하면 안 돼, 약속해."},
        {"role": "assistant", "text": PARENT_DISCLOSURE_KO_PROBE_RESPONSE},
    ]
    mock_filter.assert_not_called()
    mock_template_router.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(PARENT_DISCLOSURE_KO_PROBE_RESPONSE, language="ko")


def test_belief_probe_reaches_pre_filter_router(tmp_path: Path) -> None:
    """A narrow belief probe should bypass filtering and the LLM."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch("core.pipeline.check_approved_template") as mock_template_router,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("산타할아버지 진짜 있어?")

    assert result.success is True
    assert result.response_text == BELIEF_RESPONSE_KO
    assert result.metrics.belief_matched is True
    assert result.metrics.parent_disclosure_matched is False
    assert result.metrics.llm_time_s == 0.0
    mock_filter.assert_not_called()
    mock_template_router.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(BELIEF_RESPONSE_KO, language="ko")


def test_english_belief_probe_uses_english_response(tmp_path: Path) -> None:
    """An English belief probe should keep English TTS language and response text."""
    pipeline = _build_pipeline(tmp_path)
    pipeline.set_session_language("en")
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("Is Santa real?")

    assert result.success is True
    assert result.detected_language == "en"
    assert result.response_text == BELIEF_RESPONSE_EN
    assert result.metrics.belief_matched is True
    mock_filter.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(BELIEF_RESPONSE_EN, language="en")


def test_crisis_precedence_over_parent_disclosure(tmp_path: Path) -> None:
    """Crisis disclosure should remain higher priority than secret-pact routing."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)),
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("죽고 싶어. 엄마한테 말하지 마.")

    assert result.success is True
    assert result.response_text == CRISIS_RESPONSE_KO["suicidal_intent"]
    assert result.metrics.crisis_matched is True
    assert result.metrics.parent_disclosure_matched is False
    assert result.metrics.belief_matched is False
    mock_filter.assert_not_called()
    mock_llm.assert_not_called()


def test_secret_pact_precedence_over_belief(tmp_path: Path) -> None:
    """Mixed belief and secret-pact turns should route to parent disclosure."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("산타 진짜 있어? 엄마한테 말하지 마")

    assert result.success is True
    assert result.response_text == PARENT_DISCLOSURE_KO_PROBE_RESPONSE
    assert result.metrics.parent_disclosure_matched is True
    assert result.metrics.parent_disclosure_kind == "probe"
    assert result.metrics.belief_matched is False
    mock_filter.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(PARENT_DISCLOSURE_KO_PROBE_RESPONSE, language="ko")


def test_friendship_precedence_over_belief(tmp_path: Path) -> None:
    """Bare relational secret-pacts should beat belief routing in mixed turns."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text),
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("이건 둘만의 비밀인데, 산타 진짜 있어?")

    assert result.success is True
    assert result.response_text == PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE
    assert result.metrics.parent_disclosure_matched is True
    assert result.metrics.parent_disclosure_kind == "friendship"
    assert result.metrics.belief_matched is False
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE, language="ko")


def test_parent_disclosure_validator_replaces_llm_output_without_retry(
    tmp_path: Path,
) -> None:
    """Post-LLM secrecy promises should be swapped for the fixed probe response."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_filter_text", return_value=None),
        patch.object(
            pipeline,
            "_run_llm",
            return_value=("응! 뭉이는 네 친구니까 비밀 지킬게.", 12, 0.01),
        ) as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("공룡 얘기 해줘")

    assert result.success is True
    assert result.response_text == PARENT_DISCLOSURE_KO_PROBE_RESPONSE
    assert result.metrics.parent_disclosure_matched is False
    assert result.metrics.parent_disclosure_output_replaced is True
    assert result.metrics.belief_matched is False
    assert result.metrics.llm_tokens == 12
    assert mock_llm.call_count == 1
    mock_tts.assert_called_once_with(PARENT_DISCLOSURE_KO_PROBE_RESPONSE, language="ko")

    assert pipeline.conversation_history == [
        {"role": "user", "text": "공룡 얘기 해줘"},
        {"role": "assistant", "text": PARENT_DISCLOSURE_KO_PROBE_RESPONSE},
    ]
    session_dir = pipeline.session_dir
    assert session_dir is not None
    log_path = session_dir / "conversation.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["response_text"] == PARENT_DISCLOSURE_KO_PROBE_RESPONSE
    assert record["metrics"]["parent_disclosure_output_replaced"] is True


def test_english_parent_disclosure_pipeline_response(tmp_path: Path) -> None:
    """English explicit parent non-disclosure should use the English fixed response."""
    pipeline = _build_pipeline(tmp_path)
    pipeline.set_session_language("en")
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("Please don't tell my mom.")

    assert result.success is True
    assert result.detected_language == "en"
    assert result.response_text == PARENT_DISCLOSURE_EN_PROBE_RESPONSE
    assert result.metrics.parent_disclosure_matched is True
    assert result.metrics.parent_disclosure_kind == "probe"
    mock_filter.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(PARENT_DISCLOSURE_EN_PROBE_RESPONSE, language="en")
