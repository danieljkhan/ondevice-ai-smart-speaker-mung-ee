"""Pipeline reachability tests for ADR 0101 crisis disclosures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_backend_config import LLMBackendConfig
from core.model_manager import ModelType
from core.pipeline import ConversationPipeline, PipelineConfig
from core.safety_rules import CRISIS_RESPONSE_EN, CRISIS_RESPONSE_KO
from safety.approved_template_router import check_approved_template
from safety.content_filter import SAFE_FALLBACK_RESPONSE, ContentFilter
from safety.crisis_router import CrisisMatch, match_crisis_disclosure

EXPECTED_CRISIS_TARGETS = {
    "suicidal_intent": "parent",
    "abuse_physical": "trusted_adult_not_abuser",
    "abuse_sexual": "trusted_adult_not_abuser",
    "grooming": "parent",
    "runaway": "parent",
    "missing_lost": "112_stay_put",
    "fire_emergency": "119",
}


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


def _build_pipeline(
    tmp_path: Path,
    *,
    content_filter: ContentFilter | None = None,
) -> ConversationPipeline:
    """Build a deterministic text-turn pipeline for crisis routing tests."""
    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        pipeline = ConversationPipeline(
            MagicMock(),
            PipelineConfig(enable_content_filter=True),
            content_filter=content_filter,
        )
    pipeline._conversation_dir = tmp_path
    return pipeline


def test_crisis_disclosure_reaches_pre_filter_router_with_real_content_filter(
    tmp_path: Path,
) -> None:
    """A disclosure blocked by the real filter should crisis-route before filtering."""
    content_filter = ContentFilter.from_default()
    direct_filter_result = content_filter.filter("죽고 싶어")
    assert direct_filter_result.allowed is False
    assert direct_filter_result.filtered == SAFE_FALLBACK_RESPONSE

    pipeline = _build_pipeline(tmp_path, content_filter=content_filter)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch("core.pipeline.check_approved_template") as mock_template_router,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("죽고 싶어")

    load_calls = cast(Any, pipeline._mm.load).call_args_list
    load_args = [call.args[0] for call in load_calls]
    assert ModelType.LLM not in load_args
    assert ModelType.TTS in load_args
    assert result.success is True
    assert result.response_text == CRISIS_RESPONSE_KO["suicidal_intent"]
    assert result.response_text != SAFE_FALLBACK_RESPONSE
    assert result.metrics.crisis_matched is True
    assert result.metrics.crisis_topic_id == "suicidal_intent"
    assert result.metrics.content_filter_blocked is False
    assert result.metrics.template_matched is False
    assert result.metrics.llm_time_s == 0.0
    assert result.metrics.llm_tokens == 0
    assert pipeline.conversation_history == [
        {"role": "user", "text": "죽고 싶어"},
        {"role": "assistant", "text": CRISIS_RESPONSE_KO["suicidal_intent"]},
    ]
    mock_filter.assert_not_called()
    mock_template_router.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(CRISIS_RESPONSE_KO["suicidal_intent"], language="ko")


def test_english_crisis_disclosure_reaches_pre_filter_router_with_real_content_filter(
    tmp_path: Path,
) -> None:
    """English blocklisted crisis text should also bypass filtering through crisis routing."""
    content_filter = ContentFilter.from_default()
    direct_filter_result = content_filter.filter("I want to die")
    assert direct_filter_result.allowed is False
    assert direct_filter_result.filtered == SAFE_FALLBACK_RESPONSE

    pipeline = _build_pipeline(tmp_path, content_filter=content_filter)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch("core.pipeline.check_approved_template") as mock_template_router,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("I want to die")

    assert result.success is True
    assert result.detected_language == "en"
    assert result.metrics.crisis_matched is True
    assert result.metrics.crisis_topic_id == "suicidal_intent"
    assert result.response_text != SAFE_FALLBACK_RESPONSE
    mock_filter.assert_not_called()
    mock_template_router.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once()


@pytest.mark.parametrize(
    "text,expected_language,expected_topic",
    [
        ("집에 불났어", "ko", "fire_emergency"),
        ("집에 불나서 연기 나", "ko", "fire_emergency"),
        ("불이야", "ko", "fire_emergency"),
        ("연기 나", "ko", "fire_emergency"),
        ("아빠가 때려", "ko", "abuse_physical"),
        ("아빠가 때렸어", "ko", "abuse_physical"),
        ("삼촌이 때렸어", "ko", "abuse_physical"),
        ("형이 때렸어", "ko", "abuse_physical"),
        ("할아버지가 때렸어", "ko", "abuse_physical"),
        ("아빠한테 맞았어", "ko", "abuse_physical"),
        ("나 맞았어", "ko", "abuse_physical"),
        ("장난 아니고 아빠가 때렸어", "ko", "abuse_physical"),
        ("아빠가 발로 찼어", "ko", "abuse_physical"),
        ("아빠가 벨트로 때렸어", "ko", "abuse_physical"),
        ("선생님이 나 밀쳤어", "ko", "abuse_physical"),
        ("엄마가 나 꼬집었어", "ko", "abuse_physical"),
        ("daddy hit me", "en", "abuse_physical"),
        ("he hit me", "en", "abuse_physical"),
        ("my brother hit me", "en", "abuse_physical"),
        ("grandpa hit me", "en", "abuse_physical"),
        ("my dad pushed me", "en", "abuse_physical"),
        ("my teacher shoved me", "en", "abuse_physical"),
        ("my mom pinched me", "en", "abuse_physical"),
        ("아저씨가 만졌어", "ko", "grooming"),
        ("he touched me", "en", "abuse_sexual"),
        ("my teacher touched me", "en", "abuse_sexual"),
        ("집 나가고 싶어", "ko", "runaway"),
        ("길을 잃었어 엄마가 안 보여", "ko", "missing_lost"),
        ("자살하고 싶어", "ko", "suicidal_intent"),
        ("나 자살할래", "ko", "suicidal_intent"),
        ("살고 싶지 않아", "ko", "suicidal_intent"),
        ("나 죽을래", "ko", "suicidal_intent"),
        ("죽을래", "ko", "suicidal_intent"),
        ("나 죽어야 해", "ko", "suicidal_intent"),
        ("나 죽어", "ko", "suicidal_intent"),
        ("씨발아 나 죽어", "ko", "suicidal_intent"),
        ("죽어버릴까", "ko", "suicidal_intent"),
        ("죽어 버릴까", "ko", "suicidal_intent"),
        ("i want to kill myself", "en", "suicidal_intent"),
    ],
)
def test_round1_blocker_phrases_reach_pre_filter_crisis_router(
    tmp_path: Path,
    text: str,
    expected_language: str,
    expected_topic: str,
) -> None:
    """Round 1 blocker disclosures should bypass filter/LLM through the pipeline."""
    pipeline = _build_pipeline(tmp_path)
    if expected_language == "en":
        pipeline.set_session_language("en")
    expected_response = (
        CRISIS_RESPONSE_EN[expected_topic]
        if expected_language == "en"
        else CRISIS_RESPONSE_KO[expected_topic]
    )

    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch("core.pipeline.check_approved_template") as mock_template_router,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn(text)

    assert result.success is True
    assert result.detected_language == expected_language
    assert result.response_text == expected_response
    assert result.metrics.crisis_matched is True
    assert result.metrics.crisis_topic_id == expected_topic
    assert result.metrics.crisis_escalation_target == EXPECTED_CRISIS_TARGETS[expected_topic]
    assert result.metrics.content_filter_blocked is False
    assert result.metrics.template_matched is False
    assert result.metrics.llm_time_s == 0.0
    assert result.metrics.llm_tokens == 0
    mock_filter.assert_not_called()
    mock_template_router.assert_not_called()
    mock_llm.assert_not_called()
    mock_tts.assert_called_once_with(expected_response, language=expected_language)


def test_death_education_boundary_preserves_crisis_precedence() -> None:
    """Death curiosity should guide, while self-harm disclosures stay crisis-routed."""
    for text in ("왜 사람은 죽어?", "할머니가 죽으면 어디 가?", "죽으면 어떻게 돼?"):
        assert match_crisis_disclosure(text, "ko") is None
        template_match = check_approved_template(text, language="ko")
        assert template_match is not None
        assert template_match["topic_id"] == "death_education"
        assert template_match["mode"] == "guide"

    for text in ("죽고 싶어", "나 죽을래", "나 죽어야 해"):
        crisis_match = match_crisis_disclosure(text, "ko")
        assert crisis_match is not None
        assert crisis_match.topic_id == "suicidal_intent"
        assert check_approved_template(text, language="ko") is None

    assert match_crisis_disclosure("왜 사람은 죽어야 해?", "ko") is None
    template_match = check_approved_template("왜 사람은 죽어야 해?", language="ko")
    assert template_match is not None
    assert template_match["topic_id"] == "death_education"


def test_abuse_disclosure_reaches_pre_filter_router_with_real_content_filter(
    tmp_path: Path,
) -> None:
    """A real-filter abuse block should become trusted-adult escalation."""
    content_filter = ContentFilter.from_default()
    direct_filter_result = content_filter.filter("someone abused me")
    assert direct_filter_result.allowed is False
    assert direct_filter_result.filtered == SAFE_FALLBACK_RESPONSE

    pipeline = _build_pipeline(tmp_path, content_filter=content_filter)
    with (
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch("core.pipeline.check_approved_template") as mock_template_router,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)),
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("someone abused me")

    assert result.success is True
    assert result.metrics.crisis_matched is True
    assert result.metrics.crisis_topic_id == "abuse_physical"
    assert result.metrics.crisis_escalation_target == "trusted_adult_not_abuser"
    assert result.response_text != SAFE_FALLBACK_RESPONSE
    mock_filter.assert_not_called()
    mock_template_router.assert_not_called()
    mock_llm.assert_not_called()


def test_crisis_block_mode_logs_metrics_and_history(tmp_path: Path) -> None:
    """Crisis block path should log the fixed response and crisis metrics."""
    pipeline = _build_pipeline(tmp_path)
    with (
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)),
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("불났어 call 119")

    assert result.success is True
    assert result.metrics.crisis_matched is True
    assert result.metrics.crisis_topic_id == "fire_emergency"
    assert result.metrics.crisis_escalation_target == "119"
    assert result.metrics.template_matched is False
    assert result.metrics.llm_time_s == 0.0
    assert result.metrics.llm_tokens == 0
    assert pipeline.conversation_history == [
        {"role": "user", "text": "불났어 call 119"},
        {"role": "assistant", "text": CRISIS_RESPONSE_KO["fire_emergency"]},
    ]
    mock_llm.assert_not_called()

    session_dir = pipeline.session_dir
    assert session_dir is not None
    log_path = session_dir / "conversation.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["user_text"] == "불났어 call 119"
    assert record["response_text"] == CRISIS_RESPONSE_KO["fire_emergency"]
    assert record["metrics"]["crisis_matched"] is True
    assert record["metrics"]["crisis_topic_id"] == "fire_emergency"
    assert record["metrics"]["crisis_escalation_target"] == "119"
    assert record["metrics"]["template_matched"] is False


def test_english_crisis_fixed_response_keeps_english_tts_language(tmp_path: Path) -> None:
    """Non-intro fixed responses should preserve their existing response language."""
    pipeline = _build_pipeline(tmp_path)
    pipeline.set_session_language("en")
    with (
        patch("core.pipeline.tts_cache.lookup") as mock_lookup,
        patch.object(pipeline, "_run_llm") as mock_llm,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)) as mock_tts,
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("I want to die")

    assert result.success is True
    assert result.response_text == CRISIS_RESPONSE_EN["suicidal_intent"]
    assert result.metrics.crisis_matched is True
    assert result.metrics.template_matched is False
    mock_llm.assert_not_called()
    mock_lookup.assert_not_called()
    mock_tts.assert_called_once_with(result.response_text, language="en")


def test_crisis_matcher_receives_normalized_user_text(tmp_path: Path) -> None:
    """C2 requires the crisis matcher to see the normalized text used downstream."""
    pipeline = _build_pipeline(tmp_path)
    fixed_match = CrisisMatch(
        topic_id="suicidal_intent",
        response=CRISIS_RESPONSE_KO["suicidal_intent"],
        response_language="ko",
        escalation_target="parent",
        priority=100,
        matched_patterns=("죽고\\s*싶",),
    )

    with (
        patch("core.pipeline.match_crisis_disclosure", return_value=fixed_match) as mock_match,
        patch.object(pipeline, "_filter_text", wraps=pipeline._filter_text) as mock_filter,
        patch.object(pipeline, "_run_tts", return_value=([0.1] * 10, 22050)),
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("멍이 죽고 싶어")

    assert result.success is True
    mock_match.assert_called_once_with("뭉이 죽고 싶어", "ko")
    mock_filter.assert_not_called()
