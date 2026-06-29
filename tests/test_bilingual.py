"""Tests for bilingual language detection and prompt routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from core.language import detect_language
from core.llm_backend_config import LLMBackendConfig
from core.pipeline import ConversationPipeline, PipelineConfig
from models.llm_runner import SAFE_FALLBACK, sanitize_response


def _make_pipeline(**config_kwargs: Any) -> ConversationPipeline:
    mm = MagicMock()
    mm.tts = MagicMock()
    legacy = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path=None,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )
    with patch("core.pipeline.LLMBackendConfig.load", return_value=legacy):
        return ConversationPipeline(mm, PipelineConfig(**config_kwargs))


def test_detect_language_korean() -> None:
    """Korean text should route to Korean mode."""
    assert detect_language("안녕 뭉이!") == "ko"


def test_detect_language_english() -> None:
    """English text should route to English mode."""
    assert detect_language("Hello Mung-i!") == "en"


def test_detect_language_mixed_prefers_korean() -> None:
    """Any Hangul character should force Korean mode."""
    assert detect_language("뭉이야 hello!") == "ko"


def test_detect_language_empty_defaults_to_english() -> None:
    """Empty input should default to English when no Hangul is present."""
    assert detect_language("") == "en"


def test_detect_language_symbols_default_to_english() -> None:
    """Non-Hangul symbol-only input should default to English."""
    assert detect_language("123!@#") == "en"


def test_pipeline_bilingual_mode_off_always_uses_korean_prompt() -> None:
    """Default mode should keep the Korean prompt regardless of input language."""
    pipeline = _make_pipeline(bilingual_mode=False)

    messages = pipeline._build_messages("Hello Mung-i!")

    assert messages[0]["content"] == pipeline._config.llm_system_prompt
    assert "simple English" not in messages[0]["content"]
    assert pipeline._detect_turn_language("Hello Mung-i!") == "ko"


def test_pipeline_session_language_routes_english_prompt_and_tts_language() -> None:
    """English session mode should use the English prompt and TTS language."""
    pipeline = _make_pipeline(bilingual_mode=True)
    pipeline._mm.tts.synthesize.return_value = ([0.1], 22050)
    pipeline.set_session_language("en")

    messages = pipeline._build_messages("뭉이야 안녕!")

    assert messages[0]["content"] == pipeline._en_system_prompt
    assert "simple English" in messages[0]["content"]

    pipeline._run_tts("Hi there!", language="en")

    pipeline._mm.tts.synthesize.assert_called_once_with("Hi there!", language="en")


def test_pipeline_run_tts_passes_language_to_engine() -> None:
    """Pipeline TTS calls should pass the requested language through to the engine."""
    pipeline = _make_pipeline()
    pipeline._mm.tts.synthesize.return_value = ([0.1], 22050)

    pipeline._run_tts("hello", language="en")

    pipeline._mm.tts.synthesize.assert_called_once_with("hello", language="en")


def test_pipeline_default_session_routes_korean_prompt_for_english_text() -> None:
    """Default Korean session mode should keep Korean prompt for English input."""
    pipeline = _make_pipeline(bilingual_mode=True)

    messages = pipeline._build_messages("Hello Mung-i!")

    assert messages[0]["content"] == pipeline._config.llm_system_prompt
    assert "뭉이" in messages[0]["content"]
    assert pipeline._detect_turn_language("Hello Mung-i!") == "ko"


def test_sanitize_response_keeps_english_in_english_mode() -> None:
    """English mode should preserve English text."""
    assert sanitize_response("Hello! How are you?", language="en") == "Hello! How are you?"


def test_sanitize_response_strips_english_in_korean_mode() -> None:
    """Korean mode should keep the Korean-only sanitization behavior."""
    assert sanitize_response("Hello! How are you?", language="ko") == "! ?"


def test_sanitize_response_keeps_korean_in_korean_mode() -> None:
    """Korean text should remain unchanged in Korean mode."""
    assert sanitize_response("안녕! 잘 지내?", language="ko") == "안녕! 잘 지내?"


def test_sanitize_response_keeps_garbage_fallback_in_english_mode() -> None:
    """Garbage detection should still trigger the safe fallback in English mode."""
    assert sanitize_response("333333333", language="en") == SAFE_FALLBACK


def test_pipeline_bilingual_mode_default_is_true() -> None:
    """PipelineConfig should enable bilingual routing by default."""
    assert PipelineConfig().bilingual_mode is True


def test_pipeline_session_language_defaults_to_korean_and_can_switch() -> None:
    """Pipeline stores explicit session language independent of input text."""
    pipeline = _make_pipeline()

    assert pipeline.session_language == "ko"

    pipeline.set_session_language("en")

    assert pipeline.session_language == "en"
    assert pipeline._detect_turn_language("뭉이야 안녕!") == "en"


def test_english_prompt_has_language_constraint() -> None:
    """Guard against the Round 17 non-Latin character leak in the English prompt."""
    prompt_path = (
        Path(__file__).resolve().parents[1] / "assets" / "prompts" / "child_safe_system_en.txt"
    )
    prompt_text = prompt_path.read_text(encoding="utf-8")

    assert "LANGUAGE RULES (highest priority):" in prompt_text, (
        "English system prompt must define the LANGUAGE RULES block."
    )
    assert "NEVER use Chinese characters" in prompt_text, (
        "English system prompt must explicitly forbid Chinese characters."
    )
    assert "Korean Hangul" in prompt_text, "English system prompt must explicitly forbid Hangul."
    assert "Korean helper words" not in prompt_text
    assert "Hangul helper" not in prompt_text
    assert "사과" not in prompt_text
