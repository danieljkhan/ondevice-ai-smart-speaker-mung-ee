"""Tests for P1 fixes: TTS 2D→1D squeeze and LLM think-tag stripping.

Covers:
- SupertonicEngine.synthesize() squeeze logic for 2D audio arrays
- strip_think_tags() function for Qwen3 reasoning block removal
- Pipeline integration: _run_llm applies strip_think_tags to LLM output
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from models.llm_runner import strip_think_tags  # noqa: E402
from models.tts_runner import SupertonicEngine  # noqa: E402

# ===================================================================
# Helpers
# ===================================================================


def _make_supertonic_engine(
    model_dir: str = "/fake/supertonic",
    voice_style: str = "F1",
) -> SupertonicEngine:
    """Create a SupertonicEngine instance for testing."""
    return SupertonicEngine(model_dir=model_dir, voice_style=voice_style)


def _setup_engine_with_mock_result(
    engine: SupertonicEngine,
    result: Any,
) -> None:
    """Inject a fake model into the engine that returns the given result."""
    mock_model = MagicMock()
    mock_model.synthesize.return_value = result
    engine._model = mock_model
    engine._voice_style = MagicMock()


# ===================================================================
# TTS 2D → 1D squeeze tests (models/tts_runner.py)
# ===================================================================


class TestTTSSqueeze2DTo1D:
    """SupertonicEngine.synthesize() squeeze 동작 검증.

    Supertonic may return 2D arrays (e.g. shape (1, N) or (N, 1)).
    The squeeze logic must reduce them to 1D without breaking 1D inputs.
    """

    def test_2d_array_1_by_n_squeezed_to_1d(self) -> None:
        """Shape (1, N) → squeezed to (N,)."""
        engine = _make_supertonic_engine()
        audio_2d = np.random.randn(1, 1000).astype(np.float32)
        _setup_engine_with_mock_result(engine, audio_2d)

        audio, sr = engine.synthesize("테스트 문장")

        assert audio.ndim == 1
        assert audio.shape == (1000,)
        assert isinstance(sr, int)

    def test_2d_array_n_by_1_squeezed_to_1d(self) -> None:
        """Shape (N, 1) → squeezed to (N,)."""
        engine = _make_supertonic_engine()
        audio_2d = np.random.randn(800, 1).astype(np.float32)
        _setup_engine_with_mock_result(engine, audio_2d)

        audio, sr = engine.synthesize("세로 벡터 테스트")

        assert audio.ndim == 1
        assert audio.shape == (800,)
        assert isinstance(sr, int)

    def test_1d_array_stays_1d(self) -> None:
        """Shape (N,) stays as-is; no regression from squeeze logic."""
        engine = _make_supertonic_engine()
        audio_1d = np.random.randn(500).astype(np.float32)
        _setup_engine_with_mock_result(engine, audio_1d)

        audio, sr = engine.synthesize("1D 테스트")

        assert audio.ndim == 1
        assert audio.shape == (500,)

    def test_empty_array_stays_empty(self) -> None:
        """Empty 1D array (from blank text) should remain empty."""
        engine = _make_supertonic_engine()
        # Empty/blank text triggers early return in synthesize()
        audio, sr = engine.synthesize("")

        assert isinstance(audio, np.ndarray)
        assert audio.shape == (0,)
        assert audio.dtype == np.float32

    def test_none_text_returns_empty_array(self) -> None:
        """None text input returns empty float32 array."""
        engine = _make_supertonic_engine()
        audio, sr = engine.synthesize(None)

        assert isinstance(audio, np.ndarray)
        assert audio.shape == (0,)
        assert audio.dtype == np.float32

    def test_return_type_is_tuple_ndarray_int(self) -> None:
        """Return value must be tuple(ndarray, int)."""
        engine = _make_supertonic_engine()
        audio_1d = np.ones(100, dtype=np.float32)
        _setup_engine_with_mock_result(engine, audio_1d)

        result = engine.synthesize("타입 확인")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], int)

    def test_squeezed_audio_preserves_sample_values(self) -> None:
        """Squeeze must not alter the actual audio sample values."""
        engine = _make_supertonic_engine()
        original = np.array([[0.1, 0.5, -0.3, 0.9, -1.0]], dtype=np.float32)
        _setup_engine_with_mock_result(engine, original)

        audio, _ = engine.synthesize("값 보존 확인")

        np.testing.assert_array_almost_equal(
            audio,
            np.array([0.1, 0.5, -0.3, 0.9, -1.0], dtype=np.float32),
        )

    def test_3d_array_also_squeezed(self) -> None:
        """Shape (1, 1, N) → squeezed to (N,) as ndim > 1."""
        engine = _make_supertonic_engine()
        audio_3d = np.random.randn(1, 1, 200).astype(np.float32)
        _setup_engine_with_mock_result(engine, audio_3d)

        audio, _ = engine.synthesize("3D 테스트")

        assert audio.ndim == 1
        assert audio.shape == (200,)

    def test_2d_single_sample_flattened_to_1d(self) -> None:
        """Shape (1, 1) → flattened to (1,), not 0-D scalar."""
        engine = _make_supertonic_engine()
        audio_2d = np.array([[0.42]], dtype=np.float32)
        _setup_engine_with_mock_result(engine, audio_2d)

        audio, _ = engine.synthesize("단일 샘플")

        assert audio.ndim == 1
        assert audio.shape == (1,)
        assert audio[0] == np.float32(0.42)


# ===================================================================
# strip_think_tags tests (models/llm_runner.py)
# ===================================================================


class TestStripThinkTags:
    """strip_think_tags() 함수의 Qwen3 <think> 태그 제거 검증."""

    def test_closed_think_block_removed(self) -> None:
        """Closed <think>...</think> block is fully removed."""
        text = "<think>reasoning here</think>actual response"
        assert strip_think_tags(text) == "actual response"

    def test_multiple_think_blocks_removed(self) -> None:
        """Multiple <think> blocks are all removed."""
        text = "<think>first thought</think>hello <think>second thought</think>world"
        assert strip_think_tags(text) == "hello world"

    def test_unclosed_think_block_removed(self) -> None:
        """Unclosed <think> block (truncated generation) is removed."""
        text = "some text<think>this was truncated and never closed"
        assert strip_think_tags(text) == "some text"

    def test_text_without_think_tags_unchanged(self) -> None:
        """Text with no think tags passes through unchanged."""
        text = "안녕하세요! 오늘 뭐 하고 놀까요?"
        assert strip_think_tags(text) == text

    def test_empty_string_input(self) -> None:
        """Empty string returns empty string."""
        assert strip_think_tags("") == ""

    def test_think_with_multiline_content(self) -> None:
        """<think> block with multiline content is fully removed."""
        text = "<think>\nline 1\nline 2\nline 3\n</think>final answer"
        assert strip_think_tags(text) == "final answer"

    def test_mixed_text_before_and_after_think(self) -> None:
        """Text before + think block + text after → only surrounding text."""
        text = "before <think>reasoning goes here</think> after"
        assert strip_think_tags(text) == "before  after"

    def test_only_think_block_returns_empty(self) -> None:
        """Input that is entirely a think block returns empty string."""
        text = "<think>nothing but reasoning</think>"
        assert strip_think_tags(text) == ""

    def test_only_unclosed_think_returns_empty(self) -> None:
        """Input that is entirely an unclosed think block returns empty."""
        text = "<think>started reasoning but never finished"
        assert strip_think_tags(text) == ""

    def test_nested_think_tags_handled(self) -> None:
        """Nested/malformed <think> tags are handled gracefully.

        The regex is greedy within each closed pair (DOTALL),
        so nested inner tags are consumed along with the outer pair.
        """
        text = "<think>outer <think>inner</think> still outer</think>result"
        result = strip_think_tags(text)
        # The first regex matches from first <think> to first </think>,
        # leaving " still outer</think>result", then second pass cleans up
        # remaining </think> is just text. Verify no crash and result is clean.
        assert "<think>" not in result

    def test_whitespace_only_after_strip_returns_empty(self) -> None:
        """If only whitespace remains after stripping, return empty."""
        text = "  <think>reasoning</think>  "
        assert strip_think_tags(text) == ""

    def test_think_tag_case_sensitive(self) -> None:
        """<THINK> (uppercase) is NOT stripped — only lowercase."""
        text = "<THINK>not removed</THINK>visible"
        assert strip_think_tags(text) == "<THINK>not removed</THINK>visible"

    def test_think_with_special_characters(self) -> None:
        """Think block containing special chars is stripped cleanly."""
        text = "<think>한국어 추론 + 특수문자 !@#$%</think>결과"
        assert strip_think_tags(text) == "결과"

    def test_empty_think_block(self) -> None:
        """Empty <think></think> block is removed."""
        text = "<think></think>response"
        assert strip_think_tags(text) == "response"

    def test_think_block_followed_by_newlines(self) -> None:
        """Think block followed by newlines: result is stripped."""
        text = "<think>reasoning</think>\n\nactual answer"
        assert strip_think_tags(text) == "actual answer"


# ===================================================================
# Pipeline integration: _run_llm applies strip_think_tags
# ===================================================================


class TestPipelineRunLlmThinkStrip:
    """ConversationPipeline._run_llm이 strip_think_tags를 적용하는지 검증."""

    def _make_pipeline(self) -> Any:
        """Create a ConversationPipeline with mocked ModelManager."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        return ConversationPipeline(mm, PipelineConfig())

    def test_think_tags_stripped_from_llm_output(self) -> None:
        """LLM returning <think>...</think>응답 → pipeline returns '응답'."""
        p = self._make_pipeline()

        with patch("models.llm_runner.run_generation") as mock_gen:
            mock_gen.return_value = (
                "<think>reasoning about answer</think>멋진 응답이에요",
                10,
                0.2,
                0.5,
            )
            text, token_count, ttft = p._run_llm("test prompt")

        assert text == "멋진 응답이에요"
        assert token_count == 10
        assert ttft == 0.2

    def test_no_think_tags_pass_through(self) -> None:
        """LLM output without think tags passes through unchanged."""
        p = self._make_pipeline()

        with patch("models.llm_runner.run_generation") as mock_gen:
            mock_gen.return_value = ("깨끗한 응답이에요", 5, 0.1, 0.3)
            text, token_count, ttft = p._run_llm("prompt")

        assert text == "깨끗한 응답이에요"
        assert token_count == 5

    def test_unclosed_think_stripped_from_pipeline(self) -> None:
        """LLM truncated mid-think → only pre-think text returned."""
        p = self._make_pipeline()

        with patch("models.llm_runner.run_generation") as mock_gen:
            mock_gen.return_value = (
                "안녕!<think>truncated reasoning that never",
                15,
                0.3,
                0.8,
            )
            text, token_count, ttft = p._run_llm("prompt")

        assert text == "안녕!"
        assert token_count == 15

    def test_multiline_think_stripped_from_pipeline(self) -> None:
        """LLM output with multiline think block → clean response."""
        p = self._make_pipeline()

        with patch("models.llm_runner.run_generation") as mock_gen:
            mock_gen.return_value = (
                "<think>\nstep 1: analyze\nstep 2: formulate\n</think>아이에게 적합한 답변입니다.",
                20,
                0.4,
                1.0,
            )
            text, token_count, ttft = p._run_llm("prompt")

        assert text == "아이에게 적합한 답변입니다."

    def test_full_turn_with_think_tags_in_response(self) -> None:
        """E2E run_turn: LLM returns think-tagged output → final response is clean."""
        from core.pipeline import ConversationPipeline, PipelineConfig

        mm = MagicMock()
        p = ConversationPipeline(mm, PipelineConfig())

        fake_seg = MagicMock(start=0.0, end=0.5)

        with (
            patch.object(p, "_run_vad", return_value=[fake_seg]),
            patch.object(p, "_extract_speech", return_value=[0.1] * 8000),
            patch.object(p, "_run_stt", return_value="안녕하세요"),
            patch(
                "models.llm_runner.run_generation",
                return_value=(
                    "<think>child asked hello, respond warmly</think>안녕! 나는 뭉이야!",
                    12,
                    0.2,
                    0.6,
                ),
            ),
            patch.object(p, "_run_tts", return_value=([0.0] * 100, 22050)),
        ):
            result = p.run_turn([0.0] * 16000)

        assert result.success is True
        assert result.response_text == "안녕! 나는 뭉이야!"
        assert "<think>" not in result.response_text


# ===================================================================
# Regex pattern module-level constants verification
# ===================================================================


class TestThinkTagRegexPatterns:
    """models.llm_runner 내 _THINK_TAG_RE, _THINK_UNCLOSED_RE 상수 검증."""

    def test_think_tag_re_is_compiled(self) -> None:
        """_THINK_TAG_RE is a compiled regex pattern."""
        import re

        from models.llm_runner import _THINK_TAG_RE

        assert isinstance(_THINK_TAG_RE, re.Pattern)

    def test_think_unclosed_re_is_compiled(self) -> None:
        """_THINK_UNCLOSED_RE is a compiled regex pattern."""
        import re

        from models.llm_runner import _THINK_UNCLOSED_RE

        assert isinstance(_THINK_UNCLOSED_RE, re.Pattern)

    def test_think_tag_re_uses_dotall(self) -> None:
        """_THINK_TAG_RE uses DOTALL flag for multiline matching."""
        import re

        from models.llm_runner import _THINK_TAG_RE

        assert _THINK_TAG_RE.flags & re.DOTALL

    def test_think_unclosed_re_uses_dotall(self) -> None:
        """_THINK_UNCLOSED_RE uses DOTALL flag for multiline matching."""
        import re

        from models.llm_runner import _THINK_UNCLOSED_RE

        assert _THINK_UNCLOSED_RE.flags & re.DOTALL

    def test_strip_think_tags_importable(self) -> None:
        """strip_think_tags is importable from models.llm_runner."""
        from models.llm_runner import strip_think_tags as fn

        assert callable(fn)

    def test_residual_think_text_stripped(self) -> None:
        """Residual 'think' word at start from empty prefill is removed."""
        assert strip_think_tags("think 안녕하세요!") == "안녕하세요!"

    def test_residual_think_with_whitespace(self) -> None:
        """Residual 'think' with leading whitespace is removed."""
        assert strip_think_tags("  think  공룡이 멋져요!") == "공룡이 멋져요!"

    def test_residual_think_case_insensitive(self) -> None:
        """Residual 'Think' (capitalized) is also removed."""
        assert strip_think_tags("Think 안녕!") == "안녕!"

    def test_standalone_closing_think_tag(self) -> None:
        """Standalone </think> from empty prefill echo is removed."""
        assert strip_think_tags("</think>\n\n안녕하세요!") == "안녕하세요!"

    def test_closing_think_then_residual(self) -> None:
        """</think> followed by 'think' word both removed."""
        assert strip_think_tags("</think>\nthink 응답이에요") == "응답이에요"


class TestSanitizeResponse:
    """sanitize_response() removes foreign text, emoji, symbols."""

    def test_korean_passthrough(self) -> None:
        """Pure Korean text passes through unchanged."""
        from models.llm_runner import sanitize_response

        assert sanitize_response("안녕하세요, 좋은 날이에요.") == "안녕하세요, 좋은 날이에요."

    def test_chinese_removed(self) -> None:
        """Chinese characters are stripped."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("힘들면 말해줘好不好? 함께 해결할게!")
        assert "好" not in result
        assert "함께 해결할게" in result

    def test_emoji_removed(self) -> None:
        """Emoji characters are stripped."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("잘했어요! 😊 대단해요!")
        assert "\U0001f60a" not in result
        assert "잘했어요" in result

    def test_japanese_removed(self) -> None:
        """Japanese characters are stripped."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("안녕하세요! おはよう 좋은 아침이에요!")
        assert "おはよう" not in result
        assert "안녕하세요" in result

    def test_empty_returns_fallback(self) -> None:
        """Empty input returns safe fallback response."""
        from models.llm_runner import SAFE_FALLBACK, sanitize_response

        assert sanitize_response("") == SAFE_FALLBACK
        assert sanitize_response("😊😊😊") == SAFE_FALLBACK

    def test_english_words_removed(self) -> None:
        """English words (2+ chars) are removed per Korean-only rule."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("Hello, 안녕하세요!")
        assert "Hello" not in result
        assert "안녕하세요" in result

    def test_mixed_cleanup(self) -> None:
        """Mixed foreign text is cleaned to Korean/English only."""
        from models.llm_runner import sanitize_response

        text = "안전해, 뭉이야. 어두운 건 무서워할 수 있대요. 不要怕!"
        result = sanitize_response(text)
        assert "不" not in result
        assert "안전해" in result

    def test_multiple_spaces_collapsed(self) -> None:
        """Multiple spaces from removals are collapsed."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("좋아요  😊  대단해요")
        assert "  " not in result

    def test_english_artifact_removed(self) -> None:
        """Random English artifacts like 'keras' are stripped."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("좋겠습니다.keras")
        assert "keras" not in result
        assert "좋겠습니다" in result

    def test_english_words_stripped(self) -> None:
        """English words (2+ chars) removed per Korean-only rule."""
        from models.llm_runner import sanitize_response

        result = sanitize_response("Hello, 안녕하세요! special 대단해요")
        assert "Hello" not in result
        assert "special" not in result
        assert "안녕하세요" in result
        assert "대단해요" in result
