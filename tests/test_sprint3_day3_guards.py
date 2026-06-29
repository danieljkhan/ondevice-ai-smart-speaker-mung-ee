"""Tests for Sprint 3 Day 3 input guard changes (Lane A).

Covers:
- STT empty/None audio guard (core/pipeline.py:_run_stt)
- TTS empty/None text guard (models/tts_runner.py: Supertonic)
- LLM empty/None prompt guard (models/llm_runner.py:run_generation)

All guards ensure graceful handling of None/empty inputs with
appropriate early returns, warning logs, and no crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from core.pipeline import ConversationPipeline, PipelineConfig  # noqa: E402
from models.llm_runner import run_generation  # noqa: E402
from models.tts_runner import SupertonicEngine  # noqa: E402

# ===================================================================
# Helpers
# ===================================================================


def _make_pipeline() -> ConversationPipeline:
    mm = MagicMock()
    return ConversationPipeline(mm, PipelineConfig())


# ===================================================================
# STT empty audio guard (pipeline.py:_run_stt)
# ===================================================================


class TestSttEmptyAudioGuard:
    """STT 처리 전 빈/None 오디오 입력 방어 검증."""

    def test_none_audio_returns_empty_string(self) -> None:
        p = _make_pipeline()
        result = p._run_stt(None)
        assert result == ""

    def test_empty_list_returns_empty_string(self) -> None:
        p = _make_pipeline()
        result = p._run_stt([])
        assert result == ""

    def test_none_audio_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        p = _make_pipeline()
        with caplog.at_level(logging.WARNING):
            p._run_stt(None)
        assert "Empty audio data received" in caplog.text

    def test_empty_audio_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        p = _make_pipeline()
        with caplog.at_level(logging.WARNING):
            p._run_stt([])
        assert "Empty audio data received" in caplog.text

    def test_normal_audio_proceeds_past_guard(self, tmp_path: Path) -> None:
        """정상 오디오 → guard 통과 후 STT 처리 시도."""
        p = _make_pipeline()
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"")
        with (
            patch("tempfile.mkstemp", return_value=(0, str(wav_file))),
            patch("os.close"),
            patch.object(p, "_write_temp_wav"),
            patch("models.stt_runner.run_stt") as mock_stt,
        ):
            mock_stt.return_value = (
                [MagicMock(text="테스트")],
                MagicMock(),
            )
            result = p._run_stt([0.1, 0.2, 0.3])
        assert result == "테스트"
        mock_stt.assert_called_once()


# ===================================================================
# TTS empty text guard — SupertonicEngine
# ===================================================================


class TestSupertonicEmptyTextGuard:
    """SupertonicEngine: None/빈 텍스트 가드 검증."""

    def test_none_text_returns_empty_array(self) -> None:
        engine = SupertonicEngine(model_dir="/fake")
        audio, sr = engine.synthesize(None)
        assert len(audio) == 0
        assert audio.dtype == np.float32
        assert isinstance(sr, int)

    def test_empty_string_returns_empty_array(self) -> None:
        engine = SupertonicEngine(model_dir="/fake")
        audio, sr = engine.synthesize("")
        assert len(audio) == 0

    def test_whitespace_returns_empty_array(self) -> None:
        engine = SupertonicEngine(model_dir="/fake")
        audio, sr = engine.synthesize("   \n\t  ")
        assert len(audio) == 0

    def test_none_text_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = SupertonicEngine(model_dir="/fake")
        with caplog.at_level(logging.WARNING):
            engine.synthesize(None)
        assert "Empty text received" in caplog.text

    def test_none_text_does_not_require_loaded_model(self) -> None:
        """None 입력 시 모델 미로드 상태에서도 크래시 없음."""
        engine = SupertonicEngine(model_dir="/fake")
        assert engine._model is None
        audio, sr = engine.synthesize(None)
        assert len(audio) == 0

    def test_normal_text_requires_loaded_model(self) -> None:
        """정상 텍스트 → 모델 필요, 미로드 시 RuntimeError."""
        engine = SupertonicEngine(model_dir="/fake")
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.synthesize("안녕하세요")


# ===================================================================
# LLM empty prompt guard (llm_runner.py:run_generation)
# ===================================================================


class TestLlmEmptyPromptGuard:
    """run_generation: None/빈 프롬프트 가드 검증."""

    def test_none_prompt_returns_negative_ttft(self) -> None:
        mock_llm = MagicMock()
        text, tokens, ttft, gen_time = run_generation(mock_llm, None)
        assert text == ""
        assert tokens == 0
        assert ttft == -1.0
        assert gen_time == 0.0

    def test_empty_prompt_returns_negative_ttft(self) -> None:
        mock_llm = MagicMock()
        text, tokens, ttft, gen_time = run_generation(mock_llm, "")
        assert text == ""
        assert tokens == 0
        assert ttft == -1.0
        assert gen_time == 0.0

    def test_whitespace_prompt_returns_negative_ttft(self) -> None:
        mock_llm = MagicMock()
        text, tokens, ttft, gen_time = run_generation(mock_llm, "   \n\t  ")
        assert text == ""
        assert tokens == 0
        assert ttft == -1.0

    def test_none_prompt_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_llm = MagicMock()
        with caplog.at_level(logging.WARNING):
            run_generation(mock_llm, None)
        assert "Empty prompt received" in caplog.text

    def test_none_prompt_does_not_call_llm(self) -> None:
        """None 프롬프트 시 LLM 모델이 호출되지 않음."""
        mock_llm = MagicMock()
        run_generation(mock_llm, None)
        mock_llm.assert_not_called()

    def test_empty_prompt_does_not_call_llm(self) -> None:
        """빈 프롬프트 시 LLM 모델이 호출되지 않음."""
        mock_llm = MagicMock()
        run_generation(mock_llm, "")
        mock_llm.assert_not_called()

    def test_normal_prompt_calls_llm(self) -> None:
        """정상 프롬프트 → LLM 호출 확인, TTFT >= 0."""
        mock_llm = MagicMock()
        mock_llm.return_value = iter(
            [
                {"choices": [{"text": "응답"}]},
            ]
        )
        text, tokens, ttft, gen_time = run_generation(mock_llm, "안녕하세요")
        mock_llm.assert_called_once()
        assert text == "응답"
        assert tokens == 1
        assert ttft >= 0.0

    def test_normal_prompt_ttft_is_not_negative(self) -> None:
        """정상 프롬프트 TTFT는 -1.0이 아닌 0 이상 값."""
        mock_llm = MagicMock()
        mock_llm.return_value = iter(
            [
                {"choices": [{"text": "hi"}]},
            ]
        )
        _, _, ttft, _ = run_generation(mock_llm, "test prompt")
        assert ttft >= 0.0
        assert ttft != -1.0
