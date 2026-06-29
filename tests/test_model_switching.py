"""Tests for config-based LLM model switching."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from core.pipeline import ConversationPipeline, PipelineConfig
from models.llm_runner import (
    DEFAULT_STOP_SEQUENCES,
    GEMMA_STOP_SEQUENCES,
    MODEL_FAMILY_AUTO,
    MODEL_FAMILY_GEMMA,
    MODEL_FAMILY_QWEN,
    detect_model_family,
    stop_sequences_for_family,
)


class TestDetectModelFamily:
    """Tests for detect_model_family()."""

    def test_qwen_gguf(self) -> None:
        assert detect_model_family("/opt/mungi/ai_models/Qwen3.5-2B-DPO.Q6_K.gguf") == "qwen"

    def test_gemma_gguf(self) -> None:
        assert detect_model_family("/opt/mungi/ai_models/gemma-4-E2B-it-mungi.Q4_K_M.gguf") == (
            "gemma"
        )

    def test_gemma_case_insensitive(self) -> None:
        assert detect_model_family("/path/Gemma-4-E2B.Q6_K.gguf") == "gemma"

    def test_unknown_defaults_to_qwen(self) -> None:
        assert detect_model_family("/path/some-model.Q4_K_M.gguf") == "qwen"

    def test_empty_path(self) -> None:
        assert detect_model_family("") == "qwen"


class TestStopSequencesForFamily:
    """Tests for stop_sequences_for_family()."""

    def test_qwen_returns_default(self) -> None:
        result = stop_sequences_for_family(MODEL_FAMILY_QWEN)
        assert result == DEFAULT_STOP_SEQUENCES
        assert result is not DEFAULT_STOP_SEQUENCES

    def test_gemma_returns_gemma_sequences(self) -> None:
        result = stop_sequences_for_family(MODEL_FAMILY_GEMMA)
        assert result == GEMMA_STOP_SEQUENCES
        assert result is not GEMMA_STOP_SEQUENCES

    def test_unknown_defaults_to_qwen(self) -> None:
        assert stop_sequences_for_family("unknown") == DEFAULT_STOP_SEQUENCES


class TestManagerConfigModelFamily:
    """Tests for ManagerConfig.llm_model_family field."""

    def test_default_is_auto(self) -> None:
        from core.model_manager import ManagerConfig

        cfg = ManagerConfig()
        assert cfg.llm_model_family == MODEL_FAMILY_AUTO

    def test_env_override_gemma(self) -> None:
        from core.model_manager import ManagerConfig

        with patch.dict(os.environ, {"MUNGI_LLM_MODEL_FAMILY": "gemma"}):
            cfg = ManagerConfig()
            assert cfg.llm_model_family == MODEL_FAMILY_GEMMA

    def test_env_override_qwen(self) -> None:
        from core.model_manager import ManagerConfig

        with patch.dict(os.environ, {"MUNGI_LLM_MODEL_FAMILY": "qwen"}):
            cfg = ManagerConfig()
            assert cfg.llm_model_family == MODEL_FAMILY_QWEN

    def test_env_invalid_ignored(self) -> None:
        from core.model_manager import ManagerConfig

        with patch.dict(os.environ, {"MUNGI_LLM_MODEL_FAMILY": "invalid"}):
            cfg = ManagerConfig()
            assert cfg.llm_model_family == MODEL_FAMILY_AUTO

    def test_env_case_insensitive(self) -> None:
        from core.model_manager import ManagerConfig

        with patch.dict(os.environ, {"MUNGI_LLM_MODEL_FAMILY": "GEMMA"}):
            cfg = ManagerConfig()
            assert cfg.llm_model_family == MODEL_FAMILY_GEMMA


class TestModelManagerModelFamily:
    """Tests for ModelManager.llm_model_family detection."""

    def test_property_defaults_to_qwen_before_load(self) -> None:
        from core.model_manager import ManagerConfig, ModelManager

        mm = ModelManager(ManagerConfig(model_dir="/fake/models"))
        assert mm.llm_model_family == MODEL_FAMILY_QWEN

    def test_load_detects_model_family_when_auto(self) -> None:
        from core.model_manager import ManagerConfig, ModelManager

        mm = ModelManager(ManagerConfig(model_dir="/fake/models"))
        mm._config.llm_model_path = "/fake/gemma-4-E2B-it.Q4_K_M.gguf"

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch("models.llm_runner.load_llm_model", return_value=MagicMock()),
        ):
            mm._load_llm_full_gpu()

        assert mm.llm_model_family == MODEL_FAMILY_GEMMA

    def test_load_respects_explicit_model_family_override(self) -> None:
        from core.model_manager import ManagerConfig, ModelManager

        mm = ModelManager(
            ManagerConfig(
                model_dir="/fake/models",
                llm_model_family=MODEL_FAMILY_QWEN,
            )
        )
        mm._config.llm_model_path = "/fake/gemma-4-E2B-it.Q4_K_M.gguf"

        with (
            patch.object(mm, "_select_gpu_layers", return_value=-1),
            patch("models.llm_runner.load_llm_model", return_value=MagicMock()),
        ):
            mm._load_llm_full_gpu()

        assert mm.llm_model_family == MODEL_FAMILY_QWEN


class TestPipelineModelSwitching:
    """Tests for runtime stop-sequence override in ConversationPipeline."""

    @staticmethod
    def _make_pipeline(model_family: str) -> ConversationPipeline:
        mm = MagicMock()
        mm.llm = MagicMock()
        mm.llm_model_family = model_family
        return ConversationPipeline(mm, PipelineConfig())

    def test_run_llm_uses_gemma_stop_sequences_for_chat_generation(self) -> None:
        pipeline = self._make_pipeline(MODEL_FAMILY_GEMMA)
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with patch(
            "models.llm_runner.run_chat_generation",
            return_value=("hello", 2, 0.1, 0.2, None, None),
        ) as mock_chat:
            pipeline._run_llm(messages)

        assert mock_chat.call_args.kwargs["stop"] == GEMMA_STOP_SEQUENCES

    def test_run_llm_uses_family_stop_sequences_for_string_prompt(self) -> None:
        pipeline = self._make_pipeline(MODEL_FAMILY_GEMMA)

        with patch(
            "models.llm_runner.run_generation",
            return_value=("hello", 2, 0.1, 0.2),
        ) as mock_generation:
            pipeline._run_llm("hi")

        assert mock_generation.call_args.kwargs["stop"] == GEMMA_STOP_SEQUENCES

    def test_run_llm_fallback_uses_effective_stop_sequences(self) -> None:
        pipeline = self._make_pipeline(MODEL_FAMILY_GEMMA)
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

        with (
            patch(
                "models.llm_runner.run_chat_generation",
                return_value=("", 0, -1.0, 0.0, None, None),
            ),
            patch(
                "models.llm_runner.run_generation",
                return_value=("hello", 2, 0.1, 0.2),
            ) as mock_generation,
        ):
            pipeline._run_llm(messages)

        assert mock_generation.call_args.kwargs["stop"] == GEMMA_STOP_SEQUENCES

    def test_run_llm_preserves_custom_stop_sequences(self) -> None:
        mm = MagicMock()
        mm.llm = MagicMock()
        mm.llm_model_family = MODEL_FAMILY_GEMMA
        custom_stops = ["[END]", "<|endoftext|>"]
        pipeline = ConversationPipeline(mm, PipelineConfig(llm_stop_sequences=custom_stops))

        with patch(
            "models.llm_runner.run_generation",
            return_value=("hello", 2, 0.1, 0.2),
        ) as mock_generation:
            pipeline._run_llm("hi")

        assert mock_generation.call_args.kwargs["stop"] == custom_stops
