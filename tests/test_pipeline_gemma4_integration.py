"""Pipeline-level tests for Gemma 4 backend routing and marker checks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from core.llm_backend_config import LLMBackendConfig

_LLM_ENV_KEYS = (
    "MUNGI_LLM_BACKEND",
    "MUNGI_LLM_MODEL_PATH",
    "MUNGI_LLM_FALLBACK_MODEL_PATH",
    "MUNGI_LLM_N_CTX",
    "MUNGI_LLM_MAX_TOKENS",
    "MUNGI_LLM_TEMPERATURE",
    "MUNGI_LLM_N_GPU_LAYERS",
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove LLM env vars that affect backend config resolution."""
    for key in _LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _backend_config(backend: str) -> LLMBackendConfig:
    """Build a backend config for pipeline tests."""
    if backend == "gemma4_text":
        return LLMBackendConfig(
            backend="gemma4_text",
            model_path="/models/gemma.gguf",
            n_ctx=2048,
            max_tokens=64,
            temperature=0.4,
            n_gpu_layers=99,
        )
    return LLMBackendConfig(
        backend="qwen3_legacy",
        model_path="/models/qwen.gguf",
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )


def _make_pipeline_for_backend(backend: str) -> Any:
    """Construct a ConversationPipeline with LLMBackendConfig.load patched."""
    from core.pipeline import ConversationPipeline, PipelineConfig

    mm = MagicMock()
    config = _backend_config(backend)
    with patch("core.pipeline.LLMBackendConfig.load", return_value=config):
        return ConversationPipeline(mm, PipelineConfig())


def test_pipeline_env_gemma4_routes_to_manager_fallback_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4 backend loads through the manager-owned fallback loader."""
    from core.model_manager import ModelType
    from models.llm_runner import DEFAULT_GEMMA4_FALLBACK_MODEL_PATH

    monkeypatch.setenv("MUNGI_LLM_BACKEND", "gemma4_text")
    pipeline = _make_pipeline_for_backend("gemma4_text")
    sentinel_llm = object()
    load_result = SimpleNamespace(
        model=sentinel_llm,
        model_path_actual="/models/gemma.gguf",
        fallback_used=False,
        fallback_reason=None,
    )

    def load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        pipeline._mm.llm = sentinel_llm
        pipeline._mm._current_gpu_model = ModelType.LLM
        return load_result

    pipeline._mm.load_gemma_with_fallback.side_effect = load_gemma_with_fallback

    pipeline._load_llm_for_active_backend()

    pipeline._mm.load.assert_not_called()
    pipeline._mm.load_gemma_with_fallback.assert_called_once_with(
        "/models/gemma.gguf",
        DEFAULT_GEMMA4_FALLBACK_MODEL_PATH,
        n_gpu_layers=99,
        n_ctx=2048,
    )
    assert pipeline._mm.llm is sentinel_llm
    assert pipeline._mm._current_gpu_model == ModelType.LLM


def test_pipeline_copies_gemma_fallback_result_to_turn_metrics() -> None:
    """Gemma model fallback telemetry is copied into the turn metrics."""
    pipeline = _make_pipeline_for_backend("gemma4_text")
    sentinel_llm = object()
    load_result = SimpleNamespace(
        model=sentinel_llm,
        model_path_actual="/models/gemma-e2b.gguf",
        fallback_used=True,
        fallback_reason="primary missing",
    )

    def load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        pipeline._mm.llm = sentinel_llm
        return load_result

    pipeline._mm.load_gemma_with_fallback.side_effect = load_gemma_with_fallback

    with (
        patch("models.llm_runner.prepare_system_state_snapshot", return_value=None),
        patch.object(pipeline, "_run_llm", return_value=("reply", 5, 0.1)),
        patch.object(pipeline, "_run_tts", return_value=([0.0], 24000)),
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("hello")

    assert result.success is True
    assert result.metrics.llm_model_fallback_used is True
    assert result.metrics.llm_model_path_actual == "/models/gemma-e2b.gguf"
    assert result.metrics.llm_model_fallback_reason == "primary missing"


def test_bad_output_does_not_trigger_model_fallback() -> None:
    """Gemma bad-output fallback keeps model-path fallback load-only."""
    from models.llm_runner import SAFE_FALLBACK

    pipeline = _make_pipeline_for_backend("gemma4_text")
    sentinel_llm = object()
    load_result = SimpleNamespace(
        model=sentinel_llm,
        model_path_actual="/models/gemma-e4b.gguf",
        fallback_used=False,
        fallback_reason=None,
    )

    def load_gemma_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        pipeline._mm.llm = sentinel_llm
        return load_result

    pipeline._mm.load_gemma_with_fallback.side_effect = load_gemma_with_fallback

    with (
        patch("models.llm_runner.prepare_system_state_snapshot", return_value=None),
        patch(
            "models.llm_runner.run_chat_generation",
            return_value=("unsafe <end_of_turn> leak", 5, 0.2, 0.3, None, None),
        ),
        patch.object(pipeline, "_run_tts", return_value=([0.0], 24000)),
        patch.object(pipeline, "_play_audio_out"),
    ):
        result = pipeline.run_text_turn("안녕")

    assert result.response_text == SAFE_FALLBACK
    assert result.metrics.llm_model_fallback_used is False
    assert result.metrics.llm_model_path_actual == "/models/gemma-e4b.gguf"
    pipeline._mm.load_gemma_with_fallback.assert_called_once()


def test_pipeline_explicit_qwen3_setup_uses_legacy_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `qwen3_legacy` backend delegates loading to ModelManager.load."""
    from core.model_manager import ModelType

    monkeypatch.delenv("MUNGI_LLM_BACKEND", raising=False)
    pipeline = _make_pipeline_for_backend("qwen3_legacy")

    pipeline._load_llm_for_active_backend()

    pipeline._mm.load.assert_called_once_with(ModelType.LLM)
    pipeline._mm._do_load.assert_not_called()


def test_pipeline_explicit_qwen3_env_uses_legacy_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit qwen3_legacy env setting stays on the legacy model-manager path."""
    from core.model_manager import ModelType

    monkeypatch.setenv("MUNGI_LLM_BACKEND", "qwen3_legacy")
    pipeline = _make_pipeline_for_backend("qwen3_legacy")

    pipeline._load_llm_for_active_backend()

    pipeline._mm.load.assert_called_once_with(ModelType.LLM)


def test_qwen3_backend_syncs_explicit_runtime_config_to_manager() -> None:
    """Explicit qwen runtime config is copied into ManagerConfig before legacy load."""
    from core.model_manager import ModelType
    from core.pipeline import ConversationPipeline, PipelineConfig

    manager_config = SimpleNamespace(
        llm_model_path=None,
        llm_n_ctx=0,
        llm_n_gpu_layers=-1,
    )
    mm = MagicMock()
    mm.config = manager_config
    backend_config = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path="/models/qwen-custom.gguf",
        n_ctx=3072,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=24,
        n_ctx_explicit=True,
        n_gpu_layers_explicit=True,
    )

    with patch("core.pipeline.LLMBackendConfig.load", return_value=backend_config):
        pipeline = ConversationPipeline(mm, PipelineConfig())

    pipeline._load_llm_for_active_backend()

    assert manager_config.llm_model_path == "/models/qwen-custom.gguf"
    assert manager_config.llm_n_ctx == 3072
    assert manager_config.llm_n_gpu_layers == 24
    mm.load.assert_called_once_with(ModelType.LLM)


@pytest.mark.parametrize("source", ["env", "config"])
def test_qwen3_backend_applies_explicit_loaded_runtime_config_to_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Explicit env/config qwen runtime values are copied before legacy load."""
    from core.model_manager import ModelType
    from core.pipeline import ConversationPipeline, PipelineConfig

    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    if source == "env":
        config_path = tmp_path / "missing.json"
        monkeypatch.setenv("MUNGI_LLM_BACKEND", "qwen3_legacy")
        monkeypatch.setenv("MUNGI_LLM_MODEL_PATH", "/models/env-qwen.gguf")
        monkeypatch.setenv("MUNGI_LLM_N_CTX", "3584")
        monkeypatch.setenv("MUNGI_LLM_N_GPU_LAYERS", "12")
    else:
        config_path.write_text(
            json.dumps(
                {
                    "llm_backend": "qwen3_legacy",
                    "llm_model_path": "/models/config-qwen.gguf",
                    "llm_n_ctx": 3584,
                    "llm_n_gpu_layers": 12,
                }
            ),
            encoding="utf-8",
        )

    backend_config = LLMBackendConfig.load(config_path)
    manager_config = SimpleNamespace(
        llm_model_path=None,
        llm_n_ctx=0,
        llm_n_gpu_layers=-1,
    )
    mm = MagicMock()
    mm.config = manager_config

    with patch("core.pipeline.LLMBackendConfig.load", return_value=backend_config):
        pipeline = ConversationPipeline(mm, PipelineConfig())

    pipeline._load_llm_for_active_backend()

    assert manager_config.llm_model_path == f"/models/{source}-qwen.gguf"
    assert manager_config.llm_n_ctx == 3584
    assert manager_config.llm_n_gpu_layers == 12
    assert backend_config.n_ctx_explicit is True
    assert backend_config.n_gpu_layers_explicit is True
    mm.load.assert_called_once_with(ModelType.LLM)


def test_qwen3_backend_preserves_manager_values_when_runtime_config_implicit() -> None:
    """Implicit qwen runtime config does not clobber existing ManagerConfig values."""
    from core.model_manager import ModelType
    from core.pipeline import ConversationPipeline, PipelineConfig

    manager_config = SimpleNamespace(
        llm_model_path="/models/cli-qwen.gguf",
        llm_n_ctx=2048,
        llm_n_gpu_layers=-1,
    )
    mm = MagicMock()
    mm.config = manager_config
    backend_config = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path=None,
        n_ctx=4096,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )

    with patch("core.pipeline.LLMBackendConfig.load", return_value=backend_config):
        pipeline = ConversationPipeline(mm, PipelineConfig())

    pipeline._load_llm_for_active_backend()

    assert manager_config.llm_model_path == "/models/cli-qwen.gguf"
    assert manager_config.llm_n_ctx == 2048
    assert manager_config.llm_n_gpu_layers == -1
    mm.load.assert_called_once_with(ModelType.LLM)


def test_pipeline_gemma4_resident_llm_skips_reload() -> None:
    """Resident Gemma 4 LLMs are reused when the manager already has one loaded."""
    pipeline = _make_pipeline_for_backend("gemma4_text")
    pipeline._mm.config.llm_resident = True
    pipeline._mm.llm = object()

    pipeline._load_llm_for_active_backend()

    pipeline._mm._do_load.assert_not_called()
    pipeline._mm.load.assert_not_called()


@pytest.mark.parametrize(
    "marker",
    [
        "<|channel>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "</|think|>",
        "<start_of_turn>",
        "<end_of_turn>",
    ],
)
def test_assert_no_gemma4_marker_leak_raises_for_template_markers(marker: str) -> None:
    """Every known Gemma 4 chat-template marker is rejected."""
    from core.pipeline import Gemma4MarkerLeakError, _assert_no_gemma4_marker_leak

    response = f"safe prefix {marker} unsafe suffix"
    with pytest.raises(Gemma4MarkerLeakError) as exc_info:
        _assert_no_gemma4_marker_leak(response)

    assert exc_info.value.marker == marker


def test_assert_no_gemma4_marker_leak_allows_clean_text() -> None:
    """Clean responses pass through the marker-leak guard."""
    from core.pipeline import _assert_no_gemma4_marker_leak

    _assert_no_gemma4_marker_leak("clean child-safe response")


def test_marker_leak_error_excerpt_is_truncated_to_200_chars() -> None:
    """Marker-leak exceptions store a bounded 200-character response excerpt."""
    from core.pipeline import Gemma4MarkerLeakError, _assert_no_gemma4_marker_leak

    response = "<start_of_turn>" + ("x" * 250)
    with pytest.raises(Gemma4MarkerLeakError) as exc_info:
        _assert_no_gemma4_marker_leak(response)

    assert exc_info.value.marker == "<start_of_turn>"
    assert exc_info.value.response_excerpt.startswith("<start_of_turn>")
    assert len(exc_info.value.response_excerpt) <= 200


def test_marker_leak_check_skipped_for_qwen3_backend() -> None:
    """Qwen3 legacy backend does not apply Gemma marker leak enforcement."""
    pipeline = _make_pipeline_for_backend("qwen3_legacy")
    pipeline._current_language = "en"
    pipeline._mm.llm = object()
    pipeline._mm.llm_model_family = "qwen"
    marker_text = "safe <start_of_turn> marker"

    with patch(
        "models.llm_runner.run_chat_generation",
        return_value=(marker_text, 3, 0.1, 0.2, None, None),
    ):
        text, token_count, ttft = pipeline._run_llm([{"role": "user", "content": "hi"}])

    assert text == marker_text
    assert token_count == 3
    assert ttft == 0.1


def test_marker_leak_check_replaces_gemma4_response_with_fallback() -> None:
    """Gemma 4 backend turns leaked template markers into the safe fallback."""
    from models.llm_runner import SAFE_FALLBACK

    pipeline = _make_pipeline_for_backend("gemma4_text")
    pipeline._current_language = "en"
    pipeline._mm.llm = object()
    pipeline._mm.llm_model_family = "gemma"

    with patch(
        "models.llm_runner.run_chat_generation",
        return_value=("unsafe <end_of_turn> leak", 5, 0.2, 0.3, None, None),
    ):
        text, token_count, ttft = pipeline._run_llm([{"role": "user", "content": "hi"}])

    assert text == SAFE_FALLBACK
    assert token_count == 5
    assert ttft == 0.2


def test_gemma4_backend_applies_generation_config() -> None:
    """Gemma 4 backend config updates pipeline max token and temperature defaults."""
    pipeline = _make_pipeline_for_backend("gemma4_text")

    assert pipeline._config.llm_max_tokens == 64
    assert pipeline._config.llm_temperature == 0.4


def test_qwen3_backend_does_not_load_gemma4_persona_prompt() -> None:
    """Legacy backend keeps the Gemma 4 persona prompt inactive."""
    pipeline = _make_pipeline_for_backend("qwen3_legacy")

    assert pipeline._gemma4_persona_prompt is None


def test_build_gemma4_system_prompt_appends_persona_to_base_prompt(tmp_path: Path) -> None:
    """Gemma 4 persona markdown is appended without replacing safety rules."""
    from core.pipeline import _build_gemma4_system_prompt

    persona_path = tmp_path / "persona.md"
    persona_prompt = "# Persona\n\nverbatim body\n"
    persona_path.write_text(persona_prompt, encoding="utf-8")

    combined = _build_gemma4_system_prompt("SAFETY RULES\nANTI-ECHO", persona_path)

    assert combined == "SAFETY RULES\nANTI-ECHO\n\n---\n\n# Persona\n\nverbatim body\n"


@pytest.mark.parametrize(
    ("user_text", "detected_language", "expected_prompt"),
    [
        ("\uc548\ub155 \ubb49\uc774", "ko", "gemma"),
        ("Hello, can you speak English?", "en", "english"),
    ],
)
def test_build_messages_selects_bilingual_system_prompt_for_gemma_backend(
    user_text: str,
    detected_language: str,
    expected_prompt: str,
) -> None:
    """Gemma 4 backend selects English prompts for English bilingual turns."""
    pipeline = _make_pipeline_for_backend("gemma4_text")
    base_prompt = pipeline._config.llm_system_prompt
    pipeline._gemma4_persona_prompt = f"{base_prompt}\n\n---\n\ngemma persona"
    pipeline._en_system_prompt = "ENGLISH PROMPT"

    messages = cast(
        list[dict[str, str]],
        pipeline._build_messages(user_text, detected_language=detected_language),
    )

    assert messages[0]["role"] == "system"
    selected_prompt = messages[0]["content"]
    if expected_prompt == "english":
        assert selected_prompt == "ENGLISH PROMPT"
        assert "gemma persona" not in selected_prompt
    else:
        assert selected_prompt.startswith(base_prompt)
        assert "\n\n---\n\ngemma persona" in selected_prompt
        assert "SAFETY RULES" in selected_prompt
