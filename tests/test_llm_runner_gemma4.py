"""Tests for Gemma 4 text LLM loader dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

import models.llm_runner as llm_runner
from core.llm_backend_config import LLMBackendConfig


class RecordingLlama:
    """Fake llama_cpp.Llama that records constructor kwargs."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


def _install_fake_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake llama_cpp module at the import site."""
    fake_module = ModuleType("llama_cpp")
    fake_module.Llama = RecordingLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    RecordingLlama.calls.clear()


def _write_model_file(tmp_path: Path, name: str = "model.gguf") -> Path:
    """Create a fake GGUF path that satisfies loader existence checks."""
    model_path = tmp_path / name
    model_path.write_text("", encoding="utf-8")
    return model_path


def test_load_gemma4_text_llm_uses_gemma_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4 loader passes the expected llama.cpp constructor kwargs."""
    _install_fake_llama(monkeypatch)
    monkeypatch.setattr(llm_runner, "_patch_llama_cpp_kv_cache", lambda: None)
    monkeypatch.setattr(llm_runner, "_resolve_kv_type", lambda: None)
    model_path = _write_model_file(tmp_path, "gemma.gguf")

    llm = llm_runner.load_gemma4_text_llm(str(model_path), n_gpu_layers=99, n_ctx=2048)

    assert isinstance(llm, RecordingLlama)
    assert RecordingLlama.calls == [
        {
            "model_path": str(model_path),
            "n_gpu_layers": 99,
            "n_ctx": 2048,
            "chat_format": None,
            "flash_attn": True,
            "verbose": False,
        }
    ]


def test_load_gemma4_text_llm_applies_opt_in_kv_cache_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4 loader forwards resolved KV cache quantization when configured."""
    _install_fake_llama(monkeypatch)
    monkeypatch.setattr(llm_runner, "_patch_llama_cpp_kv_cache", lambda: None)
    monkeypatch.setattr(llm_runner, "_resolve_kv_type", lambda: 8)
    model_path = _write_model_file(tmp_path, "gemma-q8.gguf")

    llm_runner.load_gemma4_text_llm(str(model_path), n_gpu_layers=50, n_ctx=1024)

    assert RecordingLlama.calls[0]["type_k"] == 8
    assert RecordingLlama.calls[0]["type_v"] == 8


def test_load_gemma4_text_llm_missing_file_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4 loader rejects missing model paths before constructing Llama."""
    _install_fake_llama(monkeypatch)
    monkeypatch.setattr(llm_runner, "_patch_llama_cpp_kv_cache", lambda: None)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        llm_runner.load_gemma4_text_llm(str(tmp_path / "missing.gguf"))

    assert RecordingLlama.calls == []


def test_build_llm_from_config_routes_gemma4_backend() -> None:
    """Dispatcher routes gemma4_text configs to load_gemma4_text_llm."""
    sentinel = object()
    config = LLMBackendConfig(
        backend="gemma4_text",
        model_path="/models/gemma.gguf",
        n_ctx=3072,
        max_tokens=64,
        temperature=0.3,
        n_gpu_layers=88,
    )

    with patch.object(llm_runner, "load_gemma4_text_llm", return_value=sentinel) as mock_load:
        backend, llm = llm_runner.build_llm_from_config(config)

    assert backend == "gemma4_text"
    assert llm is sentinel
    mock_load.assert_called_once_with("/models/gemma.gguf", n_gpu_layers=88, n_ctx=3072)


def test_build_llm_from_config_routes_qwen3_legacy_backend() -> None:
    """Dispatcher routes qwen3_legacy configs to the legacy loader."""
    sentinel = object()
    config = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path="/models/qwen.gguf",
        n_ctx=2048,
        max_tokens=128,
        temperature=0.7,
        n_gpu_layers=10,
    )

    with patch.object(llm_runner, "load_llm_model", return_value=sentinel) as mock_load:
        backend, llm = llm_runner.build_llm_from_config(config)

    assert backend == "qwen3_legacy"
    assert llm is sentinel
    mock_load.assert_called_once_with("/models/qwen.gguf", n_gpu_layers=10, n_ctx=2048)


def test_build_llm_from_config_uses_default_gemma4_model_path() -> None:
    """A missing Gemma 4 model_path falls back to DEFAULT_GEMMA4_TEXT_MODEL_PATH."""
    config = LLMBackendConfig(
        backend="gemma4_text",
        model_path=None,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )

    with patch.object(llm_runner, "load_gemma4_text_llm", return_value=object()) as mock_load:
        llm_runner.build_llm_from_config(config)

    mock_load.assert_called_once_with(
        llm_runner.DEFAULT_GEMMA4_TEXT_MODEL_PATH,
        n_gpu_layers=99,
        n_ctx=2048,
    )


def test_gemma4_primary_and_fallback_defaults_are_distinct() -> None:
    """The repository contract uses E4B as primary and E2B as load fallback."""
    assert llm_runner.DEFAULT_GEMMA4_TEXT_MODEL_PATH.endswith("/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf")
    assert llm_runner.DEFAULT_GEMMA4_FALLBACK_MODEL_PATH.endswith("/gemma-4-E2B-it-Q5_K_M.gguf")
    assert (
        llm_runner.DEFAULT_GEMMA4_TEXT_MODEL_PATH != llm_runner.DEFAULT_GEMMA4_FALLBACK_MODEL_PATH
    )


def test_build_llm_from_config_uses_default_qwen3_model_path() -> None:
    """Explicit `qwen3_legacy` backend uses default Qwen3 model path."""
    config = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path=None,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )

    with patch.object(llm_runner, "load_llm_model", return_value=object()) as mock_load:
        llm_runner.build_llm_from_config(config)

    mock_load.assert_called_once_with(
        llm_runner.DEFAULT_QWEN3_LEGACY_MODEL_PATH,
        n_gpu_layers=config.n_gpu_layers,
        n_ctx=config.n_ctx,
    )


def test_explicit_qwen3_legacy_config_preserves_parameter_passthrough() -> None:
    """Explicit qwen3_legacy config fields are passed to the legacy loader unchanged."""
    config = LLMBackendConfig(
        backend="qwen3_legacy",
        model_path=None,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )

    with patch.object(llm_runner, "load_llm_model", return_value="legacy") as mock_load:
        backend, llm = llm_runner.build_llm_from_config(config)

    assert backend == "qwen3_legacy"
    assert llm == "legacy"
    assert mock_load.call_args.kwargs == {
        "n_gpu_layers": config.n_gpu_layers,
        "n_ctx": config.n_ctx,
    }


def test_load_gemma4_text_llm_applies_runtime_patch_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4 loader invokes the llama.cpp compatibility patch hook."""
    _install_fake_llama(monkeypatch)
    patch_calls = 0

    def fake_patch() -> None:
        nonlocal patch_calls
        patch_calls += 1

    monkeypatch.setattr(llm_runner, "_patch_llama_cpp_kv_cache", fake_patch)
    monkeypatch.setattr(llm_runner, "_resolve_kv_type", lambda: None)
    model_path = _write_model_file(tmp_path)

    llm_runner.load_gemma4_text_llm(str(model_path))

    assert patch_calls == 1


def test_load_gemma4_text_llm_import_error_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ImportError is surfaced when llama_cpp is unavailable."""
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    model_path = _write_model_file(tmp_path)

    with patch.dict(sys.modules, {"llama_cpp": None}):
        with pytest.raises(ImportError):
            llm_runner.load_gemma4_text_llm(str(model_path))
