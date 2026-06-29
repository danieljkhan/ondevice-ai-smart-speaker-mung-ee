"""Tests for the Gemma 4 default preflight script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.llm_backend_config import LLMBackendConfig


def test_preflight_uses_gemma_primary_default_when_model_path_absent(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Config-absent preflight resolves the Gemma primary default path."""
    from models import llm_runner
    from scripts import preflight_gemma4_default

    default_model_path = tmp_path / "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
    default_model_path.write_bytes(b"fake gguf")
    loaded_paths: list[str] = []

    cfg = LLMBackendConfig(
        backend="gemma4_text",
        model_path=None,
        n_ctx=4096,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )

    def fake_load(cls: type[LLMBackendConfig], config_path: Path | None = None) -> LLMBackendConfig:
        return cfg

    def fake_load_gemma4_text_llm(
        model_path: str,
        *,
        n_gpu_layers: int,
        n_ctx: int,
    ) -> object:
        loaded_paths.append(model_path)
        assert n_gpu_layers == 99
        assert n_ctx == 2048
        return object()

    monkeypatch.setattr(LLMBackendConfig, "load", classmethod(fake_load))
    monkeypatch.setattr(llm_runner, "DEFAULT_GEMMA4_TEXT_MODEL_PATH", str(default_model_path))
    monkeypatch.setattr(preflight_gemma4_default, "_read_gguf_architecture", lambda path: "gemma4")
    monkeypatch.setattr(llm_runner, "load_gemma4_text_llm", fake_load_gemma4_text_llm)

    assert preflight_gemma4_default.main() == 0
    assert loaded_paths == [str(default_model_path)]
