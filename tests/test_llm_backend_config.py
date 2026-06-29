"""Tests for LLM backend runtime configuration resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.llm_backend_config import DEFAULT_N_CTX, LLMBackendConfig
from models.llm_runner import (
    DEFAULT_GEMMA4_FALLBACK_MODEL_PATH,
    DEFAULT_GEMMA4_TEXT_MODEL_PATH,
)

_ENV_KEYS = (
    "MUNGI_LLM_BACKEND",
    "MUNGI_LLM_MODEL_PATH",
    "MUNGI_LLM_FALLBACK_MODEL_PATH",
    "MUNGI_LLM_N_CTX",
    "MUNGI_LLM_MAX_TOKENS",
    "MUNGI_LLM_TEMPERATURE",
    "MUNGI_LLM_N_GPU_LAYERS",
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LLM backend env vars that affect config resolution."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write_config(path: Path, payload: dict[str, object]) -> None:
    """Write a mock config.json payload."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_defaults_select_gemma4_text_backend() -> None:
    """Defaults select the Gemma 4 text backend and preserve Phase 1 numeric defaults."""
    cfg = LLMBackendConfig.defaults()

    assert cfg.backend == "gemma4_text"
    assert cfg.model_path is None
    assert cfg.fallback_model_path == DEFAULT_GEMMA4_FALLBACK_MODEL_PATH
    assert cfg.n_ctx == 4096
    assert cfg.max_tokens == 256
    assert cfg.temperature == 0.4
    assert cfg.n_gpu_layers == 99
    assert cfg.n_ctx_explicit is False
    assert cfg.n_gpu_layers_explicit is False


def test_default_n_ctx_is_4096() -> None:
    """The Gemma 4 default context window should fit the production prompt."""
    assert DEFAULT_N_CTX == 4096


def test_load_missing_config_uses_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing config file falls back to code defaults without warning."""
    _clear_llm_env(monkeypatch)

    cfg = LLMBackendConfig.load(tmp_path / "missing.json")

    assert cfg == LLMBackendConfig.defaults()
    assert cfg.model_path is None
    assert cfg.fallback_model_path == DEFAULT_GEMMA4_FALLBACK_MODEL_PATH
    assert DEFAULT_GEMMA4_TEXT_MODEL_PATH != cfg.fallback_model_path


def test_config_file_overrides_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present config.json overrides all matching default fields."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "llm_backend": "gemma4_text",
            "llm_model_path": "/models/gemma.gguf",
            "llm_fallback_model_path": "/models/gemma-small.gguf",
            "llm_n_ctx": 4096,
            "llm_max_tokens": 128,
            "llm_temperature": 0.2,
            "llm_n_gpu_layers": 77,
        },
    )

    cfg = LLMBackendConfig.load(config_path)

    assert cfg == LLMBackendConfig(
        backend="gemma4_text",
        model_path="/models/gemma.gguf",
        fallback_model_path="/models/gemma-small.gguf",
        n_ctx=4096,
        max_tokens=128,
        temperature=0.2,
        n_gpu_layers=77,
        n_ctx_explicit=True,
        n_gpu_layers_explicit=True,
    )


def test_env_backend_overrides_config_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend env var wins over config.json for the backend field."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"llm_backend": "qwen3_legacy"})
    monkeypatch.setenv("MUNGI_LLM_BACKEND", "gemma4_text")

    cfg = LLMBackendConfig.load(config_path)

    assert cfg.backend == "gemma4_text"


def test_env_values_override_config_per_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables override config.json per matching field."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "llm_backend": "qwen3_legacy",
            "llm_model_path": "/models/config.gguf",
            "llm_fallback_model_path": "/models/config-fallback.gguf",
            "llm_n_ctx": 2048,
            "llm_max_tokens": 128,
            "llm_temperature": 0.5,
            "llm_n_gpu_layers": 10,
        },
    )
    monkeypatch.setenv("MUNGI_LLM_BACKEND", "gemma4_text")
    monkeypatch.setenv("MUNGI_LLM_MODEL_PATH", "/models/env.gguf")
    monkeypatch.setenv("MUNGI_LLM_FALLBACK_MODEL_PATH", "/models/env-fallback.gguf")
    monkeypatch.setenv("MUNGI_LLM_N_CTX", "3072")
    monkeypatch.setenv("MUNGI_LLM_MAX_TOKENS", "64")
    monkeypatch.setenv("MUNGI_LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("MUNGI_LLM_N_GPU_LAYERS", "99")

    cfg = LLMBackendConfig.load(config_path)

    assert cfg == LLMBackendConfig(
        backend="gemma4_text",
        model_path="/models/env.gguf",
        fallback_model_path="/models/env-fallback.gguf",
        n_ctx=3072,
        max_tokens=64,
        temperature=0.1,
        n_gpu_layers=99,
        n_ctx_explicit=True,
        n_gpu_layers_explicit=True,
    )


def test_empty_env_values_are_treated_as_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank env strings do not mask lower-precedence config values."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "llm_backend": "gemma4_text",
            "llm_model_path": "/cfg.gguf",
            "llm_fallback_model_path": "/cfg-fallback.gguf",
        },
    )
    monkeypatch.setenv("MUNGI_LLM_BACKEND", "  ")
    monkeypatch.setenv("MUNGI_LLM_MODEL_PATH", "")
    monkeypatch.setenv("MUNGI_LLM_FALLBACK_MODEL_PATH", " ")

    cfg = LLMBackendConfig.load(config_path)

    assert cfg.backend == "gemma4_text"
    assert cfg.model_path == "/cfg.gguf"
    assert cfg.fallback_model_path == "/cfg-fallback.gguf"


def test_malformed_json_logs_warning_and_uses_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSON is warned about and does not raise."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text("{not-json", encoding="utf-8")

    with caplog.at_level("WARNING", logger="mungi.core.llm_backend_config"):
        cfg = LLMBackendConfig.load(config_path)

    assert cfg == LLMBackendConfig.defaults()
    assert "Malformed LLM backend config" in caplog.text


def test_non_object_json_logs_warning_and_uses_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JSON array is rejected as malformed config and defaults are used."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text("[]", encoding="utf-8")

    with caplog.at_level("WARNING", logger="mungi.core.llm_backend_config"):
        cfg = LLMBackendConfig.load(config_path)

    assert cfg == LLMBackendConfig.defaults()
    assert "expected JSON object" in caplog.text


def test_invalid_backend_in_config_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An invalid backend name logs and falls back to the default backend."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"llm_backend": "invalid_name"})

    with caplog.at_level("DEBUG", logger="mungi.core.llm_backend_config"):
        cfg = LLMBackendConfig.load(config_path)

    assert cfg.backend == "gemma4_text"
    assert "Skipping invalid LLM backend" in caplog.text


def test_invalid_env_backend_falls_back_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid env backend does not prevent a valid config backend from loading."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"llm_backend": "gemma4_text"})
    monkeypatch.setenv("MUNGI_LLM_BACKEND", "invalid_name")

    cfg = LLMBackendConfig.load(config_path)

    assert cfg.backend == "gemma4_text"


def test_invalid_positive_int_env_falls_back_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-positive env integer checks config.json before using defaults."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"llm_n_ctx": 1024, "llm_max_tokens": 99})
    monkeypatch.setenv("MUNGI_LLM_N_CTX", "0")
    monkeypatch.setenv("MUNGI_LLM_MAX_TOKENS", "-1")

    cfg = LLMBackendConfig.load(config_path)

    assert cfg.n_ctx == 1024
    assert cfg.n_ctx_explicit is True
    assert cfg.max_tokens == 99


def test_invalid_numeric_config_values_fall_back_to_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid numeric config entries are ignored in favor of defaults."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        {
            "llm_n_ctx": "not-int",
            "llm_max_tokens": 0,
            "llm_temperature": -0.1,
            "llm_n_gpu_layers": True,
        },
    )

    cfg = LLMBackendConfig.load(config_path)

    defaults = LLMBackendConfig.defaults()
    assert cfg.n_ctx == defaults.n_ctx
    assert cfg.max_tokens == defaults.max_tokens
    assert cfg.temperature == defaults.temperature
    assert cfg.n_gpu_layers == defaults.n_gpu_layers
    assert cfg.n_ctx_explicit is False
    assert cfg.n_gpu_layers_explicit is False


@pytest.mark.parametrize(
    ("env_value", "config_value", "expected_n_ctx", "expected_explicit"),
    [
        ("3072", None, 3072, True),
        (None, 2048, 2048, True),
        (None, None, DEFAULT_N_CTX, False),
        ("0", None, DEFAULT_N_CTX, False),
        ("-1", None, DEFAULT_N_CTX, False),
        ("not-int", None, DEFAULT_N_CTX, False),
        (None, 0, DEFAULT_N_CTX, False),
        (None, -1, DEFAULT_N_CTX, False),
        (None, "not-int", DEFAULT_N_CTX, False),
    ],
)
def test_n_ctx_explicit_tracks_valid_positive_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    config_value: object,
    expected_n_ctx: int,
    expected_explicit: bool,
) -> None:
    """n_ctx explicit provenance is set only for valid positive env/config values."""
    _clear_llm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    payload: dict[str, object] = {}
    if config_value is not None:
        payload["llm_n_ctx"] = config_value
    _write_config(config_path, payload)
    if env_value is not None:
        monkeypatch.setenv("MUNGI_LLM_N_CTX", env_value)

    cfg = LLMBackendConfig.load(config_path)

    assert cfg.n_ctx == expected_n_ctx
    assert cfg.n_ctx_explicit is expected_explicit
