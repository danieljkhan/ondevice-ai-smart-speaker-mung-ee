"""Runtime configuration for selecting the active LLM backend."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Literal

from models.llm_runner import DEFAULT_GEMMA4_FALLBACK_MODEL_PATH

logger = logging.getLogger("mungi.core.llm_backend_config")

LLMBackendName = Literal["qwen3_legacy", "gemma4_text"]
DEFAULT_BACKEND: LLMBackendName = "gemma4_text"
DEFAULT_CONFIG_PATH = Path("/var/lib/mungi/config/config.json")
DEFAULT_N_CTX: int = 4096
DEFAULT_MAX_TOKENS: int = 256
DEFAULT_TEMPERATURE: float = 0.4
DEFAULT_N_GPU_LAYERS: int = 99
_VALID_BACKENDS: frozenset[str] = frozenset({"qwen3_legacy", "gemma4_text"})


@dataclasses.dataclass(frozen=True)
class LLMBackendConfig:
    """Resolved runtime configuration for the LLM backend."""

    backend: LLMBackendName
    model_path: str | None
    n_ctx: int
    max_tokens: int
    temperature: float
    n_gpu_layers: int
    fallback_model_path: str | None = DEFAULT_GEMMA4_FALLBACK_MODEL_PATH
    n_ctx_explicit: bool = False
    n_gpu_layers_explicit: bool = False

    @classmethod
    def defaults(cls) -> LLMBackendConfig:
        """Return code-level defaults with the Gemma 4 text backend selected (ADR 0073)."""
        return cls(
            backend=DEFAULT_BACKEND,
            model_path=None,
            n_ctx=DEFAULT_N_CTX,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            n_gpu_layers=DEFAULT_N_GPU_LAYERS,
            fallback_model_path=DEFAULT_GEMMA4_FALLBACK_MODEL_PATH,
        )

    @classmethod
    def load(cls, config_path: Path | None = None) -> LLMBackendConfig:
        """Load config with precedence: env > config.json > code defaults.

        Env variable keys:
            MUNGI_LLM_BACKEND ("qwen3_legacy" | "gemma4_text")
            MUNGI_LLM_MODEL_PATH
            MUNGI_LLM_FALLBACK_MODEL_PATH
            MUNGI_LLM_N_CTX
            MUNGI_LLM_MAX_TOKENS
            MUNGI_LLM_TEMPERATURE
            MUNGI_LLM_N_GPU_LAYERS

        Config.json schema (optional file, safe missing):
            {
                "llm_backend": "...",
                "llm_model_path": "...",
                "llm_fallback_model_path": "...",
                "llm_n_ctx": 2048,
                "llm_max_tokens": 256,
                "llm_temperature": 0.4,
                "llm_n_gpu_layers": 99,
            }

        Precedence order per returned field:
            env (if set) > config.json (if file exists and key present) > defaults.
        """
        defaults = cls.defaults()
        path = config_path or DEFAULT_CONFIG_PATH
        config_values = _load_config_values(path)

        n_gpu_layers, n_gpu_layers_explicit = _resolve_int_field_with_explicit(
            env_key="MUNGI_LLM_N_GPU_LAYERS",
            config_key="llm_n_gpu_layers",
            config_values=config_values,
            default=defaults.n_gpu_layers,
        )
        n_ctx, n_ctx_explicit = _resolve_positive_int_field_with_explicit(
            env_key="MUNGI_LLM_N_CTX",
            config_key="llm_n_ctx",
            config_values=config_values,
            default=defaults.n_ctx,
        )

        return cls(
            backend=_resolve_backend_field(config_values, defaults.backend),
            model_path=_resolve_string_field(
                env_key="MUNGI_LLM_MODEL_PATH",
                config_key="llm_model_path",
                config_values=config_values,
                default=defaults.model_path,
            ),
            fallback_model_path=_resolve_string_field(
                env_key="MUNGI_LLM_FALLBACK_MODEL_PATH",
                config_key="llm_fallback_model_path",
                config_values=config_values,
                default=defaults.fallback_model_path,
            ),
            n_ctx=n_ctx,
            max_tokens=_resolve_positive_int_field(
                env_key="MUNGI_LLM_MAX_TOKENS",
                config_key="llm_max_tokens",
                config_values=config_values,
                default=defaults.max_tokens,
            ),
            temperature=_resolve_nonnegative_float_field(
                env_key="MUNGI_LLM_TEMPERATURE",
                config_key="llm_temperature",
                config_values=config_values,
                default=defaults.temperature,
            ),
            n_gpu_layers=n_gpu_layers,
            n_ctx_explicit=n_ctx_explicit,
            n_gpu_layers_explicit=n_gpu_layers_explicit,
        )


def _load_config_values(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}

    try:
        raw_config: object = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Malformed LLM backend config at %s: %s", config_path, exc)
        return {}
    except OSError as exc:
        logger.warning("Failed to read LLM backend config at %s: %s", config_path, exc)
        return {}

    if not isinstance(raw_config, dict):
        logger.warning("Malformed LLM backend config at %s: expected JSON object", config_path)
        return {}

    return {key: value for key, value in raw_config.items() if isinstance(key, str)}


def _read_env_value(env_key: str) -> str | None:
    raw = os.getenv(env_key)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _parse_backend(value: object, *, source: str) -> LLMBackendName | None:
    if not isinstance(value, str):
        logger.debug("Skipping non-string LLM backend from %s: %r", source, value)
        return None

    normalized = value.strip()
    if normalized not in _VALID_BACKENDS:
        logger.debug("Skipping invalid LLM backend from %s: %r", source, value)
        return None

    return "gemma4_text" if normalized == "gemma4_text" else "qwen3_legacy"


def _resolve_backend_field(
    config_values: dict[str, object],
    default: LLMBackendName,
) -> LLMBackendName:
    env_backend = _read_env_value("MUNGI_LLM_BACKEND")
    if env_backend is not None:
        parsed_env = _parse_backend(env_backend, source="MUNGI_LLM_BACKEND")
        if parsed_env is not None:
            return parsed_env

    config_backend = config_values.get("llm_backend")
    if config_backend is not None:
        parsed_config = _parse_backend(config_backend, source="config.json llm_backend")
        if parsed_config is not None:
            return parsed_config

    return default


def _parse_string(value: object, *, source: str) -> str | None:
    if not isinstance(value, str):
        logger.debug("Skipping non-string LLM config value from %s: %r", source, value)
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_string_field(
    *,
    env_key: str,
    config_key: str,
    config_values: dict[str, object],
    default: str | None,
) -> str | None:
    env_value = _read_env_value(env_key)
    if env_value is not None:
        return env_value

    if config_key in config_values:
        parsed_config = _parse_string(
            config_values[config_key],
            source=f"config.json {config_key}",
        )
        if parsed_config is not None:
            return parsed_config

    return default


def _parse_int(value: object, *, source: str) -> int | None:
    if isinstance(value, bool):
        logger.warning("Skipping boolean LLM integer value from %s: %r", source, value)
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            logger.warning("Skipping invalid LLM integer value from %s: %r", source, value)
            return None

    logger.warning("Skipping non-integer LLM config value from %s: %r", source, value)
    return None


def _resolve_int_field(
    *,
    env_key: str,
    config_key: str,
    config_values: dict[str, object],
    default: int,
) -> int:
    value, _ = _resolve_int_field_with_explicit(
        env_key=env_key,
        config_key=config_key,
        config_values=config_values,
        default=default,
    )
    return value


def _resolve_int_field_with_explicit(
    *,
    env_key: str,
    config_key: str,
    config_values: dict[str, object],
    default: int,
) -> tuple[int, bool]:
    env_value = _read_env_value(env_key)
    if env_value is not None:
        parsed_env = _parse_int(env_value, source=env_key)
        if parsed_env is not None:
            return parsed_env, True

    if config_key in config_values:
        parsed_config = _parse_int(
            config_values[config_key],
            source=f"config.json {config_key}",
        )
        if parsed_config is not None:
            return parsed_config, True

    return default, False


def _resolve_positive_int_field(
    *,
    env_key: str,
    config_key: str,
    config_values: dict[str, object],
    default: int,
) -> int:
    value, _ = _resolve_positive_int_field_with_explicit(
        env_key=env_key,
        config_key=config_key,
        config_values=config_values,
        default=default,
    )
    return value


def _resolve_positive_int_field_with_explicit(
    *,
    env_key: str,
    config_key: str,
    config_values: dict[str, object],
    default: int,
) -> tuple[int, bool]:
    env_value = _read_env_value(env_key)
    if env_value is not None:
        parsed_env = _parse_int(env_value, source=env_key)
        if parsed_env is not None:
            if parsed_env > 0:
                return parsed_env, True
            logger.warning(
                "%s resolved to %d; checking lower-precedence sources", env_key, parsed_env
            )

    if config_key in config_values:
        parsed_config = _parse_int(
            config_values[config_key],
            source=f"config.json {config_key}",
        )
        if parsed_config is not None:
            if parsed_config > 0:
                return parsed_config, True
            logger.warning(
                "%s resolved to %d; using default %d",
                config_key,
                parsed_config,
                default,
            )

    return default, False


def _parse_float(value: object, *, source: str) -> float | None:
    if isinstance(value, bool):
        logger.warning("Skipping boolean LLM float value from %s: %r", source, value)
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            logger.warning("Skipping invalid LLM float value from %s: %r", source, value)
            return None

    logger.warning("Skipping non-float LLM config value from %s: %r", source, value)
    return None


def _resolve_nonnegative_float_field(
    *,
    env_key: str,
    config_key: str,
    config_values: dict[str, object],
    default: float,
) -> float:
    env_value = _read_env_value(env_key)
    if env_value is not None:
        parsed_env = _parse_float(env_value, source=env_key)
        if parsed_env is not None:
            if parsed_env >= 0.0:
                return parsed_env
            logger.warning(
                "%s resolved to %.3f; checking lower-precedence sources",
                env_key,
                parsed_env,
            )

    if config_key in config_values:
        parsed_config = _parse_float(
            config_values[config_key],
            source=f"config.json {config_key}",
        )
        if parsed_config is not None:
            if parsed_config >= 0.0:
                return parsed_config
            logger.warning(
                "%s resolved to %.3f; using default %.3f",
                config_key,
                parsed_config,
                default,
            )

    return default
