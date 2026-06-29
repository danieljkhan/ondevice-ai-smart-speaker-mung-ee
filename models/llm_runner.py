"""Qwen LLM (llama.cpp) model loader and generation runner.

Provides functions to discover, load, and run text generation with
GGUF models via llama-cpp-python. Extracted from ``scripts/test_llm.py``
to establish correct dependency direction (``core/`` → ``models/``).
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from core.llm_backend_config import LLMBackendConfig

logger = logging.getLogger("mungi.models.llm_runner")

# ---------------------------------------------------------------------------
# llama-cpp-python KV cache compatibility patch
# ---------------------------------------------------------------------------
# llama-cpp-python 0.3.14 passes seq_id=-1 (meaning "all sequences")
# to llama_memory_seq_rm(), but the bundled llama.cpp unified KV cache
# asserts seq_id >= 0.  Upstream 0.3.16 fixes this by clamping
# seq_id < 0 to 0 in _internals.py.  This monkey-patch replicates
# that upstream fix.  Safe to apply multiple times.
#
# Refs:
#   https://github.com/ggml-org/llama.cpp/issues/14847
#   https://github.com/ggml-org/llama.cpp/issues/15215
# ---------------------------------------------------------------------------

_LLAMA_CPP_PATCHED: bool = False


def _patch_llama_cpp_kv_cache() -> None:
    """Patch llama-cpp-python 0.3.14 to clamp seq_id=-1 to 0.

    Matches the upstream fix in llama-cpp-python 0.3.16 where
    ``_internals.py`` clamps ``seq_id < 0`` to ``0``.

    This is idempotent -- calling it multiple times is safe.
    """
    global _LLAMA_CPP_PATCHED  # noqa: PLW0603
    if _LLAMA_CPP_PATCHED:
        return

    try:
        import llama_cpp._internals as internals
    except ImportError:
        return

    if not hasattr(internals.LlamaContext, "kv_cache_seq_rm"):
        return

    original_seq_rm = internals.LlamaContext.kv_cache_seq_rm

    def _patched_seq_rm(self: Any, seq_id: int, p0: int, p1: int) -> None:
        """Clamp seq_id < 0 to 0 (upstream 0.3.16 compatibility)."""
        if seq_id < 0:
            seq_id = 0
        original_seq_rm(self, seq_id, p0, p1)

    internals.LlamaContext.kv_cache_seq_rm = _patched_seq_rm  # type: ignore[method-assign]

    if hasattr(internals, "LlamaModel") and hasattr(internals.LlamaModel, "close"):
        original_close = internals.LlamaModel.close

        def _patched_close(self: Any) -> None:
            """Guard partially constructed models missing ``sampler``."""
            if not hasattr(self, "sampler"):
                self.sampler = None
            original_close(self)

        internals.LlamaModel.close = _patched_close  # type: ignore[method-assign]

    _LLAMA_CPP_PATCHED = True
    logger.info("Applied llama-cpp-python runtime compatibility patches")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR: str = "/opt/mungi/ai_models"
DEFAULT_QWEN3_LEGACY_MODEL_PATH: str = f"{DEFAULT_MODEL_DIR}/Qwen3.5-2B-DPO.Q6_K.gguf"
DEFAULT_GEMMA4_TEXT_MODEL_PATH: str = f"{DEFAULT_MODEL_DIR}/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
DEFAULT_GEMMA4_FALLBACK_MODEL_PATH: str = f"{DEFAULT_MODEL_DIR}/gemma-4-E2B-it-Q5_K_M.gguf"
DEFAULT_MAX_TOKENS: int = 64
# Jetson Orin Nano 8GB safe default for legacy load_llm() path.
# Sequential loading (ModelManager.load) uses n_gpu_layers=-1.
DEFAULT_N_GPU_LAYERS: int = 10
DEFAULT_N_CTX: int = 4096
DEFAULT_STOP_SEQUENCES: list[str] = ["<|im_end|>", "<|im_start|>"]
_KV_TYPE_MAP: dict[str, int] = {
    "f16": 1,
    "f32": 0,
    "q8_0": 8,
    "q4_0": 2,
    "q5_0": 6,
}
DEFAULT_KV_TYPE: str = "f16"

# Qwen3-1.7B FT sampling params tuned around the official Qwen3 guidance.
# Fine-tuning carries more of the desired persona behavior, so production can
# use a higher temperature and shorter answers without the previous 4B guardrails.
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.8
DEFAULT_TOP_K: int = 20
DEFAULT_MIN_P: float = 0.0
DEFAULT_PRESENCE_PENALTY: float = 1.2
DEFAULT_REPEAT_PENALTY: float = 1.0
# Qwen3.5 recommended presence_penalty for non-thinking mode.
QWEN35_PRESENCE_PENALTY: float = 1.5

# Qwen3 recommended sampling params for thinking mode (enable_thinking=True)
THINKING_TEMPERATURE: float = 0.6
THINKING_TOP_P: float = 0.95
THINKING_TOP_K: int = 20
THINKING_MIN_P: float = 0.0
THINKING_PRESENCE_PENALTY: float = 1.2

# Gemma 4 stop sequences (used by create_chat_completion internally,
# but needed as explicit stop list for run_generation string-prompt fallback).
GEMMA_STOP_SEQUENCES: list[str] = ["<end_of_turn>", "<start_of_turn>"]

# Model family identifiers
MODEL_FAMILY_QWEN: str = "qwen"
MODEL_FAMILY_GEMMA: str = "gemma"
MODEL_FAMILY_AUTO: str = "auto"

# GGUF files to skip during auto-discovery (stored but incompatible/retired)
_EXCLUDED_GGUF: set[str] = {
    "Qwen3-4B-Q4_K_M.gguf",
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen3.5-2B-Q5_K_M.gguf",
    "qwen3-1.7b.Q4_K_M.gguf",
    "Qwen3-1.7B-Q8_0.gguf",
    "Qwen3-8B-Q4_K_M.gguf",
}

# Regex for Qwen3 <think>...</think> reasoning blocks
_THINK_TAG_RE: re.Pattern[str] = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE: re.Pattern[str] = re.compile(r"<think>.*", re.DOTALL)
# Residual "think" text at start of response after empty prefill
_THINK_RESIDUAL_RE: re.Pattern[str] = re.compile(r"^\s*think\s*", re.IGNORECASE)
_LEADING_INTERJECTION_RE: re.Pattern[str] = re.compile(r"^\s*우[!~,. ]+\s*", re.IGNORECASE)
# Garbage output detection: digit-only or single-char repetition sequences
# Catches KV cache corruption artifacts like "333333..." or "111 222 333..."
_GARBAGE_DIGIT_RE: re.Pattern[str] = re.compile(r"^[\d\s.,]+$")
_GARBAGE_REPEAT_RE: re.Pattern[str] = re.compile(r"^(.{1,3})\1{10,}$")
_MUNGI_LAST_CACHE_PROMPT_ATTR: str = "_mungi_last_cache_prompt_text"
_MUNGI_LAST_CACHE_TOKENS_ATTR: str = "_mungi_last_cache_prompt_tokens"


# ---------------------------------------------------------------------------
# Think-tag post-processing
# ---------------------------------------------------------------------------


def strip_think_tags(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` reasoning blocks from output.

    Handles both closed ``<think>...</think>`` pairs and unclosed
    ``<think>`` tags (when generation was truncated mid-thought).

    Args:
        text: Raw LLM output string.

    Returns:
        Cleaned text with reasoning blocks removed.
    """
    text = _THINK_TAG_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    # Remove standalone closing </think> tags (from empty prefill echo)
    text = text.replace("</think>", "")
    text = _THINK_RESIDUAL_RE.sub("", text)
    return text.strip()


# Regex: keep Korean (Hangul), English, digits, basic punctuation, whitespace
_ALLOWED_CHARS_RE: re.Pattern[str] = re.compile(
    r"[^\uAC00-\uD7A3\u3131-\u3163\u1100-\u11FF"
    r"a-zA-Z0-9"
    r"\s.,!?~\-\u2026:;'\"()]"
)

SAFE_FALLBACK: str = "안녕! 무슨 이야기 해볼래?"
ECHO_FALLBACK: str = "뭉이가 잘 못 알아들었어. 다시 말해줘!"
_HELLO_PLACEHOLDER: str = "__MUNGI_HELLO__"
_PRESERVED_HONORIFICS: tuple[tuple[str, str], ...] = (
    ("안녕하세요", "__MUNGI_HELLO__"),
    ("잘했어요", "__MUNGI_PRAISE_GOOD_JOB__"),
    ("대단해요", "__MUNGI_PRAISE_AMAZING__"),
    ("좋아요", "__MUNGI_PRAISE_LIKE__"),
)
_HONORIFIC_REPAIRS: list[tuple[str, str]] = [
    ("해요", "해"),
    ("하세요", "해"),
    ("합니다", "해"),
    ("됩니다", "돼"),
    ("있어요", "있어"),
    ("없어요", "없어"),
    ("거예요", "거야"),
    ("줄게요", "줄게"),
    ("할게요", "할게"),
    ("할까요", "할까"),
    ("볼까요", "볼까"),
    ("인가요", "인 거야"),
    ("하나요", "하는 거야"),
    ("둥그러요", "둥글어"),
]


def detect_echo(user_text: str, response_text: str) -> bool:
    """Detect whether a response mostly repeats the user's input."""
    if not user_text or not response_text:
        return False

    clean_user = re.sub(r"[?!.,~\s]+", "", user_text)
    clean_resp = re.sub(r"[?!.,~\s]+", "", response_text)
    if not clean_user or not clean_resp:
        return False

    overlap = sum(1 for char in clean_user if char in clean_resp)
    ratio = overlap / len(clean_user) if clean_user else 0.0
    return (
        ratio > 0.8
        and len(clean_resp) < len(clean_user) * 1.5
        and (clean_user in clean_resp or clean_resp in clean_user)
    )


def repair_honorifics(text: str) -> str:
    """Replace common honorific endings with casual 반말 equivalents."""
    result = text
    for source, placeholder in _PRESERVED_HONORIFICS:
        result = result.replace(source, placeholder)
    for honorific, casual in _HONORIFIC_REPAIRS:
        result = result.replace(honorific, casual)
    for source, placeholder in _PRESERVED_HONORIFICS:
        result = result.replace(placeholder, source)
    return result


def sanitize_response(text: str, language: str = "ko") -> str:
    """Sanitize model output for the active response language.

    Processing steps:
    1. Detect garbage output artifacts and return a safe fallback
    2. In Korean mode, strip non-Korean characters and English word sequences
    3. Collapse whitespace and trim punctuation artifacts from removals
    4. In Korean mode, repair common honorific endings
    5. Return a safe fallback if the result is empty

    Args:
        text: LLM output after think-tag stripping.
        language: Active response language. Defaults to Korean mode.

    Returns:
        Sanitized text for the active language mode.
    """
    # Detect garbage output (digit-only sequences, single-char repetition)
    # Defense layer for KV cache corruption artifacts (e.g. "333..." garbage)
    stripped = text.strip()
    if stripped and (
        _GARBAGE_DIGIT_RE.fullmatch(stripped) or _GARBAGE_REPEAT_RE.fullmatch(stripped)
    ):
        logger.warning("Garbage output detected, returning safe fallback: %s", stripped[:50])
        return repair_honorifics(SAFE_FALLBACK)

    cleaned = text
    if language == "ko":
        cleaned = _ALLOWED_CHARS_RE.sub("", cleaned)
        # Remove English word sequences (system prompt: Korean only)
        cleaned = re.sub(r"[a-zA-Z]{2,}", "", cleaned)
    cleaned = _LEADING_INTERJECTION_RE.sub("", cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Clean trailing/leading punctuation from removals
    cleaned = re.sub(r"^\s*[.,]\s*", "", cleaned)
    if language == "ko":
        cleaned = repair_honorifics(cleaned)
    if not cleaned:
        if language == "ko":
            return repair_honorifics(SAFE_FALLBACK)
        return SAFE_FALLBACK
    return cleaned


# ---------------------------------------------------------------------------
# GGUF model discovery
# ---------------------------------------------------------------------------


def find_gguf_model(model_dir: str) -> Path | None:
    """Auto-discover a GGUF model file in the given directory.

    Scans for ``*.gguf`` files, excludes entries in
    ``_EXCLUDED_GGUF``, and returns the first remaining file
    (sorted alphabetically). Returns ``None`` if no files match.

    Args:
        model_dir: Directory to scan for GGUF files.

    Returns:
        Path to the first discovered GGUF file, or None.
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        logger.warning("Model directory does not exist: %s", model_dir)
        return None
    gguf_files = sorted(gf for gf in model_path.glob("*.gguf") if gf.name not in _EXCLUDED_GGUF)
    if gguf_files:
        logger.info("Found %d GGUF file(s) in %s", len(gguf_files), model_dir)
        for gf in gguf_files:
            logger.info("  - %s", gf.name)
    return gguf_files[0] if gguf_files else None


def detect_model_family(model_path: str) -> str:
    """Detect model family from GGUF filename.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        ``"qwen"`` or ``"gemma"``. Falls back to ``"qwen"`` if
        detection is ambiguous.
    """
    name = Path(model_path).stem.lower()
    if "gemma" in name:
        return MODEL_FAMILY_GEMMA
    return MODEL_FAMILY_QWEN


def stop_sequences_for_family(family: str) -> list[str]:
    """Return stop sequences appropriate for the given model family.

    Args:
        family: One of ``"qwen"``, ``"gemma"``.

    Returns:
        Stop sequence list.
    """
    if family == MODEL_FAMILY_GEMMA:
        return list(GEMMA_STOP_SEQUENCES)
    return list(DEFAULT_STOP_SEQUENCES)


# ---------------------------------------------------------------------------
# LLM model loading
# ---------------------------------------------------------------------------


def load_llm_model(
    model_path: str,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    n_ctx: int = DEFAULT_N_CTX,
) -> Any:
    """Load a GGUF model via llama-cpp-python.

    Args:
        model_path: Path to the GGUF model file.
        n_gpu_layers: Number of layers to offload to GPU.
            Use -1 for all layers.
        n_ctx: Context window size in tokens.

    Returns:
        A loaded ``llama_cpp.Llama`` instance.

    Raises:
        ImportError: If llama_cpp is not installed.
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If model loading fails.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error(
            "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
        )
        raise

    # Apply KV cache compatibility patch before first model load
    # LLAMA_SET_ROWS no longer needed with 0.3.17 (ggml_set_rows is the only path).
    # Retained as defensive no-op for any rollback to 0.3.16.
    _patch_llama_cpp_kv_cache()

    gguf_path = Path(model_path)
    if not gguf_path.exists():
        msg = f"Model file not found: {gguf_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading GGUF model: %s", gguf_path)
    logger.info("  n_gpu_layers=%d, n_ctx=%d", n_gpu_layers, n_ctx)

    kv_type = _resolve_kv_type()
    llama_kwargs: dict[str, Any] = {
        "model_path": str(gguf_path),
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "flash_attn": True,
        "verbose": False,
    }
    if kv_type is not None:
        llama_kwargs["type_k"] = kv_type
        llama_kwargs["type_v"] = kv_type

    llm = Llama(**llama_kwargs)
    return llm


def load_gemma4_text_llm(
    model_path: str,
    n_gpu_layers: int = 99,
    n_ctx: int = DEFAULT_N_CTX,
) -> Any:
    """Load the Gemma 4 text-only GGUF model via llama-cpp-python.

    Args:
        model_path: Path to the Gemma 4 GGUF model file.
        n_gpu_layers: Number of layers to offload to GPU.
        n_ctx: Context window size in tokens.

    Returns:
        A loaded ``llama_cpp.Llama`` instance using the GGUF-embedded
        Gemma chat template.

    Raises:
        ImportError: If llama_cpp is not installed.
        FileNotFoundError: If the model file does not exist.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error(
            "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
        )
        raise

    _patch_llama_cpp_kv_cache()

    gguf_path = Path(model_path)
    if not gguf_path.exists():
        msg = f"Model file not found: {gguf_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading Gemma 4 text LLM backend: %s", gguf_path)
    logger.info("  backend=gemma4_text, n_gpu_layers=%d, n_ctx=%d", n_gpu_layers, n_ctx)

    kv_type = _resolve_kv_type()
    llama_kwargs: dict[str, Any] = {
        "model_path": str(gguf_path),
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "chat_format": None,
        "flash_attn": True,
        "verbose": False,
    }
    if kv_type is not None:
        llama_kwargs["type_k"] = kv_type
        llama_kwargs["type_v"] = kv_type

    return Llama(**llama_kwargs)


def build_llm_from_config(config: LLMBackendConfig) -> tuple[str, Any]:
    """Build an LLM instance from the resolved backend configuration.

    Args:
        config: Resolved LLM backend configuration.

    Returns:
        A tuple of ``(backend_name, llama_instance)``.
    """
    if config.backend == "gemma4_text":
        model_path = config.model_path or DEFAULT_GEMMA4_TEXT_MODEL_PATH
        logger.info("Dispatching LLM load to gemma4_text backend")
        return (
            config.backend,
            load_gemma4_text_llm(
                model_path,
                n_gpu_layers=config.n_gpu_layers,
                n_ctx=config.n_ctx,
            ),
        )

    model_path = config.model_path or DEFAULT_QWEN3_LEGACY_MODEL_PATH
    logger.info("Dispatching LLM load to qwen3_legacy backend")
    return (
        config.backend,
        load_llm_model(
            model_path,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
        ),
    )


def _resolve_kv_type() -> int | None:
    """Return the KV cache ``type_k``/``type_v`` enum value.

    Honors ``MUNGI_LLM_KV_TYPE`` env override with values in
    {f16, f32, q8_0, q4_0, q5_0}. Default ``f16`` matches the
    llama-cpp-python 0.3.17 default so existing deployments
    see no behavior change unless explicitly opted in.

    Returns ``None`` for ``f16`` to let llama-cpp-python use its
    native default path. Returns the integer enum value otherwise.
    """
    raw_value = os.getenv("MUNGI_LLM_KV_TYPE", DEFAULT_KV_TYPE)
    normalized = raw_value.strip().lower()
    if normalized not in _KV_TYPE_MAP:
        logger.warning(
            "MUNGI_LLM_KV_TYPE=%r not in %s; using %s default",
            raw_value,
            sorted(_KV_TYPE_MAP),
            DEFAULT_KV_TYPE,
        )
        normalized = DEFAULT_KV_TYPE

    if normalized == DEFAULT_KV_TYPE:
        return None

    kv_type = _KV_TYPE_MAP[normalized]
    logger.info(
        "KV cache quantization: type_k=type_v=%s (%d) saves approximately 200 MB "
        "vs f16 on Qwen3.5-2B n_ctx=2048",
        normalized,
        kv_type,
    )
    return kv_type


def _supports_chat_completion_cache_prompt(llm: Any) -> bool:
    """Return ``True`` when the chat-completion API exposes ``cache_prompt``."""
    create_chat_completion = getattr(llm, "create_chat_completion", None)
    if create_chat_completion is None:
        return False
    try:
        signature = inspect.signature(create_chat_completion)
    except (TypeError, ValueError):
        return False
    return "cache_prompt" in signature.parameters


def _serialize_messages_for_cache(messages: list[dict[str, str]]) -> str:
    """Serialize chat messages into a stable string for cache estimation."""
    return "\n".join(
        f"{message.get('role', '')}\n{message.get('content', '')}" for message in messages
    )


def _longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    """Return the length of the shared prefix between two token lists."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _tokenize_cache_prompt(llm: Any, prompt_text: str) -> list[int] | None:
    """Tokenize a cache-estimation prompt when the Llama instance supports it."""
    tokenize = getattr(llm, "tokenize", None)
    if not callable(tokenize):
        return None

    prompt_bytes = prompt_text.encode("utf-8")
    tokenize_attempts: tuple[dict[str, object], ...] = (
        {"add_bos": True, "special": True},
        {"add_bos": True},
        {},
    )
    for kwargs in tokenize_attempts:
        try:
            tokens = tokenize(prompt_bytes, **kwargs)
        except TypeError:
            continue
        except Exception:
            return None
        try:
            return [int(token) for token in tokens]
        except TypeError:
            return None
    return None


def _tokenize_system_state_prompt(llm: Any, system_prompt: str) -> list[int] | None:
    """Tokenize the standalone system prompt for KV-state snapshot preparation."""
    tokenize = getattr(llm, "tokenize", None)
    if not callable(tokenize):
        return None

    prompt_bytes = system_prompt.encode("utf-8")
    tokenize_attempts: tuple[dict[str, object], ...] = (
        {"add_bos": True, "special": True},
        {"add_bos": True},
        {},
    )
    for kwargs in tokenize_attempts:
        try:
            tokens = tokenize(prompt_bytes, **kwargs)
        except TypeError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("System-state snapshot failed: %s", exc)
            return None
        try:
            return [int(token) for token in tokens]
        except TypeError:
            return None
    return None


def prepare_system_state_snapshot(
    llm: Any,
    system_prompt: str,
) -> Any | None:
    """Snapshot the KV state after evaluating the system prompt.

    Returns a ``LlamaState``-like object usable with
    :func:`restore_system_state_snapshot`, or ``None`` when the active
    Llama instance does not expose the required low-level state API.

    The helper is safe to call repeatedly. It resets the active context
    before and after snapshot preparation so the caller does not inherit
    the temporary system-prompt evaluation side effects.
    """
    save_state = getattr(llm, "save_state", None)
    reset = getattr(llm, "reset", None)
    evaluate = getattr(llm, "eval", None)
    tokenize = getattr(llm, "tokenize", None)
    if not all(callable(fn) for fn in (save_state, reset, evaluate, tokenize)):
        logger.warning(
            "Llama instance missing save_state/reset/eval/tokenize; system-state snapshot disabled",
        )
        return None
    save_state_fn = cast(Callable[[], Any], save_state)
    reset_fn = cast(Callable[[], None], reset)
    evaluate_fn = cast(Callable[[list[int]], Any], evaluate)

    try:
        reset_fn()
        tokens = _tokenize_system_state_prompt(llm, system_prompt)
        if tokens is None:
            logger.warning(
                "System-state snapshot disabled: tokenize() could not encode system prompt",
            )
            return None
        evaluate_fn(tokens)
        state = save_state_fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("System-state snapshot failed: %s", exc)
        return None
    finally:
        try:
            reset_fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug("System-state snapshot reset failed: %s", exc)

    logger.info("System-state snapshot captured (%d tokens)", len(tokens))
    return state


def restore_system_state_snapshot(llm: Any, state: Any) -> bool:
    """Restore a previously captured Llama state on the active Llama instance."""
    load_state = getattr(llm, "load_state", None)
    if not callable(load_state):
        return False

    try:
        load_state(state)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("System-state restore failed: %s", exc)
        return False


def _extract_cache_token_counts(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract cache hit/miss token counts from API metadata when present."""
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return None, None

    prompt_tokens = usage.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        prompt_tokens = None

    prompt_tokens_cached = usage.get("prompt_tokens_cached")
    if isinstance(prompt_tokens_cached, bool) or not isinstance(prompt_tokens_cached, int):
        prompt_tokens_cached = None

    if prompt_tokens is not None and prompt_tokens_cached is not None:
        return prompt_tokens_cached, max(0, prompt_tokens - prompt_tokens_cached)

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")
        if isinstance(cached_tokens, int) and prompt_tokens is not None:
            return cached_tokens, max(0, prompt_tokens - cached_tokens)

    return None, None


def _estimate_cache_token_counts(
    llm: Any,
    prompt_text: str,
    prompt_tokens: list[int] | None,
) -> tuple[int | None, int | None]:
    """Estimate cache hit/miss counts from the previous prompt prefix."""
    if prompt_tokens is not None:
        previous_tokens = getattr(llm, _MUNGI_LAST_CACHE_TOKENS_ATTR, None)
        if isinstance(previous_tokens, list):
            cache_hit_tokens = _longest_common_prefix_length(previous_tokens, prompt_tokens)
        else:
            cache_hit_tokens = 0
        return cache_hit_tokens, max(0, len(prompt_tokens) - cache_hit_tokens)

    estimated_prompt_tokens = max(1, (len(prompt_text.strip()) + 3) // 4)
    previous_prompt_text = getattr(llm, _MUNGI_LAST_CACHE_PROMPT_ATTR, None)
    if not isinstance(previous_prompt_text, str) or not prompt_text:
        return 0, estimated_prompt_tokens

    shared_prefix_chars = 0
    for left_char, right_char in zip(previous_prompt_text, prompt_text, strict=False):
        if left_char != right_char:
            break
        shared_prefix_chars += 1

    cache_hit_tokens = round(estimated_prompt_tokens * (shared_prefix_chars / len(prompt_text)))
    return cache_hit_tokens, max(0, estimated_prompt_tokens - cache_hit_tokens)


def _store_cache_prompt_state(
    llm: Any,
    prompt_text: str,
    prompt_tokens: list[int] | None,
) -> None:
    """Persist the latest prompt state for the next cache estimate."""
    setattr(llm, _MUNGI_LAST_CACHE_PROMPT_ATTR, prompt_text)
    setattr(llm, _MUNGI_LAST_CACHE_TOKENS_ATTR, prompt_tokens)


# ---------------------------------------------------------------------------
# LLM generation (streaming for TTFT measurement)
# ---------------------------------------------------------------------------


def run_generation(
    llm: Any,
    prompt: str | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stop: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    min_p: float = DEFAULT_MIN_P,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    enable_thinking: bool = False,
) -> tuple[str, int, float, float]:
    """Run text generation with streaming to measure TTFT.

    Uses the streaming API to capture time-to-first-token accurately,
    then collects all remaining tokens.

    Args:
        llm: A loaded ``llama_cpp.Llama`` instance.
        prompt: The input prompt string, or ``None``.
        max_tokens: Maximum number of tokens to generate.
        stop: Sequences that terminate generation. Defaults to
            ``DEFAULT_STOP_SEQUENCES`` if ``None``.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-K sampling.
        min_p: Minimum probability threshold.
        presence_penalty: Presence penalty for repetition control.
        repeat_penalty: Repetition penalty for recently generated
            tokens. Higher values reduce repeated phrases.
        enable_thinking: When ``True``, uses Qwen3 thinking-mode
            sampling params (temperature=0.6, top_p=0.95) instead
            of the non-thinking defaults.

    Returns:
        A tuple of (generated_text, completion_tokens, ttft_s,
        generation_time_s). Returns ``("", 0, -1.0, 0.0)`` when
        the prompt is empty or ``None``, with ``ttft=-1.0``
        distinguishing a skipped call from a real 0 ms TTFT.
    """
    if prompt is None or not prompt.strip():
        logger.warning("Empty prompt received, skipping LLM generation")
        return "", 0, -1.0, 0.0

    if enable_thinking:
        temperature = THINKING_TEMPERATURE
        top_p = THINKING_TOP_P
        top_k = THINKING_TOP_K
        min_p = THINKING_MIN_P
        presence_penalty = THINKING_PRESENCE_PENALTY

    if stop is None:
        stop = DEFAULT_STOP_SEQUENCES

    generated_parts: list[str] = []
    ttft: float = -1.0
    token_count: int = 0
    first_token_received: bool = False

    gen_start = time.monotonic()

    stream = llm(
        prompt,
        max_tokens=max_tokens,
        stop=stop,
        echo=False,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repeat_penalty=repeat_penalty,
    )

    for chunk in stream:
        token_text = chunk["choices"][0]["text"]
        if not first_token_received:
            ttft = time.monotonic() - gen_start
            first_token_received = True
        generated_parts.append(token_text)
        token_count += 1

    generation_time = time.monotonic() - gen_start

    generated_text = "".join(generated_parts)
    return generated_text, token_count, ttft, generation_time


def run_chat_generation(
    llm: Any,
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stop: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    min_p: float = DEFAULT_MIN_P,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    enable_thinking: bool = False,
    cache_prompt: bool = False,
    system_state: Any | None = None,
) -> tuple[str, int, float, float, int | None, int | None]:
    """Run chat generation with streaming to measure TTFT.

    Args:
        llm: A loaded ``llama_cpp.Llama`` instance.
        messages: Chat messages passed to ``create_chat_completion``.
        max_tokens: Maximum number of tokens to generate.
        stop: Sequences that terminate generation. Defaults to
            ``DEFAULT_STOP_SEQUENCES`` if ``None``.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-K sampling.
        min_p: Minimum probability threshold.
        presence_penalty: Presence penalty for repetition control.
        repeat_penalty: Repetition penalty for recently generated
            tokens. Higher values reduce repeated phrases.
        enable_thinking: When ``True``, uses Qwen3 thinking-mode
            sampling params (temperature=0.6, top_p=0.95) instead
            of the non-thinking defaults.
        cache_prompt: When ``True``, enable llama.cpp prefix-cache
            reuse for repeated chat prompt prefixes.
        system_state: Optional saved Llama KV state representing the
            language-specific system prompt prefix to restore before the
            chat-completion call.

    Returns:
        A tuple of (generated_text, completion_tokens, ttft_s,
        generation_time_s, cache_hit_tokens, cache_miss_tokens).
        Returns ``("", 0, -1.0, 0.0, None, None)`` when ``messages``
        is empty, with ``ttft=-1.0`` distinguishing a skipped call
        from a real 0 ms TTFT.
    """
    if not messages:
        logger.warning("Empty messages received, skipping LLM generation")
        return "", 0, -1.0, 0.0, None, None

    if enable_thinking:
        temperature = THINKING_TEMPERATURE
        top_p = THINKING_TOP_P
        top_k = THINKING_TOP_K
        min_p = THINKING_MIN_P
        presence_penalty = THINKING_PRESENCE_PENALTY

    if stop is None:
        stop = DEFAULT_STOP_SEQUENCES

    generated_parts: list[str] = []
    ttft: float = -1.0
    token_count: int = 0
    first_token_received: bool = False
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    prompt_text = _serialize_messages_for_cache(messages)
    prompt_tokens = _tokenize_cache_prompt(llm, prompt_text) if cache_prompt else None

    gen_start = time.monotonic()

    # chat_template_kwargs is only supported by llama-server REST API,
    # not the Python bindings. Qwen3.5-2B defaults to non-thinking mode.
    create_chat_completion_kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repeat_penalty": repeat_penalty,
    }
    if cache_prompt and _supports_chat_completion_cache_prompt(llm):
        create_chat_completion_kwargs["cache_prompt"] = True

    if system_state is not None:
        restored = restore_system_state_snapshot(llm, system_state)
        if not restored:
            logger.warning(
                "System-state restore failed; falling back to default prompt evaluation",
            )

    stream = llm.create_chat_completion(**create_chat_completion_kwargs)

    for chunk in stream:
        if cache_prompt and (cache_hit_tokens is None or cache_miss_tokens is None):
            cache_hit_tokens, cache_miss_tokens = _extract_cache_token_counts(chunk)
        delta = chunk["choices"][0].get("delta", {})
        token_text = delta.get("content", "")
        if not token_text:
            continue
        if not first_token_received:
            ttft = time.monotonic() - gen_start
            first_token_received = True
        generated_parts.append(token_text)
        token_count += 1

    generation_time = time.monotonic() - gen_start

    if cache_prompt:
        if cache_hit_tokens is None or cache_miss_tokens is None:
            cache_hit_tokens, cache_miss_tokens = _estimate_cache_token_counts(
                llm,
                prompt_text,
                prompt_tokens,
            )
        _store_cache_prompt_state(llm, prompt_text, prompt_tokens)

    generated_text = "".join(generated_parts)
    return (
        generated_text,
        token_count,
        ttft,
        generation_time,
        cache_hit_tokens,
        cache_miss_tokens,
    )


def _extract_formatter_from_chat_handler(chat_handler: Any) -> Callable[..., Any] | None:
    """Return the wrapped ChatFormatter when a handler exposes it via closure."""
    closure = getattr(chat_handler, "__closure__", None)
    if closure is None:
        return None

    for cell in closure:
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if callable(candidate) and candidate is not chat_handler:
            return cast(Callable[..., Any], candidate)
    return None


def _coerce_chat_prompt(formatter_response: Any) -> str | None:
    """Extract a prompt string from llama-cpp-python ChatFormatter responses."""
    if isinstance(formatter_response, str):
        return formatter_response
    if isinstance(formatter_response, bytes):
        return formatter_response.decode("utf-8", errors="replace")
    prompt = getattr(formatter_response, "prompt", None)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8", errors="replace")
    if isinstance(formatter_response, dict):
        dict_prompt = formatter_response.get("prompt")
        if isinstance(dict_prompt, str):
            return dict_prompt
        if isinstance(dict_prompt, bytes):
            return dict_prompt.decode("utf-8", errors="replace")
    return None


def _call_chat_formatter(
    formatter: Callable[..., Any],
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str | None:
    """Call a formatter across llama-cpp-python signature variants."""
    call_attempts: tuple[tuple[tuple[object, ...], dict[str, Any]], ...] = (
        (
            (),
            {
                "messages": messages,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            },
        ),
        ((), {"messages": messages, "enable_thinking": enable_thinking}),
        ((), {"messages": messages}),
        (
            (messages,),
            {
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
                "tokenize": False,
            },
        ),
        ((messages,), {"add_generation_prompt": True, "tokenize": False}),
        ((messages,), {}),
    )
    for args, kwargs in call_attempts:
        try:
            prompt = _coerce_chat_prompt(formatter(*args, **kwargs))
        except TypeError:
            continue
        if prompt is not None:
            return prompt
    return None


def _llama_token_text(llm: Any, token_id: int) -> str:
    """Return a token text string for metadata-backed ChatFormatter setup."""
    if token_id == -1:
        return ""
    model = getattr(llm, "_model", None)
    token_get_text = getattr(model, "token_get_text", None)
    if callable(token_get_text):
        token_text = token_get_text(token_id)
        if isinstance(token_text, bytes):
            return token_text.decode("utf-8", errors="replace")
        if isinstance(token_text, str):
            return token_text
    detokenize = getattr(llm, "detokenize", None)
    if callable(detokenize):
        token_bytes = detokenize([token_id])
        if isinstance(token_bytes, bytes):
            return token_bytes.decode("utf-8", errors="replace")
        if isinstance(token_bytes, str):
            return token_bytes
    return ""


def _format_chat_messages_with_llama_chat_formatter(
    llm: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    """Format chat messages through llama-cpp-python's ChatFormatter APIs."""
    model = getattr(llm, "_model", None)
    apply_chat_template = getattr(model, "apply_chat_template", None)
    if callable(apply_chat_template):
        prompt = _call_chat_formatter(
            cast(Callable[..., Any], apply_chat_template),
            messages,
            enable_thinking=enable_thinking,
        )
        if prompt is not None:
            return prompt

    from llama_cpp import llama_chat_format  # type: ignore[import-not-found, import-untyped]

    for formatter_name in (
        "format_messages",
        "format_chat_messages",
        "apply_chat_template",
    ):
        formatter = getattr(llama_chat_format, formatter_name, None)
        if callable(formatter):
            prompt = _call_chat_formatter(
                cast(Callable[..., Any], formatter),
                messages,
                enable_thinking=enable_thinking,
            )
            if prompt is not None:
                return prompt

    chat_format = getattr(llm, "chat_format", None)
    get_handler = getattr(llama_chat_format, "get_chat_completion_handler", None)
    if isinstance(chat_format, str) and callable(get_handler):
        try:
            handler = get_handler(chat_format)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not load llama chat handler %r: %s", chat_format, exc)
        else:
            formatter = _extract_formatter_from_chat_handler(handler)
            if formatter is not None:
                prompt = _call_chat_formatter(
                    formatter,
                    messages,
                    enable_thinking=enable_thinking,
                )
                if prompt is not None:
                    return prompt

    metadata = getattr(llm, "metadata", None)
    formatter_class = getattr(llama_chat_format, "Jinja2ChatFormatter", None)
    if isinstance(metadata, dict) and callable(formatter_class):
        template_choices = {
            key[10:]: value
            for key, value in metadata.items()
            if isinstance(key, str)
            and key.startswith("tokenizer.chat_template.")
            and isinstance(value, str)
        }
        default_template = metadata.get("tokenizer.chat_template")
        if isinstance(default_template, str):
            template_choices["chat_template.default"] = default_template

        template_name = chat_format if isinstance(chat_format, str) else "chat_template.default"
        template = template_choices.get(template_name) or template_choices.get(
            "chat_template.default",
        )
        if template is not None:
            eos_token_id = int(llm.token_eos())
            bos_token_id = int(llm.token_bos())
            formatter = formatter_class(
                template=template,
                eos_token=_llama_token_text(llm, eos_token_id),
                bos_token=_llama_token_text(llm, bos_token_id),
                stop_token_ids=[eos_token_id],
            )
            prompt = _call_chat_formatter(
                cast(Callable[..., Any], formatter),
                messages,
                enable_thinking=enable_thinking,
            )
            if prompt is not None:
                return prompt

    chat_handler = getattr(llm, "chat_handler", None)
    if callable(chat_handler):
        formatter = _extract_formatter_from_chat_handler(chat_handler)
        if formatter is not None:
            prompt = _call_chat_formatter(
                formatter,
                messages,
                enable_thinking=enable_thinking,
            )
            if prompt is not None:
                return prompt

    msg = "Llama instance does not expose a usable chat formatter"
    raise RuntimeError(msg)


def _strip_stop_suffix(text: str, stop: list[str]) -> tuple[str, bool]:
    """Strip a matched stop suffix from generated text."""
    for stop_sequence in stop:
        if stop_sequence and text.endswith(stop_sequence):
            return text[: -len(stop_sequence)], True
    return text, False


def run_chat_generation_lowlevel(
    llm: Any,
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stop: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    min_p: float = DEFAULT_MIN_P,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    enable_thinking: bool = False,
) -> tuple[str, int, float, float, int | None, int | None]:
    """Run low-level llama.cpp chat generation with manual sampling.

    Chat prompt rendering is delegated to llama-cpp-python's built-in
    ChatFormatter stack rather than hand-built ChatML. The formatter helper
    checks ``llm._model.apply_chat_template`` first, then
    ``llama_cpp.llama_chat_format`` module-level formatters and registered
    formatter closures, and finally a closure-backed ``llm.chat_handler``. This
    mirrors llama-cpp-python's supported chat-template machinery while avoiding
    ``create_chat_completion()`` for the token generation loop.

    Args:
        llm: A loaded ``llama_cpp.Llama`` instance.
        messages: Chat messages to format and evaluate.
        max_tokens: Maximum number of tokens to generate.
        stop: Sequences that terminate generation. Defaults to
            ``DEFAULT_STOP_SEQUENCES`` if ``None``.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-K sampling.
        min_p: Minimum probability threshold.
        presence_penalty: Presence penalty for repetition control.
        repeat_penalty: Repeat penalty for recently generated tokens.
        enable_thinking: When ``True``, uses Qwen3 thinking-mode sampling
            params instead of the non-thinking defaults.

    Returns:
        A tuple of (generated_text, completion_tokens, ttft_s,
        generation_time_s, cache_hit_tokens, cache_miss_tokens). Cache
        hit/miss counts are not tracked on this path and are returned as
        ``None``.
    """
    if not messages:
        logger.warning("Empty messages received, skipping LLM generation")
        return "", 0, -1.0, 0.0, None, None

    if enable_thinking:
        temperature = THINKING_TEMPERATURE
        top_p = THINKING_TOP_P
        top_k = THINKING_TOP_K
        min_p = THINKING_MIN_P
        presence_penalty = THINKING_PRESENCE_PENALTY

    if stop is None:
        stop = DEFAULT_STOP_SEQUENCES

    from llama_cpp import llama_cpp as llama_c  # type: ignore[import-not-found, import-untyped]

    formatted_prompt = _format_chat_messages_with_llama_chat_formatter(
        llm,
        messages,
        enable_thinking=enable_thinking,
    )
    prompt_tokens = llm.tokenize(formatted_prompt.encode("utf-8"))
    generated_parts: list[str] = []
    ttft: float = -1.0
    token_count = 0
    first_token_received = False
    chain = None
    gen_start = time.monotonic()

    llm.reset()
    llm.eval(prompt_tokens)

    ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
    if ctx is None:
        msg = "Llama instance does not expose _ctx.ctx"
        raise RuntimeError(msg)

    try:
        chain = llama_c.llama_sampler_chain_init(
            llama_c.llama_sampler_chain_default_params(),
        )
        penalty_last_n = int(getattr(llm, "last_n_tokens_size", 64))
        llama_c.llama_sampler_chain_add(
            chain,
            llama_c.llama_sampler_init_penalties(
                penalty_last_n,
                repeat_penalty,
                0.0,
                presence_penalty,
            ),
        )
        llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_top_k(top_k))
        llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_top_p(top_p, 1))
        llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_min_p(min_p, 1))
        llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_temp(temperature))
        seed = int(getattr(llm, "_seed", getattr(llama_c, "LLAMA_DEFAULT_SEED", 0xFFFFFFFF)))
        llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_dist(seed))

        sampler_accept = getattr(llama_c, "llama_sampler_accept", None)
        eos_token = int(llm.token_eos())

        for _ in range(max_tokens):
            token_id = int(llama_c.llama_sampler_sample(chain, ctx, -1))
            if token_id == eos_token:
                break

            piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
            generated_parts.append(piece)
            generated_text, stop_hit = _strip_stop_suffix("".join(generated_parts), stop)
            if stop_hit:
                generated_parts = [generated_text]
                break

            if piece:
                if not first_token_received:
                    ttft = time.monotonic() - gen_start
                    first_token_received = True
                token_count += 1

            if callable(sampler_accept):
                sampler_accept(chain, token_id)
            llm.eval([token_id])
    finally:
        if chain is not None:
            llama_c.llama_sampler_free(chain)

    generation_time = time.monotonic() - gen_start
    return "".join(generated_parts), token_count, ttft, generation_time, None, None
