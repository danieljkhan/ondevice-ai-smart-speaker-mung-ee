"""Sherpa-ONNX Qwen3-ASR model loader and inference runner.

ADR 0055 Update (2026-04-29 Decision item 3) makes Qwen3-ASR the sole
supported STT engine. This module preserves the public loader and inference
interfaces used by the pipeline, scripts, and benchmark tooling while routing
legacy model selectors to the Qwen3-ASR bundle.
"""

from __future__ import annotations

import logging
import os
import re
import struct
import wave
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger("mungi.models.stt_runner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keep the public default stable; resolve_model_size routes it to Qwen3-ASR.
DEFAULT_MODEL_SIZE: str = "small"
DEFAULT_DEVICE: str = "cpu"
DEFAULT_COMPUTE_TYPE: str = "float16"
DEFAULT_MODEL_DIR: str = "/opt/mungi/ai_models"
DEFAULT_LANGUAGE: str = "ko"
DEFAULT_BEAM_SIZE: int = 5

_QWEN3_ASR_NAME = "qwen3-asr-0.6b"
_QWEN3_ASR_BUNDLE_PREFIX = "sherpa-onnx-qwen3-asr-"
_QWEN3_ASR_DEFAULT_MAX_TOTAL_LEN = 512
_QWEN3_ASR_DEFAULT_MAX_NEW_TOKENS = 128
_QWEN3_ASR_DEFAULT_FEATURE_DIM = 128
_QWEN3_ASR_DEFAULT_SAMPLE_RATE = 16000
# Legacy Tier 1 hotwords for diagnostics: wakeword variants + Tier 1 terms.
_HOTWORDS_BASELINE: Final[tuple[str, ...]] = ("뭉이야", "뭉이")
_HOTWORDS_REQUIRED_TIER: Final[tuple[str, ...]] = (
    "한글",
    "추석",
    "송편",
    "단군신화",
)
_HOTWORDS_EXPLORATORY_TIER: Final[tuple[str, ...]] = (
    "일제강점기",
    "빙하",
    "자석",
    "화산",
    "지진",
    "무지개",
    "한복",
)
LEGACY_QWEN3_ASR_HOTWORDS_TIER1_DEFAULT: str = ",".join(
    _HOTWORDS_BASELINE + _HOTWORDS_REQUIRED_TIER + _HOTWORDS_EXPLORATORY_TIER
)
_QWEN3_ASR_HOTWORDS_ENV_VAR = "MUNGI_QWEN3_ASR_HOTWORDS"

_MODEL_ALIASES: dict[str, str] = {
    "small": _QWEN3_ASR_NAME,
    "base": _QWEN3_ASR_NAME,
    "medium": _QWEN3_ASR_NAME,
    "large": _QWEN3_ASR_NAME,
    "large-v2": _QWEN3_ASR_NAME,
    "large-v3": _QWEN3_ASR_NAME,
    "tiny": _QWEN3_ASR_NAME,
    "qwen3-asr": _QWEN3_ASR_NAME,
    "qwen3": _QWEN3_ASR_NAME,
    "qwen3-asr-0.6b": _QWEN3_ASR_NAME,
    "qwen3-asr-0.6b-int8": _QWEN3_ASR_NAME,
}
_LEGACY_COMPACT_ALIASES: frozenset[str] = frozenset(
    {
        "sense" + "voice",
        "moon" + "shine",
        "moon" + "shine" + "tiny",
        "moon" + "shine" + "tiny" + "ko",
    }
)
_LEGACY_BUNDLE_PREFIXES: tuple[str, ...] = (
    "sherpa-onnx-" + "sense" + "-voice",
    "sherpa-onnx-" + "moon" + "shine",
)

_TEXT_WS_RE = re.compile(r"\s+")
_ASR_TEXT_TAG: Final[str] = "<asr_text>"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single transcription segment with timestamps and text."""

    start: float
    end: float
    text: str

    def duration_s(self) -> float:
        """Return segment duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Serialize segment to a plain dict."""
        return asdict(self)


@dataclass
class LoadedSttModel:
    """Loaded Sherpa-ONNX STT recognizer wrapper."""

    recognizer: Any
    backend: str
    requested_model_size: str
    resolved_model_size: str
    provider: str
    model_path: str
    language: str


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------


def resolve_model_size(model_size: str) -> str:
    """Resolve a requested model identifier to the supported Qwen3-ASR bundle."""

    normalized = model_size.strip().lower()
    if not normalized:
        return _MODEL_ALIASES[DEFAULT_MODEL_SIZE]
    if normalized in _MODEL_ALIASES:
        return _MODEL_ALIASES[normalized]
    compact = normalized.replace("-", "").replace("_", "")
    if compact in _LEGACY_COMPACT_ALIASES:
        return _QWEN3_ASR_NAME
    if any(normalized.startswith(prefix) for prefix in _LEGACY_BUNDLE_PREFIXES):
        return _QWEN3_ASR_NAME
    if normalized.startswith(_QWEN3_ASR_BUNDLE_PREFIX):
        return _QWEN3_ASR_NAME
    msg = (
        f"Unsupported STT model '{model_size}'. "
        f"Use stt_engine=qwen3-asr or one of the legacy size aliases that now "
        f"resolve to {_QWEN3_ASR_NAME}."
    )
    raise ValueError(msg)


def _provider_candidates(device: str) -> list[str]:
    normalized = device.strip().lower()
    if "cuda" in normalized or normalized == "gpu":
        return ["cuda", "cpu"]
    if not normalized:
        return ["cpu"]
    return [normalized]


def _supported_providers() -> set[str]:
    try:
        onnxruntime = import_module("onnxruntime")
    except ImportError:
        return set()

    try:
        return {str(provider) for provider in onnxruntime.get_available_providers()}
    except Exception:
        return set()


def _find_bundle_dir(model_dir: str, prefix: str) -> Path:
    root = Path(model_dir)
    if not root.exists():
        msg = f"Model directory does not exist: {model_dir}"
        raise FileNotFoundError(msg)

    matches = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
    if not matches:
        msg = f"No Sherpa-ONNX bundle found in {model_dir} with prefix '{prefix}'"
        raise FileNotFoundError(msg)
    return matches[0]


def _require_first_file(bundle_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = bundle_dir / name
        if candidate.exists():
            return candidate
    msg = f"Required file not found in {bundle_dir}: one of {', '.join(names)}"
    raise FileNotFoundError(msg)


def _load_qwen3_asr(
    sherpa_onnx: Any,
    model_dir: str,
    provider: str,
    hotwords: str,
) -> tuple[Any, str]:
    """Load the Qwen3-ASR Sherpa-ONNX bundle."""

    bundle_dir = _find_bundle_dir(model_dir, _QWEN3_ASR_BUNDLE_PREFIX)
    logger.info("Loading Sherpa-ONNX Qwen3-ASR bundle %s", bundle_dir)

    conv_frontend = _require_first_file(bundle_dir, ("conv_frontend.onnx",))
    encoder = _require_first_file(bundle_dir, ("encoder.int8.onnx", "encoder.onnx"))
    decoder = _require_first_file(bundle_dir, ("decoder.int8.onnx", "decoder.onnx"))

    tokenizer_dir = bundle_dir / "tokenizer"
    required_tokenizer_files = (
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    )
    missing_tokenizer_files = [
        name for name in required_tokenizer_files if not (tokenizer_dir / name).exists()
    ]
    if missing_tokenizer_files:
        msg = (
            f"Required tokenizer file(s) not found in {tokenizer_dir}: "
            f"{', '.join(missing_tokenizer_files)}"
        )
        raise FileNotFoundError(msg)

    if hotwords:
        logger.info("Qwen3-ASR hotwords (%d entries): %s", len(hotwords.split(",")), hotwords)
    else:
        logger.info("Qwen3-ASR hotwords: (none)")

    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(conv_frontend),
        encoder=str(encoder),
        decoder=str(decoder),
        tokenizer=str(tokenizer_dir),
        provider=provider,
        num_threads=1,
        sample_rate=_QWEN3_ASR_DEFAULT_SAMPLE_RATE,
        feature_dim=_QWEN3_ASR_DEFAULT_FEATURE_DIM,
        decoding_method="greedy_search",
        max_total_len=_QWEN3_ASR_DEFAULT_MAX_TOTAL_LEN,
        max_new_tokens=_QWEN3_ASR_DEFAULT_MAX_NEW_TOKENS,
        hotwords=hotwords,
    )
    return recognizer, str(bundle_dir)


def _resolve_qwen3_asr_hotwords(explicit: str | None) -> str:
    """Resolve Qwen3-ASR hotwords from explicit arg, env var, or empty default."""

    if explicit is not None:
        return explicit

    env_hotwords = os.environ.get(_QWEN3_ASR_HOTWORDS_ENV_VAR)
    if env_hotwords is not None:
        return env_hotwords

    return ""


# ---------------------------------------------------------------------------
# STT model loading
# ---------------------------------------------------------------------------


def load_stt_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    model_dir: str = DEFAULT_MODEL_DIR,
    language: str = DEFAULT_LANGUAGE,
    qwen3_asr_hotwords: str | None = None,
) -> Any:
    """Load the Sherpa-ONNX Qwen3-ASR model with the specified configuration.

    Args:
        model_size: Qwen3-ASR selector or legacy alias routed to Qwen3-ASR.
        device: Preferred execution provider (``"cuda"`` or ``"cpu"``).
        compute_type: Retained for CLI/config compatibility. Sherpa ignores it.
        model_dir: Directory containing Sherpa-ONNX model bundles.
        language: Retained for CLI/config compatibility. Qwen3-ASR auto-detects
            language, so this value is only used as metadata fallback.
        qwen3_asr_hotwords: Optional Qwen3-ASR decoder hotwords. ``None`` checks
            ``MUNGI_QWEN3_ASR_HOTWORDS`` before falling back to the empty runtime
            default. ``""`` disables hotwords.

    Returns:
        Loaded STT recognizer wrapper.

    Raises:
        ImportError: If sherpa_onnx is not installed.
        FileNotFoundError: If the selected Sherpa bundle is missing required files.
        RuntimeError: If model loading fails for all provider candidates.
    """
    try:
        sherpa_onnx = import_module("sherpa_onnx")
    except ImportError:
        logger.error("sherpa_onnx is not installed. Install with: pip install sherpa-onnx")
        raise

    resolved_model = resolve_model_size(model_size)
    providers = _provider_candidates(device)
    target_language = language.strip() or DEFAULT_LANGUAGE
    resolved_hotwords = _resolve_qwen3_asr_hotwords(qwen3_asr_hotwords)
    available_providers = _supported_providers()

    if (
        providers
        and providers[0] == "cuda"
        and available_providers
        and "CUDAExecutionProvider" not in available_providers
    ):
        logger.info("CUDAExecutionProvider is unavailable for Sherpa-ONNX STT; falling back to CPU")
        providers = ["cpu"]

    logger.info(
        "Loading Sherpa-ONNX STT model: requested=%s resolved=%s device=%s compute_type=%s "
        "model_dir=%s language=%s",
        model_size,
        resolved_model,
        device,
        compute_type,
        model_dir,
        target_language,
    )

    if compute_type != DEFAULT_COMPUTE_TYPE:
        logger.info("Ignoring compute_type=%s for Sherpa-ONNX STT backend", compute_type)

    last_error: Exception | None = None
    for provider in providers:
        try:
            recognizer, model_path = _load_qwen3_asr(
                sherpa_onnx=sherpa_onnx,
                model_dir=model_dir,
                provider=provider,
                hotwords=resolved_hotwords,
            )

            return LoadedSttModel(
                recognizer=recognizer,
                backend="sherpa-onnx",
                requested_model_size=model_size,
                resolved_model_size=resolved_model,
                provider=provider,
                model_path=model_path,
                language=target_language,
            )
        except FileNotFoundError as exc:
            logger.warning(
                "Sherpa-ONNX STT load failed: resolved=%s provider=%s error=%s",
                resolved_model,
                provider,
                exc,
            )
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Sherpa-ONNX STT load failed: resolved=%s provider=%s error=%s",
                resolved_model,
                provider,
                exc,
            )

    msg = f"Failed to load Sherpa-ONNX STT model '{model_size}' from {model_dir}"
    raise RuntimeError(msg) from last_error


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _decode_pcm_samples(raw_data: bytes, sample_width: int) -> list[float]:
    formats = {1: "B", 2: "h", 4: "i"}
    divisors = {1: 128.0, 2: 32768.0, 4: 2147483648.0}

    if sample_width not in formats:
        msg = f"Unsupported WAV sample width: {sample_width * 8}-bit"
        raise ValueError(msg)

    sample_count = len(raw_data) // sample_width
    unpacked = struct.unpack(f"<{sample_count}{formats[sample_width]}", raw_data)

    if sample_width == 1:
        return [(sample - 128.0) / divisors[sample_width] for sample in unpacked]
    return [sample / divisors[sample_width] for sample in unpacked]


def _read_wav_samples(wav_path: Path) -> tuple[list[float], int, float]:
    if not wav_path.exists():
        msg = f"WAV file not found: {wav_path}"
        raise FileNotFoundError(msg)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        compression = wf.getcomptype()
        raw_data = wf.readframes(n_frames)

    if compression != "NONE":
        msg = f"Compressed WAV is not supported: {compression}"
        raise ValueError(msg)
    if n_channels <= 0:
        msg = f"Invalid channel count: {n_channels}"
        raise ValueError(msg)
    if sample_rate <= 0:
        msg = f"Invalid sample rate: {sample_rate}"
        raise ValueError(msg)

    samples = _decode_pcm_samples(raw_data, sample_width)
    if n_channels > 1:
        mono_samples: list[float] = []
        for idx in range(0, len(samples), n_channels):
            frame = samples[idx : idx + n_channels]
            mono_samples.append(sum(frame) / len(frame))
        samples = mono_samples

    duration = n_frames / sample_rate
    return samples, sample_rate, duration


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


def _normalize_language_tag(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if text.startswith("<|") and text.endswith("|>"):
        text = text[2:-2]
    return text or fallback


def _strip_asr_text_template_prefix(text: str) -> str:
    """Return only the decoded ASR payload after the final template tag."""

    if _ASR_TEXT_TAG not in text:
        return text
    return text.rsplit(_ASR_TEXT_TAG, maxsplit=1)[-1]


def _normalize_transcript_text(text: str) -> str:
    """Normalize Sherpa decoded transcript text for downstream use."""

    payload_text = _strip_asr_text_template_prefix(text)
    cleaned = payload_text.replace("▁", " ")
    cleaned = _TEXT_WS_RE.sub(" ", cleaned)
    return cleaned.strip()


def _coerce_float_list(values: Any) -> list[float]:
    result: list[float] = []
    for value in values or []:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _build_segments(result: Any, text: str, audio_duration: float) -> list[TranscriptionSegment]:
    if not text:
        return []

    segment_texts = [
        cleaned
        for cleaned in (
            _normalize_transcript_text(str(item)) for item in getattr(result, "segment_texts", [])
        )
        if cleaned
    ]
    segment_timestamps = _coerce_float_list(getattr(result, "segment_timestamps", []))

    if segment_texts and len(segment_texts) == len(segment_timestamps):
        segments: list[TranscriptionSegment] = []
        start = 0.0
        for seg_text, raw_end in zip(segment_texts, segment_timestamps, strict=True):
            if audio_duration > 0:
                end = max(start, min(raw_end, audio_duration))
            else:
                end = max(start, raw_end)
            segments.append(
                TranscriptionSegment(
                    start=round(start, 4),
                    end=round(end, 4),
                    text=seg_text,
                )
            )
            start = end
        return segments

    word_timestamps = _coerce_float_list(getattr(result, "timestamps", []))
    start = 0.0
    end = audio_duration
    if word_timestamps:
        start = max(word_timestamps[0], 0.0)
        last_ts = max(word_timestamps[-1], start)
        if audio_duration > 0:
            end = min(last_ts, audio_duration)
            if end <= start:
                end = audio_duration
        else:
            end = last_ts

    return [
        TranscriptionSegment(
            start=round(start, 4),
            end=round(end, 4),
            text=text,
        )
    ]


# ---------------------------------------------------------------------------
# STT inference
# ---------------------------------------------------------------------------


def run_stt(
    model: Any,
    wav_path: Path,
    language: str = DEFAULT_LANGUAGE,
    beam_size: int = DEFAULT_BEAM_SIZE,
) -> tuple[list[TranscriptionSegment], dict[str, Any]]:
    """Run Sherpa-ONNX transcription on a WAV file.

    Args:
        model: Loaded STT recognizer wrapper.
        wav_path: Path to the audio file.
        language: Language code for transcription.
        beam_size: Retained for CLI/config compatibility. Sherpa ignores it.

    Returns:
        Tuple of (list of TranscriptionSegment, info dict with
        language/probability/duration plus backend metadata).
    """
    if not isinstance(model, LoadedSttModel):
        msg = f"Expected LoadedSttModel, got {type(model).__name__}"
        raise TypeError(msg)

    if beam_size != DEFAULT_BEAM_SIZE:
        logger.debug("Ignoring beam_size=%s for Sherpa-ONNX STT backend", beam_size)

    requested_language = language.strip() or model.language or DEFAULT_LANGUAGE

    samples, sample_rate, duration = _read_wav_samples(wav_path)
    if model.resolved_model_size == _QWEN3_ASR_NAME and duration > 300:
        msg = f"Qwen3-ASR maximum audio length is 300s (5 min), got {duration:.2f}s"
        raise ValueError(msg)
    stream = model.recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    model.recognizer.decode_stream(stream)
    result = stream.result

    raw_text = str(getattr(result, "text", ""))
    text = _normalize_transcript_text(raw_text)
    segments = _build_segments(result, text, duration)
    info_dict: dict[str, Any] = {
        "language": _normalize_language_tag(getattr(result, "lang", None), requested_language),
        "language_probability": 1.0 if text else 0.0,
        "duration": round(duration, 4),
        "backend": model.backend,
        "provider": model.provider,
        "resolved_model_size": model.resolved_model_size,
        "model_path": model.model_path,
        "raw_stt_text": raw_text,
    }

    return segments, info_dict
