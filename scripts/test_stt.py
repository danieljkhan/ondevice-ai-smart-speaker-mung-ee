"""Sherpa-ONNX STT standalone test script for Jetson Orin Nano.

Loads a Sherpa-ONNX model, runs speech-to-text transcription on a WAV
file, and reports recognized text with timing and memory metrics.

Usage:
    python scripts/test_stt.py /path/to/test.wav
    python scripts/test_stt.py /path/to/test.wav --model-size sense-voice
    python scripts/test_stt.py /path/to/test.wav --device cpu --compute-type float32
    python scripts/test_stt.py /path/to/test.wav --model-dir /opt/mungi/ai_models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.stt_runner import (  # noqa: E402
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_SIZE,
    load_stt_model,
    run_stt,
)
from scripts.utils import get_peak_memory_kb  # noqa: E402

logger = logging.getLogger("mungi.scripts.test_stt")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class STTResult:
    """Full result of an STT run including transcription and metrics."""

    segments: list[dict[str, Any]]
    full_text: str
    model_load_time_s: float
    inference_time_s: float
    peak_memory_kb: int
    audio_duration_s: float
    rtf: float
    detected_language: str
    language_probability: float
    model_size: str
    device: str
    compute_type: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------


def validate_wav_file(wav_path: Path) -> float:
    """Validate that a WAV file exists and return its duration in seconds.

    The file must be readable by Sherpa-ONNX. This function performs
    a basic existence check and reads the duration via the wave module.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Audio duration in seconds.

    Raises:
        FileNotFoundError: If the WAV file does not exist.
        ValueError: If the file cannot be read as a valid WAV.
    """
    import wave

    if not wav_path.exists():
        msg = f"WAV file not found: {wav_path}"
        raise FileNotFoundError(msg)

    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            if framerate <= 0:
                msg = f"Invalid framerate: {framerate}"
                raise ValueError(msg)
            return n_frames / framerate
    except wave.Error as exc:
        msg = f"Cannot read WAV file: {wav_path} ({exc})"
        raise ValueError(msg) from exc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the STT test script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=("Sherpa-ONNX STT standalone test -- load model and transcribe a WAV file."),
    )
    parser.add_argument(
        "wav_path",
        type=Path,
        help="Path to a WAV audio file (16kHz mono recommended).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        help=(
            f"Sherpa STT model selector (default: {DEFAULT_MODEL_SIZE}). "
            "Examples: small, sense-voice, moonshine-tiny-ko."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device for inference (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        help=(
            f"Compatibility-only compute type flag (default: {DEFAULT_COMPUTE_TYPE}). "
            "Sherpa-ONNX ignores this value."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=(f"Directory for model cache/download (default: {DEFAULT_MODEL_DIR})."),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help=(f"Language code for transcription (default: {DEFAULT_LANGUAGE})."),
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size for decoding (default: {DEFAULT_BEAM_SIZE}).",
    )
    return parser


def main() -> int:
    """Run the Sherpa-ONNX STT test and report results.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    # Validate audio file
    try:
        logger.info("Validating WAV file: %s", args.wav_path)
        audio_duration = validate_wav_file(args.wav_path)
        logger.info("Audio duration: %.2f seconds", audio_duration)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Audio validation failed: %s", exc)
        return 1

    # Load model
    try:
        mem_before = get_peak_memory_kb()
        t0 = time.monotonic()
        model = load_stt_model(
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            model_dir=args.model_dir,
            language=args.language,
        )
        load_time = time.monotonic() - t0
        logger.info("Model loaded in %.3f seconds", load_time)
        logger.info(
            "STT backend: %s (resolved=%s, provider=%s)",
            getattr(model, "backend", "unknown"),
            getattr(model, "resolved_model_size", args.model_size),
            getattr(model, "provider", args.device),
        )
    except (ImportError, RuntimeError, OSError) as exc:
        logger.error("Model load failed: %s", exc)
        return 1

    # Run inference
    try:
        t0 = time.monotonic()
        segments, info_dict = run_stt(
            model,
            args.wav_path,
            language=args.language,
            beam_size=args.beam_size,
        )
        inference_time = time.monotonic() - t0
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        return 1

    peak_memory = get_peak_memory_kb()

    # Assemble full text
    full_text = " ".join(seg.text for seg in segments)

    # Compute real-time factor
    rtf = round(inference_time / audio_duration, 4) if audio_duration > 0 else 0.0

    # Build result
    result = STTResult(
        segments=[seg.to_dict() for seg in segments],
        full_text=full_text,
        model_load_time_s=round(load_time, 4),
        inference_time_s=round(inference_time, 4),
        peak_memory_kb=max(peak_memory - mem_before, 0),
        audio_duration_s=round(audio_duration, 4),
        rtf=rtf,
        detected_language=info_dict.get("language", args.language),
        language_probability=info_dict.get("language_probability", 0.0),
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    # Report
    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    logger.info("STT result:\n%s", output)

    logger.info("--- Summary ---")
    logger.info("Audio duration:      %.2f s", audio_duration)
    logger.info("Model size:          %s", args.model_size)
    logger.info("Device:              %s", args.device)
    logger.info("Compute type:        %s", args.compute_type)
    logger.info(
        "Backend:             %s (%s via %s)",
        info_dict.get("backend", "unknown"),
        info_dict.get("resolved_model_size", args.model_size),
        info_dict.get("provider", args.device),
    )
    logger.info("Model load time:     %.3f s", load_time)
    logger.info("Inference time:      %.3f s", inference_time)
    logger.info("RTF:                 %.4f", rtf)
    logger.info("Peak memory delta:   %d KB", result.peak_memory_kb)
    logger.info(
        "Detected language:   %s (prob: %.2f%%)",
        result.detected_language,
        result.language_probability * 100,
    )
    logger.info("Segments:            %d", len(segments))
    logger.info("Full text:           %s", full_text)

    for i, seg in enumerate(segments, 1):
        logger.info(
            "  Segment %d: [%.2fs -> %.2fs] %s",
            i,
            seg.start,
            seg.end,
            seg.text,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
