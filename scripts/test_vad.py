"""Silero VAD standalone test script for Jetson Orin Nano.

Loads a Silero VAD model, runs voice activity detection on a WAV file,
and reports detected speech segments with timing and memory metrics.

Usage:
    python scripts/test_vad.py /path/to/test.wav
    python scripts/test_vad.py /path/to/test.wav --model-path /opt/mungi/ai_models/silero_vad.jit
"""

from __future__ import annotations

import argparse
import logging
import struct
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.vad_runner import (
    DEFAULT_THRESHOLD,
    MIN_SILENCE_DURATION_MS,
    MIN_SPEECH_DURATION_MS,
    SAMPLE_RATE,
    load_vad_model,
    run_vad,
)
from scripts.utils import get_peak_memory_kb

logger = logging.getLogger("mungi.scripts.test_vad")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VADResult:
    """Full result of a VAD run including segments and metrics."""

    segments: list[dict[str, float]]
    model_load_time_s: float
    inference_time_s: float
    peak_memory_kb: int
    audio_duration_s: float
    total_speech_s: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------


def read_wav_mono_16k(wav_path: Path) -> list[float]:
    """Read a WAV file and return audio samples as float32 in [-1, 1].

    Validates that the file is 16kHz mono 16-bit PCM.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        List of float samples normalized to [-1.0, 1.0].

    Raises:
        ValueError: If the WAV format is not 16kHz mono.
        FileNotFoundError: If the WAV file does not exist.
    """
    if not wav_path.exists():
        msg = f"WAV file not found: {wav_path}"
        raise FileNotFoundError(msg)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        if n_channels != 1:
            msg = f"Expected mono audio, got {n_channels} channels"
            raise ValueError(msg)
        if framerate != SAMPLE_RATE:
            msg = f"Expected {SAMPLE_RATE}Hz, got {framerate}Hz"
            raise ValueError(msg)
        if sample_width != 2:
            msg = f"Expected 16-bit audio, got {sample_width * 8}-bit"
            raise ValueError(msg)

        raw_data = wf.readframes(n_frames)

    # Unpack 16-bit signed integers
    sample_count = len(raw_data) // 2
    samples_int = struct.unpack(f"<{sample_count}h", raw_data)

    # Normalize to [-1.0, 1.0]
    return [s / 32768.0 for s in samples_int]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the VAD test script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Silero VAD standalone test — load model and detect speech in a WAV file.",
    )
    parser.add_argument(
        "wav_path",
        type=Path,
        help="Path to a 16kHz mono WAV file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a local silero_vad.jit file. If not provided, downloads via torch.hub.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Speech probability threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=MIN_SPEECH_DURATION_MS,
        help=f"Minimum speech segment duration in ms (default: {MIN_SPEECH_DURATION_MS}).",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=MIN_SILENCE_DURATION_MS,
        help=f"Minimum silence to split segments in ms (default: {MIN_SILENCE_DURATION_MS}).",
    )
    return parser


def main() -> int:
    """Run the Silero VAD test and report results.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    # Read audio
    try:
        logger.info("Reading WAV file: %s", args.wav_path)
        audio_samples = read_wav_mono_16k(args.wav_path)
        audio_duration = len(audio_samples) / SAMPLE_RATE
        logger.info(
            "Audio loaded: %.2f seconds, %d samples",
            audio_duration,
            len(audio_samples),
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Audio read failed: %s", exc)
        return 1

    # Load model
    try:
        mem_before = get_peak_memory_kb()
        t0 = time.monotonic()
        model = load_vad_model(args.model_path)
        load_time = time.monotonic() - t0
        logger.info("Model loaded in %.3f seconds", load_time)
    except (ImportError, FileNotFoundError, OSError) as exc:
        logger.error("Model load failed: %s", exc)
        return 1

    # Run inference
    t0 = time.monotonic()
    segments = run_vad(
        audio_samples,
        model,
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
    )
    inference_time = time.monotonic() - t0
    peak_memory = get_peak_memory_kb()

    total_speech = sum(seg.duration_ms() for seg in segments) / 1000.0

    # Build result
    result = VADResult(
        segments=[seg.to_dict() for seg in segments],
        model_load_time_s=round(load_time, 4),
        inference_time_s=round(inference_time, 4),
        peak_memory_kb=max(peak_memory - mem_before, 0),
        audio_duration_s=round(audio_duration, 4),
        total_speech_s=round(total_speech, 4),
    )

    # Report
    import json

    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    logger.info("VAD result:\n%s", output)

    logger.info("--- Summary ---")
    logger.info("Audio duration:  %.2f s", audio_duration)
    logger.info("Speech segments: %d", len(segments))
    logger.info("Total speech:    %.2f s", total_speech)
    logger.info("Model load time: %.3f s", load_time)
    logger.info("Inference time:  %.3f s", inference_time)
    logger.info("Peak memory:     %d KB", result.peak_memory_kb)

    for i, seg in enumerate(segments, 1):
        logger.info(
            "  Segment %d: %.3f s - %.3f s (%.0f ms)",
            i,
            seg.start,
            seg.end,
            seg.duration_ms(),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
