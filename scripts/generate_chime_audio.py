"""Generate the touchscreen long-press feedback chime WAV."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

LOGGER = logging.getLogger("mungi.scripts.generate_chime_audio")

DEFAULT_OUTPUT = Path("assets/sounds/feedback/long_press_chime.wav")
DEFAULT_DURATION_S = 0.20
DEFAULT_FREQUENCY_HZ = 880.0
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_DB_VS_VOICE = -6.0
FADE_DURATION_S = 0.02
INT16_MAX = 32_767.0


def positive_float(raw_value: str) -> float:
    """Parse a positive float for argparse."""
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        msg = f"must be a number, got {raw_value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed <= 0.0:
        msg = f"must be > 0, got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def positive_int(raw_value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        msg = f"must be an integer, got {raw_value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed <= 0:
        msg = f"must be > 0, got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the chime generator."""
    parser = argparse.ArgumentParser(
        description="Generate the deterministic touchscreen long-press chime WAV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output WAV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--duration-s",
        type=positive_float,
        default=DEFAULT_DURATION_S,
        help=f"Chime duration in seconds (default: {DEFAULT_DURATION_S}).",
    )
    parser.add_argument(
        "--frequency-hz",
        type=positive_float,
        default=DEFAULT_FREQUENCY_HZ,
        help=f"Sine frequency in Hz (default: {DEFAULT_FREQUENCY_HZ}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=positive_int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Output sample rate in Hz (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--db-vs-voice",
        type=float,
        default=DEFAULT_DB_VS_VOICE,
        help=f"Peak level relative to voice full scale in dB (default: {DEFAULT_DB_VS_VOICE}).",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Log the generation plan without writing a WAV file.",
    )
    return parser


def build_envelope(frame_count: int, sample_rate: int) -> np.ndarray:
    """Build a short linear attack/release envelope for a click-free chime."""
    envelope = np.ones(frame_count, dtype=np.float64)
    fade_frames = int(round(FADE_DURATION_S * sample_rate))
    fade_frames = max(1, min(fade_frames, frame_count // 2))
    envelope[:fade_frames] = np.linspace(0.0, 1.0, fade_frames, endpoint=False)
    envelope[-fade_frames:] = np.linspace(1.0, 0.0, fade_frames, endpoint=True)
    return envelope


def generate_chime_samples(
    *,
    duration_s: float = DEFAULT_DURATION_S,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    db_vs_voice: float = DEFAULT_DB_VS_VOICE,
) -> np.ndarray:
    """Return deterministic float32 mono chime samples in [-1.0, 1.0]."""
    frame_count = int(round(duration_s * sample_rate))
    if frame_count <= 0:
        msg = f"duration_s produced no frames: {duration_s}"
        raise ValueError(msg)

    times = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    waveform = np.sin(2.0 * math.pi * frequency_hz * times)
    waveform *= build_envelope(frame_count, sample_rate)

    peak = float(np.max(np.abs(waveform)))
    if peak <= 0.0:
        msg = "generated chime waveform is silent"
        raise RuntimeError(msg)

    target_peak = 10.0 ** (db_vs_voice / 20.0)
    scaled = waveform / peak * target_peak
    return cast(np.ndarray, scaled.astype(np.float32))


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(clipped * INT16_MAX).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def configure_logging() -> None:
    """Configure CLI logging to stdout for smoke-test capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chime generator CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)

    output = Path(args.output)
    duration_s = float(args.duration_s)
    frequency_hz = float(args.frequency_hz)
    sample_rate = int(args.sample_rate)
    db_vs_voice = float(args.db_vs_voice)

    LOGGER.info(
        "Chime plan: output=%s duration_s=%.3f frequency_hz=%.1f sample_rate=%d db_vs_voice=%.1f",
        output,
        duration_s,
        frequency_hz,
        sample_rate,
        db_vs_voice,
    )
    if bool(args.dry_run_only):
        LOGGER.info("Dry run only; no WAV file written")
        return 0

    try:
        samples = generate_chime_samples(
            duration_s=duration_s,
            frequency_hz=frequency_hz,
            sample_rate=sample_rate,
            db_vs_voice=db_vs_voice,
        )
        write_wav(output, samples, sample_rate)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Failed to generate chime WAV: %s", exc)
        return 1

    LOGGER.info("WAV saved: %s (%d samples, %d Hz)", output, len(samples), sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
