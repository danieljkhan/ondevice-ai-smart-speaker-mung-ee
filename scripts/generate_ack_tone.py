"""Generate the touchscreen tap acknowledgement ding-dong WAV."""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

LOGGER = logging.getLogger("mungi.scripts.generate_ack_tone")

DEFAULT_OUTPUT = Path("assets/sounds/feedback/ack.wav")
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_PEAK_AMPLITUDE = 0.42
DEFAULT_NOTE1_FREQUENCY_HZ = 659.25
DEFAULT_NOTE1_DURATION_MS = 170.0
DEFAULT_NOTE1_ATTACK_MS = 28.0
DEFAULT_NOTE1_TAU_MS = 140.0
DEFAULT_NOTE1_RELEASE_MS = 20.0
DEFAULT_NOTE2_FREQUENCY_HZ = 493.88
DEFAULT_NOTE2_DURATION_MS = 280.0
DEFAULT_NOTE2_ATTACK_MS = 15.0
DEFAULT_NOTE2_TAU_MS = 200.0
DEFAULT_NOTE2_RELEASE_MS = 20.0
INT16_MAX = 32_767.0


@dataclass(frozen=True)
class NoteSpec:
    """Configuration for one normalized acknowledgement tone note."""

    frequency_hz: float
    duration_ms: float
    attack_ms: float
    tau_ms: float
    release_ms: float


@dataclass(frozen=True)
class WavStats:
    """Statistics for generated 16-bit PCM acknowledgement audio."""

    sample_count: int
    duration_seconds: float
    peak_int16: int
    saturated_count: int


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
    """Build the command-line parser for the acknowledgement tone generator."""
    parser = argparse.ArgumentParser(
        description="Generate the deterministic touchscreen tap acknowledgement tone WAV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output WAV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=positive_int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Output sample rate in Hz (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--note1-freq",
        type=positive_float,
        default=DEFAULT_NOTE1_FREQUENCY_HZ,
        help=f"First note frequency in Hz (default: {DEFAULT_NOTE1_FREQUENCY_HZ}).",
    )
    parser.add_argument(
        "--note1-duration-ms",
        type=positive_float,
        default=DEFAULT_NOTE1_DURATION_MS,
        help=f"First note duration in milliseconds (default: {DEFAULT_NOTE1_DURATION_MS}).",
    )
    parser.add_argument(
        "--note1-attack-ms",
        type=positive_float,
        default=DEFAULT_NOTE1_ATTACK_MS,
        help=f"First note attack in milliseconds (default: {DEFAULT_NOTE1_ATTACK_MS}).",
    )
    parser.add_argument(
        "--note1-tau-ms",
        type=positive_float,
        default=DEFAULT_NOTE1_TAU_MS,
        help=f"First note exponential decay tau in milliseconds (default: {DEFAULT_NOTE1_TAU_MS}).",
    )
    parser.add_argument(
        "--note1-release-ms",
        type=positive_float,
        default=DEFAULT_NOTE1_RELEASE_MS,
        help=f"First note release in milliseconds (default: {DEFAULT_NOTE1_RELEASE_MS}).",
    )
    parser.add_argument(
        "--note2-freq",
        type=positive_float,
        default=DEFAULT_NOTE2_FREQUENCY_HZ,
        help=f"Second note frequency in Hz (default: {DEFAULT_NOTE2_FREQUENCY_HZ}).",
    )
    parser.add_argument(
        "--note2-duration-ms",
        type=positive_float,
        default=DEFAULT_NOTE2_DURATION_MS,
        help=f"Second note duration in milliseconds (default: {DEFAULT_NOTE2_DURATION_MS}).",
    )
    parser.add_argument(
        "--note2-attack-ms",
        type=positive_float,
        default=DEFAULT_NOTE2_ATTACK_MS,
        help=f"Second note attack in milliseconds (default: {DEFAULT_NOTE2_ATTACK_MS}).",
    )
    parser.add_argument(
        "--note2-tau-ms",
        type=positive_float,
        default=DEFAULT_NOTE2_TAU_MS,
        help=f"Second note exponential decay tau in milliseconds (default: {DEFAULT_NOTE2_TAU_MS}).",
    )
    parser.add_argument(
        "--note2-release-ms",
        type=positive_float,
        default=DEFAULT_NOTE2_RELEASE_MS,
        help=f"Second note release in milliseconds (default: {DEFAULT_NOTE2_RELEASE_MS}).",
    )
    parser.add_argument(
        "--peak-amplitude",
        type=positive_float,
        default=DEFAULT_PEAK_AMPLITUDE,
        help=f"Peak amplitude in full-scale units (default: {DEFAULT_PEAK_AMPLITUDE}).",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Log the generation plan without writing a WAV file.",
    )
    return parser


def build_envelope(
    frame_count: int,
    sample_rate: int,
    *,
    attack_ms: float,
    tau_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Build a click-free attack, exponential decay, and release envelope."""
    if frame_count <= 0:
        msg = f"frame_count must be > 0, got {frame_count}"
        raise ValueError(msg)

    times = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    envelope = np.ones(frame_count, dtype=np.float64)

    attack_frames = int(sample_rate * attack_ms / 1000.0)
    if attack_frames > 0:
        envelope[: min(attack_frames, frame_count)] = np.linspace(
            0.0,
            1.0,
            min(attack_frames, frame_count),
            endpoint=True,
        )

    envelope *= np.exp(-times / (tau_ms / 1000.0))

    release_frames = int(sample_rate * release_ms / 1000.0)
    if release_frames > 0:
        envelope[-min(release_frames, frame_count) :] *= np.linspace(
            1.0,
            0.0,
            min(release_frames, frame_count),
            endpoint=True,
        )
    return envelope


def generate_note_samples(
    note: NoteSpec,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    peak_amplitude: float = DEFAULT_PEAK_AMPLITUDE,
) -> np.ndarray:
    """Return one deterministic normalized float32 note in [-1.0, 1.0]."""
    if peak_amplitude > 1.0:
        msg = f"peak_amplitude must be <= 1.0, got {peak_amplitude}"
        raise ValueError(msg)

    frame_count = int(sample_rate * note.duration_ms / 1000.0)
    if frame_count <= 0:
        msg = f"note duration produced no frames: {note.duration_ms}"
        raise ValueError(msg)

    times = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    waveform = np.sin(2.0 * np.pi * note.frequency_hz * times)
    waveform *= build_envelope(
        frame_count,
        sample_rate,
        attack_ms=note.attack_ms,
        tau_ms=note.tau_ms,
        release_ms=note.release_ms,
    )

    peak = float(np.max(np.abs(waveform)))
    if peak <= 0.0:
        msg = "generated acknowledgement note is silent"
        raise RuntimeError(msg)

    scaled = waveform / peak * peak_amplitude
    return cast(np.ndarray, scaled.astype(np.float32))


def generate_ack_samples(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    peak_amplitude: float = DEFAULT_PEAK_AMPLITUDE,
    note1: NoteSpec | None = None,
    note2: NoteSpec | None = None,
) -> np.ndarray:
    """Return deterministic float32 mono two-note acknowledgement samples."""
    first_note = note1 or NoteSpec(
        frequency_hz=DEFAULT_NOTE1_FREQUENCY_HZ,
        duration_ms=DEFAULT_NOTE1_DURATION_MS,
        attack_ms=DEFAULT_NOTE1_ATTACK_MS,
        tau_ms=DEFAULT_NOTE1_TAU_MS,
        release_ms=DEFAULT_NOTE1_RELEASE_MS,
    )
    second_note = note2 or NoteSpec(
        frequency_hz=DEFAULT_NOTE2_FREQUENCY_HZ,
        duration_ms=DEFAULT_NOTE2_DURATION_MS,
        attack_ms=DEFAULT_NOTE2_ATTACK_MS,
        tau_ms=DEFAULT_NOTE2_TAU_MS,
        release_ms=DEFAULT_NOTE2_RELEASE_MS,
    )
    return cast(
        np.ndarray,
        np.concatenate(
            [
                generate_note_samples(
                    first_note,
                    sample_rate=sample_rate,
                    peak_amplitude=peak_amplitude,
                ),
                generate_note_samples(
                    second_note,
                    sample_rate=sample_rate,
                    peak_amplitude=peak_amplitude,
                ),
            ],
        ),
    )


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> WavStats:
    """Write mono float samples to a 16-bit PCM WAV file and return output stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(clipped * INT16_MAX).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    pcm16_abs = np.abs(pcm16.astype(np.int32))
    return WavStats(
        sample_count=int(pcm16.size),
        duration_seconds=float(pcm16.size / sample_rate),
        peak_int16=int(np.max(pcm16_abs)) if pcm16_abs.size else 0,
        saturated_count=int(np.count_nonzero(pcm16_abs >= int(INT16_MAX))),
    )


def configure_logging() -> None:
    """Configure CLI logging to stdout for smoke-test capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the acknowledgement tone generator CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)

    output = Path(args.output)
    sample_rate = int(args.sample_rate)
    note1 = NoteSpec(
        frequency_hz=float(args.note1_freq),
        duration_ms=float(args.note1_duration_ms),
        attack_ms=float(args.note1_attack_ms),
        tau_ms=float(args.note1_tau_ms),
        release_ms=float(args.note1_release_ms),
    )
    note2 = NoteSpec(
        frequency_hz=float(args.note2_freq),
        duration_ms=float(args.note2_duration_ms),
        attack_ms=float(args.note2_attack_ms),
        tau_ms=float(args.note2_tau_ms),
        release_ms=float(args.note2_release_ms),
    )
    peak_amplitude = float(args.peak_amplitude)

    LOGGER.info(
        (
            "Ack tone plan: output=%s sample_rate=%d "
            "note1=%.2fHz/%.1fms/attack%.1fms/tau%.1fms/release%.1fms "
            "note2=%.2fHz/%.1fms/attack%.1fms/tau%.1fms/release%.1fms "
            "peak_amplitude=%.2f"
        ),
        output,
        sample_rate,
        note1.frequency_hz,
        note1.duration_ms,
        note1.attack_ms,
        note1.tau_ms,
        note1.release_ms,
        note2.frequency_hz,
        note2.duration_ms,
        note2.attack_ms,
        note2.tau_ms,
        note2.release_ms,
        peak_amplitude,
    )
    if bool(args.dry_run_only):
        LOGGER.info("Dry run only; no WAV file written")
        return 0

    try:
        samples = generate_ack_samples(
            sample_rate=sample_rate,
            peak_amplitude=peak_amplitude,
            note1=note1,
            note2=note2,
        )
        stats = write_wav(output, samples, sample_rate)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Failed to generate acknowledgement tone WAV: %s", exc)
        return 1

    LOGGER.info(
        (
            "WAV saved: %s (samples=%d duration=%.3fs sample_rate=%d "
            "peak_int16=%d saturated_count=%d)"
        ),
        output,
        stats.sample_count,
        stats.duration_seconds,
        sample_rate,
        stats.peak_int16,
        stats.saturated_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
