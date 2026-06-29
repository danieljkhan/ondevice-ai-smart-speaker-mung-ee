"""Generate the non-verbal language-switch cue WAV."""

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

LOGGER = logging.getLogger("mungi.scripts.generate_language_switch_cue")

DEFAULT_ASSET_DIR = Path("assets") / "sounds" / "feedback"
OUTPUT_SUBDIR = "language_switch"
OUTPUT_FILENAME = "switch_01.wav"
DEFAULT_PEAK_AMPLITUDE = 0.30
INT16_MAX = 32_767.0


@dataclass(frozen=True)
class NoteSpec:
    """Configuration for one normalized language-switch cue note."""

    frequency_hz: float
    duration_ms: float
    attack_ms: float
    tau_ms: float
    release_ms: float


@dataclass(frozen=True)
class WavStats:
    """Statistics for generated 16-bit PCM cue audio."""

    sample_count: int
    duration_seconds: float
    peak_int16: int
    saturated_count: int


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for language-switch cue generation."""
    parser = argparse.ArgumentParser(
        description="Generate the short non-verbal KO/EN language-switch cue WAV.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help=f"Feedback asset directory containing ack.wav (default: {DEFAULT_ASSET_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing language-switch cue WAV.",
    )
    return parser


def read_wav_sample_rate(path: Path) -> int:
    """Read the sample rate from a PCM WAV file."""
    with wave.open(str(path), "rb") as wav_file:
        return int(wav_file.getframerate())


def build_envelope(
    frame_count: int,
    sample_rate: int,
    *,
    attack_ms: float,
    tau_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Build a smooth attack, exponential decay, and release envelope."""
    if frame_count <= 0:
        msg = f"frame_count must be > 0, got {frame_count}"
        raise ValueError(msg)

    times = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    envelope = np.exp(-times / (tau_ms / 1000.0))

    attack_frames = int(round(sample_rate * attack_ms / 1000.0))
    if attack_frames > 0:
        width = min(attack_frames, frame_count)
        envelope[:width] *= np.sin(np.linspace(0.0, np.pi / 2.0, width, endpoint=True))

    release_frames = int(round(sample_rate * release_ms / 1000.0))
    if release_frames > 0:
        width = min(release_frames, frame_count)
        envelope[-width:] *= np.cos(np.linspace(0.0, np.pi / 2.0, width, endpoint=True))
    return cast(np.ndarray, envelope)


def generate_note_samples(
    note: NoteSpec,
    *,
    sample_rate: int,
    peak_amplitude: float = DEFAULT_PEAK_AMPLITUDE,
) -> np.ndarray:
    """Return one deterministic float32 sine note in [-1.0, 1.0]."""
    frame_count = int(round(sample_rate * note.duration_ms / 1000.0))
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
        msg = "generated language-switch note is silent"
        raise RuntimeError(msg)
    return cast(np.ndarray, (waveform / peak * peak_amplitude).astype(np.float32))


def generate_language_switch_samples(sample_rate: int) -> np.ndarray:
    """Return deterministic mono samples for a warm ascending two-note cue."""
    first_note = NoteSpec(
        frequency_hz=523.25,
        duration_ms=170.0,
        attack_ms=22.0,
        tau_ms=150.0,
        release_ms=28.0,
    )
    second_note = NoteSpec(
        frequency_hz=659.25,
        duration_ms=240.0,
        attack_ms=20.0,
        tau_ms=210.0,
        release_ms=45.0,
    )
    gap = np.zeros(int(round(sample_rate * 0.025)), dtype=np.float32)
    return cast(
        np.ndarray,
        np.concatenate(
            [
                generate_note_samples(first_note, sample_rate=sample_rate),
                gap,
                generate_note_samples(second_note, sample_rate=sample_rate),
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


def generate_language_switch_cue(asset_dir: Path, *, force: bool = False) -> Path:
    """Generate the language-switch cue asset and return the written path."""
    ack_path = asset_dir / "ack.wav"
    output_path = asset_dir / OUTPUT_SUBDIR / OUTPUT_FILENAME
    if output_path.exists() and not force:
        LOGGER.info("skipped existing cue %s", output_path)
        return output_path

    sample_rate = read_wav_sample_rate(ack_path)
    samples = generate_language_switch_samples(sample_rate)
    stats = write_wav(output_path, samples, sample_rate)
    LOGGER.info(
        "WAV saved: %s (samples=%d duration=%.3fs sample_rate=%d peak_int16=%d saturated=%d)",
        output_path,
        stats.sample_count,
        stats.duration_seconds,
        sample_rate,
        stats.peak_int16,
        stats.saturated_count,
    )
    return output_path


def configure_logging() -> None:
    """Configure CLI logging to stdout for smoke-test capture."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the language-switch cue generator CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        generate_language_switch_cue(Path(args.asset_dir), force=bool(args.force))
    except (OSError, RuntimeError, ValueError, wave.Error) as exc:
        LOGGER.error("Failed to generate language-switch cue: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
