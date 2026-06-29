"""Tests for the touchscreen chime generator."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

from scripts import generate_chime_audio


def _md5(path: Path) -> str:
    """Return the MD5 digest for a small test artifact."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _read_pcm16(path: Path) -> tuple[np.ndarray, int, int, int]:
    """Read mono PCM16 WAV data for assertions."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32_767.0
    return samples, sample_rate, channels, sample_width


def test_chime_wav_is_deterministic(tmp_path: Path) -> None:
    """Two runs with identical args should produce byte-identical WAV files."""
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"

    assert generate_chime_audio.main(["--output", str(first)]) == 0
    assert generate_chime_audio.main(["--output", str(second)]) == 0

    assert _md5(first) == _md5(second)


def test_chime_wav_format_and_level(tmp_path: Path) -> None:
    """Generated chime should be 44.1 kHz mono PCM16 with a -6 dB peak."""
    output = tmp_path / "long_press_chime.wav"

    assert generate_chime_audio.main(["--output", str(output)]) == 0

    samples, sample_rate, channels, sample_width = _read_pcm16(output)
    assert sample_rate == 44_100
    assert channels == 1
    assert sample_width == 2
    assert len(samples) == int(0.20 * 44_100)
    assert float(np.max(np.abs(samples))) == pytest_approx_db_level(-6.0)


def test_dry_run_only_does_not_write(tmp_path: Path) -> None:
    """Dry-run mode should validate CLI args without writing the target file."""
    output = tmp_path / "dry_run.wav"

    assert generate_chime_audio.main(["--output", str(output), "--dry-run-only"]) == 0

    assert not output.exists()


def test_chime_parser_defaults() -> None:
    """Parser defaults should match the Phase 1 chime contract."""
    parser = generate_chime_audio.build_parser()
    args = parser.parse_args([])

    assert args.output == Path("assets/sounds/feedback/long_press_chime.wav")
    assert args.duration_s == 0.20
    assert args.frequency_hz == 880.0
    assert args.sample_rate == 44_100
    assert args.db_vs_voice == -6.0


def pytest_approx_db_level(db_value: float) -> object:
    """Return a pytest-compatible comparator for a dB full-scale peak."""
    import pytest

    target = 10.0 ** (db_value / 20.0)
    quantization_tolerance = 2.0 / 32_767.0
    return pytest.approx(target, abs=quantization_tolerance)
