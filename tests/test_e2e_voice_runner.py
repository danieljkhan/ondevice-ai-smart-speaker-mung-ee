"""Tests for the E2E voice runner helpers and CLI contract."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from scripts.e2e_voice_runner import (
    DEFAULT_END_GAP_S,
    DEFAULT_ENERGY_THRESHOLD_DB,
    DEFAULT_INPUT_DEVICE,
    DEFAULT_MAX_CAPTURE_S,
    DEFAULT_MIN_CAPTURE_S,
    DEFAULT_PROBE_WINDOW_S,
    DEFAULT_ROUNDS,
    RunnerState,
    build_parser,
    find_usb_input,
    rms_db,
)


def test_rms_db_silence() -> None:
    """Return negative infinity for silent audio."""
    assert rms_db(np.zeros(128, dtype=np.float32)) == float("-inf")


def test_rms_db_known_signal() -> None:
    """Match the expected RMS level for a known sine wave."""
    sample_rate = 16000
    samples = np.arange(sample_rate, dtype=np.float32)
    sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * samples / sample_rate)

    measured = rms_db(sine)
    expected = 20.0 * math.log10(0.5 / math.sqrt(2.0))

    assert measured == pytest.approx(expected, abs=0.1)


def test_rms_db_loud_signal() -> None:
    """Report 0 dB for a full-scale constant signal."""
    loud = np.ones(256, dtype=np.float32)

    assert rms_db(loud) == pytest.approx(0.0, abs=1e-6)


def test_runner_state_values() -> None:
    """Expose the expected observable runner state values."""
    assert RunnerState.STARTUP.value == "startup"
    assert RunnerState.ARMED.value == "armed"
    assert RunnerState.CAPTURING.value == "capturing"
    assert RunnerState.PROCESS.value == "process"
    assert RunnerState.SHUTDOWN.value == "shutdown"


def test_build_parser_defaults() -> None:
    """Populate CLI defaults from the exported module constants."""
    args = build_parser().parse_args(["--lang", "ko", "--output-dir", "out"])

    assert args.rounds == DEFAULT_ROUNDS
    assert args.energy_threshold_db == DEFAULT_ENERGY_THRESHOLD_DB
    assert args.end_gap_s == DEFAULT_END_GAP_S
    assert args.max_capture_s == DEFAULT_MAX_CAPTURE_S
    assert args.min_capture_s == DEFAULT_MIN_CAPTURE_S
    assert args.probe_window_s == DEFAULT_PROBE_WINDOW_S
    assert args.input_device == DEFAULT_INPUT_DEVICE
    assert args.skip_preflight is False
    assert args.warmup is True


def test_build_parser_custom() -> None:
    """Accept explicit CLI overrides for all public options."""
    args = build_parser().parse_args(
        [
            "--lang",
            "en",
            "--rounds",
            "12",
            "--output-dir",
            "custom",
            "--manifest",
            "manifest.json",
            "--energy-threshold-db",
            "-32.5",
            "--end-gap-s",
            "1.2",
            "--max-capture-s",
            "8.0",
            "--min-capture-s",
            "1.1",
            "--probe-window-s",
            "0.4",
            "--input-device",
            "Mic",
            "--skip-preflight",
            "--no-warmup",
        ],
    )

    assert args.lang == "en"
    assert args.rounds == 12
    assert str(args.output_dir) == "custom"
    assert str(args.manifest) == "manifest.json"
    assert args.energy_threshold_db == -32.5
    assert args.end_gap_s == 1.2
    assert args.max_capture_s == 8.0
    assert args.min_capture_s == 1.1
    assert args.probe_window_s == 0.4
    assert args.input_device == "Mic"
    assert args.skip_preflight is True
    assert args.warmup is False


def test_find_usb_input_not_found() -> None:
    """Raise a clear error when the requested USB input device is absent."""
    devices = [
        {"name": "Built-in Output", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "Laptop Mic", "max_input_channels": 1, "default_samplerate": 44100.0},
    ]

    with patch("sounddevice.query_devices", return_value=devices):
        with pytest.raises(RuntimeError, match="USB input device 'USB PnP Audio Device' not found"):
            find_usb_input(DEFAULT_INPUT_DEVICE)


def test_find_usb_input_found() -> None:
    """Return the expected device tuple for a matching USB microphone."""
    devices = [
        {"name": "Output", "max_input_channels": 0, "default_samplerate": 48000.0},
        {
            "name": "USB PnP Audio Device Analog Stereo",
            "max_input_channels": 4,
            "default_samplerate": 44100.0,
        },
    ]

    with patch("sounddevice.query_devices", return_value=devices):
        assert find_usb_input("usb pnp audio device") == (1, 44100, 2)
