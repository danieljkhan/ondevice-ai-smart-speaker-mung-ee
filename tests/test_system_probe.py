"""Tests for system probe helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core import system_probe


def install_sounddevice(monkeypatch: Any, devices: list[dict[str, Any]]) -> None:
    """Install a fake sounddevice module with query_devices."""
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(query_devices=lambda: devices))


def test_find_usb_input_honors_env_override(monkeypatch: Any) -> None:
    """The env override takes precedence over the function argument."""
    install_sounddevice(
        monkeypatch,
        [
            {"name": "USB PnP Audio Device", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "Custom Touch Mic", "max_input_channels": 4, "default_samplerate": 44100},
        ],
    )
    monkeypatch.setenv("MUNGI_AUDIO_INPUT_DEVICE", "Custom")

    result = system_probe.find_usb_input("USB PnP")

    assert result == {
        "index": 1,
        "name": "Custom Touch Mic",
        "sample_rate": 44_100,
        "channels": 2,
    }


def test_find_usb_input_uses_default_name(monkeypatch: Any) -> None:
    """The default USB input name is used when no override is provided."""
    monkeypatch.delenv("MUNGI_AUDIO_INPUT_DEVICE", raising=False)
    install_sounddevice(
        monkeypatch,
        [
            {"name": "Other", "max_input_channels": 1, "default_samplerate": 16000},
            {
                "name": system_probe.DEFAULT_INPUT_DEVICE_NAME,
                "max_input_channels": 1,
                "default_samplerate": 48000,
            },
        ],
    )

    result = system_probe.find_usb_input()

    assert result is not None
    assert result["index"] == 1
    assert result["channels"] == 1


def test_find_usb_input_returns_none_when_no_match(monkeypatch: Any) -> None:
    """No input-capable device returns None."""
    monkeypatch.delenv("MUNGI_AUDIO_INPUT_DEVICE", raising=False)
    install_sounddevice(
        monkeypatch,
        [{"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000}],
    )

    assert system_probe.find_usb_input("missing") is None


def test_find_usb_input_falls_back_to_first_input_device(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Missing named input falls back to the first input-capable device."""
    monkeypatch.delenv("MUNGI_AUDIO_INPUT_DEVICE", raising=False)
    install_sounddevice(
        monkeypatch,
        [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Fallback Mic", "max_input_channels": 1, "default_samplerate": 16000},
            {"name": "Second Mic", "max_input_channels": 4, "default_samplerate": 44100},
        ],
    )
    caplog.set_level(logging.WARNING, logger="core.system_probe")

    result = system_probe.find_usb_input("missing")

    assert result == {
        "index": 1,
        "name": "Fallback Mic",
        "sample_rate": 16_000,
        "channels": 1,
    }
    assert "falling back" in caplog.text


def test_find_usb_input_can_disable_fallback(monkeypatch: Any) -> None:
    """Callers can request named-device lookup without fallback."""
    monkeypatch.delenv("MUNGI_AUDIO_INPUT_DEVICE", raising=False)
    install_sounddevice(
        monkeypatch,
        [{"name": "Fallback Mic", "max_input_channels": 1, "default_samplerate": 16000}],
    )

    assert system_probe.find_usb_input("missing", allow_fallback=False) is None


def test_read_max_thermal_c(monkeypatch: Any, tmp_path: Path) -> None:
    """Thermal probe returns the highest valid zone value in Celsius."""
    thermal_root = tmp_path / "thermal"
    (thermal_root / "thermal_zone0").mkdir(parents=True)
    (thermal_root / "thermal_zone1").mkdir()
    (thermal_root / "thermal_zone0" / "temp").write_text("42000\n", encoding="utf-8")
    (thermal_root / "thermal_zone1" / "temp").write_text("55000\n", encoding="utf-8")
    monkeypatch.setattr(system_probe, "THERMAL_ROOT", thermal_root)

    assert system_probe.read_max_thermal_c() == 55.0


def test_read_meminfo_snapshot(monkeypatch: Any, tmp_path: Path) -> None:
    """Meminfo parsing keeps numeric kB fields."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree:       123 kB\nCached: abc kB\nBuffers: 456 kB\n", encoding="utf-8")
    monkeypatch.setattr(system_probe, "MEMINFO_PATH", meminfo)

    assert system_probe.read_meminfo_snapshot() == {"MemFree": 123, "Buffers": 456}
