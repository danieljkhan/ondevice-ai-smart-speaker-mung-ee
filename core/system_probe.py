"""Small runtime probes used by live scripts and diagnostics."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DEVICE_NAME = "USB PnP Audio Device"
THERMAL_ROOT = Path("/sys/class/thermal")
MEMINFO_PATH = Path("/proc/meminfo")


def find_usb_input(
    device_name: str | None = None,
    *,
    allow_fallback: bool = True,
) -> dict[str, Any] | None:
    """Find an input-capable sounddevice by configured/default name or fallback."""
    import sounddevice as sd  # type: ignore[import-not-found, import-untyped]

    requested = (
        os.getenv("MUNGI_AUDIO_INPUT_DEVICE", "").strip()
        or (device_name or "").strip()
        or DEFAULT_INPUT_DEVICE_NAME
    )
    devices = sd.query_devices()
    first_input: dict[str, Any] | None = None
    for idx, info in enumerate(devices):
        max_channels = int(info.get("max_input_channels", 0))
        if max_channels <= 0:
            continue
        name = str(info.get("name", ""))
        device_info = {
            "index": idx,
            "name": name,
            "sample_rate": int(float(info.get("default_samplerate", 48_000))),
            "channels": min(max_channels, 2),
        }
        if first_input is None:
            first_input = device_info
        if requested.lower() not in name.lower():
            continue
        return device_info
    if allow_fallback and first_input is not None:
        logger.warning(
            "Input device %r not found; falling back to input-capable device %r (index=%d)",
            requested,
            first_input["name"],
            first_input["index"],
        )
        return first_input
    return None


def read_max_thermal_c() -> float | None:
    """Return the highest thermal-zone temperature in Celsius."""
    readings: list[float] = []
    for temp_path in THERMAL_ROOT.glob("thermal_zone*/temp"):
        try:
            raw = temp_path.read_text(encoding="utf-8").strip()
            readings.append(int(raw) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(readings) if readings else None


def read_meminfo_snapshot() -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a mapping of field names to integer kB values."""
    snapshot: dict[str, int] = {}
    try:
        lines = MEMINFO_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return snapshot
    for line in lines:
        if ":" not in line:
            continue
        name, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            snapshot[name] = int(parts[0])
        except ValueError:
            continue
    return snapshot
