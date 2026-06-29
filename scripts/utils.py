"""Shared utility functions for Mungi test and benchmark scripts."""

from __future__ import annotations

from pathlib import Path


def get_peak_memory_kb() -> int:
    """Read peak resident set size from /proc/self/status on Linux.

    Returns 0 on non-Linux platforms where /proc is unavailable.
    """
    try:
        status_path = Path("/proc/self/status")
        if not status_path.exists():
            return 0
        text = status_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (OSError, ValueError):
        pass
    return 0
