from __future__ import annotations

from pathlib import Path
from typing import Any


def _read_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def probe_jetson() -> dict[str, Any]:
    """Return Jetson probe details: device model, nv_tegra_release, and is_jetson."""
    model = _read_optional(Path("/proc/device-tree/model"))
    release = _read_optional(Path("/etc/nv_tegra_release"))
    return {
        "device_model": model,
        "nv_tegra_release": release,
        "is_jetson": bool(model and "jetson" in model.lower()),
    }
