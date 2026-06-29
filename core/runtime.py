from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _normalize_path(raw: str) -> str:
    return str(Path(raw)).replace("\\", "/")


@dataclass(frozen=True)
class RuntimePaths:
    """Immutable container for Mungi runtime directory paths."""

    source_root: str
    mutable_root: str
    log_root: str
    model_root: str

    def to_dict(self) -> dict[str, str]:
        """Return all paths as a plain dictionary."""
        return asdict(self)


def detect_runtime_paths() -> RuntimePaths:
    """Detect runtime paths with defaults for the split Jetson setup."""
    source_root = os.getenv("MUNGI_SOURCE_ROOT", "/opt/mungi-repo")
    mutable_root = os.getenv("MUNGI_MUTABLE_ROOT", "/var/lib/mungi")
    log_root = os.getenv("MUNGI_LOG_ROOT", "/var/log/mungi")
    model_root = os.getenv("MUNGI_MODEL_ROOT", "/opt/mungi/ai_models")

    return RuntimePaths(
        source_root=_normalize_path(source_root),
        mutable_root=_normalize_path(mutable_root),
        log_root=_normalize_path(log_root),
        model_root=_normalize_path(model_root),
    )
