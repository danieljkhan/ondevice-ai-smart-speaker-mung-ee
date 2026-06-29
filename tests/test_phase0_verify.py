from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime import detect_runtime_paths
from scripts.phase0_verify import build_report


def test_runtime_defaults() -> None:
    runtime = detect_runtime_paths().to_dict()
    assert runtime["source_root"] == "/opt/mungi-repo"
    assert runtime["mutable_root"] == "/var/lib/mungi"
    assert runtime["log_root"] == "/var/log/mungi"
    assert runtime["model_root"] == "/opt/mungi/ai_models"


def test_phase0_report_shape() -> None:
    report = build_report()
    assert "runtime_paths" in report
    assert "onnxruntime" in report
    assert "torch" in report
    assert "jetson" in report
