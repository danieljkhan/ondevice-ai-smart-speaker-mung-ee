from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.runtime import detect_runtime_paths
from hardware.jetson_probe import probe_jetson
from models.inference_probe import (
    probe_onnxruntime,
    probe_optional_inference_libs,
    probe_torch_cuda,
)

logger = logging.getLogger(__name__)


def build_report() -> dict[str, object]:
    runtime_paths = detect_runtime_paths()
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_paths": runtime_paths.to_dict(),
        "jetson": probe_jetson(),
        "torch": probe_torch_cuda(),
        "onnxruntime": probe_onnxruntime(),
        "optional_inference": probe_optional_inference_libs(),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mungi Phase 0 environment verification (Jetson-oriented)."
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output JSON path. Example: reports/phase0-verify.json",
    )
    args = parser.parse_args()

    report = build_report()
    logger.info(json.dumps(report, indent=2, ensure_ascii=False))

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved report: %s", args.save)

    onnx = report["onnxruntime"]
    has_cuda = bool(isinstance(onnx, dict) and onnx.get("has_cuda_provider", False))
    return 0 if has_cuda else 2


if __name__ == "__main__":
    raise SystemExit(main())
