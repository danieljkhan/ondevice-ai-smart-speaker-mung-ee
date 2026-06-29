"""Jetson-only tegrastats memory gate harness."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MUNGI_JETSON_MEMORY"),
    reason="Jetson memory gate - PM-run only",
)

_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")


def _parse_tegrastats_ram_mib(line: str) -> int | None:
    """Extract used RAM in MiB from one tegrastats line."""
    match = _RAM_RE.search(line)
    if match is None:
        return None
    return int(match.group(1))


def _read_meminfo_mib() -> dict[str, int]:
    """Read /proc/meminfo values converted to MiB."""
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if parts and parts[0].isdigit():
            meminfo[key] = int(parts[0]) // 1024
    return meminfo


def _start_tegrastats() -> subprocess.Popen[str]:
    """Start tegrastats at 500 ms cadence."""
    if shutil.which("tegrastats") is None:
        pytest.fail("missing runtime deps: tegrastats")
    return subprocess.Popen(
        ["tegrastats", "--interval", "500"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _stop_tegrastats(process: subprocess.Popen[str]) -> list[str]:
    """Stop tegrastats and return captured output lines."""
    process.terminate()
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _stderr = process.communicate(timeout=5)
    return stdout.splitlines()


def _summary_path(prefix: str) -> Path:
    """Return a timestamped JSON evidence path under the system temp dir."""
    return Path(tempfile.gettempdir()) / f"{prefix}_{int(time.time())}.json"


def test_gemma4_full_pipeline_memory_gate() -> None:
    """Run the PM 20-turn Gemma 4 full-pipeline tegrastats gate."""
    try:
        from core.model_manager import ManagerConfig, ModelManager
        from core.pipeline import ConversationPipeline, PipelineConfig
        from models.stt_runner import _read_wav_samples
    except ImportError as exc:
        pytest.fail(f"missing runtime deps: {exc}")

    bench_dir = Path.home() / "gemma4_bench_wavs"
    wav_paths = sorted(bench_dir.glob("audio_*.wav"))
    assert wav_paths

    random.seed(20260421)
    selected_wavs = [random.choice(wav_paths) for _ in range(20)]
    long_utterance_turns = 0
    max_output_turns = 0
    stage_snapshots: list[dict[str, Any]] = [{"stage": "start", "meminfo": _read_meminfo_mib()}]
    process = _start_tegrastats()

    try:
        manager = ModelManager(
            ManagerConfig(stt_model_size=os.getenv("MUNGI_STT_MODEL_SIZE", "qwen3-asr"))
        )
        manager.initialize()
        pipeline = ConversationPipeline(
            manager,
            PipelineConfig(play_tts_audio=False, llm_max_tokens=256),
        )
        stage_snapshots.append({"stage": "before_turns", "meminfo": _read_meminfo_mib()})

        rows: list[dict[str, Any]] = []
        for index, wav_path in enumerate(selected_wavs, start=1):
            samples, sample_rate, duration_s = _read_wav_samples(wav_path)
            stage_snapshots.append(
                {
                    "stage": f"before_turn_{index}",
                    "wav": str(wav_path),
                    "meminfo": _read_meminfo_mib(),
                }
            )
            result = pipeline.run_turn(samples, sample_rate=sample_rate)
            stage_snapshots.append(
                {
                    "stage": f"after_turn_{index}",
                    "wav": str(wav_path),
                    "meminfo": _read_meminfo_mib(),
                }
            )
            long_utterance_turns += int(duration_s > 15.0)
            max_output_turns += int(result.metrics.llm_tokens >= 230)
            rows.append(
                {
                    "turn": index,
                    "wav": str(wav_path),
                    "duration_s": duration_s,
                    "success": result.success,
                    "llm_tokens": result.metrics.llm_tokens,
                    "total_time_s": result.metrics.total_time_s,
                    "error": result.error,
                }
            )
    finally:
        tegrastats_lines = _stop_tegrastats(process)

    ram_samples = [
        used_mib
        for used_mib in (_parse_tegrastats_ram_mib(line) for line in tegrastats_lines)
        if used_mib is not None
    ]
    assert ram_samples
    peak_ram_mib = max(ram_samples)
    summary = {
        "peak_ram_mib": peak_ram_mib,
        "tegrastats_lines": tegrastats_lines,
        "stage_snapshots": stage_snapshots,
        "long_utterance_turns": long_utterance_turns,
        "max_output_turns": max_output_turns,
        "rows": rows,
    }
    output_path = _summary_path("gemma4_memory_gate")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    assert 5600 <= peak_ram_mib <= 6600
    assert len(rows) == 20
    assert all(row["success"] for row in rows)
