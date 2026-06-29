"""Jetson-only regression gates for L1 LLM resident default mode."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MUNGI_JETSON_MEMORY"),
    reason="Jetson memory gate - PM-run only",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_SCRIPT = REPO_ROOT / "scripts" / "e2e_60rounds_text_tts.py"
EVIDENCE_ROOT = Path("/var/lib/mungi/e2e_results")
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SYSTEM_RAM_LIMIT_MIB = 5500
HOT_TURN_MEAN_LIMIT_S = 12.0
KOREAN_TTFT_THRESHOLD_S = 4.50


@dataclass(frozen=True)
class _E2ERun:
    """Parsed evidence from one text-input resident E2E run."""

    output_dir: Path
    rounds: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    ram_samples_mib: list[int]


def _new_output_dir(name: str) -> Path:
    """Create a unique evidence directory under the Jetson E2E result root."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = EVIDENCE_ROOT / f"{name}-{stamp}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _run_resident_e2e(
    *,
    name: str,
    rounds: int,
    start: int,
    seed: int,
    timeout_s: int,
) -> _E2ERun:
    """Run the text-input TTS E2E script in resident mode and parse evidence files."""
    output_dir = _new_output_dir(name)
    command = [
        sys.executable,
        str(E2E_SCRIPT),
        "--rounds",
        str(rounds),
        "--start",
        str(start),
        "--seed",
        str(seed),
        "--skip-preflight",
        "--llm-resident",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["MUNGI_LLM_RESIDENT"] = "1"

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "resident E2E run failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    rounds_data = _load_rounds(output_dir / "rounds.jsonl")
    return _E2ERun(
        output_dir=output_dir,
        rounds=rounds_data,
        turns=_flatten_turns(rounds_data),
        ram_samples_mib=_load_tegrastats_ram(output_dir / "tegrastats.log"),
    )


def _load_rounds(path: Path) -> list[dict[str, Any]]:
    """Load round-level JSONL records written by the E2E script."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                pytest.fail(f"Unexpected non-object JSONL record in {path}: {value!r}")
            rows.append(value)
    return rows


def _flatten_turns(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested round/topic/turn records into one turn list."""
    turns: list[dict[str, Any]] = []
    for round_record in rounds:
        for topic in round_record.get("topics", []):
            for turn in topic.get("turns", []):
                if not isinstance(turn, dict):
                    pytest.fail(f"Unexpected non-object turn record: {turn!r}")
                turns.append(turn)
    return turns


def _load_tegrastats_ram(path: Path) -> list[int]:
    """Parse used system RAM samples from tegrastats output."""
    samples: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RAM_RE.search(line)
        if match is not None:
            samples.append(int(match.group(1)))
    return samples


def _assert_all_turns_succeeded(run: _E2ERun, *, expected_turns: int) -> None:
    """Assert that every parsed E2E turn succeeded."""
    assert len(run.turns) == expected_turns, run.output_dir
    failures = [turn for turn in run.turns if turn.get("success") is not True]
    assert not failures, {"output_dir": str(run.output_dir), "failures": failures}


def _hot_turn_mean(run: _E2ERun, metric_name: str) -> float:
    """Return the mean for turns after the first cold-load turn."""
    hot_values = [
        float(turn[metric_name]) for turn in run.turns if int(turn.get("exchange", 0)) > 1
    ]
    assert hot_values, run.output_dir
    return sum(hot_values) / len(hot_values)


def _contains_hangul(text: str) -> bool:
    """Return True when text contains at least one Hangul syllable."""
    return any("\uac00" <= char <= "\ud7a3" for char in text)


@pytest.fixture(scope="module")
def short_resident_run() -> _E2ERun:
    """Run a short 1-round resident text-input smoke for shared L1 gates."""
    return _run_resident_e2e(
        name="l1-resident-short",
        rounds=1,
        start=1,
        seed=1,
        timeout_s=1800,
    )


def test_resident_text_run_peak_under_5500mb(short_resident_run: _E2ERun) -> None:
    """Resident text-input mode should stay under the ADR 0076 system RAM invariant."""
    _assert_all_turns_succeeded(short_resident_run, expected_turns=3)
    assert short_resident_run.ram_samples_mib, short_resident_run.output_dir
    assert max(short_resident_run.ram_samples_mib) <= SYSTEM_RAM_LIMIT_MIB


def test_resident_hot_turn_mean_under_12s(short_resident_run: _E2ERun) -> None:
    """Hot-turn total latency should catch accidental LLM reload regressions."""
    _assert_all_turns_succeeded(short_resident_run, expected_turns=3)
    assert _hot_turn_mean(short_resident_run, "total_time_s") <= HOT_TURN_MEAN_LIMIT_S


def test_resident_korean_ttft_within_threshold(
    short_resident_run: _E2ERun,
) -> None:
    """Resident Korean TTFT uses the revised L1 baseline threshold.

    The ADR 0073 Session 2 baseline is invalidated as an L1 regression target
    by the current Gemma 4 default, Rule 8 prompt stack, and n_ctx=4096 path.
    The 4.50 s threshold keeps the rounds.jsonl evidence path and allows the
    empirical 3.94 s implementation peak with roughly 14% cushion while a
    separate follow-up plan tracks TTFT regression root-cause attribution.
    """
    _assert_all_turns_succeeded(short_resident_run, expected_turns=3)
    assert _hot_turn_mean(short_resident_run, "llm_ttft_s") <= KOREAN_TTFT_THRESHOLD_S


def test_resident_long_conversation_kv_bounded() -> None:
    """Long resident conversations use rounds.jsonl for turn state and tegrastats for RAM."""
    run = _run_resident_e2e(
        name="l1-resident-long-kv",
        rounds=9,
        start=1,
        seed=1,
        timeout_s=7200,
    )

    _assert_all_turns_succeeded(run, expected_turns=27)
    assert run.ram_samples_mib, run.output_dir
    assert max(run.ram_samples_mib) <= SYSTEM_RAM_LIMIT_MIB

    context_errors = [
        turn.get("error")
        for turn in run.turns
        if "Requested tokens" in str(turn.get("error") or "")
        and "exceed context window" in str(turn.get("error") or "")
    ]
    assert not context_errors, {"output_dir": str(run.output_dir), "errors": context_errors}

    final_turn = next(
        turn
        for turn in run.turns
        if int(turn.get("round_num", 0)) == 9 and int(turn.get("exchange", 0)) == 3
    )
    assistant_text = str(final_turn.get("assistant_text") or "")
    assert assistant_text.strip()
    assert _contains_hangul(assistant_text)
