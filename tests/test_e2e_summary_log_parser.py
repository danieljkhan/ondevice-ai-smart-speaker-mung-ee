"""Tests for E2E run.log stability counter parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts.e2e_qwen3_asr_mix import (
    STABILITY_COUNTER_KEYS,
    parse_run_log_stability_counters,
)

LOGGER_NAME = "mungi.scripts.e2e_qwen3_asr_mix"


def _zero_counters() -> dict[str, int]:
    return {key: 0 for key in STABILITY_COUNTER_KEYS}


def _write_log(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "run.log"
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_stability_counters_empty_log(tmp_path: Path) -> None:
    """Empty logs should produce all-zero stability counters."""
    counters = parse_run_log_stability_counters(_write_log(tmp_path, ""))

    assert counters == _zero_counters()


def test_parse_stability_counters_one_of_each(tmp_path: Path) -> None:
    """Each known log message should increment its own counter once."""
    log_path = _write_log(
        tmp_path,
        "\n".join(
            [
                "CRITICAL memory: 6700 MB exceeds critical threshold",
                "forcing STT unload to reclaim memory headroom",
                "LLM prompt cache flushed",
                "System-state snapshot captured (42 tokens)",
            ],
        ),
    )

    counters = parse_run_log_stability_counters(log_path)

    assert counters == {
        "critical_memory_events": 1,
        "stt_force_unload_count": 1,
        "llm_prompt_cache_flush_count": 1,
        "system_state_snapshot_count": 1,
    }


def test_parse_stability_counters_multiple_of_each(tmp_path: Path) -> None:
    """Repeated log messages should be counted independently by line."""
    log_path = _write_log(
        tmp_path,
        "\n".join(
            [
                *(["CRITICAL memory: 6700 MB over limit"] * 3),
                *(["forcing STT unload before LLM load"] * 2),
                *(["LLM prompt cache flushed"] * 5),
                *(["System-state snapshot captured (128 tokens)"] * 4),
            ],
        ),
    )

    counters = parse_run_log_stability_counters(log_path)

    assert counters == {
        "critical_memory_events": 3,
        "stt_force_unload_count": 2,
        "llm_prompt_cache_flush_count": 5,
        "system_state_snapshot_count": 4,
    }


def test_parse_stability_counters_ignores_irrelevant_lines(tmp_path: Path) -> None:
    """Unrelated runner log lines should not affect stability counters."""
    log_path = _write_log(
        tmp_path,
        "\n".join(
            [
                "INFO: starting",
                "DEBUG: heartbeat",
                "INFO mungi.core.pipeline: Round 1 complete",
                "WARNING mungi.models.llm_runner: prompt cache unchanged",
            ],
        ),
    )

    counters = parse_run_log_stability_counters(log_path)

    assert counters == _zero_counters()


def test_parse_stability_counters_realistic_prefixed_lines(tmp_path: Path) -> None:
    """Stdlib logging prefixes should not prevent exact message matching."""
    log_path = _write_log(
        tmp_path,
        "\n".join(
            [
                (
                    "2026-04-15 22:06:48,123 WARNING mungi.core.model_manager: "
                    "CRITICAL memory: 6700 MB available headroom is low"
                ),
                (
                    "2026-04-15 22:06:49,456 INFO mungi.core.model_manager: "
                    "forcing STT unload before LLM load"
                ),
                ("2026-04-15 22:06:50,789 INFO mungi.models.llm_runner: LLM prompt cache flushed"),
                (
                    "2026-04-15 22:06:51,012 DEBUG mungi.models.llm_runner: "
                    "System-state snapshot captured (16 tokens)"
                ),
            ],
        ),
    )

    counters = parse_run_log_stability_counters(log_path)

    assert counters == {
        "critical_memory_events": 1,
        "stt_force_unload_count": 1,
        "llm_prompt_cache_flush_count": 1,
        "system_state_snapshot_count": 1,
    }


def test_parse_stability_counters_missing_file_warns_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing run.log files should warn once and return all-zero counters."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    counters = parse_run_log_stability_counters(tmp_path / "missing.log")

    warning_records = [
        record
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]
    assert counters == _zero_counters()
    assert len(warning_records) == 1
    assert "Missing run.log for stability counters" in warning_records[0].message


def test_parse_stability_counters_prompt_cache_variants(tmp_path: Path) -> None:
    """Prompt-cache flush messages with and without byte counts should match."""
    log_path = _write_log(
        tmp_path,
        "\n".join(
            [
                "LLM prompt cache flushed (freed ~12345 bytes)",
                "LLM prompt cache flushed",
            ],
        ),
    )

    counters = parse_run_log_stability_counters(log_path)

    assert counters == {
        **_zero_counters(),
        "llm_prompt_cache_flush_count": 2,
    }
