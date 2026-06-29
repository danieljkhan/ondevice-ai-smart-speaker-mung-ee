"""Generate a markdown E2E report from a 60-round evaluation directory."""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_model import parse_tegrastats_line  # noqa: E402

logger = logging.getLogger("mungi.scripts.generate_e2e_report")

STABILITY_COUNTER_KEYS = (
    "critical_memory_events",
    "stt_force_unload_count",
    "llm_prompt_cache_flush_count",
    "system_state_snapshot_count",
)
CANONICAL_TEMPLATE_PATH = REPO_ROOT / "docs" / "templates" / "e2e-report-format.md"


def _split_markdown_row(row: str) -> list[str]:
    """Split a markdown table row into stripped cell values."""
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _parse_canonical_latency_table_header(
    template_path: Path = CANONICAL_TEMPLATE_PATH,
) -> tuple[str, str, list[str]]:
    """Parse the canonical E2E latency table header from the docs template."""
    try:
        lines = template_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        msg = f"Failed to read canonical latency template: {template_path}"
        raise RuntimeError(msg) from exc
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| Turn |"):
            cells = _split_markdown_row(stripped)
            divider_cells = ["---", *(["---:"] * (len(cells) - 1))]
            return stripped, "| " + " | ".join(divider_cells) + " |", cells
    msg = f"Canonical latency table header not found in {template_path}"
    raise RuntimeError(msg)


@dataclass(frozen=True)
class TurnRecord:
    """Flattened turn-level data extracted from rounds.jsonl."""

    round_num: int
    topic: str
    exchange: int
    user_text: str
    assistant_text: str
    llm_tokens: int | None
    llm_ttft_s: float | None
    llm_time_s: float | None
    tts_time_s: float | None
    total_time_s: float | None
    peak_memory_kb: int | None
    success: bool
    error: str | None
    language: str
    pass_id: str | None = None
    global_turn_id: int | None = None
    source_round_id: int | None = None
    vad_time_s: float | None = None
    stt_total_time_s: float | None = None
    stt_load_time_s: float | None = None
    llm_load_time_s: float | None = None
    tts_load_time_s: float | None = None
    playback_time_s: float | None = None
    first_sound_time_s: float | None = None


@dataclass(frozen=True)
class RoundRecord:
    """Round-level rollup for the generated report."""

    round_num: int
    topic: str
    turn_count: int
    success_count: int
    failure_count: int
    llm_tokens: int
    avg_ttft_s: float | None
    avg_llm_time_s: float | None
    avg_tts_time_s: float | None
    avg_total_time_s: float | None
    peak_memory_kb: int | None
    errors: list[str]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the report generator."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown E2E report from evaluation artifacts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing rounds.jsonl, tegrastats.log, summary.json, thermal_summary.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Optional previous evaluation directory for A/B comparison.",
    )
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Render additional bilingual language-stratified sections.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning ``None`` when the file is missing or invalid."""
    if not path.exists():
        logger.warning("Missing JSON file: %s", path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Expected JSON object in %s but found %s", path, type(payload).__name__)
        return None
    return cast(dict[str, Any], payload)


def _read_jsonl_rounds(path: Path) -> list[dict[str, Any]]:
    """Read rounds.jsonl as a list of raw round dictionaries."""
    if not path.exists():
        logger.warning("Missing rounds file: %s", path)
        return []

    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL line in %s: %s", path, exc)
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return records


def _as_int(value: Any) -> int | None:
    """Best-effort integer conversion."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    """Best-effort float conversion."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms_to_seconds(value: Any) -> float | None:
    """Convert a millisecond value to seconds when available."""
    value_ms = _as_float(value)
    if value_ms is None:
        return None
    return value_ms / 1000.0


def _is_mix_runner_format(raw_rounds: list[dict[str, Any]]) -> bool:
    """Return ``True`` when the JSONL payload matches the flat mix-runner schema."""
    if not raw_rounds:
        return False
    first_record = raw_rounds[0]
    return (
        isinstance(first_record, dict)
        and "round_id" in first_record
        and "lang" in first_record
        and "timings_ms" in first_record
    )


def _flatten_mix_jsonl(raw_rounds: list[dict[str, Any]]) -> list[TurnRecord]:
    """Extract turn records from the flat mix-runner JSONL structure."""
    logger.info("Detected mix-runner JSONL format (%d rounds)", len(raw_rounds))
    turns: list[TurnRecord] = []
    missing_language_count = 0
    for raw_round in raw_rounds:
        timings_ms = raw_round.get("timings_ms")
        timings = timings_ms if isinstance(timings_ms, dict) else {}
        raw_language = raw_round.get("lang")
        if raw_language is None:
            missing_language_count += 1
            language = "ko"
        else:
            language = str(raw_language).strip().lower() or "ko"
        turns.append(
            TurnRecord(
                round_num=_as_int(raw_round.get("round_id")) or 0,
                topic=str(raw_round.get("lang") or raw_round.get("detected_language") or "mix"),
                exchange=_as_int(raw_round.get("turn_index_per_lang")) or 0,
                user_text=str(raw_round.get("stt_pred") or ""),
                assistant_text=str(raw_round.get("llm_response") or ""),
                llm_tokens=_as_int(raw_round.get("llm_tokens")),
                vad_time_s=_ms_to_seconds(timings.get("vad_ms")),
                stt_load_time_s=_ms_to_seconds(timings.get("stt_load_ms")),
                stt_total_time_s=_ms_to_seconds(timings.get("stt_total_ms")),
                llm_load_time_s=_ms_to_seconds(timings.get("llm_load_ms")),
                llm_ttft_s=_ms_to_seconds(timings.get("llm_ttft_ms")),
                llm_time_s=_ms_to_seconds(timings.get("llm_ms")),
                tts_load_time_s=_ms_to_seconds(timings.get("tts_load_ms")),
                tts_time_s=_ms_to_seconds(timings.get("tts_ms")),
                playback_time_s=_ms_to_seconds(timings.get("playback_ms")),
                first_sound_time_s=_ms_to_seconds(timings.get("first_sound_ms")),
                total_time_s=_ms_to_seconds(timings.get("total_ms")),
                peak_memory_kb=_as_int(raw_round.get("peak_memory_kb")),
                success=bool(raw_round.get("success", False)),
                error=str(raw_round.get("error")) if raw_round.get("error") else None,
                language=language,
                pass_id=str(raw_round.get("pass_id")) if raw_round.get("pass_id") else None,
                global_turn_id=_as_int(raw_round.get("global_turn_id")),
                source_round_id=_as_int(raw_round.get("source_round_id")),
            ),
        )
    if missing_language_count:
        logger.warning(
            "Missing language field on %d turn(s); defaulting those turns to 'ko'.",
            missing_language_count,
        )
    turns.sort(
        key=lambda item: (
            item.global_turn_id if item.global_turn_id is not None else 10**9,
            item.round_num,
            item.exchange,
        ),
    )
    return turns


def _flatten_turns(raw_rounds: list[dict[str, Any]]) -> list[TurnRecord]:
    """Extract turn records from the nested rounds JSONL structure."""
    turns: list[TurnRecord] = []
    missing_language_count = 0
    for raw_round in raw_rounds:
        default_round_num = _as_int(raw_round.get("round"))
        topics = raw_round.get("topics") or []
        if not isinstance(topics, list):
            continue
        for topic_item in topics:
            if not isinstance(topic_item, dict):
                continue
            topic_name = str(topic_item.get("topic") or raw_round.get("topic") or "")
            topic_turns = topic_item.get("turns") or []
            if not isinstance(topic_turns, list):
                continue
            for turn in topic_turns:
                if not isinstance(turn, dict):
                    continue
                round_num = _as_int(turn.get("round_num")) or default_round_num or 0
                raw_language = turn.get("language")
                if raw_language is None:
                    missing_language_count += 1
                    language = "ko"
                else:
                    language = str(raw_language).strip().lower() or "ko"
                turns.append(
                    TurnRecord(
                        round_num=round_num,
                        topic=str(turn.get("topic") or topic_name),
                        exchange=_as_int(turn.get("exchange")) or 0,
                        user_text=str(turn.get("user_text") or ""),
                        assistant_text=str(turn.get("assistant_text") or ""),
                        llm_tokens=_as_int(turn.get("llm_tokens")),
                        llm_ttft_s=_as_float(turn.get("llm_ttft_s")),
                        llm_time_s=_as_float(turn.get("llm_time_s")),
                        tts_time_s=_as_float(turn.get("tts_time_s")),
                        total_time_s=_as_float(turn.get("total_time_s")),
                        peak_memory_kb=_as_int(turn.get("peak_memory_kb")),
                        success=bool(turn.get("success", False)),
                        error=str(turn.get("error")) if turn.get("error") else None,
                        language=language,
                    ),
                )
    if missing_language_count:
        logger.warning(
            "Missing language field on %d turn(s); defaulting those turns to 'ko'.",
            missing_language_count,
        )
    turns.sort(key=lambda item: (item.round_num, item.exchange))
    return turns


def _group_rounds(turns: list[TurnRecord]) -> list[RoundRecord]:
    """Aggregate flattened turn records into round-level summaries."""
    grouped: dict[int, list[TurnRecord]] = defaultdict(list)
    for turn in turns:
        grouped[turn.round_num].append(turn)

    rounds: list[RoundRecord] = []
    for round_num in sorted(grouped):
        items = sorted(grouped[round_num], key=lambda item: item.exchange)
        topic_counts = Counter(item.topic for item in items if item.topic)
        topic = topic_counts.most_common(1)[0][0] if topic_counts else ""
        success_count = sum(1 for item in items if item.success)
        failure_count = len(items) - success_count
        rounds.append(
            RoundRecord(
                round_num=round_num,
                topic=topic,
                turn_count=len(items),
                success_count=success_count,
                failure_count=failure_count,
                llm_tokens=sum(item.llm_tokens or 0 for item in items),
                avg_ttft_s=_mean([item.llm_ttft_s for item in items]),
                avg_llm_time_s=_mean([item.llm_time_s for item in items]),
                avg_tts_time_s=_mean([item.tts_time_s for item in items]),
                avg_total_time_s=_mean([item.total_time_s for item in items]),
                peak_memory_kb=max(
                    (item.peak_memory_kb for item in items if item.peak_memory_kb is not None),
                    default=None,
                ),
                errors=[item.error for item in items if item.error],
            ),
        )
    return rounds


def _mean(values: Sequence[float | None]) -> float | None:
    """Compute the arithmetic mean over defined values."""
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return statistics.fmean(filtered)


def _percentile(values: list[float], percentile: float) -> float | None:
    """Compute an interpolated percentile from a list of values."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    weight = position - lower_index
    return lower + (upper - lower) * weight


def _metric_series(turns: list[TurnRecord], attr: str) -> list[float]:
    """Collect a numeric series from turn records."""
    series: list[float] = []
    for turn in turns:
        value = getattr(turn, attr)
        if value is not None:
            series.append(float(value))
    return series


def _format_seconds(value: float | None) -> str:
    """Format a time value for markdown tables."""
    return "-" if value is None else f"{value:.3f}s"


def _format_number(value: float | int | None, digits: int = 1) -> str:
    """Format a numeric value for markdown output."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _format_memory_kb(value: int | None) -> str:
    """Format a peak memory value in KB."""
    return "-" if value is None else f"{value:,} KB"


def _format_memory_mb(value: int | None) -> str:
    """Format a peak memory value in MB."""
    return "-" if value is None else f"{value / 1024.0:,.1f} MB"


def _format_latency_seconds_cell(value: float | None) -> str:
    """Format a canonical latency-table cell in seconds."""
    return "-" if value is None else f"{value:.3f}"


def _canonical_turn_label(turn: TurnRecord) -> str:
    """Return the Stage-2 canonical per-turn label."""
    pass_id = turn.pass_id or "pass1"
    source_round_id = turn.source_round_id if turn.source_round_id is not None else turn.round_num
    return f"{pass_id}.sr{source_round_id:02d}.{turn.language}"


def _render_mix_latency_table(turns: list[TurnRecord]) -> list[str]:
    """Render the canonical 11-column per-turn latency table for mix-runner data."""
    if not turns or not any(turn.pass_id or turn.source_round_id is not None for turn in turns):
        return []
    header, divider, _cells = _parse_canonical_latency_table_header()
    fields = (
        "vad_time_s",
        "stt_total_time_s",
        "llm_load_time_s",
        "llm_ttft_s",
        "llm_time_s",
        "tts_load_time_s",
        "tts_time_s",
        "playback_time_s",
        "first_sound_time_s",
        "total_time_s",
    )
    lines = [header, divider]
    for turn in turns:
        values = [_format_latency_seconds_cell(getattr(turn, field)) for field in fields]
        lines.append(f"| {_canonical_turn_label(turn)} | " + " | ".join(values) + " |")

    avg_values = [
        _format_latency_seconds_cell(_mean([getattr(turn, field) for turn in turns]))
        for field in fields
    ]
    lines.append("| AVG | " + " | ".join(avg_values) + " |")
    return lines


def _render_pass_aggregates(turns: list[TurnRecord], thermal: dict[str, Any]) -> list[str]:
    """Render repeat-pass aggregate metrics in a separate table."""
    pass_ids = sorted({turn.pass_id for turn in turns if turn.pass_id})
    if not pass_ids:
        return []
    thermal_max_c = _as_float(thermal.get("thermal_max_c"))
    lines = [
        "| pass_id | turns | avg_total_s | success_rate | avg_ttft_ko_s | thermal_max_c |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pass_id in pass_ids:
        items = [turn for turn in turns if turn.pass_id == pass_id]
        successes = sum(1 for turn in items if turn.success)
        success_rate = successes / len(items) * 100.0 if items else None
        ko_ttft = [turn.llm_ttft_s for turn in items if turn.language == "ko"]
        lines.append(
            (
                f"| {pass_id} | {len(items)} | "
                f"{_format_latency_seconds_cell(_mean([turn.total_time_s for turn in items]))} | "
                f"{'-' if success_rate is None else f'{success_rate:.1f}%'} | "
                f"{_format_latency_seconds_cell(_mean(ko_ttft))} | "
                f"{_format_number(thermal_max_c, 3)} |"
            ),
        )
    return lines


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, returning an empty list when unavailable."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return []
    if isinstance(payload, dict):
        payload = payload.get("samples", [])
    if not isinstance(payload, list):
        logger.warning("Expected JSON array in %s but found %s", path, type(payload).__name__)
        return []
    return [item for item in payload if isinstance(item, dict)]


def _render_thermal_curve(curve: list[dict[str, Any]]) -> list[str]:
    """Render the downsampled thermal curve table."""
    if not curve:
        return ["Thermal curve not available."]
    lines = [
        "| t_s | CPU°C | GPU°C | RAM MB | GR3D % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for sample in curve:
        lines.append(
            (
                f"| {_format_number(_as_float(sample.get('t_s')), 3)} | "
                f"{_format_number(_as_float(sample.get('cpu_temp_c')), 3)} | "
                f"{_format_number(_as_float(sample.get('gpu_temp_c')), 3)} | "
                f"{_format_number(_as_float(sample.get('ram_used_mb')), 1)} | "
                f"{_format_number(_as_float(sample.get('gr3d_freq_pct')), 1)} |"
            ),
        )
    return lines


def _raw_round_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    """Sort raw mix rows by global turn id with a round-id fallback."""
    global_turn_id = _as_int(row.get("global_turn_id"))
    round_id = _as_int(row.get("round_id")) or 0
    return (global_turn_id if global_turn_id is not None else 10**9, round_id)


def _render_memory_envelope(raw_rounds: list[dict[str, Any]]) -> list[str]:
    """Render per-turn and per-source-round memory envelope tables."""
    mix_rows = [row for row in raw_rounds if "system_ram_mb" in row or "process_rss_mb" in row]
    if not mix_rows:
        return ["Memory envelope not available."]

    sorted_rows = sorted(mix_rows, key=_raw_round_sort_key)
    lines = [
        "### Per-Turn Envelope",
        "| global_turn_id | pass_id | source_round_id | round_id | lang | system_ram_mb | "
        "process_rss_mb | delta_from_previous_mb |",
        "|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    previous_rss: float | None = None
    for row in sorted_rows:
        rss = _as_float(row.get("process_rss_mb"))
        delta = rss - previous_rss if rss is not None and previous_rss is not None else None
        if rss is not None:
            previous_rss = rss
        lines.append(
            (
                f"| {_format_number(_as_int(row.get('global_turn_id')), 0)} | "
                f"{row.get('pass_id', '-')} | "
                f"{_format_number(_as_int(row.get('source_round_id')), 0)} | "
                f"{_format_number(_as_int(row.get('round_id')), 0)} | "
                f"{row.get('lang', '-')} | "
                f"{_format_number(_as_float(row.get('system_ram_mb')), 3)} | "
                f"{_format_number(rss, 3)} | "
                f"{_format_number(delta, 3)} |"
            ),
        )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted_rows:
        source_round_id = _as_int(row.get("source_round_id"))
        if source_round_id is not None:
            grouped[source_round_id].append(row)

    lines.extend(
        [
            "",
            "### Per-Source-Round Rollup",
            "| source_round_id | passes | peak_system_ram_mb | peak_process_rss_mb |",
            "|---:|---:|---:|---:|",
        ],
    )
    for source_round_id in sorted(grouped):
        items = grouped[source_round_id]
        pass_count = len({str(item.get("pass_id", "")) for item in items})
        source_system_values = [
            value for item in items if (value := _as_float(item.get("system_ram_mb"))) is not None
        ]
        source_rss_values = [
            value for item in items if (value := _as_float(item.get("process_rss_mb"))) is not None
        ]
        peak_system = max(source_system_values, default=None)
        peak_rss = max(source_rss_values, default=None)
        lines.append(
            (
                f"| {source_round_id} | {pass_count} | "
                f"{_format_number(peak_system, 3)} | {_format_number(peak_rss, 3)} |"
            ),
        )

    system_values = [
        value for row in sorted_rows if (value := _as_float(row.get("system_ram_mb"))) is not None
    ]
    rss_values = [
        value for row in sorted_rows if (value := _as_float(row.get("process_rss_mb"))) is not None
    ]
    lines.extend(
        [
            "",
            "### Run Peaks",
            f"- run_peak_system_ram_mb: {_format_number(max(system_values, default=None), 3)}",
            f"- run_peak_process_rss_mb: {_format_number(max(rss_values, default=None), 3)}",
            "| pass_id | peak_system_ram_mb | peak_process_rss_mb |",
            "|---|---:|---:|",
        ],
    )
    for pass_id in sorted({str(row.get("pass_id", "")) for row in sorted_rows}):
        items = [row for row in sorted_rows if str(row.get("pass_id", "")) == pass_id]
        pass_system_values = [
            value for row in items if (value := _as_float(row.get("system_ram_mb"))) is not None
        ]
        pass_rss_values = [
            value for row in items if (value := _as_float(row.get("process_rss_mb"))) is not None
        ]
        lines.append(
            (
                f"| {pass_id or '-'} | "
                f"{_format_number(max(pass_system_values, default=None), 3)} | "
                f"{_format_number(max(pass_rss_values, default=None), 3)} |"
            ),
        )
    return lines


def _gate_verdict(
    observed: float | int | None,
    predicate: Any,
) -> str:
    """Return PASS/FAIL/SKIP for a numeric gate."""
    if observed is None:
        return "SKIP"
    return "PASS" if predicate(observed) else "FAIL"


def _summary_has_field(summary: dict[str, Any], field: str) -> bool:
    """Return whether summary.json explicitly recorded a field."""
    return field in summary


def _stt_provider_for_gate(summary: dict[str, Any]) -> str | None:
    """Return the actual STT provider for G10, with a legacy-summary fallback."""
    if _summary_has_field(summary, "stt_provider_actual"):
        provider = summary.get("stt_provider_actual")
    else:
        # Legacy summaries predate per-turn actual-provider telemetry.
        provider = summary.get("stt_provider_resolved")
    if provider is None:
        return None
    return str(provider)


def _render_gate_verdicts(
    raw_rounds: list[dict[str, Any]],
    summary: dict[str, Any],
    thermal: dict[str, Any],
) -> list[str]:
    """Render the mechanical gate verdict scaffold."""
    rows: list[tuple[str, str, str, str, str]] = []
    system_values: list[float] = []
    for row in raw_rounds:
        value = _as_float(row.get("system_ram_mb"))
        if value is not None:
            system_values.append(value)
    peak_system = max(system_values) if system_values else None
    rows.append(
        (
            "G1",
            "< 5500 MB",
            _format_number(peak_system, 3),
            _gate_verdict(peak_system, lambda value: value < 5500),
            "rounds.jsonl",
        ),
    )
    rows.append(
        (
            "G2a",
            "< 6000 MB",
            _format_number(peak_system, 3),
            _gate_verdict(peak_system, lambda value: value < 6000),
            "rounds.jsonl",
        ),
    )
    critical_events = _as_int(summary.get("critical_memory_events"))
    rows.append(
        (
            "G2b",
            "== 0 CRITICAL events",
            _format_number(critical_events, 0),
            _gate_verdict(critical_events, lambda value: value == 0),
            "run.log",
        ),
    )
    thermal_max = _as_float(thermal.get("thermal_max_c"))
    rows.append(
        (
            "G3",
            "< 80.0 °C",
            _format_number(thermal_max, 3),
            _gate_verdict(thermal_max, lambda value: value < 80.0),
            "thermal_summary.json",
        ),
    )
    success_flags = [bool(row.get("success", False)) for row in raw_rounds]
    success_rate = sum(success_flags) / len(success_flags) if success_flags else None
    rows.append(
        (
            "G4",
            ">= 95%",
            "-" if success_rate is None else f"{success_rate * 100.0:.1f}%",
            _gate_verdict(success_rate, lambda value: value >= 0.95),
            "rounds.jsonl",
        ),
    )
    vad_miss_available = any(
        "vad_miss" in row or row.get("failure_reason") == "vad_miss" for row in raw_rounds
    )
    vad_miss_count = sum(
        1
        for row in raw_rounds
        if row.get("vad_miss") is True or row.get("failure_reason") == "vad_miss"
    )
    rows.append(
        (
            "G_VAD_MISS",
            "== 0 rows with failure_reason=vad_miss",
            str(vad_miss_count) if vad_miss_available else "-",
            _gate_verdict(vad_miss_count, lambda value: value == 0)
            if vad_miss_available
            else "SKIP",
            "rounds.jsonl",
        ),
    )
    first_timings = raw_rounds[0].get("timings_ms") if raw_rounds else None
    first_timings = first_timings if isinstance(first_timings, dict) else {}
    g6_available = "stt_load_count" in summary and "tts_load_count" in summary and bool(raw_rounds)
    g6_pass = (
        _as_float(first_timings.get("stt_load_ms")) is not None
        and (_as_float(first_timings.get("stt_load_ms")) or 0.0) > 0.0
        and _as_float(first_timings.get("tts_load_ms")) is not None
        and (_as_float(first_timings.get("tts_load_ms")) or 0.0) > 0.0
        and _as_int(summary.get("stt_load_count")) == 1
        and _as_int(summary.get("tts_load_count")) == 1
    )
    rows.append(
        (
            "G6",
            "first load_ms > 0 and load counts == 1",
            "available" if g6_available else "-",
            ("PASS" if g6_pass else "FAIL") if g6_available else "SKIP",
            "summary.json + rounds.jsonl",
        ),
    )
    swimming_guides = sum(
        1
        for row in raw_rounds
        if row.get("template_topic_id") == "swimming"
        and row.get("template_mode") == "guide"
        and row.get("template_matched") is True
    )
    rows.append(
        (
            "G7",
            "swimming guide template hits >= 5",
            str(swimming_guides) if raw_rounds else "-",
            ("PASS" if swimming_guides >= 5 else "FAIL") if raw_rounds else "SKIP",
            "rounds.jsonl",
        ),
    )
    hot_ttft = [
        _ms_to_seconds(row.get("timings_ms", {}).get("llm_ttft_ms"))
        for row in raw_rounds
        if row.get("lang") == "ko" and row.get("template_matched") is not True
    ]
    hot_ttft = [value for value in hot_ttft if value is not None]
    hot_ttft_mean = _mean(hot_ttft)
    rows.append(
        (
            "G8",
            "KO hot-turn TTFT mean <= 4.50 s",
            _format_latency_seconds_cell(hot_ttft_mean),
            _gate_verdict(hot_ttft_mean, lambda value: value <= 4.50),
            "rounds.jsonl",
        ),
    )
    g9_available = bool(raw_rounds)
    g9_pass = (
        _as_int(summary.get("tts_synth_error_count")) == 0
        and _as_int(summary.get("tts_load_error_count")) == 0
        and all(
            not row.get("success")
            or (
                (_as_int(row.get("tts_wav_bytes")) or 0) > 0
                and (_as_int(row.get("tts_wav_frames")) or 0) > 0
            )
            for row in raw_rounds
        )
    )
    rows.append(
        (
            "G9",
            "TTS errors == 0 and success rows have WAV bytes/frames",
            "available" if g9_available else "-",
            ("PASS" if g9_pass else "FAIL") if g9_available else "SKIP",
            "summary.json + rounds.jsonl",
        ),
    )
    g10_available = bool(summary)
    stt_provider_for_gate = _stt_provider_for_gate(summary)
    g10_pass = (
        stt_provider_for_gate == "cpu"
        and bool(summary.get("sherpa_onnx_version"))
        and _as_int(summary.get("stt_load_count")) == 1
    )
    rows.append(
        (
            "G10",
            "STT provider == cpu, version recorded, stt_load_count == 1",
            f"provider={stt_provider_for_gate or '-'}" if g10_available else "-",
            ("PASS" if g10_pass else "FAIL") if g10_available else "SKIP",
            "summary.json",
        ),
    )
    lines = [
        "| gate_id | threshold | observed_value | verdict | evidence_artifact_path |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {gate} | {threshold} | {observed} | {verdict} | {path} |"
        for gate, threshold, observed, verdict, path in rows
    )
    return lines


def _format_recorded(value: Any) -> str:
    """Format reproducibility fields with a stable fallback."""
    if value is None:
        return "not recorded"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_reproducibility_appendix(summary: dict[str, Any]) -> list[str]:
    """Render Stage-2 reproducibility metadata from summary.json."""
    fields = [
        "mungi_llm_resident",
        "mungi_stt_resident",
        "mungi_tts_resident",
        "llm_n_gpu_layers_resolved",
        "commit_sha",
        "sherpa_onnx_version",
        "stt_provider_actual",
        "stt_provider_configured",
        "stt_provider_requested",
    ]
    if _summary_has_field(summary, "stt_provider_resolved"):
        fields.append("stt_provider_resolved")
    fields.extend(
        [
            "repeat_passes",
            "model_sha256",
        ],
    )
    lines = ["| field | value |", "|---|---|"]
    for field in fields:
        lines.append(f"| {field} | {_format_recorded(summary.get(field))} |")
    return lines


def _summarize_turns(turns: list[TurnRecord]) -> dict[str, Any]:
    """Compute overall performance metrics for a turn set."""
    total_turns = len(turns)
    success_count = sum(1 for turn in turns if turn.success)
    failures = total_turns - success_count
    ttft = _metric_series(turns, "llm_ttft_s")
    llm = _metric_series(turns, "llm_time_s")
    tts = _metric_series(turns, "tts_time_s")
    total = _metric_series(turns, "total_time_s")
    tokens = [turn.llm_tokens for turn in turns if turn.llm_tokens is not None]
    memories = [turn.peak_memory_kb for turn in turns if turn.peak_memory_kb is not None]
    return {
        "turns": total_turns,
        "success_count": success_count,
        "failure_count": failures,
        "success_rate": (success_count / total_turns * 100.0) if total_turns else None,
        "tokens_total": sum(tokens),
        "tokens_avg": statistics.fmean(tokens) if tokens else None,
        "ttft_avg": _mean([float(value) for value in ttft]) if ttft else None,
        "ttft_med": statistics.median(ttft) if ttft else None,
        "ttft_p5": _percentile(ttft, 5) if ttft else None,
        "ttft_p95": _percentile(ttft, 95) if ttft else None,
        "llm_avg": _mean([float(value) for value in llm]) if llm else None,
        "llm_med": statistics.median(llm) if llm else None,
        "llm_p5": _percentile(llm, 5) if llm else None,
        "llm_p95": _percentile(llm, 95) if llm else None,
        "tts_avg": _mean([float(value) for value in tts]) if tts else None,
        "tts_med": statistics.median(tts) if tts else None,
        "tts_p5": _percentile(tts, 5) if tts else None,
        "tts_p95": _percentile(tts, 95) if tts else None,
        "total_avg": _mean([float(value) for value in total]) if total else None,
        "total_med": statistics.median(total) if total else None,
        "total_p5": _percentile(total, 5) if total else None,
        "total_p95": _percentile(total, 95) if total else None,
        "memory_peak_kb": max(memories) if memories else None,
        "memory_avg_kb": _mean([float(value) for value in memories]) if memories else None,
        "rounds": _group_rounds(turns),
    }


def _summarize_segments(rounds: list[RoundRecord]) -> list[dict[str, Any]]:
    """Summarize rounds in 10-round segments."""
    segments: list[dict[str, Any]] = []
    for start in range(1, 61, 10):
        end = start + 9
        segment_rounds = [item for item in rounds if start <= item.round_num <= end]
        if not segment_rounds:
            segments.append(
                {
                    "label": f"R{start}-{end}",
                    "turns": 0,
                    "success_rate": None,
                    "avg_ttft_s": None,
                    "avg_llm_time_s": None,
                    "avg_tts_time_s": None,
                    "avg_total_time_s": None,
                    "peak_memory_kb": None,
                },
            )
            continue

        successes = sum(item.success_count for item in segment_rounds)
        total_turns = sum(item.turn_count for item in segment_rounds)
        ttft_values = [
            value
            for item in segment_rounds
            for value in ([item.avg_ttft_s] if item.avg_ttft_s is not None else [])
        ]
        llm_values = [
            value
            for item in segment_rounds
            for value in ([item.avg_llm_time_s] if item.avg_llm_time_s is not None else [])
        ]
        tts_values = [
            value
            for item in segment_rounds
            for value in ([item.avg_tts_time_s] if item.avg_tts_time_s is not None else [])
        ]
        total_values = [
            value
            for item in segment_rounds
            for value in ([item.avg_total_time_s] if item.avg_total_time_s is not None else [])
        ]
        peak_memory = max(
            (item.peak_memory_kb for item in segment_rounds if item.peak_memory_kb is not None),
            default=None,
        )
        segments.append(
            {
                "label": f"R{start}-{end}",
                "turns": total_turns,
                "success_rate": (successes / total_turns * 100.0) if total_turns else None,
                "avg_ttft_s": _mean(ttft_values),
                "avg_llm_time_s": _mean(llm_values),
                "avg_tts_time_s": _mean(tts_values),
                "avg_total_time_s": _mean(total_values),
                "peak_memory_kb": peak_memory,
            },
        )
    return segments


def _normalize_text(text: str) -> str:
    """Normalize free-form text for heuristic comparisons."""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _truncate_text(text: str, limit: int = 40) -> str:
    """Trim long text snippets for markdown summaries."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _series_summary(values: list[float]) -> dict[str, float]:
    """Summarize a numeric series with start/end/min/max/avg/delta."""
    return {
        "start": values[0],
        "end": values[-1],
        "min": min(values),
        "max": max(values),
        "avg": statistics.fmean(values),
        "delta": values[-1] - values[0],
    }


def _detect_honorific_issues(turns: list[TurnRecord]) -> dict[str, Any]:
    """Heuristically estimate honorific, repetition, and context leak issues."""
    polite_endings = (
        "요",
        "세요",
        "까요",
        "입니다",
        "습니다",
        "해요",
        "예요",
        "이에요",
        "군요",
        "네요",
        "죠",
    )
    polite_phrases = ("안녕하세요", "감사해요", "괜찮아요")
    repeated_responses: Counter[str] = Counter()
    honorific_examples: list[TurnRecord] = []
    context_leak_examples: list[dict[str, Any]] = []

    prior_topics_by_round: dict[int, list[str]] = {}
    seen_topics: list[str] = []
    for round_item in _group_rounds(turns):
        prior_topics_by_round[round_item.round_num] = list(seen_topics)
        if round_item.topic and round_item.topic not in seen_topics:
            seen_topics.append(round_item.topic)

    for turn in turns:
        text = turn.assistant_text.strip()
        normalized = _normalize_text(text)
        if normalized:
            repeated_responses[normalized] += 1

        sentence_candidates = [
            candidate.strip() for candidate in re.split(r"[.!?\n~]+", text) if candidate.strip()
        ]
        has_honorific = any(
            any(candidate.endswith(ending) for ending in polite_endings)
            for candidate in sentence_candidates
        ) or any(phrase in text for phrase in polite_phrases)
        if has_honorific:
            honorific_examples.append(turn)

        current_topic = turn.topic.strip()
        for previous_topic in prior_topics_by_round.get(turn.round_num, []):
            candidate = previous_topic.strip()
            if len(candidate) < 2 or candidate == current_topic:
                continue
            if candidate in text:
                context_leak_examples.append(
                    {
                        "round_num": turn.round_num,
                        "current_topic": current_topic,
                        "leaked_topic": candidate,
                        "assistant_text": turn.assistant_text,
                    }
                )
                break

    repeated_pairs: list[tuple[str, int]] = [
        (_truncate_text(text), count)
        for text, count in repeated_responses.most_common()
        if count > 1
    ][:8]
    repetition_count = sum(count - 1 for _, count in repeated_pairs)
    return {
        "honorific_turn_count": len(honorific_examples),
        "honorific_examples": honorific_examples[:5],
        "repeated_responses": [{"text": text, "count": count} for text, count in repeated_pairs],
        "repetition_count": repetition_count,
        "context_leak_count": len(context_leak_examples),
        "context_leak_examples": context_leak_examples[:5],
    }


def _summarize_thermal(input_dir: Path) -> dict[str, Any]:
    """Load or derive thermal summary data."""
    thermal_path = input_dir / "thermal_summary.json"
    thermal = _read_json(thermal_path) or {}

    tegrastats_path = input_dir / "tegrastats.log"
    if not tegrastats_path.exists():
        if not thermal:
            logger.warning("Missing tegrastats log: %s", tegrastats_path)
            return {"snapshots_count": 0}
        return thermal

    snapshots: list[dict[str, Any]] = []
    try:
        for line in tegrastats_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = parse_tegrastats_line(stripped)
            if parsed:
                snapshots.append(parsed)
    except OSError as exc:
        logger.warning("Failed to read %s: %s", tegrastats_path, exc)
        return thermal or {"snapshots_count": 0}

    summary: dict[str, Any] = dict(thermal)
    summary["snapshots_count"] = max(
        int(summary.get("snapshots_count", 0) or 0),
        len(snapshots),
    )
    for key in ("cpu_temp_c", "gpu_temp_c", "ram_used_mb", "gr3d_freq_pct"):
        values = [float(snapshot[key]) for snapshot in snapshots if key in snapshot]
        if values:
            summary[key] = _series_summary(values)
    temp_max_values: list[float] = []
    for key in ("cpu_temp_c", "gpu_temp_c"):
        item = summary.get(key)
        if isinstance(item, dict):
            value = _as_float(item.get("max"))
            if value is not None:
                temp_max_values.append(value)
    if temp_max_values and "thermal_max_c" not in summary:
        summary["thermal_max_c"] = max(temp_max_values)
    return summary


def _load_dataset(input_dir: Path) -> dict[str, Any]:
    """Load and summarize an evaluation directory."""
    raw_rounds = _read_jsonl_rounds(input_dir / "rounds.jsonl")
    if _is_mix_runner_format(raw_rounds):
        turns = _flatten_mix_jsonl(raw_rounds)
    else:
        turns = _flatten_turns(raw_rounds)
    rounds = _group_rounds(turns)
    summary = _read_json(input_dir / "summary.json")
    thermal = _summarize_thermal(input_dir)
    thermal_curve = _read_json_array(input_dir / "thermal_curve.json")
    return {
        "input_dir": input_dir,
        "summary": summary,
        "thermal": thermal,
        "thermal_curve": thermal_curve,
        "raw_rounds": raw_rounds,
        "turns": turns,
        "rounds": rounds,
        "metrics": _summarize_turns(turns),
        "quality": _detect_honorific_issues(turns),
    }


def _comparison_rows(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Build A/B comparison rows."""
    current_metrics = current["metrics"]
    previous_metrics = previous["metrics"]
    rows: list[tuple[str, str, str]] = []

    def add_row(label: str, current_value: Any, previous_value: Any, unit: str = "") -> None:
        if current_value is None or previous_value is None:
            rows.append((label, "-", "-"))
            return
        delta = float(current_value) - float(previous_value)
        rows.append(
            (
                label,
                f"{_format_number(current_value, 3)}{unit}",
                f"{_format_number(previous_value, 3)}{unit}" + f" ({delta:+.3f}{unit})",
            ),
        )

    add_row("성공률", current_metrics["success_rate"], previous_metrics["success_rate"], "%")
    add_row("평균 TTFT", current_metrics["ttft_avg"], previous_metrics["ttft_avg"], "s")
    add_row("평균 LLM", current_metrics["llm_avg"], previous_metrics["llm_avg"], "s")
    add_row("평균 TTS", current_metrics["tts_avg"], previous_metrics["tts_avg"], "s")
    add_row("평균 전체", current_metrics["total_avg"], previous_metrics["total_avg"], "s")
    add_row("p95 전체", current_metrics["total_p95"], previous_metrics["total_p95"], "s")
    add_row("피크 메모리", current_metrics["memory_peak_kb"], previous_metrics["memory_peak_kb"])
    return rows


def _language_turns(turns: list[TurnRecord]) -> dict[str, list[TurnRecord]]:
    """Split turns into Korean and English buckets."""
    grouped: dict[str, list[TurnRecord]] = {"ko": [], "en": []}
    for turn in turns:
        language = "en" if turn.language == "en" else "ko"
        grouped[language].append(turn)
    return grouped


def _format_delta(current: float | int | None, baseline: float | int | None, unit: str = "") -> str:
    """Format a delta value for markdown tables."""
    if current is None or baseline is None:
        return "-"
    delta = float(current) - float(baseline)
    digits = 0 if isinstance(current, int) and isinstance(baseline, int) else 3
    return f"{delta:+.{digits}f}{unit}"


def _render_bilingual_summary(turns: list[TurnRecord]) -> list[str]:
    """Render a Korean/English stratified performance table."""
    grouped_turns = _language_turns(turns)
    ko_turns = grouped_turns["ko"]
    en_turns = grouped_turns["en"]
    ko_metrics = _summarize_turns(ko_turns)
    en_metrics = _summarize_turns(en_turns)
    ko_rounds = len({turn.round_num for turn in ko_turns})
    en_rounds = len({turn.round_num for turn in en_turns})

    rows = [
        ("Rounds count", ko_rounds, en_rounds, ""),
        ("Turn count", ko_metrics["turns"], en_metrics["turns"], ""),
        ("Success rate", ko_metrics["success_rate"], en_metrics["success_rate"], "%"),
        ("Avg TTFT", ko_metrics["ttft_avg"], en_metrics["ttft_avg"], "s"),
        ("p50 TTFT", ko_metrics["ttft_med"], en_metrics["ttft_med"], "s"),
        ("p95 TTFT", ko_metrics["ttft_p95"], en_metrics["ttft_p95"], "s"),
        ("Avg LLM time", ko_metrics["llm_avg"], en_metrics["llm_avg"], "s"),
        ("p50 LLM", ko_metrics["llm_med"], en_metrics["llm_med"], "s"),
        ("p95 LLM", ko_metrics["llm_p95"], en_metrics["llm_p95"], "s"),
        ("Avg TTS", ko_metrics["tts_avg"], en_metrics["tts_avg"], "s"),
        ("Avg total", ko_metrics["total_avg"], en_metrics["total_avg"], "s"),
        ("p50 total", ko_metrics["total_med"], en_metrics["total_med"], "s"),
        ("p95 total", ko_metrics["total_p95"], en_metrics["total_p95"], "s"),
        ("Avg tokens", ko_metrics["tokens_avg"], en_metrics["tokens_avg"], ""),
        ("Total tokens", ko_metrics["tokens_total"], en_metrics["tokens_total"], ""),
    ]

    lines = [
        "| Metric | KO | EN | Δ (EN vs KO) |",
        "|---|---:|---:|---:|",
    ]
    for label, ko_value, en_value, unit in rows:
        if unit == "s":
            ko_rendered = _format_seconds(cast(float | None, ko_value))
            en_rendered = _format_seconds(cast(float | None, en_value))
        elif unit == "%":
            ko_rendered = "-" if ko_value is None else f"{float(ko_value):.3f}%"
            en_rendered = "-" if en_value is None else f"{float(en_value):.3f}%"
        else:
            ko_rendered = _format_number(cast(float | int | None, ko_value), digits=3)
            en_rendered = _format_number(cast(float | int | None, en_value), digits=3)
        lines.append(
            f"| {label} | {ko_rendered} | {en_rendered} | "
            f"{_format_delta(cast(float | int | None, en_value), cast(float | int | None, ko_value), unit)} |",
        )
    return lines


def _render_error_details(turns: list[TurnRecord]) -> list[str]:
    """Render detailed failure bullets for each unsuccessful turn."""
    failures = [turn for turn in turns if not turn.success]
    if not failures:
        return ["- 실패한 turn이 없습니다."]

    lines: list[str] = []
    for turn in failures:
        error_message = turn.error or "-"
        lines.append(
            (
                f"- R{turn.round_num:02d} [{turn.language}] topic=`{turn.topic or '-'}` "
                f"turn={turn.exchange}: user=`{turn.user_text or '-'}`; "
                f"assistant=`{turn.assistant_text or '-'}`; error=`{error_message}`"
            ),
        )
    return lines


def _rounds_table(rounds: list[RoundRecord]) -> list[str]:
    """Render the round-by-round markdown table."""
    lines = [
        (
            "| 라운드 | 토픽 | 턴 수 | 성공 | 평균 TTFT | 평균 LLM | 평균 TTS | "
            "평균 전체 | 피크 메모리 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rounds:
        lines.append(
            (
                f"| R{item.round_num:02d} | {item.topic or '-'} | {item.turn_count} | "
                f"{item.success_count} | {_format_seconds(item.avg_ttft_s)} | "
                f"{_format_seconds(item.avg_llm_time_s)} | "
                f"{_format_seconds(item.avg_tts_time_s)} | "
                f"{_format_seconds(item.avg_total_time_s)} | "
                f"{_format_memory_kb(item.peak_memory_kb)} |"
            ),
        )
    return lines


def _segment_table(segments: list[dict[str, Any]]) -> list[str]:
    """Render the 10-round segment breakdown table."""
    lines = [
        "| 구간 | 턴 수 | 성공률 | 평균 TTFT | 평균 LLM | 평균 TTS | 평균 전체 | 피크 메모리 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for segment in segments:
        success_rate = "-"
        if segment["success_rate"] is not None:
            success_rate = f"{segment['success_rate']:.1f}%"
        lines.append(
            (
                f"| {segment['label']} | {segment['turns']} | "
                f"{success_rate} | "
                f"{_format_seconds(segment['avg_ttft_s'])} | "
                f"{_format_seconds(segment['avg_llm_time_s'])} | "
                f"{_format_seconds(segment['avg_tts_time_s'])} | "
                f"{_format_seconds(segment['avg_total_time_s'])} | "
                f"{_format_memory_kb(segment['peak_memory_kb'])} |"
            ),
        )
    return lines


def _render_thermal_section(thermal: dict[str, Any]) -> list[str]:
    """Render the thermal summary section."""
    lines = [f"- 샘플 수: {thermal.get('snapshots_count', 0):,}"]
    thermal_max_c = _as_float(thermal.get("thermal_max_c"))
    if thermal_max_c is not None:
        lines.append(f"- thermal_max_c: {thermal_max_c:.3f}°C")
    for key, label, unit in (
        ("cpu_temp_c", "CPU", "°C"),
        ("gpu_temp_c", "GPU", "°C"),
        ("ram_used_mb", "RAM", "MB"),
        ("gr3d_freq_pct", "GR3D_FREQ", "%"),
    ):
        value = thermal.get(key)
        if not isinstance(value, dict):
            continue
        start = value.get("start")
        end = value.get("end")
        minimum = value.get("min")
        maximum = value.get("max")
        average = value.get("avg")
        delta = value.get("delta")
        start_value = _as_float(start)
        end_value = _as_float(end)
        minimum_value = _as_float(minimum)
        maximum_value = _as_float(maximum)
        average_value = _as_float(average)
        delta_value = _as_float(delta)
        if any(
            item is None
            for item in (
                start_value,
                end_value,
                minimum_value,
                maximum_value,
                average_value,
                delta_value,
            )
        ):
            continue
        lines.append(
            f"- {label}: start {start_value:.3f}{unit}, end {end_value:.3f}{unit}, "
            f"min {minimum_value:.3f}{unit}, max {maximum_value:.3f}{unit}, "
            f"avg {average_value:.3f}{unit}, delta {delta_value:.3f}{unit}",
        )
    if len(lines) == 1:
        lines.append("- 온도 요약 데이터가 없습니다.")
    return lines


def _summary_counter_value(summary: dict[str, Any], key: str) -> int | None:
    if key not in summary:
        return None
    value = _as_int(summary.get(key))
    if value is None or value < 0:
        return None
    return value


def _render_stability_section(
    summary: dict[str, Any],
    previous_summary: dict[str, Any] | None,
) -> list[str]:
    """Render summary.json stability counters for Wave 3 gate checks."""
    current_available = all(key in summary for key in STABILITY_COUNTER_KEYS)
    previous_available = previous_summary is not None and all(
        key in previous_summary for key in STABILITY_COUNTER_KEYS
    )
    lines = [
        "| 지표 | 값 | Wave 2 final | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key in STABILITY_COUNTER_KEYS:
        current_value = _summary_counter_value(summary, key) if current_available else None
        baseline_value = (
            _summary_counter_value(previous_summary, key)
            if previous_summary is not None and previous_available
            else None
        )
        lines.append(
            f"| {key} | {_format_number(current_value, 0)} | "
            f"{_format_number(baseline_value, 0)} | "
            f"{_format_delta(current_value, baseline_value)} |",
        )
    if not current_available:
        lines.append("")
        lines.append("(summary.json predates T3.0 - counters unavailable)")
    return lines


def _render_quality_section(quality: dict[str, Any]) -> list[str]:
    """Render the quality analysis section."""
    lines = [
        f"- 존댓말 누출 의심 turn 수: {quality['honorific_turn_count']:,}",
        f"- 반복 응답 관측 수: {quality['repetition_count']:,}",
        f"- 문맥 누출 의심 turn 수: {quality['context_leak_count']:,}",
    ]
    honorific_examples = quality["honorific_examples"]
    if honorific_examples:
        joined = "; ".join(
            f"R{item.round_num:02d} `{_truncate_text(item.assistant_text)}`"
            for item in honorific_examples[:3]
        )
        lines.append(f"- 존댓말 예시: {joined}")
    repeated_responses = quality["repeated_responses"]
    if repeated_responses:
        joined = "; ".join(f"`{item['text']}` x{item['count']}" for item in repeated_responses[:3])
        lines.append(f"- 반복 응답 상위: {joined}")
    context_examples = quality["context_leak_examples"]
    if context_examples:
        joined = "; ".join(
            (
                f"R{item['round_num']:02d} 현재 `{item['current_topic']}` "
                f"-> 이전 `{item['leaked_topic']}`"
            )
            for item in context_examples[:3]
        )
        lines.append(f"- 문맥 누출 예시: {joined}")
    return lines


def _render_late_round_analysis(rounds: list[RoundRecord]) -> list[str]:
    """Render a short late-round analysis for rounds 51-60."""
    late_rounds = [item for item in rounds if item.round_num >= 51]
    if not late_rounds:
        return ["- 후반 라운드 데이터가 없습니다."]

    avg_total = _mean([item.avg_total_time_s for item in late_rounds])
    avg_llm = _mean([item.avg_llm_time_s for item in late_rounds])
    avg_ttft = _mean([item.avg_ttft_s for item in late_rounds])
    early_rounds = [item for item in rounds if 1 <= item.round_num <= 10]
    early_total = _mean([item.avg_total_time_s for item in early_rounds])
    success_rate = (
        sum(item.success_count for item in late_rounds)
        / sum(item.turn_count for item in late_rounds)
        * 100.0
    )
    peak_memory = max(
        (item.peak_memory_kb for item in late_rounds if item.peak_memory_kb is not None),
        default=None,
    )
    lines = [
        f"- R51-60 평균 전체: {_format_seconds(avg_total)}",
        f"- R51-60 평균 LLM: {_format_seconds(avg_llm)}",
        f"- R51-60 평균 TTFT: {_format_seconds(avg_ttft)}",
        f"- R51-60 성공률: {success_rate:.1f}%",
        f"- R51-60 피크 메모리: {_format_memory_kb(peak_memory)}",
    ]
    if avg_total is not None and early_total is not None and early_total > 0:
        slowdown_pct = (avg_total - early_total) / early_total * 100.0
        lines.append(f"- R1-10 대비 전체 지연 변화: {slowdown_pct:+.1f}%")
    return lines


def _render_performance_summary(metrics: dict[str, Any]) -> list[str]:
    """Render the overall performance summary section."""
    lines = [
        f"- 평균 TTFT: {_format_seconds(metrics['ttft_avg'])}",
        f"- 중앙값 TTFT: {_format_seconds(metrics['ttft_med'])}",
        (
            f"- p5 / p95 TTFT: {_format_seconds(metrics['ttft_p5'])} / "
            f"{_format_seconds(metrics['ttft_p95'])}"
        ),
        f"- 평균 LLM: {_format_seconds(metrics['llm_avg'])}",
        f"- 중앙값 LLM: {_format_seconds(metrics['llm_med'])}",
        (
            f"- p5 / p95 LLM: {_format_seconds(metrics['llm_p5'])} / "
            f"{_format_seconds(metrics['llm_p95'])}"
        ),
        f"- 평균 TTS: {_format_seconds(metrics['tts_avg'])}",
        f"- 중앙값 TTS: {_format_seconds(metrics['tts_med'])}",
        (
            f"- p5 / p95 TTS: {_format_seconds(metrics['tts_p5'])} / "
            f"{_format_seconds(metrics['tts_p95'])}"
        ),
        f"- 평균 전체: {_format_seconds(metrics['total_avg'])}",
        f"- 중앙값 전체: {_format_seconds(metrics['total_med'])}",
        (
            f"- p5 / p95 전체: {_format_seconds(metrics['total_p5'])} / "
            f"{_format_seconds(metrics['total_p95'])}"
        ),
        f"- 총 토큰 수: {metrics['tokens_total']:,}",
        f"- 평균 토큰 수: {_format_number(metrics['tokens_avg'])}",
        f"- 피크 메모리: {_format_memory_mb(metrics['memory_peak_kb'])}",
    ]
    return lines


def _render_test_scale(metrics: dict[str, Any], rounds: list[RoundRecord]) -> list[str]:
    """Render the test scale section."""
    if not rounds:
        return ["- 라운드 데이터가 없습니다."]
    total_turns = metrics["turns"]
    return [
        f"- 총 라운드 수: {len(rounds):,}",
        f"- 총 턴 수: {total_turns:,}",
        f"- 성공 턴 수: {metrics['success_count']:,}",
        f"- 실패 턴 수: {metrics['failure_count']:,}",
        f"- 성공률: {metrics['success_rate']:.1f}%" if metrics["success_rate"] is not None else "-",
        f"- 라운드 범위: R{rounds[0].round_num:02d} ~ R{rounds[-1].round_num:02d}",
    ]


def _format_ms_and_seconds(value_ms: float | None) -> str:
    """Format a millisecond metric with a seconds companion value."""
    if value_ms is None:
        return "n/a"
    return f"{value_ms:.3f} ms ({value_ms / 1000.0:.3f} s)"


def _render_first_sound_breakdown(summary: dict[str, Any]) -> list[str]:
    """Render the first-sound breakdown section when mix-runner metrics exist."""
    avg_tts_first_chunk_ms = _as_float(summary.get("avg_tts_first_chunk_ms"))
    if avg_tts_first_chunk_ms is None:
        return []
    avg_first_sound_ms = _as_float(summary.get("avg_first_sound_ms"))
    return [
        f"- Avg first sound: {_format_ms_and_seconds(avg_first_sound_ms)}",
        f"- Avg TTS first chunk: {_format_ms_and_seconds(avg_tts_first_chunk_ms)}",
    ]


def _render_ttft_turn_index_split(summary: dict[str, Any]) -> list[str]:
    """Render TTFT split metrics for first and later same-language turns."""
    first_turn = _as_float(summary.get("avg_llm_ttft_ms_first_turn"))
    after_first = _as_float(summary.get("avg_llm_ttft_ms_after_first"))
    if first_turn is None and after_first is None:
        return []
    return [
        f"- First turn per language: {_format_ms_and_seconds(first_turn)}",
        f"- After first turn: {_format_ms_and_seconds(after_first)}",
    ]


def _render_llm_cache_hit_rate(summary: dict[str, Any]) -> list[str]:
    """Render the aggregate LLM cache hit-rate section when present."""
    avg_cache_hit_rate = _as_float(summary.get("avg_llm_cache_hit_rate"))
    if avg_cache_hit_rate is None:
        return []
    return [f"- Avg cache hit rate: {avg_cache_hit_rate * 100.0:.1f}%"]


def _render_improvement_suggestions(
    analysis: dict[str, Any],
    rounds: list[RoundRecord],
) -> list[str]:
    """Render action-oriented improvement suggestions."""
    late_rounds = [item for item in rounds if item.round_num >= 51]
    late_total = _mean([item.avg_total_time_s for item in late_rounds])
    early_rounds = [item for item in rounds if item.round_num <= 10]
    early_total = _mean([item.avg_total_time_s for item in early_rounds])
    lines: list[str] = []

    if analysis["honorific_turn_count"] > 0:
        lines.append("- 존댓말 규칙을 더 강하게 고정하고, 금지 어미에 대한 후처리를 추가하세요.")
    if analysis["repetition_count"] > 0:
        lines.append("- 반복 prefix가 보이므로 anti-echo 규칙과 반복 패널티를 강화하세요.")
    if analysis["context_leak_count"] > 0:
        lines.append("- 이전 발화/이전 라운드 참조를 줄이도록 컨텍스트 필터를 조정하세요.")
    if late_total is not None and early_total is not None and late_total > early_total * 1.2:
        lines.append(
            "- 후반 라운드 지연이 커서 history truncation 또는 KV cache 관리가 필요합니다."
        )
    if not lines:
        lines.append("- 현재 데이터만으로는 우선순위가 높은 개선 신호가 적습니다.")
    return lines


def _render_report(
    data: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    bilingual: bool = False,
) -> str:
    """Render the final markdown report."""
    metrics = data["metrics"]
    rounds = data["rounds"]
    segments = _summarize_segments(rounds)
    quality = data["quality"]
    thermal = data["thermal"]
    summary = data["summary"] or {}
    raw_rounds = data.get("raw_rounds", [])
    thermal_curve = data.get("thermal_curve", [])
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines: list[str] = []
    lines.append("# E2E 60라운드 리포트")
    lines.append("")
    lines.append(f"> 생성 시각: {timestamp}")
    lines.append(f"> 입력 디렉터리: `{data['input_dir']}`")
    if summary:
        lines.append(
            f"> 실행 설정: rounds={summary.get('rounds', '-')}, "
            f"playback={summary.get('playback_enabled', '-')}"
        )
        lines.append(f"> summary.json peak memory: `{summary.get('peak_memory_kb', '-')}`")
    lines.append("")
    lines.append("## 개요")
    if summary:
        lines.append("- summary.json 존재: 예")
        lines.append(
            "- thermal_summary.json 존재: "
            f"{'예' if thermal.get('snapshots_count', 0) else '아니오'}"
        )
        lines.append(f"- baseline 비교: {'예' if previous is not None else '아니오'}")
    else:
        lines.append("- summary.json이 없어 메타데이터 일부를 생략했습니다.")
    lines.append("")
    lines.append("## 테스트 규모")
    lines.extend(_render_test_scale(metrics, rounds))
    lines.append("")
    lines.append("## 성능 요약")
    lines.extend(_render_performance_summary(metrics))
    lines.append("")
    mix_latency_table = _render_mix_latency_table(data["turns"])
    if mix_latency_table:
        lines.append("## Canonical Per-Turn Latency")
        lines.extend(mix_latency_table)
        lines.append("")
    pass_aggregates = _render_pass_aggregates(data["turns"], thermal)
    if pass_aggregates:
        lines.append("## Per-Pass Aggregates")
        lines.extend(pass_aggregates)
        lines.append("")
    first_sound_breakdown = _render_first_sound_breakdown(summary)
    if first_sound_breakdown:
        lines.append("## First Sound Breakdown")
        lines.extend(first_sound_breakdown)
        lines.append("")
    ttft_turn_index_split = _render_ttft_turn_index_split(summary)
    if ttft_turn_index_split:
        lines.append("## TTFT Turn Index Split")
        lines.extend(ttft_turn_index_split)
        lines.append("")
    llm_cache_hit_rate = _render_llm_cache_hit_rate(summary)
    if llm_cache_hit_rate:
        lines.append("## LLM Cache Hit Rate")
        lines.extend(llm_cache_hit_rate)
        lines.append("")
    if bilingual:
        lines.append("## 언어별 성능 요약")
        lines.extend(_render_bilingual_summary(data["turns"]))
        lines.append("")
    lines.append("## 구간별 요약")
    lines.append("### 10라운드 구간")
    lines.extend(_segment_table(segments))
    lines.append("")
    lines.append("## 라운드별 표")
    lines.extend(_rounds_table(rounds))
    lines.append("")
    lines.append("## 품질 분석")
    lines.extend(_render_quality_section(quality))
    lines.append("")
    lines.append("## 후반 라운드 분석")
    lines.extend(_render_late_round_analysis(rounds))
    lines.append("")
    lines.append("## 열 분석")
    lines.extend(_render_thermal_section(thermal))
    lines.append("")
    lines.append("## Thermal Curve")
    lines.extend(_render_thermal_curve(thermal_curve))
    lines.append("")
    lines.append("## Memory Envelope")
    lines.extend(_render_memory_envelope(raw_rounds))
    lines.append("")
    lines.append("## Stability")
    previous_summary = previous["summary"] if previous is not None else None
    lines.extend(_render_stability_section(summary, previous_summary))
    lines.append("")
    lines.append("## Gate Verdicts")
    lines.extend(_render_gate_verdicts(raw_rounds, summary, thermal))
    lines.append("")
    lines.append("## Reproducibility Appendix")
    lines.extend(_render_reproducibility_appendix(summary))

    if bilingual:
        lines.append("")
        lines.append("## 오류 상세")
        lines.extend(_render_error_details(data["turns"]))

    if previous is not None:
        lines.append("")
        lines.append("## A/B 비교")
        rows = _comparison_rows(data, previous)
        lines.append("| 항목 | 현재 | 이전 |")
        lines.append("|---|---:|---:|")
        for label, current_value, previous_value in rows:
            lines.append(f"| {label} | {current_value} | {previous_value} |")

    lines.append("")
    lines.append("## 개선 제안")
    lines.extend(_render_improvement_suggestions(quality, rounds))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Entry point for the report generator CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    logger.info("Generating report from %s", input_dir)

    current = _load_dataset(input_dir)
    previous = _load_dataset(args.previous.resolve()) if args.previous is not None else None
    report = _render_report(current, previous, bilingual=args.bilingual)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
