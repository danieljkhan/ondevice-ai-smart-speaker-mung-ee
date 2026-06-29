"""Generate a 12-model E2E comparison report and conversation script bundle."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger("mungi.scripts.generate_12model_report")

DEFAULT_OUTPUT_DIR = Path("docs/runbooks/weekly")
DEFAULT_INPUT_BASE = Path("/tmp")
REPORT_DATE = date(2026, 4, 5)
JETSON_MEMORY_BUDGET_MB = 8 * 1024
HONORIFIC_SUFFIXES: tuple[str, ...] = ("요", "습니다", "세요", "해요", "죠", "까요", "네요")
EVASION_PATTERNS: tuple[str, ...] = (
    "모르겠",
    "같이 알아",
    "같이 생각",
    "같이 찾아",
    "궁금해",
    "엄마아빠한테",
)


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for one evaluated model."""

    label: str
    relative_jsonl: Path
    family: str
    quant: str
    finetune: str
    file_size: str


@dataclass(frozen=True)
class TurnRecord:
    """Flattened per-turn record extracted from rounds.jsonl."""

    round_num: int
    exchange: int
    topic: str
    user_text: str
    assistant_text: str
    success: bool
    error: str | None
    total_time_s: float | None
    llm_time_s: float | None
    tts_time_s: float | None
    llm_ttft_s: float | None
    llm_tokens: int | None
    peak_memory_kb: int | None


@dataclass(frozen=True)
class MetricSummary:
    """Summary statistics for one numeric metric."""

    count: int
    average: float | None
    median: float | None
    p5: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class SegmentSummary:
    """Aggregated metrics for a 10-round segment."""

    label: str
    round_start: int
    round_end: int
    avg_total_time_s: float | None
    success_rate_pct: float | None


@dataclass(frozen=True)
class ModelReport:
    """Computed metrics and source data for one model."""

    spec: ModelSpec
    source_path: Path
    turns: list[TurnRecord]
    available: bool
    warning: str | None
    total_time: MetricSummary
    llm_time: MetricSummary
    tts_time: MetricSummary
    ttft_time: MetricSummary
    llm_tokens: MetricSummary
    memory_mb: MetricSummary
    success_count: int
    failure_count: int
    success_rate_pct: float | None
    honorific_count: int
    honorific_pct: float | None
    evasion_count: int
    evasion_pct: float | None
    unique_responses: int
    segments: list[SegmentSummary]
    late_early_ratio: float | None
    kpi_p50_pass: bool
    kpi_p95_pass: bool
    kpi_success_pass: bool
    kpi_honorific_pass: bool
    ranking_score: float


TESTS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "Q3 Q4_K_M SFT", Path("e2e_q3_q4_sft/rounds.jsonl"), "Qwen3-1.7B", "Q4_K_M", "SFT", "1.1GB"
    ),
    ModelSpec(
        "Q3 Q4_K_M DPO", Path("e2e_q3_q4_dpo/rounds.jsonl"), "Qwen3-1.7B", "Q4_K_M", "DPO", "1.1GB"
    ),
    ModelSpec(
        "Q3 Q6_K SFT", Path("e2e_q3_q6_sft/rounds.jsonl"), "Qwen3-1.7B", "Q6_K", "SFT", "1.4GB"
    ),
    ModelSpec(
        "Q3 Q6_K DPO", Path("e2e_q3_q6_dpo/rounds.jsonl"), "Qwen3-1.7B", "Q6_K", "DPO", "1.4GB"
    ),
    ModelSpec(
        "Q3 Q8_0 SFT", Path("e2e_q3_q8_sft/rounds.jsonl"), "Qwen3-1.7B", "Q8_0", "SFT", "1.8GB"
    ),
    ModelSpec(
        "Q3 Q8_0 DPO", Path("e2e_q3_q8_dpo/rounds.jsonl"), "Qwen3-1.7B", "Q8_0", "DPO", "1.8GB"
    ),
    ModelSpec(
        "Q35 Q4_K_M SFT",
        Path("e2e_q35_q4_sft/rounds.jsonl"),
        "Qwen3.5-2B",
        "Q4_K_M",
        "SFT",
        "1.2GB",
    ),
    ModelSpec(
        "Q35 Q6_K SFT", Path("e2e_q35_q6_sft/rounds.jsonl"), "Qwen3.5-2B", "Q6_K", "SFT", "1.5GB"
    ),
    ModelSpec(
        "Q35 Q8_0 SFT", Path("e2e_q35_q8_sft/rounds.jsonl"), "Qwen3.5-2B", "Q8_0", "SFT", "1.9GB"
    ),
    ModelSpec(
        "Q35 Q4_K_M DPO",
        Path("e2e_q35_q4_dpo/rounds.jsonl"),
        "Qwen3.5-2B",
        "Q4_K_M",
        "DPO",
        "1.2GB",
    ),
    ModelSpec(
        "Q35 Q6_K DPO", Path("e2e_q35_q6_dpo/rounds.jsonl"), "Qwen3.5-2B", "Q6_K", "DPO", "1.5GB"
    ),
    ModelSpec(
        "Q35 Q8_0 DPO", Path("e2e_q35_q8_dpo/rounds.jsonl"), "Qwen3.5-2B", "Q8_0", "DPO", "1.9GB"
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate 12-model E2E comparison and conversation markdown outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for markdown outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--input-base",
        type=Path,
        default=DEFAULT_INPUT_BASE,
        help=f"Base directory that contains copied E2E result folders (default: {DEFAULT_INPUT_BASE}).",
    )
    return parser


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return parsed objects."""
    if not path.exists():
        logger.warning("Missing JSONL file: %s", path)
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSONL line in %s: %s", path, exc)
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                else:
                    logger.warning("Skipping non-object JSONL entry in %s", path)
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return rows


def _flatten_turns(raw_rounds: Sequence[dict[str, Any]]) -> list[TurnRecord]:
    """Flatten nested JSONL rounds into a turn list."""
    turns: list[TurnRecord] = []
    for raw_round in raw_rounds:
        default_round = _as_int(raw_round.get("round")) or 0
        topics = raw_round.get("topics")
        if not isinstance(topics, list):
            continue
        for topic_item in topics:
            if not isinstance(topic_item, dict):
                continue
            topic_name = str(topic_item.get("topic") or raw_round.get("topic") or "")
            topic_turns = topic_item.get("turns")
            if not isinstance(topic_turns, list):
                continue
            for turn in topic_turns:
                if not isinstance(turn, dict):
                    continue
                round_num = _as_int(turn.get("round_num")) or default_round
                exchange = _as_int(turn.get("exchange")) or 0
                assistant_text = str(turn.get("assistant_text") or turn.get("assistant") or "")
                user_text = str(turn.get("user_text") or turn.get("user") or "")
                turns.append(
                    TurnRecord(
                        round_num=round_num,
                        exchange=exchange,
                        topic=str(turn.get("topic") or topic_name),
                        user_text=user_text,
                        assistant_text=assistant_text,
                        success=bool(turn.get("success", False)),
                        error=str(turn.get("error")) if turn.get("error") else None,
                        total_time_s=_as_float(turn.get("total_time_s") or turn.get("time_s")),
                        llm_time_s=_as_float(turn.get("llm_time_s")),
                        tts_time_s=_as_float(turn.get("tts_time_s")),
                        llm_ttft_s=_as_float(turn.get("llm_ttft_s") or turn.get("ttft_s")),
                        llm_tokens=_as_int(turn.get("llm_tokens") or turn.get("tokens")),
                        peak_memory_kb=_as_int(turn.get("peak_memory_kb")),
                    ),
                )
    turns.sort(key=lambda item: (item.round_num, item.exchange))
    return turns


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """Compute an interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    weight = position - lower_index
    return lower + (upper - lower) * weight


def _summarize_numeric(values: Sequence[float]) -> MetricSummary:
    """Build a summary object for numeric values."""
    if not values:
        return MetricSummary(0, None, None, None, None, None, None)
    numeric = [float(value) for value in values]
    return MetricSummary(
        count=len(numeric),
        average=statistics.fmean(numeric),
        median=statistics.median(numeric),
        p5=_percentile(numeric, 5),
        p95=_percentile(numeric, 95),
        minimum=min(numeric),
        maximum=max(numeric),
    )


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the arithmetic mean over defined values."""
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return statistics.fmean(filtered)


def _count_honorific(turns: Sequence[TurnRecord]) -> int:
    """Count assistant turns that end with honorific endings."""
    count = 0
    for turn in turns:
        text = turn.assistant_text.strip()
        if not text:
            continue
        compact = text.rstrip(" .,!?:;~\"'")
        if any(compact.endswith(suffix) for suffix in HONORIFIC_SUFFIXES):
            count += 1
    return count


def _count_evasion(turns: Sequence[TurnRecord]) -> int:
    """Count assistant turns that match evasion phrases."""
    count = 0
    for turn in turns:
        text = turn.assistant_text
        if any(pattern in text for pattern in EVASION_PATTERNS):
            count += 1
    return count


def _compute_segments(turns: Sequence[TurnRecord]) -> list[SegmentSummary]:
    """Compute R1-10 to R51-60 segment summaries."""
    segments: list[SegmentSummary] = []
    for start in range(1, 61, 10):
        end = start + 9
        in_segment = [turn for turn in turns if start <= turn.round_num <= end]
        total_values = [turn.total_time_s for turn in in_segment if turn.total_time_s is not None]
        success_count = sum(1 for turn in in_segment if turn.success)
        success_rate = (success_count / len(in_segment) * 100.0) if in_segment else None
        segments.append(
            SegmentSummary(
                label=f"R{start}-{end}",
                round_start=start,
                round_end=end,
                avg_total_time_s=_mean(total_values),
                success_rate_pct=success_rate,
            ),
        )
    return segments


def _compute_late_early_ratio(segments: Sequence[SegmentSummary]) -> float | None:
    """Compute the R51-60 / R1-10 average total-time ratio."""
    early = next(
        (segment.avg_total_time_s for segment in segments if segment.round_start == 1),
        None,
    )
    late = next(
        (segment.avg_total_time_s for segment in segments if segment.round_start == 51),
        None,
    )
    if early is None or late is None or early == 0:
        return None
    return late / early


def _score_model(
    success_rate_pct: float | None,
    honorific_pct: float | None,
    evasion_pct: float | None,
    total_p95: float | None,
    total_average: float | None,
    unique_responses: int,
    memory_peak_mb: float | None,
) -> float:
    """Build a pragmatic ranking score for recommendation ordering."""
    score = 0.0
    score += (success_rate_pct or 0.0) * 4.0
    score -= (honorific_pct or 0.0) * 6.0
    score -= (evasion_pct or 0.0) * 2.0
    score -= (total_p95 or 99.0) * 7.0
    score -= (total_average or 99.0) * 5.0
    score += unique_responses * 0.2
    score -= (memory_peak_mb or 9999.0) * 0.03
    return score


def _load_model_report(spec: ModelSpec, input_base: Path) -> ModelReport:
    """Load one model dataset and compute all report metrics."""
    source_path = input_base / spec.relative_jsonl
    raw_rounds = _read_jsonl(source_path)
    turns = _flatten_turns(raw_rounds)

    if not turns:
        warning = f"입력 파일이 없거나 유효한 turn 데이터가 없습니다: {source_path}"
        return ModelReport(
            spec=spec,
            source_path=source_path,
            turns=[],
            available=False,
            warning=warning,
            total_time=_summarize_numeric([]),
            llm_time=_summarize_numeric([]),
            tts_time=_summarize_numeric([]),
            ttft_time=_summarize_numeric([]),
            llm_tokens=_summarize_numeric([]),
            memory_mb=_summarize_numeric([]),
            success_count=0,
            failure_count=0,
            success_rate_pct=None,
            honorific_count=0,
            honorific_pct=None,
            evasion_count=0,
            evasion_pct=None,
            unique_responses=0,
            segments=_compute_segments([]),
            late_early_ratio=None,
            kpi_p50_pass=False,
            kpi_p95_pass=False,
            kpi_success_pass=False,
            kpi_honorific_pass=False,
            ranking_score=float("-inf"),
        )

    total_turns = len(turns)
    success_count = sum(1 for turn in turns if turn.success)
    failure_count = total_turns - success_count
    success_rate_pct = success_count / total_turns * 100.0
    honorific_count = _count_honorific(turns)
    evasion_count = _count_evasion(turns)
    honorific_pct = honorific_count / total_turns * 100.0
    evasion_pct = evasion_count / total_turns * 100.0
    unique_responses = len({turn.assistant_text for turn in turns if turn.assistant_text.strip()})

    total_time = _summarize_numeric(
        [value for value in (turn.total_time_s for turn in turns) if value is not None]
    )
    llm_time = _summarize_numeric(
        [value for value in (turn.llm_time_s for turn in turns) if value is not None]
    )
    tts_time = _summarize_numeric(
        [value for value in (turn.tts_time_s for turn in turns) if value is not None]
    )
    ttft_time = _summarize_numeric(
        [value for value in (turn.llm_ttft_s for turn in turns) if value is not None]
    )
    llm_tokens = _summarize_numeric(
        [float(value) for value in (turn.llm_tokens for turn in turns) if value is not None]
    )
    memory_mb = _summarize_numeric(
        [value / 1024.0 for value in (turn.peak_memory_kb for turn in turns) if value is not None]
    )
    segments = _compute_segments(turns)
    late_early_ratio = _compute_late_early_ratio(segments)
    ranking_score = _score_model(
        success_rate_pct,
        honorific_pct,
        evasion_pct,
        total_time.p95,
        total_time.average,
        unique_responses,
        memory_mb.maximum,
    )

    return ModelReport(
        spec=spec,
        source_path=source_path,
        turns=turns,
        available=True,
        warning=None,
        total_time=total_time,
        llm_time=llm_time,
        tts_time=tts_time,
        ttft_time=ttft_time,
        llm_tokens=llm_tokens,
        memory_mb=memory_mb,
        success_count=success_count,
        failure_count=failure_count,
        success_rate_pct=success_rate_pct,
        honorific_count=honorific_count,
        honorific_pct=honorific_pct,
        evasion_count=evasion_count,
        evasion_pct=evasion_pct,
        unique_responses=unique_responses,
        segments=segments,
        late_early_ratio=late_early_ratio,
        kpi_p50_pass=(total_time.median or float("inf")) <= 8.0,
        kpi_p95_pass=(total_time.p95 or float("inf")) <= 10.0,
        kpi_success_pass=math.isclose(success_rate_pct, 100.0, rel_tol=0.0, abs_tol=1e-9),
        kpi_honorific_pass=(honorific_pct or float("inf")) <= 1.0,
        ranking_score=ranking_score,
    )


def _fmt_float(value: float | None, digits: int = 2) -> str:
    """Format a float for markdown tables."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    """Format a percent value for markdown tables."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def _fmt_ratio(value: float | None) -> str:
    """Format a ratio for markdown tables."""
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _status(pass_flag: bool) -> str:
    """Return a Korean pass/fail marker."""
    return "통과" if pass_flag else "미통과"


def _best_model(models: Sequence[ModelReport]) -> ModelReport | None:
    """Return the top-ranked available model."""
    available = [model for model in models if model.available]
    if not available:
        return None
    return max(available, key=lambda item: item.ranking_score)


def _top_models(models: Sequence[ModelReport], count: int = 3) -> list[ModelReport]:
    """Return the top-ranked available models."""
    available = [model for model in models if model.available]
    return sorted(available, key=lambda item: item.ranking_score, reverse=True)[:count]


def _render_test_matrix(models: Sequence[ModelReport]) -> list[str]:
    """Render the test matrix section."""
    lines = [
        "| 모델 | 계열 | 양자화 | FT | 파일 크기 | 입력 파일 | 상태 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for model in models:
        status = "수집 완료" if model.available else "누락"
        lines.append(
            f"| {model.spec.label} | {model.spec.family} | {model.spec.quant} | "
            f"{model.spec.finetune} | {model.spec.file_size} | `{model.source_path}` | {status} |"
        )
    return lines


def _render_performance_table(models: Sequence[ModelReport]) -> list[str]:
    """Render the main performance comparison table."""
    lines = [
        "| 모델 | 평균 전체 | p50 전체 | p95 전체 | 평균 LLM | 평균 TTS | 평균 TTFT | 평균 토큰 | Peak MB | 성공률 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| {model.spec.label} | {_fmt_float(model.total_time.average)} | "
            f"{_fmt_float(model.total_time.median)} | {_fmt_float(model.total_time.p95)} | "
            f"{_fmt_float(model.llm_time.average)} | {_fmt_float(model.tts_time.average)} | "
            f"{_fmt_float(model.ttft_time.average)} | {_fmt_float(model.llm_tokens.average, 1)} | "
            f"{_fmt_float(model.memory_mb.maximum, 0)} | {_fmt_pct(model.success_rate_pct)} |"
        )
    return lines


def _render_family_comparison(models: Sequence[ModelReport]) -> list[str]:
    """Render Qwen3 vs Qwen3.5 comparisons."""
    lines = [
        "| 비교 | 평균 전체 차이 | 평균 토큰 차이 | Peak MB 차이 | 성공률 차이 |",
        "|---|---:|---:|---:|---:|",
    ]
    found = False
    for quant in ("Q4_K_M", "Q6_K", "Q8_0"):
        for finetune in ("SFT", "DPO"):
            q3 = next(
                (
                    m
                    for m in models
                    if m.spec.family == "Qwen3-1.7B"
                    and m.spec.quant == quant
                    and m.spec.finetune == finetune
                    and m.available
                ),
                None,
            )
            q35 = next(
                (
                    m
                    for m in models
                    if m.spec.family == "Qwen3.5-2B"
                    and m.spec.quant == quant
                    and m.spec.finetune == finetune
                    and m.available
                ),
                None,
            )
            if q3 is None or q35 is None:
                continue
            found = True
            lines.append(
                f"| {quant} {finetune} (Q3.5 - Q3) | "
                f"{_fmt_float((q35.total_time.average or 0.0) - (q3.total_time.average or 0.0))} | "
                f"{_fmt_float((q35.llm_tokens.average or 0.0) - (q3.llm_tokens.average or 0.0), 1)} | "
                f"{_fmt_float((q35.memory_mb.maximum or 0.0) - (q3.memory_mb.maximum or 0.0), 0)} | "
                f"{_fmt_pct((q35.success_rate_pct or 0.0) - (q3.success_rate_pct or 0.0))} |"
            )
    return lines if found else ["- 비교 가능한 Qwen3 / Qwen3.5 짝이 없습니다."]


def _render_ft_comparison(models: Sequence[ModelReport]) -> list[str]:
    """Render SFT vs DPO comparisons for the same family and quant."""
    rows = [
        "| 비교 | 평균 전체 차이 (DPO-SFT) | 존댓말 차이 | 회피 차이 | 고유 응답 차이 |",
        "|---|---:|---:|---:|---:|",
    ]
    found = False
    for family in ("Qwen3-1.7B", "Qwen3.5-2B"):
        for quant in ("Q4_K_M", "Q6_K", "Q8_0"):
            sft = next(
                (
                    m
                    for m in models
                    if m.spec.family == family
                    and m.spec.quant == quant
                    and m.spec.finetune == "SFT"
                    and m.available
                ),
                None,
            )
            dpo = next(
                (
                    m
                    for m in models
                    if m.spec.family == family
                    and m.spec.quant == quant
                    and m.spec.finetune == "DPO"
                    and m.available
                ),
                None,
            )
            if sft is None or dpo is None:
                continue
            found = True
            rows.append(
                f"| {family} {quant} | {_fmt_float((dpo.total_time.average or 0.0) - (sft.total_time.average or 0.0))} | "
                f"{_fmt_pct((dpo.honorific_pct or 0.0) - (sft.honorific_pct or 0.0))} | "
                f"{_fmt_pct((dpo.evasion_pct or 0.0) - (sft.evasion_pct or 0.0))} | "
                f"{dpo.unique_responses - sft.unique_responses} |"
            )
    return rows if found else ["- 비교 가능한 SFT / DPO 짝이 없습니다."]


def _render_quant_comparison(models: Sequence[ModelReport]) -> list[str]:
    """Render quantization comparisons within the same family and finetune."""
    rows = [
        "| 그룹 | Q4 평균 전체 | Q6 평균 전체 | Q8 평균 전체 | Q4 Peak MB | Q6 Peak MB | Q8 Peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    found = False
    for family in ("Qwen3-1.7B", "Qwen3.5-2B"):
        for finetune in ("SFT", "DPO"):
            candidates = {
                quant: next(
                    (
                        m
                        for m in models
                        if m.spec.family == family
                        and m.spec.quant == quant
                        and m.spec.finetune == finetune
                        and m.available
                    ),
                    None,
                )
                for quant in ("Q4_K_M", "Q6_K", "Q8_0")
            }
            if sum(1 for item in candidates.values() if item is not None) < 2:
                continue
            found = True
            rows.append(
                f"| {family} {finetune} | {_fmt_float(candidates['Q4_K_M'].total_time.average if candidates['Q4_K_M'] else None)} | "
                f"{_fmt_float(candidates['Q6_K'].total_time.average if candidates['Q6_K'] else None)} | "
                f"{_fmt_float(candidates['Q8_0'].total_time.average if candidates['Q8_0'] else None)} | "
                f"{_fmt_float(candidates['Q4_K_M'].memory_mb.maximum if candidates['Q4_K_M'] else None, 0)} | "
                f"{_fmt_float(candidates['Q6_K'].memory_mb.maximum if candidates['Q6_K'] else None, 0)} | "
                f"{_fmt_float(candidates['Q8_0'].memory_mb.maximum if candidates['Q8_0'] else None, 0)} |"
            )
    return rows if found else ["- 비교 가능한 Q4 / Q6 / Q8 조합이 부족합니다."]


def _render_quality_table(models: Sequence[ModelReport]) -> list[str]:
    """Render quality heuristics table."""
    lines = ["| 모델 | 존댓말 | 회피 | 고유 응답 | 실패 수 |", "|---|---:|---:|---:|---:|"]
    for model in models:
        lines.append(
            f"| {model.spec.label} | {_fmt_pct(model.honorific_pct)} | "
            f"{_fmt_pct(model.evasion_pct)} | {model.unique_responses} | {model.failure_count} |"
        )
    return lines


def _render_memory_analysis(models: Sequence[ModelReport]) -> list[str]:
    """Render memory section."""
    lines = [
        f"- Jetson Orin Nano Super 8GB 기준 예산: 약 {JETSON_MEMORY_BUDGET_MB:,} MB",
        "",
        "| 모델 | Peak MB | 예산 대비 비율 |",
        "|---|---:|---:|",
    ]
    for model in models:
        ratio = (
            (model.memory_mb.maximum / JETSON_MEMORY_BUDGET_MB * 100.0)
            if model.memory_mb.maximum is not None
            else None
        )
        lines.append(
            f"| {model.spec.label} | {_fmt_float(model.memory_mb.maximum, 0)} | {_fmt_pct(ratio)} |"
        )
    return lines


def _render_stability_table(models: Sequence[ModelReport]) -> list[str]:
    """Render section stability ratios and segment averages."""
    lines = ["| 모델 | R1-10 평균 | R51-60 평균 | 후반/초반 비율 |", "|---|---:|---:|---:|"]
    for model in models:
        early = next(
            (segment.avg_total_time_s for segment in model.segments if segment.round_start == 1),
            None,
        )
        late = next(
            (segment.avg_total_time_s for segment in model.segments if segment.round_start == 51),
            None,
        )
        lines.append(
            f"| {model.spec.label} | {_fmt_float(early)} | {_fmt_float(late)} | {_fmt_ratio(model.late_early_ratio)} |"
        )
    return lines


def _render_kpi_table(models: Sequence[ModelReport]) -> list[str]:
    """Render KPI pass/fail section."""
    lines = [
        "| 모델 | p50 <= 8s | p95 <= 10s | Success 100% | Honorific <= 1% |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        lines.append(
            f"| {model.spec.label} | {_status(model.kpi_p50_pass)} | {_status(model.kpi_p95_pass)} | "
            f"{_status(model.kpi_success_pass)} | {_status(model.kpi_honorific_pass)} |"
        )
    return lines


def _ranking_reason(model: ModelReport) -> str:
    """Build a concise Korean rationale for ranking output."""
    parts: list[str] = []
    if model.success_rate_pct is not None:
        parts.append(f"성공률 {_fmt_pct(model.success_rate_pct)}")
    if model.total_time.p95 is not None:
        parts.append(f"p95 {_fmt_float(model.total_time.p95)}초")
    if model.honorific_pct is not None:
        parts.append(f"존댓말 {_fmt_pct(model.honorific_pct)}")
    if model.unique_responses:
        parts.append(f"고유 응답 {model.unique_responses}개")
    return ", ".join(parts)


def _render_ranking(models: Sequence[ModelReport]) -> list[str]:
    """Render the final ranking section."""
    top_three = _top_models(models, 3)
    if not top_three:
        return ["- 순위를 산정할 수 있는 유효 데이터가 없습니다."]
    lines = ["| 순위 | 모델 | 근거 |", "|---:|---|---|"]
    for index, model in enumerate(top_three, start=1):
        lines.append(f"| {index} | {model.spec.label} | {_ranking_reason(model)} |")
    return lines


def _render_detail_stats(models: Sequence[ModelReport]) -> list[str]:
    """Render detailed statistics required by the task."""
    lines: list[str] = ["## 상세 통계", ""]
    for model in models:
        lines.append(f"### {model.spec.label}")
        lines.append("")
        lines.append("| 지표 | 개수 | 평균 | 중앙값 | p5 | p95 | 최소 | 최대 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label, summary in (
            ("전체", model.total_time),
            ("LLM", model.llm_time),
            ("TTS", model.tts_time),
            ("TTFT", model.ttft_time),
            ("토큰", model.llm_tokens),
            ("메모리 MB", model.memory_mb),
        ):
            lines.append(
                f"| {label} | {summary.count} | {_fmt_float(summary.average)} | {_fmt_float(summary.median)} | "
                f"{_fmt_float(summary.p5)} | {_fmt_float(summary.p95)} | {_fmt_float(summary.minimum)} | "
                f"{_fmt_float(summary.maximum)} |"
            )
        lines.append("")
    return lines


def _render_executive_summary(models: Sequence[ModelReport]) -> list[str]:
    """Render the executive summary section."""
    best = _best_model(models)
    available_count = sum(1 for model in models if model.available)
    missing_count = len(models) - available_count
    if best is None:
        return [
            f"- 유효한 입력 데이터가 없어 추천 모델을 산정하지 못했다. 누락 세트: {missing_count}개"
        ]
    return [
        f"- 분석 대상: 총 {len(models)}개 중 유효 세트 {available_count}개, 누락 {missing_count}개",
        f"- 종합 추천: **{best.spec.label}**",
        (
            f"- 추천 근거: 성공률 {_fmt_pct(best.success_rate_pct)}, 평균 전체 {_fmt_float(best.total_time.average)}초, "
            f"p95 {_fmt_float(best.total_time.p95)}초, 존댓말 {_fmt_pct(best.honorific_pct)}, "
            f"Peak {_fmt_float(best.memory_mb.maximum, 0)}MB"
        ),
    ]


def _render_conclusion(models: Sequence[ModelReport]) -> list[str]:
    """Render the conclusion section."""
    best = _best_model(models)
    if best is None:
        return ["- 유효 데이터가 없어 결론을 보류한다."]
    return [
        f"- 최종 추천은 **{best.spec.label}** 이다. 품질 휴리스틱과 지연 시간, 메모리 예산을 함께 고려했을 때 가장 균형이 좋다.",
        "- 운영 후보 선정 시에는 이 리포트의 1차 순위를 기준으로, 실기기 장시간 반복 테스트와 안전 필터 체인 검증을 추가로 수행하는 것이 적절하다.",
    ]


def _build_report(models: Sequence[ModelReport], input_base: Path) -> str:
    """Build the markdown comparison report."""
    lines: list[str] = [
        f"# {REPORT_DATE.isoformat()} 12개 모델 비교 리포트",
        "",
        f"- 입력 기준 경로: `{input_base}`",
        "- 생성 파일 수: 비교 리포트 1개, 대화 스크립트 1개",
        "",
        "## 요약",
        "",
    ]
    lines.extend(_render_executive_summary(models))
    lines.extend(["", "## 테스트 매트릭스", ""])
    lines.extend(_render_test_matrix(models))
    lines.extend(["", "## 성능 비교", ""])
    lines.extend(_render_performance_table(models))
    lines.extend(["", "## 교차 분석", "", "### Qwen3 vs Qwen3.5", ""])
    lines.extend(_render_family_comparison(models))
    lines.extend(["", "### SFT vs DPO", ""])
    lines.extend(_render_ft_comparison(models))
    lines.extend(["", "### Q4 vs Q6 vs Q8", ""])
    lines.extend(_render_quant_comparison(models))
    lines.extend(["", "## 품질 비교", ""])
    lines.extend(_render_quality_table(models))
    lines.extend(["", "## 메모리 분석", ""])
    lines.extend(_render_memory_analysis(models))
    lines.extend(["", "## 구간 안정성", ""])
    lines.extend(_render_stability_table(models))
    lines.extend(["", "## Phase 1 MVP KPI 점검", ""])
    lines.extend(_render_kpi_table(models))
    lines.extend(["", "## 최종 순위", ""])
    lines.extend(_render_ranking(models))
    lines.extend(["", "## 결론 및 권고", ""])
    lines.extend(_render_conclusion(models))
    lines.extend([""])
    lines.extend(_render_detail_stats(models))
    missing = [model for model in models if not model.available and model.warning]
    if missing:
        lines.extend(["## 누락 또는 경고", ""])
        for model in missing:
            lines.append(f"- {model.spec.label}: {model.warning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _group_turns_by_round(turns: Sequence[TurnRecord]) -> dict[int, list[TurnRecord]]:
    """Group flattened turns by round number."""
    grouped: dict[int, list[TurnRecord]] = {}
    for turn in turns:
        grouped.setdefault(turn.round_num, []).append(turn)
    for round_turns in grouped.values():
        round_turns.sort(key=lambda item: item.exchange)
    return grouped


def _build_conversation_scripts(models: Sequence[ModelReport], input_base: Path) -> str:
    """Build the markdown conversation script compilation."""
    lines: list[str] = [
        f"# {REPORT_DATE.isoformat()} 12개 모델 대화 스크립트",
        "",
        f"- 입력 기준 경로: `{input_base}`",
        "",
    ]
    first_section = True
    for model in models:
        if not first_section:
            lines.extend(["", "---", ""])
        first_section = False
        lines.extend(
            [
                f"## {model.spec.label}",
                "",
                f"- 계열: {model.spec.family}",
                f"- 양자화: {model.spec.quant}",
                f"- FT: {model.spec.finetune}",
                f"- 파일 크기: {model.spec.file_size}",
                f"- 입력 파일: `{model.source_path}`",
                "",
            ]
        )
        if not model.available:
            lines.append("- 데이터가 없어 대화 스크립트를 생성하지 못했다.")
            continue
        grouped = _group_turns_by_round(model.turns)
        for round_num in sorted(grouped):
            lines.extend([f"### R{round_num:02d}", ""])
            for turn in grouped[round_num]:
                lines.append(f"아이: {turn.user_text}")
                lines.append(f"뭉이: {turn.assistant_text}")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(
    output_dir: Path,
    models: Sequence[ModelReport],
    input_base: Path,
) -> tuple[Path, Path]:
    """Write both markdown outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{REPORT_DATE.isoformat()}-12model-comparison-report.md"
    scripts_path = output_dir / f"{REPORT_DATE.isoformat()}-12model-conversation-scripts.md"
    report_path.write_text(_build_report(models, input_base), encoding="utf-8")
    scripts_path.write_text(_build_conversation_scripts(models, input_base), encoding="utf-8")
    return report_path, scripts_path


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    input_base = args.input_base.resolve()
    output_dir = args.output_dir.resolve()
    models = [_load_model_report(spec, input_base) for spec in TESTS]
    report_path, scripts_path = _write_outputs(output_dir, models, input_base)
    logger.info("Comparison report written to %s", report_path)
    logger.info("Conversation scripts written to %s", scripts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
