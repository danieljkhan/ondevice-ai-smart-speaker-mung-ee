"""Run the Phase 0 fact-shortlist placement A/B harness."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Literal, cast
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.llm_backend_config import LLMBackendConfig
from core.pipeline import ConversationPipeline, PipelineConfig
from scripts.score_fact_holdout import (
    HoldoutRow,
    ResponseRow,
    load_holdout_rows,
    score_response_rows,
    write_json,
    write_jsonl,
)

CellInjectionState = Literal["off", "on"]
Placement = Literal["p1", "p2"]
PLACEMENTS: Final[tuple[Placement, ...]] = ("p1", "p2")
INJECTION_STATES: Final[tuple[CellInjectionState, ...]] = ("off", "on")
AGE_BANDS: Final[tuple[str, ...]] = ("under_10", "under_15")


@dataclasses.dataclass(frozen=True)
class ShortlistTopic:
    """One shortlist topic with the metadata needed by holdout validation."""

    category: str
    age_band: str


class GenerationBackend(ABC):
    """Abstract generation backend for the Phase 0 harness."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> tuple[str, float | None, float | None]:
        """Generate one response and return text plus optional latency timings."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources when applicable."""


class MockBackend(GenerationBackend):
    """Deterministic backend used by unit tests."""

    def __init__(self, responder: Callable[[list[dict[str, str]]], str]) -> None:
        self._responder = responder

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, float | None, float | None]:
        return self._responder(messages), None, None

    def close(self) -> None:
        """Release backend resources when applicable."""


class LlamaCppBackend(GenerationBackend):
    """llama-cpp backend that mirrors pipeline chat generation."""

    def __init__(self, backend_config: LLMBackendConfig, pipeline_config: PipelineConfig) -> None:
        from models.llm_runner import build_llm_from_config

        self._backend_name, self._llm = build_llm_from_config(backend_config)
        self._pipeline_config = pipeline_config
        self._family = "gemma" if self._backend_name == "gemma4_text" else "qwen"

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, float | None, float | None]:
        from models.llm_runner import (
            SAFE_FALLBACK,
            run_chat_generation,
            run_generation,
            sanitize_response,
            stop_sequences_for_family,
            strip_think_tags,
        )

        stop_sequences = stop_sequences_for_family(self._family)
        (
            text,
            token_count,
            ttft,
            gen_time,
            _cache_hit_tokens,
            _cache_miss_tokens,
        ) = run_chat_generation(
            self._llm,
            messages,
            max_tokens=self._pipeline_config.llm_max_tokens,
            stop=stop_sequences,
            temperature=self._pipeline_config.llm_temperature,
            top_p=self._pipeline_config.llm_top_p,
            top_k=self._pipeline_config.llm_top_k,
            min_p=self._pipeline_config.llm_min_p,
            presence_penalty=self._pipeline_config.llm_presence_penalty,
            repeat_penalty=self._pipeline_config.llm_repeat_penalty,
            enable_thinking=False,
        )

        if not text and token_count == 0:
            legacy_prompt = ConversationPipeline._messages_to_prompt(messages)
            text, _token_count, ttft, gen_time = run_generation(
                self._llm,
                legacy_prompt,
                max_tokens=self._pipeline_config.llm_max_tokens,
                stop=stop_sequences,
                temperature=self._pipeline_config.llm_temperature,
                top_p=self._pipeline_config.llm_top_p,
                top_k=self._pipeline_config.llm_top_k,
                min_p=self._pipeline_config.llm_min_p,
                presence_penalty=self._pipeline_config.llm_presence_penalty,
                repeat_penalty=self._pipeline_config.llm_repeat_penalty,
            )

        cleaned = strip_think_tags(text)
        return sanitize_response(cleaned, language="ko") or SAFE_FALLBACK, ttft, gen_time

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()


def create_backend(
    backend_name: str,
    *,
    model_path: str | None,
    pipeline_config: PipelineConfig,
) -> GenerationBackend:
    """Create the requested generation backend."""

    if backend_name != "llama-cpp":
        msg = f"Unsupported backend: {backend_name}"
        raise ValueError(msg)

    resolved_config = LLMBackendConfig.load()
    if model_path is not None:
        resolved_config = dataclasses.replace(resolved_config, model_path=model_path)
    return LlamaCppBackend(resolved_config, pipeline_config)


def run_phase0_ab_experiment(
    *,
    holdout_rows: Sequence[HoldoutRow],
    backend: GenerationBackend,
    backend_config: LLMBackendConfig,
    placements: Sequence[Placement],
    output_dir: Path,
) -> dict[str, Any]:
    """Run the four Phase 0 cells and return the aggregate summary payload."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_cells: dict[str, Any] = {}
    placement_results: dict[str, Any] = {}

    for placement in placements:
        for injection_state in INJECTION_STATES:
            mode = placement if injection_state == "on" else "disabled"
            pipeline = _build_pipeline_for_mode(mode=mode, backend_config=backend_config)
            response_rows = _generate_response_rows(holdout_rows, pipeline, backend)
            response_path = output_dir / f"responses_{placement}_{injection_state}.jsonl"
            write_jsonl(response_path, response_rows)

            scored_rows, cell_summary = score_response_rows(
                holdout_rows,
                [ResponseRow(**row) for row in response_rows],
                injection_state=injection_state,
            )
            rows_path = output_dir / f"rows_{placement}_{injection_state}.jsonl"
            summary_path = output_dir / f"summary_{placement}_{injection_state}.json"
            write_jsonl(rows_path, scored_rows)
            write_json(summary_path, cell_summary)
            summary_cells[f"{placement}_{injection_state}"] = {
                "placement": placement,
                "injection_state": injection_state,
                "response_path": str(response_path.relative_to(output_dir)),
                "rows_path": str(rows_path.relative_to(output_dir)),
                "summary_path": str(summary_path.relative_to(output_dir)),
                "summary": cell_summary,
            }

    for placement in placements:
        off_summary = summary_cells[f"{placement}_off"]["summary"]
        on_summary = summary_cells[f"{placement}_on"]["summary"]
        relative_reduction = _relative_failure_reduction(
            off_summary["failure_rate"],
            on_summary["failure_rate"],
        )
        verdict = "go" if relative_reduction >= 0.5 else "stop"
        placement_results[placement] = {
            "off_failure_rate": off_summary["failure_rate"],
            "on_failure_rate": on_summary["failure_rate"],
            "relative_failure_reduction": relative_reduction,
            "verdict": verdict,
        }

    winner_payload = determine_placement_winner(placement_results)
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells": summary_cells,
        "placements": placement_results,
        "winner": winner_payload,
    }
    write_json(output_dir / "phase0_summary.json", summary_payload)
    write_json(output_dir / "placement_winner.json", winner_payload)
    return summary_payload


def determine_placement_winner(placement_results: dict[str, Any]) -> dict[str, Any]:
    """Return the GO/STOP verdict for the Phase 0 placement comparison."""

    eligible = [
        (placement, payload)
        for placement, payload in placement_results.items()
        if payload["verdict"] == "go"
    ]
    if not eligible:
        return {"verdict": "STOP", "winner": None, "placements": placement_results}
    winner, _ = max(
        eligible,
        key=lambda item: (
            item[1]["relative_failure_reduction"],
            1 if item[0] == "p1" else 0,
        ),
    )
    return {"verdict": "GO", "winner": winner, "placements": placement_results}


def build_parser() -> argparse.ArgumentParser:
    """Build the A/B harness CLI parser."""

    parser = argparse.ArgumentParser(description="Run the Phase 0 placement A/B harness.")
    parser.add_argument("--holdout", type=Path, required=True, help="Holdout JSONL path.")
    parser.add_argument("--shortlist", type=Path, required=True, help="Shortlist JSON path.")
    parser.add_argument(
        "--backend",
        choices=("llama-cpp",),
        required=True,
        help="Generation backend to use.",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Optional GGUF model path.")
    parser.add_argument(
        "--placements",
        type=str,
        default="p1,p2",
        help="Comma-separated placement list, e.g. p1,p2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Artifact output dir.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Phase 0 harness CLI."""

    args = build_parser().parse_args(argv)
    placements = parse_placements(args.placements)
    holdout_rows = load_holdout_rows(args.holdout)
    shortlist_topics = load_shortlist_topics(args.shortlist)
    validate_holdout_against_shortlist(holdout_rows, shortlist_topics)
    pipeline_config = PipelineConfig()
    backend = create_backend(
        args.backend,
        model_path=args.model_path,
        pipeline_config=pipeline_config,
    )
    backend_config = LLMBackendConfig.load()
    if args.model_path is not None:
        backend_config = dataclasses.replace(backend_config, model_path=args.model_path)

    try:
        run_phase0_ab_experiment(
            holdout_rows=holdout_rows,
            backend=backend,
            backend_config=backend_config,
            placements=placements,
            output_dir=args.output_dir,
        )
    finally:
        backend.close()
    return 0


def parse_placements(raw: str) -> list[Placement]:
    """Parse and validate a comma-separated placement string."""

    placements = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not placements:
        msg = "At least one placement must be provided"
        raise ValueError(msg)
    invalid = [item for item in placements if item not in PLACEMENTS]
    if invalid:
        msg = f"Unsupported placements: {', '.join(invalid)}"
        raise ValueError(msg)
    deduplicated = list(dict.fromkeys(placements))
    return [cast(Placement, item) for item in deduplicated]


def load_shortlist_topics(path: Path) -> dict[str, ShortlistTopic]:
    """Load shortlist topics, categories, and age bands from the shortlist artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = "Shortlist root must be a JSON array"
        raise ValueError(msg)

    topics: dict[str, ShortlistTopic] = {}
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            msg = f"Shortlist entry {index} must be a JSON object"
            raise ValueError(msg)
        topic = row.get("topic")
        category = row.get("category")
        age_band = row.get("age_band", "under_10")
        if not isinstance(topic, str) or not topic.strip():
            msg = f"Shortlist entry {index} has invalid topic"
            raise ValueError(msg)
        if not isinstance(category, str) or not category.strip():
            msg = f"Shortlist entry {index} has invalid category"
            raise ValueError(msg)
        if not isinstance(age_band, str) or age_band.strip() not in AGE_BANDS:
            allowed = ", ".join(AGE_BANDS)
            msg = f"Shortlist entry {index} has invalid age_band; expected one of: {allowed}"
            raise ValueError(msg)
        normalized_topic = topic.strip()
        if normalized_topic in topics:
            msg = f"Duplicate shortlist topic: {normalized_topic}"
            raise ValueError(msg)
        topics[normalized_topic] = ShortlistTopic(
            category=category.strip(),
            age_band=age_band.strip(),
        )
    return topics


def validate_holdout_against_shortlist(
    holdout_rows: Sequence[HoldoutRow],
    shortlist_topics: dict[str, ShortlistTopic],
) -> None:
    """Validate that matched holdout rows align with the shortlist coverage contract."""

    matched_rows = [row for row in holdout_rows if row.axis == "matched"]
    shortlist_cells = {
        (shortlist_topic.category, shortlist_topic.age_band)
        for shortlist_topic in shortlist_topics.values()
    }
    matched_cells = {(row.category, row.age_band) for row in matched_rows}
    missing_cells = sorted(shortlist_cells - matched_cells)
    if missing_cells:
        formatted_cells = ", ".join(
            f"{category}/{age_band}" for category, age_band in missing_cells
        )
        msg = f"Matched holdout missing shortlist cells: {formatted_cells}"
        raise ValueError(msg)

    for row in holdout_rows:
        if row.axis == "unmatched":
            continue
        shortlist_topic = shortlist_topics.get(row.topic)
        if shortlist_topic is None:
            msg = f"Matched holdout topic not found in shortlist: {row.topic}"
            raise ValueError(msg)
        if row.category != shortlist_topic.category:
            msg = (
                f"Holdout category mismatch for {row.topic}: "
                f"{row.category!r} != {shortlist_topic.category!r}"
            )
            raise ValueError(msg)
        if row.age_band != shortlist_topic.age_band:
            msg = (
                f"Holdout age_band mismatch for {row.topic}: "
                f"{row.age_band!r} != {shortlist_topic.age_band!r}"
            )
            raise ValueError(msg)


def _generate_response_rows(
    holdout_rows: Sequence[HoldoutRow],
    pipeline: ConversationPipeline,
    backend: GenerationBackend,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for holdout_row in holdout_rows:
        messages = pipeline._build_messages(holdout_row.question, detected_language="ko")
        response, ttft_s, gen_time_s = backend.generate(messages)
        rows.append(
            {
                "topic": holdout_row.topic,
                "question": holdout_row.question,
                "response": response,
                "ttft_s": ttft_s,
                "gen_time_s": gen_time_s,
            }
        )
    return rows


def _build_pipeline_for_mode(
    *,
    mode: str,
    backend_config: LLMBackendConfig,
) -> ConversationPipeline:
    with _temporary_env("MUNGI_FACT_SHORTLIST", mode):
        with patch("core.pipeline.LLMBackendConfig.load", return_value=backend_config):
            pipeline = ConversationPipeline(cast(Any, SimpleNamespace()), PipelineConfig())
    return pipeline


def _relative_failure_reduction(off_failure_rate: float, on_failure_rate: float) -> float:
    if off_failure_rate <= 0:
        return 0.0 if on_failure_rate <= 0 else -1.0
    return (off_failure_rate - on_failure_rate) / off_failure_rate


@contextlib.contextmanager
def _temporary_env(name: str, value: str) -> Iterator[None]:
    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


if __name__ == "__main__":
    raise SystemExit(main())
