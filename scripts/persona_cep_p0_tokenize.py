"""Offline P0 prompt-token post-processor for Persona CEP measurements."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pipeline import GEMMA4_PERSONA_PROMPT_PATH, PipelineConfig, _build_gemma4_system_prompt
from safety.approved_template_router import strip_emoji

CSV_COLUMNS: Sequence[str] = (
    "turn_id",
    "language",
    "backend",
    "safety_guide_topic_id",
    "prompt_len_chars",
    "tokens_heuristic",
    "tokens_tokenizer",
    "deviation_pct",
    "deviation_pct_abs",
)
LANGUAGE_KEYS: Sequence[str] = ("language", "detected_language", "lang")
BACKEND_KEYS: Sequence[str] = ("backend", "llm_backend")
SAFETY_TOPIC_KEYS: Sequence[str] = ("safety_guide_topic_id", "template_topic_id")
TURN_ID_KEYS: Sequence[str] = ("turn_id", "global_turn_id", "round_id")
NULL_VALUE = "NULL"
EMPTY_TOKENIZER_VALUE = ""
Tokenizer = Callable[[str], int]


@dataclass(frozen=True)
class _PromptSources:
    ko_base_prompt: str
    gemma4_ko_prompt: str
    en_prompt: str
    safety_templates: Mapping[str, Mapping[str, Any]]


def estimate_tokens(text: str) -> int:
    """Return the existing pipeline's chars-per-three token estimate."""

    return max(1, (len(text) + 2) // 3)


def process_rounds(
    rounds_jsonl: Path,
    output_csv: Path,
    gemma_model_path: Path | None = None,
) -> int:
    """Process ``rounds.jsonl`` into ``prompt_tokens.csv`` and return a CLI code."""

    records = _read_round_records(rounds_jsonl)
    if not records:
        if rounds_jsonl.stat().st_size == 0:
            _warn("rounds.jsonl is empty; wrote header-only CSV.")
        else:
            _warn("rounds.jsonl contained no valid turn records; wrote header-only CSV.")
        _write_csv(output_csv, [])
        return 0

    sources = _load_prompt_sources(REPO_ROOT)
    tokenizer = _load_tokenizer(gemma_model_path)
    _write_csv(output_csv, _build_rows(records, sources, tokenizer))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the prompt-token post-processor."""

    parser = argparse.ArgumentParser(
        description="Post-process Persona CEP P0 rounds.jsonl into prompt_tokens.csv."
    )
    parser.add_argument("--rounds-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--gemma-model-path", type=Path, default=None)
    args = parser.parse_args(argv)
    return process_rounds(args.rounds_jsonl, args.output_csv, args.gemma_model_path)


def _load_prompt_sources(repo_root: Path) -> _PromptSources:
    config = PipelineConfig()
    ko_base_prompt = config.llm_system_prompt
    persona_path = repo_root / GEMMA4_PERSONA_PROMPT_PATH
    gemma4_ko_prompt = _build_gemma4_system_prompt(ko_base_prompt, persona_path)
    en_prompt = _read_text_or_warn(
        repo_root / config.en_system_prompt_path,
        fallback=ko_base_prompt,
        label="English system prompt",
    ).strip()
    return _PromptSources(
        ko_base_prompt=ko_base_prompt,
        gemma4_ko_prompt=gemma4_ko_prompt,
        en_prompt=en_prompt,
        safety_templates=_load_safety_templates(
            repo_root / "assets" / "filters" / "approved_templates.json"
        ),
    )


def _load_tokenizer(gemma_model_path: Path | None) -> Tokenizer | None:
    if gemma_model_path is None:
        _warn(
            "--gemma-model-path not provided; tokens_tokenizer and deviation columns will be empty."
        )
        return None
    if not gemma_model_path.exists():
        _warn(
            f"Gemma GGUF model path does not exist: {gemma_model_path}; "
            "tokens_tokenizer and deviation columns will be empty."
        )
        return None
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        _warn(
            "llama-cpp-python is unavailable; tokens_tokenizer and deviation columns "
            f"will be empty: {exc}"
        )
        return None
    try:
        llm = Llama(model_path=str(gemma_model_path), n_gpu_layers=0, verbose=False)
    except (OSError, RuntimeError, ValueError) as exc:
        _warn(
            f"failed to load Gemma tokenizer from {gemma_model_path}: {exc}; "
            "tokens_tokenizer and deviation columns will be empty."
        )
        return None

    def tokenize_prompt(text: str) -> int:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

    return tokenize_prompt


def _read_round_records(rounds_jsonl: Path) -> list[tuple[int, Mapping[str, Any]]]:
    records: list[tuple[int, Mapping[str, Any]]] = []
    with rounds_jsonl.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                _warn(f"line {line_number}: empty JSONL line skipped")
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                _warn(f"line {line_number}: malformed JSON skipped: {exc}")
                continue
            if isinstance(parsed, dict):
                records.append((line_number, parsed))
            else:
                _warn(f"line {line_number}: expected JSON object, got {type(parsed).__name__}")
    return records


def _build_rows(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    sources: _PromptSources,
    tokenizer: Tokenizer | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, record in records:
        prompt = _prompt_for_record(record, sources, line_number)
        row = _base_row(record, line_number)
        row.update(_null_metrics() if prompt is None else _metrics(prompt, tokenizer))
        rows.append(row)
    rows.append(_summary_row(rows, tokenizer is not None))
    return rows


def _prompt_for_record(
    record: Mapping[str, Any],
    sources: _PromptSources,
    line_number: int,
) -> str | None:
    if "prompt" in record:
        prompt = record["prompt"]
        if isinstance(prompt, str):
            return prompt
        _warn(f"line {line_number}: prompt field is not a string; emitted NULL metrics")
        return None

    missing = [
        label
        for label, keys in (
            ("language", LANGUAGE_KEYS),
            ("backend", BACKEND_KEYS),
            ("safety_guide_topic_id", SAFETY_TOPIC_KEYS),
        )
        if not any(key in record for key in keys)
    ]
    if missing:
        _warn(
            f"line {line_number}: missing {', '.join(missing)}; "
            "cannot reconstruct prompt, emitted NULL metrics"
        )
        return None

    language = _field_text(record, LANGUAGE_KEYS) or "ko"
    backend = _field_text(record, BACKEND_KEYS) or ""
    prompt = _base_prompt(language, backend, sources)
    topic_id = _field_text(record, SAFETY_TOPIC_KEYS)
    if topic_id is None:
        return prompt
    guide = _guide_response(topic_id, language, sources)
    if guide is None:
        _warn(f"line {line_number}: safety guide topic {topic_id!r} could not be resolved")
        return None
    return _append_safety_guide(prompt, guide, language)


def _base_prompt(language: str, backend: str, sources: _PromptSources) -> str:
    if language.casefold() == "en" and sources.en_prompt:
        return sources.en_prompt
    if backend.casefold() == "gemma4_text":
        return sources.gemma4_ko_prompt
    return sources.ko_base_prompt


def _guide_response(topic_id: str, language: str, sources: _PromptSources) -> str | None:
    topic = sources.safety_templates.get(topic_id)
    if topic is None or topic.get("mode", "block") != "guide":
        return None
    response_key = "response_en" if language.casefold() == "en" else "response_ko"
    response = topic.get(response_key)
    if not isinstance(response, str) or not response.strip():
        return None
    return strip_emoji(response)


def _append_safety_guide(base_prompt: str, guide: str, language: str) -> str:
    if language.casefold() == "en":
        return (
            f"{base_prompt}\n\n[Safety Guide] {guide}\n"
            "Refer to the above safety information, but answer the child's "
            "question with an educational and age-appropriate explanation."
        )
    return (
        f"{base_prompt}\n\n[안전 가이드] {guide}\n"
        "위 안전 정보를 참고하되, 아이의 질문에 맞는 교육적이고 이해하기 쉬운 답변을 해주세요."
    )


def _metrics(prompt: str, tokenizer: Tokenizer | None) -> dict[str, str]:
    heuristic = estimate_tokens(prompt)
    row = {
        "prompt_len_chars": str(len(prompt)),
        "tokens_heuristic": str(heuristic),
        "tokens_tokenizer": EMPTY_TOKENIZER_VALUE,
        "deviation_pct": EMPTY_TOKENIZER_VALUE,
        "deviation_pct_abs": EMPTY_TOKENIZER_VALUE,
    }
    if tokenizer is None:
        return row
    actual = tokenizer(prompt)
    deviation = (actual - heuristic) / heuristic * 100
    row.update(
        {
            "tokens_tokenizer": str(actual),
            "deviation_pct": _format_number(deviation),
            "deviation_pct_abs": _format_number(abs(deviation)),
        }
    )
    return row


def _summary_row(rows: Sequence[Mapping[str, str]], include_tokenizer: bool) -> dict[str, str]:
    return {
        "turn_id": "summary",
        "language": NULL_VALUE,
        "backend": NULL_VALUE,
        "safety_guide_topic_id": NULL_VALUE,
        "prompt_len_chars": NULL_VALUE,
        "tokens_heuristic": _summary_stats(_numeric_column(rows, "tokens_heuristic")),
        "tokens_tokenizer": _summary_stats(_numeric_column(rows, "tokens_tokenizer"))
        if include_tokenizer
        else EMPTY_TOKENIZER_VALUE,
        "deviation_pct": NULL_VALUE,
        "deviation_pct_abs": _summary_stats(_numeric_column(rows, "deviation_pct_abs"))
        if include_tokenizer
        else EMPTY_TOKENIZER_VALUE,
    }


def _summary_stats(values: Sequence[float]) -> str:
    if not values:
        return NULL_VALUE
    ordered = sorted(values)
    return ";".join(
        (
            f"mean={_format_number(statistics.mean(ordered))}",
            f"median={_format_number(statistics.median(ordered))}",
            f"p90={_format_number(_nearest_rank_percentile(ordered, 90))}",
            f"p99={_format_number(_nearest_rank_percentile(ordered, 99))}",
            f"max={_format_number(max(ordered))}",
        )
    )


def _nearest_rank_percentile(ordered_values: Sequence[float], percentile: int) -> float:
    rank = max(1, (len(ordered_values) * percentile + 99) // 100)
    return ordered_values[min(rank, len(ordered_values)) - 1]


def _read_text_or_warn(path: Path, *, fallback: str, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _warn(f"failed to load {label} from {path}: {exc}; using fallback")
        return fallback


def _load_safety_templates(path: Path) -> Mapping[str, Mapping[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"failed to load approved templates from {path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        _warn(f"approved templates at {path} are not a JSON object")
        return {}
    return {str(topic_id): topic for topic_id, topic in raw.items() if isinstance(topic, dict)}


def _write_csv(output_csv: Path, rows: Sequence[Mapping[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(record: Mapping[str, Any], line_number: int) -> dict[str, str]:
    return {
        "turn_id": _field_text(record, TURN_ID_KEYS) or str(line_number),
        "language": _field_text(record, LANGUAGE_KEYS) or NULL_VALUE,
        "backend": _field_text(record, BACKEND_KEYS) or NULL_VALUE,
        "safety_guide_topic_id": _field_text(record, SAFETY_TOPIC_KEYS) or NULL_VALUE,
    }


def _null_metrics() -> dict[str, str]:
    return {
        "prompt_len_chars": NULL_VALUE,
        "tokens_heuristic": NULL_VALUE,
        "tokens_tokenizer": NULL_VALUE,
        "deviation_pct": NULL_VALUE,
        "deviation_pct_abs": NULL_VALUE,
    }


def _numeric_column(rows: Sequence[Mapping[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column)
        if raw is None or raw in ("", NULL_VALUE):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _field_text(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in record:
            value = record[key]
            if value is None:
                return None
            text = str(value).strip()
            return text or None
    return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
