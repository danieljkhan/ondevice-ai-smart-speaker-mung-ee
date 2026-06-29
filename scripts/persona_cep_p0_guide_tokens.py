#!/usr/bin/env python3
"""Measure approved safety guide template injection token counts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

CSV_COLUMNS = [
    "topic_id",
    "mode",
    "priority",
    "lang",
    "response_chars",
    "injection_chars",
    "injection_tokens_heuristic",
    "injection_tokens_tokenizer",
    "deviation_pct_abs",
    "sentence_count",
    "mandatory_sentence_count",
    "mandatory_floor_tokens",
    "max_sentence_tokens",
    "comments",
]
KEYWORDS_KO = tuple("대피 피해 위험 엄마 아빠 안전 즉시 빨리 어른 들어 따라 이동 움직".split())
KEYWORDS_EN = (
    "evacuate",
    "shelter",
    "danger",
    "parents",
    "adult",
    "immediately",
    "safely",
    "grown-up",
    "grown up",
    "listen to",
    "follow",
    "move",
    "safe place",
    "stay safe",
    "cover your head",
    "under a",
)
SENTENCE_RE = re.compile(r".+?(?:[.!?]+|(?<=[다야어해])(?=\s|$))", re.DOTALL)
Stats = dict[str, dict[str, list[int]]]


class _Tokenizer(Protocol):
    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        """Tokenize UTF-8 bytes without running inference."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure guide-mode approved-template token counts for P0.",
    )
    parser.add_argument("--templates-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--gemma-model-path", type=Path)
    return parser


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 2) // 3)


def _load_templates(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"templates JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"templates JSON is malformed at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise SystemExit(f"failed to read templates JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("templates JSON must be an object keyed by topic_id")
    for topic_id, template in raw.items():
        if not isinstance(topic_id, str) or not isinstance(template, dict):
            raise SystemExit("templates entries must map string topic IDs to objects")
    return cast(dict[str, dict[str, Any]], raw)


def _warn_no_tokenizer(reason: str) -> None:
    print(
        f"warning: {reason}; tokenizer columns will be blank and mandatory floor "
        "counts will use the heuristic",
        file=sys.stderr,
    )


def _load_tokenizer(model_path: Path | None) -> _Tokenizer | None:
    if model_path is None:
        _warn_no_tokenizer("--gemma-model-path not supplied")
        return None
    try:
        from llama_cpp import Llama  # type: ignore[import-not-found]
    except ImportError as exc:
        _warn_no_tokenizer(f"llama_cpp is unavailable ({exc})")
        return None

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_gpu_layers": 0,
        "n_ctx": 128,
        "verbose": False,
        "vocab_only": True,
    }
    try:
        return cast(_Tokenizer, Llama(**kwargs))
    except TypeError:
        kwargs.pop("vocab_only", None)
        try:
            return cast(_Tokenizer, Llama(**kwargs))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _warn_no_tokenizer(f"failed to load tokenizer from {model_path}: {exc}")
            return None
    except (OSError, RuntimeError, ValueError) as exc:
        _warn_no_tokenizer(f"failed to load tokenizer from {model_path}: {exc}")
        return None


def _count_tokens(text: str, tokenizer: _Tokenizer | None) -> int:
    if tokenizer is None:
        return _estimate_tokens(text)
    return len(tokenizer.tokenize(text.encode("utf-8"), add_bos=False))


def _injection(response: str, lang: str) -> str:
    if lang == "en":
        return (
            f"\n\n[Safety Guide] {response}\n"
            "Refer to the above safety information, but answer the child's "
            "question with an educational and age-appropriate explanation."
        )
    return (
        f"\n\n[안전 가이드] {response}\n"
        "위 안전 정보를 참고하되, 아이의 질문에 맞는 교육적이고 "
        "이해하기 쉬운 답변을 해주세요."
    )


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    last_end = 0
    for match in SENTENCE_RE.finditer(text):
        if sentence := match.group(0).strip():
            sentences.append(sentence)
        last_end = match.end()
    if remainder := text[last_end:].strip():
        sentences.append(remainder)
    return sentences


def _score(sentence: str, lang: str) -> int:
    if lang == "en":
        lowered = sentence.lower()
        return sum(1 for keyword in KEYWORDS_EN if keyword in lowered)
    return sum(1 for keyword in KEYWORDS_KO if keyword in sentence)


def _floor_counts(
    response: str, lang: str, tokenizer: _Tokenizer | None
) -> tuple[int, int, int, int]:
    sentences = _split_sentences(response)
    mandatory = [sentence for sentence in sentences if _score(sentence, lang) > 0]
    mandatory_tokens = [_count_tokens(sentence, tokenizer) for sentence in mandatory]
    sentence_tokens = [_count_tokens(sentence, tokenizer) for sentence in sentences]
    return len(sentences), len(mandatory), sum(mandatory_tokens), max(sentence_tokens, default=0)


def _blank_row(
    topic_id: str, template: dict[str, Any], lang: str, response_key: str
) -> dict[str, str]:
    return {
        "topic_id": topic_id,
        "mode": "guide",
        "priority": _priority(template),
        "lang": lang,
        "response_chars": "0",
        "injection_chars": "0",
        "injection_tokens_heuristic": "0",
        "injection_tokens_tokenizer": "",
        "deviation_pct_abs": "",
        "sentence_count": "0",
        "mandatory_sentence_count": "0",
        "mandatory_floor_tokens": "0",
        "max_sentence_tokens": "0",
        "comments": f"missing_{response_key}",
    }


def _measure_row(
    topic_id: str,
    template: dict[str, Any],
    lang: str,
    response: str,
    tokenizer: _Tokenizer | None,
) -> tuple[dict[str, str], int | None, int]:
    injection = _injection(response, lang)
    heuristic = _estimate_tokens(injection)
    tokenizer_tokens = _count_tokens(injection, tokenizer) if tokenizer is not None else None
    sentence_count, mandatory_count, floor_tokens, max_sentence_tokens = _floor_counts(
        response,
        lang,
        tokenizer,
    )
    deviation = (
        ""
        if tokenizer_tokens is None
        else f"{abs((tokenizer_tokens - heuristic) / heuristic * 100):.2f}"
    )
    row = {
        "topic_id": topic_id,
        "mode": "guide",
        "priority": _priority(template),
        "lang": lang,
        "response_chars": str(len(response)),
        "injection_chars": str(len(injection)),
        "injection_tokens_heuristic": str(heuristic),
        "injection_tokens_tokenizer": "" if tokenizer_tokens is None else str(tokenizer_tokens),
        "deviation_pct_abs": deviation,
        "sentence_count": str(sentence_count),
        "mandatory_sentence_count": str(mandatory_count),
        "mandatory_floor_tokens": str(floor_tokens),
        "max_sentence_tokens": str(max_sentence_tokens),
        "comments": "" if tokenizer is not None else "tokenizer_missing;floor_uses_heuristic",
    }
    return row, tokenizer_tokens, floor_tokens


def _priority(template: dict[str, Any]) -> str:
    value = template.get("priority")
    return "" if value is None else str(value)


def _measurement_rows(
    templates: dict[str, dict[str, Any]],
    tokenizer: _Tokenizer | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    stats: Stats = {"ko": {"tokenizer": [], "floor": []}, "en": {"tokenizer": [], "floor": []}}
    for topic_id in sorted(templates):
        template = templates[topic_id]
        if template.get("mode") != "guide":
            continue
        for lang, response_key in (("ko", "response_ko"), ("en", "response_en")):
            response = template.get(response_key)
            if not isinstance(response, str):
                rows.append(_blank_row(topic_id, template, lang, response_key))
                continue
            row, tokenizer_tokens, floor_tokens = _measure_row(
                topic_id,
                template,
                lang,
                response,
                tokenizer,
            )
            rows.append(row)
            stats[lang]["floor"].append(floor_tokens)
            if tokenizer_tokens is not None:
                stats[lang]["tokenizer"].append(tokenizer_tokens)
    rows.extend(_summary_rows(stats))
    return rows


def _summary_rows(stats: Stats) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lang in ("ko", "en"):
        for metric_name, reducer in (("mean", _mean), ("max", _max)):
            rows.append(
                {
                    "topic_id": f"summary_{metric_name}",
                    "mode": "summary",
                    "priority": "",
                    "lang": lang,
                    "response_chars": "",
                    "injection_chars": "",
                    "injection_tokens_heuristic": "",
                    "injection_tokens_tokenizer": _optional_int(reducer(stats[lang]["tokenizer"])),
                    "deviation_pct_abs": "",
                    "sentence_count": "",
                    "mandatory_sentence_count": "",
                    "mandatory_floor_tokens": _optional_int(reducer(stats[lang]["floor"])),
                    "max_sentence_tokens": "",
                    "comments": metric_name,
                }
            )
    return rows


def _mean(values: Sequence[int]) -> int | None:
    return None if not values else round(statistics.fmean(values))


def _max(values: Sequence[int]) -> int | None:
    return None if not values else max(values)


def _optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _write_csv(rows: Sequence[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guide token measurement CLI."""
    args = _build_parser().parse_args(argv)
    templates = _load_templates(args.templates_json)
    tokenizer = _load_tokenizer(args.gemma_model_path)
    _write_csv(_measurement_rows(templates, tokenizer), args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
