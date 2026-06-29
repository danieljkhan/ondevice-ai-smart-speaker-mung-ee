"""Post-hoc F31-3 wakeword-collapse classifier for archived rounds.jsonl rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pipeline import _tokenize_raw_stt

WAKEWORDS: Final[tuple[str, ...]] = ("뭉이야", "뭉이")
VERDICTS: Final[tuple[str, ...]] = ("full_collapse", "partial_injection", "clean")


def _row_text(row: Mapping[str, Any], field_names: tuple[str, ...]) -> str:
    """Return the first string value from a row for the requested field names."""
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str):
            return value
    return ""


def _count_wakeword_tokens(text: str, wakewords: tuple[str, ...] = WAKEWORDS) -> int:
    """Count exact wakeword tokens in text after raw-STT token preparation."""
    wakeword_set = set(wakewords)
    return sum(1 for token in _tokenize_raw_stt(text) if token in wakeword_set)


def f31_3_classify(row: dict[str, Any]) -> tuple[str, int]:
    """Classify one archived row using Plan v4 F31-3 post-hoc semantics."""
    raw_stt_text = _row_text(row, ("raw_stt_text", "user_text", "stt_text"))
    query = _row_text(row, ("query",))
    wakeword_token_count = _count_wakeword_tokens(raw_stt_text)

    if _count_wakeword_tokens(query) > 0:
        return "clean", wakeword_token_count
    if wakeword_token_count >= 5:
        return "full_collapse", wakeword_token_count
    if 2 <= wakeword_token_count < 5:
        return "partial_injection", wakeword_token_count
    return "clean", wakeword_token_count


def iter_round_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows from a rounds.jsonl file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise ValueError(msg)
            yield row


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return per-row F31-3 verdicts and aggregate counts."""
    per_row: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        verdict, wakeword_token_count = f31_3_classify(row)
        counts[verdict] += 1
        per_row.append(
            {
                "index": index,
                "id": row.get("id", row.get("query_id")),
                "verdict": verdict,
                "wakeword_token_count": wakeword_token_count,
            }
        )

    return {
        "rows_scanned": len(rows),
        "counts": {verdict: counts.get(verdict, 0) for verdict in VERDICTS},
        "rows": per_row,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Classify F31-3 wakeword-collapse rows in a rounds.jsonl file.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to rounds.jsonl.")
    parser.add_argument("--output", type=Path, default=None, help="Optional summary JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the post-hoc classifier CLI."""
    args = build_parser().parse_args(argv)
    summary = summarize_rows(list(iter_round_rows(args.input)))
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(summary_json + "\n", encoding="utf-8")
    else:
        sys.stdout.write(summary_json)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
