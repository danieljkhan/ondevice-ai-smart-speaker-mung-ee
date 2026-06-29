"""Aggregate-only vocabulary baseline scan for local conversation JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.pipeline import LEGACY_TIER1_HOTWORDS_VOCABULARY, _tokenize_raw_stt

DEFAULT_CORPUS_DIR = Path("/var/lib/mungi/conversations")


def _extract_utterance_text(row: Mapping[str, Any]) -> str:
    """Return the best available utterance text field from one conversation row."""
    for field_name in ("raw_stt_text", "user_text", "text", "transcript", "query"):
        value = row.get(field_name)
        if isinstance(value, str):
            return value
    return ""


def count_vocabulary_entries(
    text: str,
    vocabulary: tuple[str, ...] = LEGACY_TIER1_HOTWORDS_VOCABULARY,
) -> int:
    """Count distinct legacy hotword-vocabulary entries in one utterance."""
    return len(set(_tokenize_raw_stt(text)) & set(vocabulary))


def iter_conversation_rows(corpus_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows from all JSONL files under a conversation directory."""
    for path in sorted(corpus_dir.glob("*.jsonl")):
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


def scan_corpus(
    corpus_dir: Path,
    vocabulary: tuple[str, ...] = LEGACY_TIER1_HOTWORDS_VOCABULARY,
    count_threshold: int = 6,
) -> dict[str, Any]:
    """Scan a conversation corpus and return aggregate-only vocabulary counts."""
    distribution: Counter[int] = Counter()
    files_scanned = len(list(corpus_dir.glob("*.jsonl")))
    utterances_scanned = 0
    max_entries = 0

    for row in iter_conversation_rows(corpus_dir):
        utterances_scanned += 1
        count = count_vocabulary_entries(_extract_utterance_text(row), vocabulary)
        distribution[count] += 1
        max_entries = max(max_entries, count)

    threshold_violations = sum(
        utterance_count
        for count, utterance_count in distribution.items()
        if count >= count_threshold
    )
    seven_or_more = sum(
        utterance_count for count, utterance_count in distribution.items() if count >= 7
    )
    if threshold_violations == 0:
        recommendation = "threshold OK"
    elif seven_or_more == 0:
        recommendation = "raise to 7"
    else:
        recommendation = "fraction-only mode recommended"

    return {
        "files_scanned": files_scanned,
        "utterances_scanned": utterances_scanned,
        "max_vocabulary_entries_in_single_utterance": max_entries,
        "distribution": {
            str(count): distribution.get(count, 0) for count in range(len(vocabulary) + 1)
        },
        "count_threshold_validation": {
            "current_threshold_6_violations": threshold_violations,
            "recommendation": recommendation,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Scan conversation JSONL files for aggregate hotword-vocabulary counts.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory containing conversation *.jsonl files.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional aggregate JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregate-only corpus scan CLI."""
    args = build_parser().parse_args(argv)
    summary = scan_corpus(args.corpus_dir)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(summary_json + "\n", encoding="utf-8")
    else:
        sys.stdout.write(summary_json)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
