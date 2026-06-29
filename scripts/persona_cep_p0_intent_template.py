"""Generate Persona CEP P0 intent-label templates from rounds.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety.approved_template_router import check_approved_template  # noqa: E402

NULL = "NULL"
COLUMNS = (
    "turn_id",
    "language",
    "user_text",
    "is_fact_query",
    "is_emotional",
    "is_greeting",
    "is_curious",
    "is_help_request",
    "safety_topic_match",
    "auto_or_manual",
    "notes",
)

KO_KEYWORDS = {
    "is_fact_query": (
        "\ubb50\uc57c",
        "\uc65c",
        "\uc5b4\ub5bb\uac8c",
        "\uc5b4\ub514",
        "\uc5b8\uc81c",
    ),
    "is_emotional": (
        "\uc18d\uc0c1",
        "\uc2ac\ud504",
        "\ubb34\uc11c",
        "\ud654\ub098",
        "\uc678\ub85c",
        "\uc2eb\uc5b4",
    ),
    "is_greeting": ("\uc548\ub155",),
    "is_curious": ("\uad81\uae08", "\uc2e0\uae30"),
    "is_help_request": ("\ub3c4\uc640\uc918",),
}
EN_KEYWORDS = {
    "is_fact_query": ("what", "why", "how", "where", "when"),
    "is_emotional": ("sad", "scared", "angry", "lonely", "hate"),
    "is_greeting": ("hi", "hello"),
    "is_curious": ("curious", "wonder"),
    "is_help_request": ("help", "can you"),
}
USER_TEXT_FIELDS = (
    "user_text",
    "stt_pred",
    "stt_text",
    "transcript",
    "input_transcript",
    "input_text",
    "gt_text",
)
LANGUAGE_FIELDS = ("language", "lang", "detected_language")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create a Persona CEP P0 intent-label CSV template.",
    )
    parser.add_argument("--rounds-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def read_rounds_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON object records, warning on malformed lines and continuing."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: malformed JSONL line {line_number}: {exc.msg}",
                    file=sys.stderr,
                )
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                print(
                    f"Warning: malformed JSONL line {line_number}: expected object",
                    file=sys.stderr,
                )
    return records


def extract_user_text(turn: dict[str, Any]) -> str | None:
    """Extract user text from plausible STT or transcript fields."""
    for field_name in USER_TEXT_FIELDS:
        value = turn.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_language(turn: dict[str, Any]) -> str:
    """Extract a router-compatible language code from one turn."""
    for field_name in LANGUAGE_FIELDS:
        value = turn.get(field_name)
        if isinstance(value, str) and value.strip():
            return "en" if value.strip().casefold().startswith("en") else "ko"
    return "ko"


def extract_turn_id(turn: dict[str, Any], fallback: int) -> int:
    """Extract a one-based turn id from common rounds.jsonl fields."""
    turn_id = _as_int(turn.get("turn_id"))
    if turn_id is not None:
        return turn_id
    global_turn_id = _as_int(turn.get("global_turn_id"))
    if global_turn_id is not None:
        return global_turn_id + 1
    for field_name in ("round_id", "segment_idx", "exchange", "turn_index_per_lang"):
        value = _as_int(turn.get(field_name))
        if value is not None:
            return value
    return fallback


def label_turn(user_text: str, language: str, turn_id: int) -> dict[str, str]:
    """Auto-label one turn with deterministic keyword heuristics."""
    labels = {
        name: _csv_bool(_matches(user_text, KO_KEYWORDS[name], EN_KEYWORDS[name]))
        for name in KO_KEYWORDS
    }
    if turn_id == 1:
        labels["is_greeting"] = "True"

    safety_match = check_approved_template(user_text, language)
    labels["safety_topic_match"] = str(safety_match["topic_id"]) if safety_match else NULL
    return labels


def write_intent_template(records: list[dict[str, Any]], output_csv: Path) -> int:
    """Write the intent-label template CSV and return the row count."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        for fallback_id, turn in enumerate(records, start=1):
            turn_id = extract_turn_id(turn, fallback_id)
            language = extract_language(turn)
            user_text = extract_user_text(turn)
            if user_text is None:
                writer.writerow(_missing_row(turn_id, language))
                continue
            writer.writerow(_labeled_row(turn_id, language, user_text))
    return len(records)


def _labeled_row(turn_id: int, language: str, user_text: str) -> dict[str, str | int]:
    labels = label_turn(user_text, language, turn_id)
    return {
        "turn_id": turn_id,
        "language": language,
        "user_text": user_text,
        "is_fact_query": labels["is_fact_query"],
        "is_emotional": labels["is_emotional"],
        "is_greeting": labels["is_greeting"],
        "is_curious": labels["is_curious"],
        "is_help_request": labels["is_help_request"],
        "safety_topic_match": labels["safety_topic_match"],
        "auto_or_manual": "auto",
        "notes": "",
    }


def _missing_row(turn_id: int, language: str) -> dict[str, str | int]:
    return {
        "turn_id": turn_id,
        "language": language,
        "user_text": "",
        "is_fact_query": NULL,
        "is_emotional": NULL,
        "is_greeting": NULL,
        "is_curious": NULL,
        "is_help_request": NULL,
        "safety_topic_match": NULL,
        "auto_or_manual": "auto",
        "notes": "missing_user_text",
    }


def _matches(text: str, ko_keywords: tuple[str, ...], en_keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in ko_keywords) or any(
        re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE)
        for keyword in en_keywords
    )


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _csv_bool(value: bool) -> str:
    return "True" if value else "False"


def main(argv: list[str] | None = None) -> int:
    """Run the intent-template generator CLI."""
    args = build_parser().parse_args(argv)
    try:
        records = read_rounds_jsonl(args.rounds_jsonl)
        row_count = write_intent_template(records, args.output_csv)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{row_count} turns auto-labeled; orchestrator must review and finalize "
        "for intent_rules.json seeding.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
