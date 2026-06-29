"""Generate a markdown conversation dump from E2E rounds data."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mungi.scripts.generate_e2e_conversation_dump")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the conversation dump generator."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown conversation dump from rounds.jsonl.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing rounds.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for the report header. Defaults to the input directory name.",
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


def _read_rounds(path: Path) -> list[dict[str, Any]]:
    """Read rounds.jsonl records from disk."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def _normalize_language(value: Any) -> str:
    """Normalize a turn language value to ko/en buckets."""
    if value is None:
        return "ko"
    normalized = str(value).strip().lower()
    return "en" if normalized == "en" else "ko"


def _format_seconds(value: float | None) -> str:
    """Format a seconds value for inline markdown metadata."""
    return "-" if value is None else f"{value:.3f}s"


def generate_markdown(rounds: list[dict[str, Any]], source_path: Path, label: str) -> str:
    """Render the conversation dump markdown document."""
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    total_turns = 0
    ko_turns = 0
    en_turns = 0
    lines = [f"# Conversation Script - {label}", ""]
    lines.append(f"> Source: `{source_path}`")
    lines.append(f"> Generated: {timestamp}")
    lines.append(f"> Total rounds: {len(rounds)}")

    round_sections: list[str] = []
    for raw_round in rounds:
        round_num = _as_int(raw_round.get("round")) or 0
        topics = raw_round.get("topics")
        if not isinstance(topics, list) or not topics:
            continue
        topic_name = ""
        round_turns: list[dict[str, Any]] = []
        for topic_entry in topics:
            if not isinstance(topic_entry, dict):
                continue
            if not topic_name:
                topic_name = str(topic_entry.get("topic") or raw_round.get("topic") or "")
            topic_turns = topic_entry.get("turns")
            if not isinstance(topic_turns, list):
                continue
            for turn in topic_turns:
                if isinstance(turn, dict):
                    round_turns.append(turn)
        if not round_turns:
            continue

        language = "ko"
        for item in round_turns:
            language = _normalize_language(item.get("language"))
            break

        round_sections.append(f'## Round {round_num} - "{topic_name}" ({language})')
        round_sections.append("")
        for turn in round_turns:
            if not isinstance(turn, dict):
                continue
            total_turns += 1
            turn_language = _normalize_language(turn.get("language"))
            if turn_language == "en":
                en_turns += 1
            else:
                ko_turns += 1
            exchange = _as_int(turn.get("exchange")) or 0
            llm_tokens = _as_int(turn.get("llm_tokens"))
            total_time_s = _as_float(turn.get("total_time_s"))
            round_sections.append(
                (
                    f"**Turn {exchange}** [{turn_language}, "
                    f"llm_tokens={llm_tokens if llm_tokens is not None else '-'}, "
                    f"total={_format_seconds(total_time_s)}]"
                ),
            )
            round_sections.append(f"- User: {str(turn.get('user_text') or '')}")
            round_sections.append(f"- Mung-i: {str(turn.get('assistant_text') or '')}")
            if not bool(turn.get("success", False)):
                error = str(turn.get("error") or "-")
                round_sections.append(f"**FAILED:** {error}")
            round_sections.append("")
        round_sections.append("---")
        round_sections.append("")

    lines.append(f"> Total turns: {total_turns}")
    lines.append(f"> KO turns: {ko_turns}, EN turns: {en_turns}")
    lines.append("")
    lines.extend(round_sections)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Run the conversation dump generator CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()

    input_dir = args.input_dir.resolve()
    rounds_path = input_dir / "rounds.jsonl"
    if not rounds_path.exists():
        logger.error("Missing rounds file: %s", rounds_path)
        return 1

    output_path = args.output.resolve()
    label = args.label or input_dir.name
    logger.info("Generating conversation dump from %s", rounds_path)

    rounds = _read_rounds(rounds_path)
    markdown = generate_markdown(rounds, rounds_path, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Conversation dump written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
