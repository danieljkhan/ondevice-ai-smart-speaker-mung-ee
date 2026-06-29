#!/usr/bin/env python3
"""Generate an ADR draft from CLI inputs.

This script creates a markdown ADR draft in the style of the repository's
existing ADR documents. It can print the draft to stdout or write it to a file.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from textwrap import fill

LOGGER = logging.getLogger(__name__)

DEFAULT_WIDTH = 100


@dataclass(frozen=True)
class AdrDraftInput:
    """Input values used to render an ADR draft."""

    number: int
    title: str
    decision: str
    context: str
    status: str
    draft_date: date
    consequences: str | None
    related_adrs: list[str]
    references: list[str]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser."""

    parser = argparse.ArgumentParser(description="Generate an ADR draft markdown file.")
    parser.add_argument("--number", type=int, required=True, help="ADR number.")
    parser.add_argument("--title", required=True, help="ADR title.")
    parser.add_argument("--decision", required=True, help="Decision summary.")
    parser.add_argument("--context", required=True, help="Context section content.")
    parser.add_argument(
        "--status",
        default="Proposed",
        help="ADR status label. Defaults to Proposed.",
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=date.today(),
        help="ADR date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--consequences",
        help="Optional consequences section body.",
    )
    parser.add_argument(
        "--related-adr",
        dest="related_adrs",
        action="append",
        default=[],
        help="Related ADR entry. Repeat for multiple values.",
    )
    parser.add_argument(
        "--reference",
        dest="references",
        action="append",
        default=[],
        help="Reference entry. Repeat for multiple values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. If omitted, the draft is printed to stdout.",
    )
    return parser


def _parse_date(value: str) -> date:
    """Parse an ISO-8601 date string."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def _normalize_list(values: Sequence[str]) -> list[str]:
    """Normalize list-style CLI values by trimming empties."""

    return [item.strip() for item in values if item.strip()]


def _format_heading(number: int, title: str) -> str:
    """Format the ADR title heading."""

    return f"# ADR {number:04d}: {title.strip()}"


def _wrap_paragraph(text: str) -> str:
    """Wrap a paragraph while preserving blank lines."""

    paragraphs = [part.strip() for part in text.strip().split("\n\n")]
    if not any(paragraphs):
        return ""
    return "\n".join(
        fill(paragraph.replace("\n", " "), width=DEFAULT_WIDTH) if paragraph else ""
        for paragraph in paragraphs
    )


def _format_bullets(items: Sequence[str], empty_label: str) -> str:
    """Format bullet items or a placeholder when no items are present."""

    normalized = _normalize_list(items)
    if not normalized:
        return f"- {empty_label}"
    return "\n".join(f"- {item}" for item in normalized)


def render_draft(data: AdrDraftInput) -> str:
    """Render the ADR draft markdown."""

    consequences = data.consequences
    if consequences is None:
        consequences = (
            f"This decision will be treated as the default approach for ADR "
            f"{data.number:04d} unless superseded by a later ADR."
        )

    related_adrs = _format_bullets(data.related_adrs, "None yet")
    references = _format_bullets(data.references, "None yet")

    sections = [
        _format_heading(data.number, data.title),
        "",
        f"- **Status**: {data.status.strip() or 'Proposed'}",
        f"- **Date**: {data.draft_date.isoformat()}",
        "",
        "## Context",
        "",
        _wrap_paragraph(data.context),
        "",
        "## Decision",
        "",
        _wrap_paragraph(data.decision),
        "",
        "## Consequences",
        "",
        _wrap_paragraph(consequences),
        "",
        "## Related ADRs",
        "",
        related_adrs,
        "",
        "## References",
        "",
        references,
    ]
    return "\n".join(sections).rstrip() + "\n"


def _build_input(args: argparse.Namespace) -> AdrDraftInput:
    """Convert parsed CLI arguments into a typed input object."""

    return AdrDraftInput(
        number=args.number,
        title=args.title,
        decision=args.decision,
        context=args.context,
        status=args.status,
        draft_date=args.date,
        consequences=args.consequences,
        related_adrs=_normalize_list(args.related_adrs),
        references=_normalize_list(args.references),
    )


def _write_output(text: str, output: Path | None) -> None:
    """Write the rendered draft to a file or stdout."""

    if output is None:
        print(text, end="")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    LOGGER.info("Wrote ADR draft to %s", output)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ADR draft generator CLI."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.title.strip():
        parser.error("--title must not be empty")
    if not args.decision.strip():
        parser.error("--decision must not be empty")
    if not args.context.strip():
        parser.error("--context must not be empty")

    draft_input = _build_input(args)
    draft = render_draft(draft_input)

    try:
        _write_output(draft, args.output)
    except OSError as exc:
        LOGGER.error("Failed to write ADR draft: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
