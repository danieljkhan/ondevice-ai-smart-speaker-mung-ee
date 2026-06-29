"""Audit confirmable-fact shortlist entries against exclude-family safety terms."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SHORTLIST_PATH: Final[Path] = REPO_ROOT / "assets" / "prompts" / "confirmable_facts.json"
PATTERN_PATH: Final[Path] = REPO_ROOT / "scripts" / "_phaseA_category_patterns.json"
OUTPUT_ROOT: Final[Path] = REPO_ROOT / "artifacts" / "phaseA-a2"
WORLD_HISTORY_CATEGORY: Final[str] = "world_history_light"


@dataclass(frozen=True)
class AuditRecord:
    """One per-entry safety audit result."""

    topic: str
    category: str
    age_band: str
    flagged: bool
    flag_reasons: list[str]
    family_hits: dict[str, list[dict[str, Any]]]
    violence_saturation_count: int


def build_parser() -> argparse.ArgumentParser:
    """Build the shortlist safety-audit CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=SHORTLIST_PATH,
        help="Shortlist JSON path to audit.",
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=PATTERN_PATH,
        help="Category-pattern JSON containing exclude-family term lists.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Artifact root directory for safety-audit output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the shortlist safety audit and write the artifact payload."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    entries = load_shortlist_entries(args.shortlist)
    exclude_terms, violence_terms = load_term_lists(args.patterns)
    records = [
        audit_entry(entry=entry, exclude_terms=exclude_terms, violence_terms=violence_terms)
        for entry in entries
    ]
    flagged_count = sum(1 for record in records if record.flagged)

    output_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "safety_audit.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shortlist_path": str(args.shortlist.relative_to(REPO_ROOT)),
        "pattern_path": str(args.patterns.relative_to(REPO_ROOT)),
        "total_entries": len(records),
        "flagged_entries": flagged_count,
        "records": [asdict(record) for record in records],
    }
    write_json(output_path, payload)

    logger.info(
        "safety_audit_completed flagged=%s total=%s output=%s",
        flagged_count,
        len(records),
        output_path.relative_to(REPO_ROOT),
    )
    sys.stdout.write(f"{flagged_count}/{len(records)} entries flagged\n")
    return 0 if flagged_count == 0 else 1


def load_shortlist_entries(path: Path) -> list[dict[str, Any]]:
    """Load shortlist entries from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = "Shortlist root must be a JSON array"
        raise ValueError(msg)
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            msg = f"Shortlist entry {index} must be a JSON object"
            raise ValueError(msg)
        entries.append(row)
    return entries


def load_term_lists(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load exclude-family and violence-saturation term lists."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "Pattern payload must be a JSON object"
        raise ValueError(msg)

    exclude_terms = payload.get("exclude_terms")
    violence_terms = payload.get("violence_saturation_terms")
    if not isinstance(exclude_terms, dict) or not isinstance(violence_terms, dict):
        msg = "Pattern payload missing exclude_terms or violence_saturation_terms"
        raise ValueError(msg)
    return exclude_terms, violence_terms


def audit_entry(
    *,
    entry: dict[str, Any],
    exclude_terms: dict[str, Any],
    violence_terms: dict[str, Any],
) -> AuditRecord:
    """Audit one shortlist entry against the configured safety lists."""

    topic = _string_field(entry, "topic")
    category = _string_field(entry, "category")
    age_band = str(entry.get("age_band", "under_10")).strip() or "under_10"

    family_hits = collect_family_hits(entry=entry, exclude_terms=exclude_terms)
    flag_reasons = [f"exclude:{family}" for family in sorted(family_hits)]

    violence_saturation_count = 0
    if category == WORLD_HISTORY_CATEGORY:
        violence_saturation_count = count_violence_saturation_hits(
            entry=entry,
            violence_terms=violence_terms,
        )
        if violence_saturation_count > 3:
            flag_reasons.append("violence_saturation")

    return AuditRecord(
        topic=topic,
        category=category,
        age_band=age_band,
        flagged=bool(flag_reasons),
        flag_reasons=flag_reasons,
        family_hits=family_hits,
        violence_saturation_count=violence_saturation_count,
    )


def collect_family_hits(
    *,
    entry: dict[str, Any],
    exclude_terms: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Collect per-family term hits for one entry."""

    family_hits: dict[str, list[dict[str, Any]]] = {}
    for lang in ("ko", "en"):
        text_fields = list(iter_text_fields(entry=entry, lang=lang))
        language_terms = exclude_terms.get(lang)
        if not isinstance(language_terms, dict):
            continue
        for family, terms in language_terms.items():
            if not isinstance(terms, list):
                continue
            hits_for_family = family_hits.setdefault(str(family), [])
            for term in terms:
                if not isinstance(term, str):
                    continue
                fields = [
                    field_name
                    for field_name, text in text_fields
                    if term_matches_text(term=term, text=text, lang=lang)
                ]
                if fields:
                    hits_for_family.append({"lang": lang, "term": term, "fields": fields})
            if not hits_for_family:
                family_hits.pop(str(family), None)
    return family_hits


def count_violence_saturation_hits(
    *,
    entry: dict[str, Any],
    violence_terms: dict[str, Any],
) -> int:
    """Count violence-saturation term hits across one world-history entry."""

    total_hits = 0
    for lang in ("ko", "en"):
        text_fields = list(iter_text_fields(entry=entry, lang=lang))
        language_terms = violence_terms.get(lang)
        if not isinstance(language_terms, list):
            continue
        for term in language_terms:
            if not isinstance(term, str):
                continue
            for _field_name, text in text_fields:
                if term_matches_text(term=term, text=text, lang=lang):
                    total_hits += 1
    return total_hits


def iter_text_fields(*, entry: dict[str, Any], lang: str) -> list[tuple[str, str]]:
    """Return shortlist text fields for the requested language."""

    if lang == "ko":
        fields: list[tuple[str, str]] = [
            ("fact_ko", _optional_string(entry.get("fact_ko"))),
        ]
        fields.extend(
            (
                f"triggers_ko[{index}]",
                value,
            )
            for index, value in enumerate(_string_list(entry.get("triggers_ko")))
        )
        return [(field_name, text) for field_name, text in fields if text]

    fields = [("fact_en", _optional_string(entry.get("fact_en")))]
    fields.extend(
        (
            f"triggers_en[{index}]",
            value,
        )
        for index, value in enumerate(_string_list(entry.get("triggers_en")))
    )
    return [(field_name, text) for field_name, text in fields if text]


def term_matches_text(*, term: str, text: str, lang: str) -> bool:
    """Return whether one configured safety term matches the given text."""

    if lang == "ko":
        return term.casefold() in text.casefold()

    pattern = re.compile(rf"\b{re.escape(term.casefold())}\b")
    return bool(pattern.search(text.casefold()))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with repo-standard formatting."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _string_field(entry: dict[str, Any], field_name: str) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        msg = f"Shortlist entry field {field_name} must be a non-empty string"
        raise ValueError(msg)
    return value.strip()


def _optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
