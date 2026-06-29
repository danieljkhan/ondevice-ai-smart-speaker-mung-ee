"""Deterministic explicit-recall query routing.

Detects child-spoken "what did I say / what's my name / what do I like"
questions that explicitly request the device recall a prior statement. The
trigger phrases live in ``assets/filters/recall_query_templates.json`` as
whole-turn anchored regexes; this module loads and validates them once per
process, then reports the matched intent (``name`` / ``general_recall``) so
the pipeline can answer from conversation memory without invoking the LLM.

The ``sub_kind`` drives how the recalled snippet is phrased. False-positive
guards reject third-party recall ("what did mom say"), general-knowledge
idioms ("what is memory"), and store-requests ("remember this") before any
recall section is evaluated.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_RECALL_QUERY_TEMPLATES_PATH: Final[Path] = (
    _PROJECT_ROOT / "assets" / "filters" / "recall_query_templates.json"
)
_FP_GUARDS_SECTION: Final[str] = "fp_guards"
# Recall sections in evaluation order; each names the sub_kind that drives the
# answer phrasing. ``general_recall`` is last so the narrower ``name`` intent
# wins over the generic "what did I say" intent.
_QUERY_SECTIONS: Final[tuple[str, ...]] = ("name", "general_recall")
_REQUIRED_SECTION_SET: Final[frozenset[str]] = frozenset({*_QUERY_SECTIONS, _FP_GUARDS_SECTION})
_RECALL_SECTION_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"sub_kind", "patterns"})
_FP_GUARD_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"patterns"})


@dataclass(frozen=True)
class RecallQueryMatch:
    """Result of a verified whole-turn explicit-recall query."""

    kind: str
    sub_kind: str
    matched_patterns: tuple[str, ...]


def match_recall_query(text: str) -> RecallQueryMatch | None:
    """Return a deterministic explicit-recall match, if the turn requests one.

    Args:
        text: The user's whole-turn transcript.

    Returns:
        A :class:`RecallQueryMatch` naming the recall intent, or ``None`` when
        the turn is not an explicit-recall question.
    """
    if not text or not text.strip():
        return None

    normalized_input = text.casefold().strip()
    templates = _load_recall_query_templates()
    if _regex_hits(templates[_FP_GUARDS_SECTION]["patterns"], normalized_input):
        return None

    for kind in _QUERY_SECTIONS:
        section = templates[kind]
        hits = _regex_hits(section["patterns"], normalized_input)
        if not hits:
            continue
        matched_patterns = tuple(sorted({hit.pattern for hit in hits}))
        logger.info("Recall query matched: sub_kind=%s", section["sub_kind"])
        return RecallQueryMatch(
            kind="recall",
            sub_kind=section["sub_kind"],
            matched_patterns=matched_patterns,
        )
    return None


@dataclass(frozen=True)
class _RegexHit:
    """Internal regex hit used for deterministic recall-query matching."""

    pattern: str
    matched_text: str


@lru_cache(maxsize=1)
def _load_recall_query_templates() -> dict[str, dict[str, Any]]:
    """Load and validate recall-query templates from disk once per process."""
    logger.debug("Loading recall-query templates from %s", _DEFAULT_RECALL_QUERY_TEMPLATES_PATH)
    with _DEFAULT_RECALL_QUERY_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("recall_query_templates.json must contain a top-level object")

    section_ids = set(data)
    if section_ids != _REQUIRED_SECTION_SET:
        missing = sorted(_REQUIRED_SECTION_SET - section_ids)
        extra = sorted(section_ids - _REQUIRED_SECTION_SET)
        raise ValueError(
            "recall_query_templates.json sections must match the expected schema "
            f"(missing={missing}, extra={extra})"
        )

    validated: dict[str, dict[str, Any]] = {}
    for section_id, raw_section in data.items():
        if not isinstance(raw_section, dict):
            raise ValueError(f"recall-query section {section_id!r} must be an object")
        if section_id == _FP_GUARDS_SECTION:
            section = _validate_fp_guard_section(section_id, raw_section)
        else:
            section = _validate_recall_section(section_id, raw_section)
        validated[section_id] = section
    return validated


def _validate_recall_section(section_id: str, raw_section: dict[str, Any]) -> dict[str, Any]:
    """Validate one explicit-recall query section."""
    _validate_field_set(section_id, set(raw_section), _RECALL_SECTION_REQUIRED_FIELDS)
    sub_kind = raw_section["sub_kind"]
    if not isinstance(sub_kind, str) or sub_kind != section_id:
        raise ValueError(f"recall-query section {section_id!r} sub_kind must equal the section id")
    patterns = _coerce_pattern_list(section_id, "patterns", raw_section["patterns"])
    return {"sub_kind": sub_kind, "patterns": patterns}


def _validate_fp_guard_section(section_id: str, raw_section: dict[str, Any]) -> dict[str, Any]:
    """Validate the false-positive guard section."""
    _validate_field_set(section_id, set(raw_section), _FP_GUARD_REQUIRED_FIELDS)
    return {"patterns": _coerce_pattern_list(section_id, "patterns", raw_section["patterns"])}


def _validate_field_set(
    section_id: str,
    actual_fields: set[str],
    required_fields: frozenset[str],
) -> None:
    """Validate that a JSON section contains exactly the expected fields."""
    missing_fields = sorted(required_fields - actual_fields)
    if missing_fields:
        raise ValueError(f"recall-query section {section_id!r} missing fields: {missing_fields}")
    extra_fields = sorted(actual_fields - required_fields)
    if extra_fields:
        raise ValueError(f"recall-query section {section_id!r} has unknown fields: {extra_fields}")


def _coerce_pattern_list(section_id: str, field: str, value: Any) -> list[str]:
    """Validate one anchored regex pattern list."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"recall-query section {section_id!r} {field} must be a list[str]")
    if not value:
        raise ValueError(f"recall-query section {section_id!r} {field} must not be empty")
    for pattern in value:
        if not pattern.startswith(r"\A") or not pattern.endswith(r"\Z"):
            raise ValueError(
                f"recall-query section {section_id!r} pattern must be whole-turn anchored: "
                f"{pattern!r}"
            )
        _compile_pattern(section_id, field, pattern)
    return value


def _regex_hits(patterns: list[str], normalized_input: str) -> list[_RegexHit]:
    """Return regex hits for patterns over already-casefolded input."""
    hits: list[_RegexHit] = []
    for pattern in patterns:
        compiled = _compile_pattern("", "", pattern)
        match = compiled.search(normalized_input)
        if match is None:
            continue
        matched_text = match.group(0)
        if not matched_text:
            continue
        hits.append(_RegexHit(pattern=pattern, matched_text=matched_text))
    return hits


@cache
def _compile_pattern(section_id: str, field: str, pattern: str) -> re.Pattern[str]:
    """Compile one regex pattern with a useful validation error."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        location = f" in {section_id}.{field}" if section_id and field else ""
        raise ValueError(f"invalid recall-query regex{location}: {pattern!r}") from exc


__all__ = ["RecallQueryMatch", "match_recall_query"]
