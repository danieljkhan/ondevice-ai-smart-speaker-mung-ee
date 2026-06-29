"""Deterministic spoken-query routing for current time / day / date.

Answers child-spoken "what time is it / what day is it / what's the date"
questions from the device's local KST clock without invoking the LLM. The
trigger phrases live in ``assets/filters/datetime_templates.json`` as
whole-turn anchored regexes; this module loads and validates them once per
process, then builds a fully spelled-out Korean answer via
:mod:`core.korean_datetime`.

Composite "date and day" requests are checked before the single date/day
sections so the combined answer wins.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Final

from core.conversation_memory_schema import KST
from core.korean_datetime import (
    append_copula,
    format_date_ko,
    format_day_ko,
    format_time_ko,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_DATETIME_TEMPLATES_PATH: Final[Path] = (
    _PROJECT_ROOT / "assets" / "filters" / "datetime_templates.json"
)
_FP_GUARDS_SECTION: Final[str] = "fp_guards"
# Query sections in evaluation order; ``date_day`` precedes single date/day so
# composite requests win over their narrower siblings.
_QUERY_SECTIONS: Final[tuple[str, ...]] = ("time", "date_day", "day", "date")
_REQUIRED_SECTION_SET: Final[frozenset[str]] = frozenset({*_QUERY_SECTIONS, _FP_GUARDS_SECTION})


@dataclass(frozen=True)
class DateTimeMatch:
    """Result of a verified whole-turn date/time query."""

    kind: str
    response_text: str
    matched_patterns: tuple[str, ...]


def match_datetime_query(text: str, *, now: datetime | None = None) -> DateTimeMatch | None:
    """Return a deterministic time/day/date answer, if the turn requests one.

    Args:
        text: The user's whole-turn transcript.
        now: Optional clock override (KST is used when omitted) for testing.

    Returns:
        A :class:`DateTimeMatch` with a spelled-out Korean answer, or ``None``
        when the turn is not a clock/calendar query.
    """
    if not text or not text.strip():
        return None

    normalized_input = text.casefold().strip()
    templates = _load_datetime_templates()
    if _regex_hits(templates[_FP_GUARDS_SECTION]["patterns"], normalized_input):
        return None

    moment = now or datetime.now(KST)
    for kind in _QUERY_SECTIONS:
        hits = _regex_hits(templates[kind]["patterns"], normalized_input)
        if not hits:
            continue
        matched_patterns = tuple(sorted({hit.pattern for hit in hits}))
        response_text = _build_response(kind, moment)
        logger.info("Datetime query matched: kind=%s", kind)
        return DateTimeMatch(
            kind=kind,
            response_text=response_text,
            matched_patterns=matched_patterns,
        )
    return None


# Builders keyed by query kind. Each returns the bare phrase (no copula) so the
# copula can be appended uniformly with batchim-aware agreement.
_RESPONSE_BUILDERS: Final[dict[str, Callable[[datetime], str]]] = {
    "time": lambda moment: f"지금 {format_time_ko(moment)}",
    "day": lambda moment: f"오늘은 {format_day_ko(moment)}",
    "date": lambda moment: f"오늘은 {format_date_ko(moment)}",
    "date_day": lambda moment: f"오늘은 {format_date_ko(moment)} {format_day_ko(moment)}",
}


def _build_response(kind: str, moment: datetime) -> str:
    """Build the spelled-out Korean answer for one query kind plus copula."""
    return append_copula(_RESPONSE_BUILDERS[kind](moment))


@dataclass(frozen=True)
class _RegexHit:
    """Internal regex hit used for deterministic datetime matching."""

    pattern: str
    matched_text: str


@lru_cache(maxsize=1)
def _load_datetime_templates() -> dict[str, dict[str, Any]]:
    """Load and validate datetime templates from disk once per process."""
    logger.debug("Loading datetime templates from %s", _DEFAULT_DATETIME_TEMPLATES_PATH)
    with _DEFAULT_DATETIME_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("datetime_templates.json must contain a top-level object")

    section_ids = set(data)
    if section_ids != _REQUIRED_SECTION_SET:
        missing = sorted(_REQUIRED_SECTION_SET - section_ids)
        extra = sorted(section_ids - _REQUIRED_SECTION_SET)
        raise ValueError(
            "datetime_templates.json sections must match the expected schema "
            f"(missing={missing}, extra={extra})"
        )

    validated: dict[str, dict[str, Any]] = {}
    for section_id, raw_section in data.items():
        if not isinstance(raw_section, dict):
            raise ValueError(f"datetime section {section_id!r} must be an object")
        if set(raw_section) != {"patterns"}:
            raise ValueError(f"datetime section {section_id!r} must have only 'patterns'")
        validated[section_id] = {
            "patterns": _coerce_pattern_list(section_id, raw_section["patterns"])
        }
    return validated


def _coerce_pattern_list(section_id: str, value: Any) -> list[str]:
    """Validate one anchored regex pattern list."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"datetime section {section_id!r} patterns must be a list[str]")
    if not value:
        raise ValueError(f"datetime section {section_id!r} patterns must not be empty")
    for pattern in value:
        if not pattern.startswith(r"\A") or not pattern.endswith(r"\Z"):
            raise ValueError(
                f"datetime section {section_id!r} pattern must be whole-turn anchored: {pattern!r}"
            )
        _compile_pattern(section_id, pattern)
    return value


def _regex_hits(patterns: list[str], normalized_input: str) -> list[_RegexHit]:
    """Return regex hits for patterns over already-casefolded input."""
    hits: list[_RegexHit] = []
    for pattern in patterns:
        compiled = _compile_pattern("", pattern)
        match = compiled.search(normalized_input)
        if match is None:
            continue
        matched_text = match.group(0)
        if not matched_text:
            continue
        hits.append(_RegexHit(pattern=pattern, matched_text=matched_text))
    return hits


@cache
def _compile_pattern(section_id: str, pattern: str) -> re.Pattern[str]:
    """Compile one regex pattern with a useful validation error."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        location = f" in {section_id}" if section_id else ""
        raise ValueError(f"invalid datetime regex{location}: {pattern!r}") from exc


__all__ = ["DateTimeMatch", "match_datetime_query"]
