"""Deterministic Funny English mode entry routing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Final, Literal

logger = logging.getLogger(__name__)

FunnyEnglishConfirmationLanguage = Literal["ko"]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_FUNNY_ENGLISH_TEMPLATES_PATH: Final[Path] = (
    _PROJECT_ROOT / "assets" / "filters" / "funny_english_templates.json"
)
_FUNNY_ENGLISH_SECTION: Final[str] = "funny_english"
_FP_GUARDS_SECTION: Final[str] = "fp_guards"
_REQUIRED_SECTION_SET: Final[frozenset[str]] = frozenset(
    {_FUNNY_ENGLISH_SECTION, _FP_GUARDS_SECTION}
)
_FUNNY_ENGLISH_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"confirmation_language", "confirmation_text", "patterns"}
)
_FP_GUARD_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"patterns"})


@dataclass(frozen=True)
class FunnyEnglishMatch:
    """Result of a verified whole-turn Funny English entry command."""

    confirmation_language: FunnyEnglishConfirmationLanguage
    confirmation_text: str
    matched_patterns: tuple[str, ...]


def match_funny_english(text: str) -> FunnyEnglishMatch | None:
    """Return a deterministic Funny English entry match, if one is present."""
    if not text or not text.strip():
        return None

    normalized_input = text.casefold().strip()
    templates = _load_funny_english_templates()
    fp_hits = _regex_hits(templates[_FP_GUARDS_SECTION]["patterns"], normalized_input)
    if fp_hits:
        return None

    section = templates[_FUNNY_ENGLISH_SECTION]
    hits = _regex_hits(section["patterns"], normalized_input)
    if not hits:
        return None

    matched_patterns = tuple(sorted({hit.pattern for hit in hits}))
    logger.info("Funny English mode entry matched")
    return FunnyEnglishMatch(
        confirmation_language=section["confirmation_language"],
        confirmation_text=section["confirmation_text"],
        matched_patterns=matched_patterns,
    )


@dataclass(frozen=True)
class _RegexHit:
    """Internal regex hit used for deterministic Funny English matching."""

    pattern: str
    matched_text: str


@lru_cache(maxsize=1)
def _load_funny_english_templates() -> dict[str, dict[str, Any]]:
    """Load and validate Funny English templates from disk once per process."""
    logger.debug("Loading Funny English templates from %s", _DEFAULT_FUNNY_ENGLISH_TEMPLATES_PATH)
    with _DEFAULT_FUNNY_ENGLISH_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("funny_english_templates.json must contain a top-level object")

    section_ids = set(data)
    if section_ids != _REQUIRED_SECTION_SET:
        missing = sorted(_REQUIRED_SECTION_SET - section_ids)
        extra = sorted(section_ids - _REQUIRED_SECTION_SET)
        raise ValueError(
            "funny_english_templates.json sections must match the expected schema "
            f"(missing={missing}, extra={extra})"
        )

    validated: dict[str, dict[str, Any]] = {}
    for section_id, raw_section in data.items():
        if not isinstance(raw_section, dict):
            raise ValueError(f"Funny English section {section_id!r} must be an object")
        if section_id == _FP_GUARDS_SECTION:
            section = _validate_fp_guard_section(section_id, raw_section)
        else:
            section = _validate_funny_english_section(section_id, raw_section)
        validated[section_id] = section
    return validated


def _validate_funny_english_section(section_id: str, raw_section: dict[str, Any]) -> dict[str, Any]:
    """Validate the Funny English entry section."""
    _validate_field_set(section_id, set(raw_section), _FUNNY_ENGLISH_REQUIRED_FIELDS)
    confirmation_language = raw_section["confirmation_language"]
    if confirmation_language != "ko":
        raise ValueError(f"Funny English section {section_id!r} confirmation_language must be 'ko'")
    confirmation_text = raw_section["confirmation_text"]
    if not isinstance(confirmation_text, str) or not confirmation_text.strip():
        raise ValueError(f"Funny English section {section_id!r} confirmation_text must be text")
    patterns = _coerce_pattern_list(section_id, "patterns", raw_section["patterns"])
    return {
        "confirmation_language": confirmation_language,
        "confirmation_text": confirmation_text,
        "patterns": patterns,
    }


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
        raise ValueError(f"Funny English section {section_id!r} missing fields: {missing_fields}")
    extra_fields = sorted(actual_fields - required_fields)
    if extra_fields:
        raise ValueError(f"Funny English section {section_id!r} has unknown fields: {extra_fields}")


def _coerce_pattern_list(section_id: str, field: str, value: Any) -> list[str]:
    """Validate one anchored regex pattern list."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Funny English section {section_id!r} {field} must be a list[str]")
    if not value:
        raise ValueError(f"Funny English section {section_id!r} {field} must not be empty")
    for pattern in value:
        if not pattern.startswith(r"\A") or not pattern.endswith(r"\Z"):
            raise ValueError(
                f"Funny English section {section_id!r} pattern must be whole-turn anchored: "
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
        raise ValueError(f"invalid Funny English regex{location}: {pattern!r}") from exc


__all__ = ["FunnyEnglishMatch", "match_funny_english"]
