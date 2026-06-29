"""Deterministic Korean-English session language switch routing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Final, Literal

logger = logging.getLogger(__name__)

LanguageCode = Literal["ko", "en"]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_LANGUAGE_SWITCH_TEMPLATES_PATH: Final[Path] = (
    _PROJECT_ROOT / "assets" / "filters" / "language_switch_templates.json"
)
_KO_TO_EN_SECTION: Final[str] = "ko_to_en"
_EN_TO_KO_SECTION: Final[str] = "en_to_ko"
_FP_GUARDS_SECTION: Final[str] = "fp_guards"
_REQUIRED_SECTION_SET: Final[frozenset[str]] = frozenset(
    {_KO_TO_EN_SECTION, _EN_TO_KO_SECTION, _FP_GUARDS_SECTION}
)
_SWITCH_SECTION_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "current_language",
        "target_language",
        "confirmation_language",
        "confirmation_text",
        "patterns",
    }
)
_FP_GUARD_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"patterns"})
_SECTION_BY_CURRENT_LANGUAGE: Final[dict[LanguageCode, str]] = {
    "ko": _KO_TO_EN_SECTION,
    "en": _EN_TO_KO_SECTION,
}


@dataclass(frozen=True)
class LanguageSwitchMatch:
    """Result of a verified whole-turn language switch command."""

    target_language: LanguageCode
    confirmation_language: LanguageCode
    confirmation_text: str
    matched_patterns: tuple[str, ...]


def match_language_switch(text: str, current_language: str) -> LanguageSwitchMatch | None:
    """Return a deterministic session-language switch match, if one is present.

    Matching is whole-turn anchored and direction-gated. A Korean-to-English
    trigger can only fire while the session is Korean, and an English-to-Korean
    trigger can only fire while the session is English.
    """
    if not text or not text.strip():
        return None

    normalized_input = text.casefold().strip()
    templates = _load_language_switch_templates()
    fp_hits = _regex_hits(templates[_FP_GUARDS_SECTION]["patterns"], normalized_input)
    if fp_hits:
        return None

    current = _normalize_language(current_language)
    section_id = _SECTION_BY_CURRENT_LANGUAGE[current]
    section = templates[section_id]
    hits = _regex_hits(section["patterns"], normalized_input)
    if not hits:
        return None

    target_language = section["target_language"]
    matched_patterns = tuple(sorted({hit.pattern for hit in hits}))
    logger.info("Language switch matched %s -> %s", current, target_language)
    return LanguageSwitchMatch(
        target_language=target_language,
        confirmation_language=section["confirmation_language"],
        confirmation_text=section["confirmation_text"],
        matched_patterns=matched_patterns,
    )


def get_switch_confirmation(target_language: LanguageCode) -> tuple[str, LanguageCode]:
    """Return the confirmation text and TTS language for a target language."""
    templates = _load_language_switch_templates()
    for section_id in (_KO_TO_EN_SECTION, _EN_TO_KO_SECTION):
        section = templates[section_id]
        if section["target_language"] == target_language:
            return section["confirmation_text"], section["confirmation_language"]
    raise ValueError(f"unsupported language switch target: {target_language!r}")


@dataclass(frozen=True)
class _RegexHit:
    """Internal regex hit used for deterministic language-switch matching."""

    pattern: str
    matched_text: str


@lru_cache(maxsize=1)
def _load_language_switch_templates() -> dict[str, dict[str, Any]]:
    """Load and validate language-switch templates from disk once per process."""
    logger.debug(
        "Loading language-switch templates from %s", _DEFAULT_LANGUAGE_SWITCH_TEMPLATES_PATH
    )
    with _DEFAULT_LANGUAGE_SWITCH_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("language_switch_templates.json must contain a top-level object")

    section_ids = set(data)
    if section_ids != _REQUIRED_SECTION_SET:
        missing = sorted(_REQUIRED_SECTION_SET - section_ids)
        extra = sorted(section_ids - _REQUIRED_SECTION_SET)
        raise ValueError(
            "language_switch_templates.json sections must match the expected schema "
            f"(missing={missing}, extra={extra})"
        )

    validated: dict[str, dict[str, Any]] = {}
    for section_id, raw_section in data.items():
        if not isinstance(raw_section, dict):
            raise ValueError(f"language-switch section {section_id!r} must be an object")
        if section_id == _FP_GUARDS_SECTION:
            section = _validate_fp_guard_section(section_id, raw_section)
        else:
            section = _validate_switch_section(section_id, raw_section)
        validated[section_id] = section

    if validated[_KO_TO_EN_SECTION]["current_language"] != "ko":
        raise ValueError("ko_to_en.current_language must be 'ko'")
    if validated[_KO_TO_EN_SECTION]["target_language"] != "en":
        raise ValueError("ko_to_en.target_language must be 'en'")
    if validated[_EN_TO_KO_SECTION]["current_language"] != "en":
        raise ValueError("en_to_ko.current_language must be 'en'")
    if validated[_EN_TO_KO_SECTION]["target_language"] != "ko":
        raise ValueError("en_to_ko.target_language must be 'ko'")
    return validated


def _validate_switch_section(section_id: str, raw_section: dict[str, Any]) -> dict[str, Any]:
    """Validate one directional switch section."""
    _validate_field_set(section_id, set(raw_section), _SWITCH_SECTION_REQUIRED_FIELDS)
    current_language = _coerce_language(
        section_id, "current_language", raw_section["current_language"]
    )
    target_language = _coerce_language(
        section_id, "target_language", raw_section["target_language"]
    )
    confirmation_language = _coerce_language(
        section_id,
        "confirmation_language",
        raw_section["confirmation_language"],
    )
    if current_language == target_language:
        raise ValueError(f"language-switch section {section_id!r} must change language")
    confirmation_text = raw_section["confirmation_text"]
    if not isinstance(confirmation_text, str) or not confirmation_text.strip():
        raise ValueError(f"language-switch section {section_id!r} confirmation_text must be text")
    patterns = _coerce_pattern_list(section_id, "patterns", raw_section["patterns"])
    return {
        "current_language": current_language,
        "target_language": target_language,
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
        raise ValueError(f"language-switch section {section_id!r} missing fields: {missing_fields}")
    extra_fields = sorted(actual_fields - required_fields)
    if extra_fields:
        raise ValueError(
            f"language-switch section {section_id!r} has unknown fields: {extra_fields}"
        )


def _coerce_language(section_id: str, field: str, value: Any) -> LanguageCode:
    """Validate one language code."""
    if value == "ko":
        return "ko"
    if value == "en":
        return "en"
    raise ValueError(f"language-switch section {section_id!r} {field} must be 'ko' or 'en'")


def _coerce_pattern_list(section_id: str, field: str, value: Any) -> list[str]:
    """Validate one anchored regex pattern list."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"language-switch section {section_id!r} {field} must be a list[str]")
    if not value:
        raise ValueError(f"language-switch section {section_id!r} {field} must not be empty")
    for pattern in value:
        if not pattern.startswith(r"\A") or not pattern.endswith(r"\Z"):
            raise ValueError(
                f"language-switch section {section_id!r} pattern must be whole-turn anchored: "
                f"{pattern!r}"
            )
        _compile_pattern(section_id, field, pattern)
    return value


def _normalize_language(language: str) -> LanguageCode:
    """Normalize a caller language into the supported session language set."""
    return "en" if language.lower() == "en" else "ko"


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
    """Compile one regex pattern (case-insensitive) with a useful validation error."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        location = f" in {section_id}.{field}" if section_id and field else ""
        raise ValueError(f"invalid language-switch regex{location}: {pattern!r}") from exc


__all__ = ["LanguageSwitchMatch", "get_switch_confirmation", "match_language_switch"]
