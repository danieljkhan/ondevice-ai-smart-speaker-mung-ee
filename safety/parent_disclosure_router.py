"""Deterministic parent-disclosure and belief-probe routing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Final, Literal

# Sanctioned safety -> core.safety_rules leaf import: shared constants only, no cycle.
from core.safety_rules import (
    BELIEF_RESPONSE_EN,
    BELIEF_RESPONSE_KO,
    PARENT_DISCLOSURE_EN_BLOCKER_PATTERNS,
    PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
    PARENT_DISCLOSURE_EN_PROHIBITED_PREFIXES,
    PARENT_DISCLOSURE_KO_BLOCKER_PATTERNS,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES,
)

logger = logging.getLogger(__name__)

ParentDisclosureKind = Literal["probe", "friendship"]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_PARENT_DISCLOSURE_TEMPLATES_PATH: Final[Path] = (
    _PROJECT_ROOT / "assets" / "filters" / "parent_disclosure_templates.json"
)
_PARENT_DISCLOSURE_SECTION: Final[str] = "parent_disclosure"
_FRIENDSHIP_SECTION: Final[str] = "friendship"
_BELIEF_SECTION: Final[str] = "belief"
_FP_GUARDS_SECTION: Final[str] = "fp_guards"
_VALIDATOR_SECTION: Final[str] = "validator"
_REQUIRED_SECTION_SET: Final[frozenset[str]] = frozenset(
    {
        _PARENT_DISCLOSURE_SECTION,
        _FRIENDSHIP_SECTION,
        _BELIEF_SECTION,
        _FP_GUARDS_SECTION,
        _VALIDATOR_SECTION,
    }
)
_SECTION_REQUIRED_FIELDS: Final[dict[str, frozenset[str]]] = {
    _PARENT_DISCLOSURE_SECTION: frozenset(
        {
            "probe_patterns_ko",
            "probe_patterns_en",
            "friendship_patterns_ko",
            "friendship_patterns_en",
            "response_ko",
            "response_en",
        }
    ),
    _FRIENDSHIP_SECTION: frozenset({"response_ko", "response_en"}),
    _BELIEF_SECTION: frozenset({"patterns_ko", "patterns_en", "response_ko", "response_en"}),
    _FP_GUARDS_SECTION: frozenset({"patterns_ko", "patterns_en"}),
    _VALIDATOR_SECTION: frozenset(
        {
            "output_blocker_patterns_ko",
            "output_blocker_patterns_en",
            "secret_context_pattern",
        }
    ),
}
_PATTERN_FIELDS_BY_SECTION: Final[dict[str, tuple[str, ...]]] = {
    _PARENT_DISCLOSURE_SECTION: (
        "probe_patterns_ko",
        "probe_patterns_en",
        "friendship_patterns_ko",
        "friendship_patterns_en",
    ),
    _BELIEF_SECTION: ("patterns_ko", "patterns_en"),
    _FP_GUARDS_SECTION: ("patterns_ko", "patterns_en"),
    _VALIDATOR_SECTION: ("output_blocker_patterns_ko", "output_blocker_patterns_en"),
}


@dataclass(frozen=True)
class ParentDisclosureMatch:
    """Result of a deterministic parent-disclosure secret-pact match."""

    kind: ParentDisclosureKind
    response: str
    response_language: str
    matched_patterns: tuple[str, ...]


def match_parent_disclosure(user_text: str, language: str) -> ParentDisclosureMatch | None:
    """Return a fixed parent-disclosure match for secret-pact probes.

    The matcher checks Korean and English patterns on every turn, regardless of
    detected language. Explicit parent non-disclosure probes are checked before
    the FP-guarded friendship tier, matching the validated prototype ordering.
    """
    if not user_text or not user_text.strip():
        return None

    normalized_input = user_text.casefold()
    response_language = _response_language(language)
    templates = _load_parent_disclosure_templates()
    parent_disclosure = templates[_PARENT_DISCLOSURE_SECTION]

    probe_hits = _regex_hits(
        parent_disclosure["probe_patterns_ko"] + parent_disclosure["probe_patterns_en"],
        normalized_input,
    )
    if probe_hits:
        return _build_match(
            "probe",
            response_language,
            parent_disclosure[f"response_{response_language}"],
            probe_hits,
        )

    fp_guards = templates[_FP_GUARDS_SECTION]
    fp_hits = _regex_hits(fp_guards["patterns_ko"] + fp_guards["patterns_en"], normalized_input)
    if fp_hits:
        return None

    friendship_hits = _regex_hits(
        parent_disclosure["friendship_patterns_ko"] + parent_disclosure["friendship_patterns_en"],
        normalized_input,
    )
    if friendship_hits:
        friendship = templates[_FRIENDSHIP_SECTION]
        return _build_match(
            "friendship",
            response_language,
            friendship[f"response_{response_language}"],
            friendship_hits,
        )

    return None


def match_belief_probe(user_text: str, language: str) -> str | None:
    """Return the fixed belief response for narrow direct belief probes.

    Secret-pact probes take precedence over belief probes, so mixed turns such
    as "Is Santa real? Don't tell my mom" return ``None`` here and route through
    :func:`match_parent_disclosure`.
    """
    if match_parent_disclosure(user_text, language) is not None:
        return None
    if not user_text or not user_text.strip():
        return None

    normalized_input = user_text.casefold()
    response_language = _response_language(language)
    belief = _load_parent_disclosure_templates()[_BELIEF_SECTION]
    belief_hits = _regex_hits(belief["patterns_ko"] + belief["patterns_en"], normalized_input)
    if not belief_hits:
        return None

    return str(belief[f"response_{response_language}"])


def validate_parent_disclosure_output(response: str, language: str) -> tuple[bool, str]:
    """Validate one LLM response against the parent-disclosure secrecy invariant.

    Returns ``(False, fixed_probe_response)`` when the response promises secrecy
    or starts with a prohibited yes/no-style prefix in a secret-pact context.
    """
    response_language = _response_language(language)
    probe_response = _load_parent_disclosure_templates()[_PARENT_DISCLOSURE_SECTION][
        f"response_{response_language}"
    ]
    normalized_response = response.casefold()
    validator = _load_parent_disclosure_templates()[_VALIDATOR_SECTION]

    blocker_hits = _regex_hits(
        validator["output_blocker_patterns_ko"] + validator["output_blocker_patterns_en"],
        normalized_response,
    )
    if blocker_hits:
        return (False, probe_response)

    prefixes = (
        PARENT_DISCLOSURE_EN_PROHIBITED_PREFIXES
        if response_language == "en"
        else PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES
    )
    stripped = normalized_response.lstrip(" \t\n\"'…·-—~")
    secret_context = _compile_pattern(
        _VALIDATOR_SECTION,
        "secret_context_pattern",
        validator["secret_context_pattern"],
    )
    if any(stripped.startswith(prefix.casefold()) for prefix in prefixes):
        if secret_context.search(normalized_response):
            return (False, probe_response)

    return (True, response)


@dataclass(frozen=True)
class _RegexHit:
    """Internal regex hit used for deterministic parent-disclosure matching."""

    pattern: str
    matched_text: str


def _build_match(
    kind: ParentDisclosureKind,
    response_language: str,
    response: str,
    hits: list[_RegexHit],
) -> ParentDisclosureMatch:
    """Build a public match object from regex hits."""
    matched_patterns = tuple(sorted({hit.pattern for hit in hits}))
    logger.info("Parent disclosure matched kind '%s'", kind)
    return ParentDisclosureMatch(
        kind=kind,
        response=response,
        response_language=response_language,
        matched_patterns=matched_patterns,
    )


def _response_language(language: str) -> str:
    """Normalize a detected language into the supported response language set."""
    return "en" if language.lower() == "en" else "ko"


@lru_cache(maxsize=1)
def _load_parent_disclosure_templates() -> dict[str, dict[str, Any]]:
    """Load and validate parent-disclosure templates from disk once per process."""
    logger.debug(
        "Loading parent-disclosure templates from %s",
        _DEFAULT_PARENT_DISCLOSURE_TEMPLATES_PATH,
    )
    with _DEFAULT_PARENT_DISCLOSURE_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("parent_disclosure_templates.json must contain a top-level object")

    section_ids = set(data)
    if section_ids != _REQUIRED_SECTION_SET:
        missing = sorted(_REQUIRED_SECTION_SET - section_ids)
        extra = sorted(section_ids - _REQUIRED_SECTION_SET)
        raise ValueError(
            "parent_disclosure_templates.json sections must match the expected schema "
            f"(missing={missing}, extra={extra})",
        )

    validated: dict[str, dict[str, Any]] = {}
    for section_id, raw_section in data.items():
        if not isinstance(raw_section, dict):
            raise ValueError(f"parent-disclosure section {section_id!r} must be an object")

        required_fields = _SECTION_REQUIRED_FIELDS[section_id]
        missing_fields = sorted(required_fields - set(raw_section))
        if missing_fields:
            raise ValueError(
                f"parent-disclosure section {section_id!r} missing fields: {missing_fields}"
            )
        extra_fields = sorted(set(raw_section) - required_fields)
        if extra_fields:
            raise ValueError(
                f"parent-disclosure section {section_id!r} has unknown fields: {extra_fields}"
            )

        section = dict(raw_section)
        for field in _PATTERN_FIELDS_BY_SECTION.get(section_id, ()):
            section[field] = _coerce_pattern_list(section_id, field, section[field])
            for pattern in section[field]:
                _compile_pattern(section_id, field, pattern)
        if "secret_context_pattern" in section:
            if not isinstance(section["secret_context_pattern"], str):
                raise ValueError("validator secret_context_pattern must be a string")
            _compile_pattern(
                section_id,
                "secret_context_pattern",
                section["secret_context_pattern"],
            )

        validated[section_id] = section

    _validate_responses(validated)
    _validate_blocker_patterns(validated)
    return validated


def _validate_responses(templates: dict[str, dict[str, Any]]) -> None:
    """Validate template response strings against core safety-rule constants."""
    parent_disclosure = templates[_PARENT_DISCLOSURE_SECTION]
    friendship = templates[_FRIENDSHIP_SECTION]
    belief = templates[_BELIEF_SECTION]
    expected_pairs = (
        (
            parent_disclosure["response_ko"],
            PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
            "parent_disclosure.response_ko",
        ),
        (
            parent_disclosure["response_en"],
            PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
            "parent_disclosure.response_en",
        ),
        (
            friendship["response_ko"],
            PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
            "friendship.response_ko",
        ),
        (
            friendship["response_en"],
            PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
            "friendship.response_en",
        ),
        (belief["response_ko"], BELIEF_RESPONSE_KO, "belief.response_ko"),
        (belief["response_en"], BELIEF_RESPONSE_EN, "belief.response_en"),
    )
    for actual, expected, field in expected_pairs:
        if actual != expected:
            raise ValueError(f"parent_disclosure_templates.json {field} must match safety_rules")


def _validate_blocker_patterns(templates: dict[str, dict[str, Any]]) -> None:
    """Validate production output-blocker patterns against safety-rule constants."""
    validator = templates[_VALIDATOR_SECTION]
    expected_pairs = (
        (
            tuple(validator["output_blocker_patterns_ko"]),
            PARENT_DISCLOSURE_KO_BLOCKER_PATTERNS,
            "validator.output_blocker_patterns_ko",
        ),
        (
            tuple(validator["output_blocker_patterns_en"]),
            PARENT_DISCLOSURE_EN_BLOCKER_PATTERNS,
            "validator.output_blocker_patterns_en",
        ),
    )
    for actual, expected, field in expected_pairs:
        if actual != expected:
            raise ValueError(f"parent_disclosure_templates.json {field} must match safety_rules")


def _coerce_pattern_list(section_id: str, field: str, value: Any) -> list[str]:
    """Validate one pattern field as a string list."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"parent-disclosure section {section_id!r} {field} must be a list[str]")
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
    """Compile one casefolded regex pattern with a useful validation error."""
    try:
        return re.compile(pattern.casefold())
    except re.error as exc:
        location = f" in {section_id}.{field}" if section_id and field else ""
        raise ValueError(f"invalid parent-disclosure regex{location}: {pattern!r}") from exc


__all__ = [
    "ParentDisclosureMatch",
    "match_belief_probe",
    "match_parent_disclosure",
    "validate_parent_disclosure_output",
]
