"""Deterministic crisis-disclosure routing for child-safety escalation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Final

# Sanctioned safety -> core.safety_rules leaf import: shared constants only, no cycle.
from core.safety_rules import (
    CRISIS_DISTRESS_CATEGORIES,
    CRISIS_RESPONSE_EN,
    CRISIS_RESPONSE_KO,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_CRISIS_TEMPLATES_PATH: Final[Path] = (
    _PROJECT_ROOT / "assets" / "filters" / "crisis_templates.json"
)
_PATTERN_FIELDS: Final[tuple[str, ...]] = (
    "disclosure_patterns_ko",
    "disclosure_patterns_en",
    "request_excludes_ko",
    "request_excludes_en",
)
# Optional additive 2-tier fields (ADR 0114). Missing/empty values behave
# exactly like the legacy single-tier code, so untouched topics are unchanged.
#   - fiction_excludes_*: fiction/request-frame phrases that suppress a topic
#     EVEN when a hard disclosure is present (fiction is fiction).
#   - hard_disclosure_patterns_*: unambiguous first-person crisis disclosures
#     that ALWAYS escalate unless a fiction-exclude is present, bypassing the
#     plain request_excludes (which were nullifying genuine disclosures that
#     incidentally contained a common noun — the fail-open bug, ADR 0114).
_OPTIONAL_PATTERN_FIELDS: Final[tuple[str, ...]] = (
    "fiction_excludes_ko",
    "fiction_excludes_en",
    "hard_disclosure_patterns_ko",
    "hard_disclosure_patterns_en",
)
_ALL_PATTERN_FIELDS: Final[tuple[str, ...]] = (
    *_PATTERN_FIELDS,
    *_OPTIONAL_PATTERN_FIELDS,
)
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    *_PATTERN_FIELDS,
    "priority",
    "response_ko",
    "response_en",
    "escalation_target",
)
_REQUIRED_FIELD_SET: Final[frozenset[str]] = frozenset(_REQUIRED_FIELDS)
_ALLOWED_FIELD_SET: Final[frozenset[str]] = frozenset(
    (*_REQUIRED_FIELDS, *_OPTIONAL_PATTERN_FIELDS),
)
_ESCALATION_TARGET_BY_TOPIC: Final[dict[str, str]] = {
    "self_harm": "parent",
    "suicidal_intent": "parent",
    "neglect": "parent",
    "abuse_physical": "trusted_adult_not_abuser",
    "abuse_sexual": "trusted_adult_not_abuser",
    "domestic_violence": "trusted_adult_not_abuser",
    "bullying": "parent_teacher",
    "threat_intimidation": "parent_teacher",
    "runaway": "parent",
    "grooming": "parent",
    "drug_solicitation": "parent",
    "missing_lost": "112_stay_put",
    "fire_emergency": "119",
}


@dataclass(frozen=True)
class CrisisMatch:
    """Result of a successful crisis-disclosure match."""

    topic_id: str
    response: str
    response_language: str
    escalation_target: str
    priority: int
    matched_patterns: tuple[str, ...]


def match_crisis_disclosure(user_text: str, language: str) -> CrisisMatch | None:
    """Return a fixed crisis escalation match for first-person disclosures.

    The matcher checks both Korean and English disclosure/exclude fields on
    every turn, regardless of detected language. A topic matches only when at
    least one disclosure pattern hits and no request-exclude pattern hits.

    Args:
        user_text: The normalized turn text seen by the content filter/router.
        language: Detected turn language. Only exact ``"en"`` selects the
            English response; all other values select Korean.

    Returns:
        A :class:`CrisisMatch` when a crisis disclosure is matched, otherwise
        ``None``.
    """
    if not user_text or not user_text.strip():
        return None

    normalized_input = user_text.casefold()
    response_language = "en" if language.lower() == "en" else "ko"
    response_field = f"response_{response_language}"

    candidates: list[tuple[int, int, int, int, CrisisMatch]] = []
    for original_index, (topic_id, topic) in enumerate(_load_crisis_templates().items()):
        disclosure_hits = _select_topic_disclosure_hits(topic, normalized_input)
        if not disclosure_hits:
            continue

        matched_patterns = tuple(sorted({hit.pattern for hit in disclosure_hits}))
        priority = int(topic["priority"])
        longest_match = max(len(hit.pattern) for hit in disclosure_hits)
        distinct_match_count = len(matched_patterns)
        match = CrisisMatch(
            topic_id=topic_id,
            response=topic[response_field],
            response_language=response_language,
            escalation_target=topic["escalation_target"],
            priority=priority,
            matched_patterns=matched_patterns,
        )
        candidates.append(
            (
                -priority,
                -longest_match,
                -distinct_match_count,
                original_index,
                match,
            ),
        )

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[:4])
    best_match = candidates[0][4]
    logger.info(
        "Crisis disclosure matched topic '%s' (target=%s)",
        best_match.topic_id,
        best_match.escalation_target,
    )
    return best_match


def _select_topic_disclosure_hits(
    topic: dict[str, Any],
    normalized_input: str,
) -> list[_RegexHit]:
    """Resolve the winning disclosure hits for one topic via the 2-tier gate.

    Implements the ADR 0114 fail-open fix. The tiers are evaluated in a fixed
    order; the first decisive tier determines the outcome:

    1. ``fiction_excludes_*`` hit -> fiction/request frame; skip the topic
       (returns no hits) EVEN if a hard disclosure is present.
    2. ``hard_disclosure_patterns_*`` hit -> unambiguous first-person crisis;
       MATCH on those hits, bypassing the plain ``request_excludes_*`` so that
       an incidental common noun can no longer nullify a genuine disclosure.
    3. ``request_excludes_*`` hit -> ordinary request frame; skip the topic.
    4. ``disclosure_patterns_*`` hit -> MATCH on those hits.

    Missing or empty optional fields collapse this to the legacy single-tier
    behaviour (request_excludes -> disclosure_patterns), so topics that do not
    define the new fields are unchanged.

    Args:
        topic: A validated crisis topic mapping.
        normalized_input: The already-casefolded turn text.

    Returns:
        The list of regex hits that should be scored for this topic, or an
        empty list when the topic does not match.
    """

    def field(name: str) -> list[str]:
        # Optional fields may be absent on hand-built test fixtures or topics
        # that predate ADR 0114; treat them as empty (legacy behaviour).
        value = topic.get(name, [])
        return value if isinstance(value, list) else []

    if _regex_hits(
        field("fiction_excludes_ko") + field("fiction_excludes_en"),
        normalized_input,
    ):
        return []

    hard_hits = _regex_hits(
        field("hard_disclosure_patterns_ko") + field("hard_disclosure_patterns_en"),
        normalized_input,
    )
    if hard_hits:
        return hard_hits

    if _regex_hits(
        field("request_excludes_ko") + field("request_excludes_en"),
        normalized_input,
    ):
        return []

    return _regex_hits(
        field("disclosure_patterns_ko") + field("disclosure_patterns_en"),
        normalized_input,
    )


@dataclass(frozen=True)
class _RegexHit:
    """Internal regex hit used for deterministic crisis scoring."""

    pattern: str
    matched_text: str


@lru_cache(maxsize=1)
def _load_crisis_templates() -> dict[str, dict[str, Any]]:
    """Load and validate crisis templates from disk once per process."""
    logger.debug("Loading crisis templates from %s", _DEFAULT_CRISIS_TEMPLATES_PATH)
    with _DEFAULT_CRISIS_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("crisis_templates.json must contain a top-level object")

    topic_ids = set(data)
    if topic_ids != CRISIS_DISTRESS_CATEGORIES:
        missing = sorted(CRISIS_DISTRESS_CATEGORIES - topic_ids)
        extra = sorted(topic_ids - CRISIS_DISTRESS_CATEGORIES)
        raise ValueError(
            "crisis_templates.json topic IDs must match CRISIS_DISTRESS_CATEGORIES "
            f"(missing={missing}, extra={extra})",
        )

    validated: dict[str, dict[str, Any]] = {}
    for topic_id, raw_topic in data.items():
        if not isinstance(raw_topic, dict):
            raise ValueError(f"crisis topic {topic_id!r} must be an object")
        missing_fields = [field for field in _REQUIRED_FIELDS if field not in raw_topic]
        if missing_fields:
            raise ValueError(f"crisis topic {topic_id!r} missing fields: {missing_fields}")
        extra_fields = sorted(set(raw_topic) - _ALLOWED_FIELD_SET)
        if extra_fields:
            raise ValueError(f"crisis topic {topic_id!r} has unknown fields: {extra_fields}")

        topic = dict(raw_topic)
        # Optional ADR 0114 fields default to empty so missing values behave
        # exactly like the legacy single-tier code.
        for field in _OPTIONAL_PATTERN_FIELDS:
            topic.setdefault(field, [])
        for field in _ALL_PATTERN_FIELDS:
            topic[field] = _coerce_pattern_list(topic_id, field, topic[field])
            for pattern in topic[field]:
                _compile_pattern(topic_id, field, pattern)

        priority = topic["priority"]
        if not isinstance(priority, int):
            raise ValueError(f"crisis topic {topic_id!r} priority must be an integer")
        if priority != 100:
            raise ValueError(f"crisis topic {topic_id!r} priority must be 100")

        for field in ("response_ko", "response_en", "escalation_target"):
            if not isinstance(topic[field], str) or not topic[field].strip():
                raise ValueError(f"crisis topic {topic_id!r} {field} must be a non-empty string")
        if topic["response_ko"] != CRISIS_RESPONSE_KO[topic_id]:
            raise ValueError(f"crisis topic {topic_id!r} response_ko must match safety_rules")
        if topic["response_en"] != CRISIS_RESPONSE_EN[topic_id]:
            raise ValueError(f"crisis topic {topic_id!r} response_en must match safety_rules")
        if topic["escalation_target"] != _ESCALATION_TARGET_BY_TOPIC[topic_id]:
            raise ValueError(
                f"crisis topic {topic_id!r} escalation_target must be "
                f"{_ESCALATION_TARGET_BY_TOPIC[topic_id]!r}",
            )

        validated[topic_id] = topic
    return validated


def _coerce_pattern_list(topic_id: str, field: str, value: Any) -> list[str]:
    """Validate one pattern field as a string list."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"crisis topic {topic_id!r} {field} must be a list[str]")
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
def _compile_pattern(topic_id: str, field: str, pattern: str) -> re.Pattern[str]:
    """Compile one casefolded regex pattern with a useful validation error."""
    try:
        return re.compile(pattern.casefold())
    except re.error as exc:
        location = f" in {topic_id}.{field}" if topic_id and field else ""
        raise ValueError(f"invalid crisis regex{location}: {pattern!r}") from exc


__all__ = ["CrisisMatch", "match_crisis_disclosure"]
