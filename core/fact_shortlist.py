"""Deterministic confirmable-fact shortlist matching for prompt grounding."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

logger = logging.getLogger(__name__)

AgeBand = Literal["under_10", "under_15"]
_ALLOWED_AGE_BANDS: Final[tuple[AgeBand, ...]] = (
    "under_10",
    "under_15",
)
_AGE_BAND_PRIORITY: Final[dict[AgeBand, int]] = {
    "under_10": 0,
    "under_15": 1,
}
_MAX_AGE_BAND_ENV: Final[str] = "MUNGI_FACT_SHORTLIST_MAX_BAND"
_DEFAULT_MAX_AGE_BAND: Final[AgeBand] = "under_10"

ConfidenceSource = Literal["pm_audited", "wikidata_verified", "keep_pool_seeded"]
_ALLOWED_CONFIDENCE_SOURCES: Final[tuple[ConfidenceSource, ...]] = (
    "pm_audited",
    "wikidata_verified",
    "keep_pool_seeded",
)

_ALLOWED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "animal",
        "plant",
        "science",
        "culture",
        "music",
        "sports",
        "vehicle",
        "weather",
        "body_health",
        "math",
        "nature",
        "story",
        "world_geography",
        "world_history_light",
        "technology_intro",
        "science_intro_deeper",
        "arts_appreciation_intro",
    }
)
_DATA_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "assets" / "prompts" / "confirmable_facts.json"
)


@dataclass(frozen=True)
class FactEntry:
    """One curated confirmable-fact shortlist entry."""

    topic: str
    category: str
    triggers_ko: tuple[str, ...]
    triggers_en: tuple[str, ...]
    fact_ko: str
    fact_en: str | None
    source_pm: str
    numeric_tolerance: int | None
    age_band: AgeBand
    confidence_source: ConfidenceSource


@dataclass(frozen=True)
class FactMatch:
    """A matched shortlist fact ready for prompt injection."""

    topic: str
    fact_ko: str
    fact_en: str | None
    matched_trigger: str


def get_fact_entries() -> tuple[FactEntry, ...]:
    """Return the validated shortlist entries loaded at module import time."""

    return _FACT_ENTRIES


def match_fact(turn_text: str, lang: Literal["ko", "en"]) -> FactMatch | None:
    """Return the best shortlist match for one user turn.

    Matching is a conservative, case-insensitive substring check over
    whitespace-normalized text. When multiple entries match, the longest
    matched trigger wins; ties then prefer the lower-age-band tier in the
    order ``under_10 > under_15``; remaining ties break by lexicographic
    ``topic`` order.
    """

    if lang not in {"ko", "en"}:
        msg = f"Unsupported shortlist language: {lang}"
        raise ValueError(msg)

    normalized_turn = _normalize_text(turn_text)
    if not normalized_turn:
        logger.debug("fact_shortlist_no_match reason=empty_turn lang=%s", lang)
        return None

    best_match: tuple[int, int, str, FactEntry, str] | None = None
    for entry in _FACT_ENTRIES:
        triggers = entry.triggers_ko if lang == "ko" else entry.triggers_en
        for trigger in triggers:
            normalized_trigger = _normalize_text(trigger)
            if normalized_trigger and normalized_trigger in normalized_turn:
                candidate = (
                    len(normalized_trigger),
                    _AGE_BAND_PRIORITY[entry.age_band],
                    entry.topic,
                    entry,
                    trigger,
                )
                if (
                    best_match is None
                    or candidate[0] > best_match[0]
                    or (candidate[0] == best_match[0] and candidate[1] < best_match[1])
                    or (
                        candidate[0] == best_match[0]
                        and candidate[1] == best_match[1]
                        and candidate[2] < best_match[2]
                    )
                ):
                    best_match = candidate

    if best_match is None:
        logger.debug("fact_shortlist_no_match lang=%s turn_text=%s", lang, turn_text)
        return None

    _, _, _, entry, matched_trigger = best_match
    logger.info(
        "fact_shortlist_match topic=%s trigger=%s lang=%s",
        entry.topic,
        matched_trigger,
        lang,
        extra={
            "event": "fact_shortlist_match",
            "topic": entry.topic,
            "matched_trigger": matched_trigger,
            "lang": lang,
        },
    )
    return FactMatch(
        topic=entry.topic,
        fact_ko=entry.fact_ko,
        fact_en=entry.fact_en,
        matched_trigger=matched_trigger,
    )


def _load_fact_entries(path: Path) -> tuple[FactEntry, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _parse_fact_entries(payload)


def _load_max_age_band() -> AgeBand:
    value = os.environ.get(_MAX_AGE_BAND_ENV, _DEFAULT_MAX_AGE_BAND).strip().casefold()
    if value not in _ALLOWED_AGE_BANDS:
        allowed = ", ".join(_ALLOWED_AGE_BANDS)
        msg = f"Unsupported {_MAX_AGE_BAND_ENV} value: {value}. Expected one of: {allowed}"
        raise ValueError(msg)
    return cast(AgeBand, value)


def _filter_fact_entries(
    entries: tuple[FactEntry, ...], max_age_band: AgeBand
) -> tuple[FactEntry, ...]:
    max_rank = _AGE_BAND_PRIORITY[max_age_band]
    return tuple(entry for entry in entries if _AGE_BAND_PRIORITY[entry.age_band] <= max_rank)


def _parse_fact_entries(payload: object) -> tuple[FactEntry, ...]:
    if not isinstance(payload, list):
        msg = "confirmable_facts.json root must be a JSON array"
        raise ValueError(msg)

    entries: list[FactEntry] = []
    seen_topics: set[str] = set()
    for index, raw_entry in enumerate(payload, start=1):
        if not isinstance(raw_entry, dict):
            msg = f"Entry {index} must be a JSON object"
            raise ValueError(msg)

        topic = _require_non_empty_string(raw_entry, "topic", index=index)
        if topic in seen_topics:
            msg = f"Duplicate shortlist topic: {topic}"
            raise ValueError(msg)
        seen_topics.add(topic)

        category = _require_non_empty_string(raw_entry, "category", index=index)
        if category not in _ALLOWED_CATEGORIES:
            msg = f"Entry {index} has unsupported category: {category}"
            raise ValueError(msg)

        triggers_ko = _require_trigger_list(
            raw_entry, "triggers_ko", index=index, allow_empty=False
        )
        triggers_en = _require_trigger_list(raw_entry, "triggers_en", index=index, allow_empty=True)
        fact_ko = _require_non_empty_string(raw_entry, "fact_ko", index=index)
        fact_en = _optional_string(raw_entry.get("fact_en"), field_name="fact_en", index=index)
        source_pm = _require_non_empty_string(raw_entry, "source_pm", index=index)
        numeric_tolerance = raw_entry.get("numeric_tolerance")
        if numeric_tolerance is not None and not isinstance(numeric_tolerance, int):
            msg = f"Entry {index} field numeric_tolerance must be an int or null"
            raise ValueError(msg)
        age_band = _optional_age_band(raw_entry.get("age_band"), index=index)
        confidence_source = _optional_confidence_source(
            raw_entry.get("confidence_source"),
            index=index,
        )

        entries.append(
            FactEntry(
                topic=topic,
                category=category,
                triggers_ko=tuple(triggers_ko),
                triggers_en=tuple(triggers_en),
                fact_ko=fact_ko,
                fact_en=fact_en,
                source_pm=source_pm,
                numeric_tolerance=cast(int | None, numeric_tolerance),
                age_band=age_band,
                confidence_source=confidence_source,
            )
        )

    return tuple(entries)


def _require_non_empty_string(payload: dict[str, Any], field_name: str, *, index: int) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        msg = f"Entry {index} field {field_name} must be a non-empty string"
        raise ValueError(msg)
    return value.strip()


def _optional_string(value: object, *, field_name: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"Entry {index} field {field_name} must be a non-empty string or null"
        raise ValueError(msg)
    return value.strip()


def _optional_age_band(value: object, *, index: int) -> AgeBand:
    if value is None:
        return "under_10"
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _ALLOWED_AGE_BANDS:
            return cast(AgeBand, normalized)
    allowed = ", ".join(_ALLOWED_AGE_BANDS)
    msg = f"Entry {index} field age_band must be one of: {allowed}"
    raise ValueError(msg)


def _optional_confidence_source(value: object, *, index: int) -> ConfidenceSource:
    if value is None:
        return "pm_audited"
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _ALLOWED_CONFIDENCE_SOURCES:
            return cast(ConfidenceSource, normalized)
    allowed = ", ".join(_ALLOWED_CONFIDENCE_SOURCES)
    msg = f"Entry {index} field confidence_source must be one of: {allowed}"
    raise ValueError(msg)


def _require_trigger_list(
    payload: dict[str, Any],
    field_name: str,
    *,
    index: int,
    allow_empty: bool,
) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"Entry {index} field {field_name} must be a string list"
        raise ValueError(msg)
    cleaned = [_normalize_text(item) for item in value if item.strip()]
    if not allow_empty and not cleaned:
        msg = f"Entry {index} field {field_name} must not be empty"
        raise ValueError(msg)
    if len(set(cleaned)) != len(cleaned):
        msg = f"Entry {index} field {field_name} contains duplicate triggers"
        raise ValueError(msg)
    return cleaned


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


_FACT_ENTRIES: Final[tuple[FactEntry, ...]] = _filter_fact_entries(
    _load_fact_entries(_DATA_PATH),
    _load_max_age_band(),
)
