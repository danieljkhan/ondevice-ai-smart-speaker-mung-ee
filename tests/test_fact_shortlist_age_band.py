"""Tests for fact shortlist age-band precedence and filtering."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

import core.fact_shortlist as fact_shortlist


def _build_entry(*, topic: str, age_band: str | None) -> dict[str, Any]:
    return {
        "topic": topic,
        "category": "body_health",
        "triggers_ko": ["같은 트리거"],
        "triggers_en": [],
        "fact_ko": f"{topic} fact",
        "fact_en": None,
        "source_pm": "test",
        "numeric_tolerance": 0,
        **({"age_band": age_band} if age_band is not None else {}),
    }


def _reload_fact_shortlist(monkeypatch: pytest.MonkeyPatch, value: str | None) -> Any:
    if value is None:
        monkeypatch.delenv("MUNGI_FACT_SHORTLIST_MAX_BAND", raising=False)
    else:
        monkeypatch.setenv("MUNGI_FACT_SHORTLIST_MAX_BAND", value)
    return importlib.reload(fact_shortlist)


def test_parse_fact_entries_defaults_missing_age_band_to_under_10() -> None:
    entries = fact_shortlist._parse_fact_entries(
        [_build_entry(topic="default_band", age_band=None)]
    )

    assert entries[0].age_band == "under_10"


def test_match_fact_prefers_lower_age_band_when_trigger_lengths_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = fact_shortlist._parse_fact_entries(
        [
            _build_entry(topic="under_15_topic", age_band="under_15"),
            _build_entry(topic="under_10_topic", age_band="under_10"),
        ]
    )
    monkeypatch.setattr(fact_shortlist, "_FACT_ENTRIES", entries)

    match = fact_shortlist.match_fact("같은 트리거 알려줘", "ko")

    assert match is not None
    assert match.topic == "under_10_topic"


def test_match_fact_falls_back_to_lexicographic_topic_after_age_band_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = fact_shortlist._parse_fact_entries(
        [
            _build_entry(topic="zeta_topic", age_band="under_15"),
            _build_entry(topic="alpha_topic", age_band="under_15"),
        ]
    )
    monkeypatch.setattr(fact_shortlist, "_FACT_ENTRIES", entries)

    match = fact_shortlist.match_fact("같은 트리거 맞지?", "ko")

    assert match is not None
    assert match.topic == "alpha_topic"


def test_match_fact_prefers_under_10_sibling_for_shared_trigger_under_default_max_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_fact_shortlist(monkeypatch, None)
    entries = module._parse_fact_entries(
        [
            _build_entry(topic="planet_u10", age_band="under_10"),
            _build_entry(topic="planet_u15", age_band="under_15"),
        ]
    )

    filtered_entries = module._filter_fact_entries(entries, module._load_max_age_band())
    monkeypatch.setattr(module, "_FACT_ENTRIES", filtered_entries)

    match = module.match_fact("같은 트리거 알려줘", "ko")

    assert [entry.topic for entry in filtered_entries] == ["planet_u10"]
    assert match is not None
    assert match.topic == "planet_u10"


def test_max_age_band_defaults_to_under_10_and_preserves_phase0_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_fact_shortlist(monkeypatch, None)

    entries = module.get_fact_entries()

    assert len(entries) == 412
    assert {entry.age_band for entry in entries} == {"under_10"}


def test_max_age_band_filters_out_higher_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_fact_shortlist(monkeypatch, "under_10")

    sample_entries = module._parse_fact_entries(
        [
            _build_entry(topic="under_10_topic", age_band="under_10"),
            _build_entry(topic="under_15_topic", age_band="under_15"),
        ]
    )

    filtered = module._filter_fact_entries(sample_entries, module._load_max_age_band())

    assert [entry.topic for entry in filtered] == ["under_10_topic"]


def test_max_age_band_under_15_keeps_both_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_fact_shortlist(monkeypatch, "under_15")

    sample_entries = module._parse_fact_entries(
        [
            _build_entry(topic="under_10_topic", age_band="under_10"),
            _build_entry(topic="under_15_topic", age_band="under_15"),
        ]
    )

    filtered = module._filter_fact_entries(sample_entries, module._load_max_age_band())

    assert [entry.topic for entry in filtered] == ["under_10_topic", "under_15_topic"]


def test_max_age_band_under_15_loads_full_repo_shortlist(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_fact_shortlist(monkeypatch, "under_15")

    entries = module.get_fact_entries()

    assert len(entries) == 522
    assert {entry.age_band for entry in entries} == {"under_10", "under_15"}


def test_default_max_band_excludes_under_15_only_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_fact_shortlist(monkeypatch, None)

    assert module.match_fact("척추동물 몇 무리?", "ko") is None


def test_under_15_band_matches_under_15_repo_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_fact_shortlist(monkeypatch, "under_15")

    match = module.match_fact("척추동물 몇 무리?", "ko")

    assert match is not None
    assert match.topic == "vertebrate_classes"


def test_under_15_band_keeps_unmatched_holdout_question_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_fact_shortlist(monkeypatch, "under_15")

    assert module.match_fact("멕시코의 수도는 어디야?", "ko") is None


def test_invalid_max_age_band_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNGI_FACT_SHORTLIST_MAX_BAND", "college")

    with pytest.raises(ValueError, match="Unsupported MUNGI_FACT_SHORTLIST_MAX_BAND value"):
        importlib.reload(fact_shortlist)

    monkeypatch.delenv("MUNGI_FACT_SHORTLIST_MAX_BAND", raising=False)
    importlib.reload(fact_shortlist)
