"""Regression suite for Session 19 Phase 2a router algorithm redesign (2026-04-28).

Plan: Dev_Plan/2026-04-28-safety-filter-phase2a-router-redesign-plan.md (v3).

Phase 2a switched the safety router from substring + first-match-wins to
substring + 4-tier tiebreak (priority DESC -> longest matched keyword DESC ->
distinct match count DESC -> JSON order ASC) and added an optional `priority`
field on the JSON template inventory. This test file covers four groups:

- Group A: Phase 1 PR #69 false-positive cases must remain resolved.
- Group B: Phase 2a longest-match wins on mixed-context utterances.
- Group C: Patched-fixture tie tests - JSON-order tiebreak + priority precedence.
- Group D: Single-match invariants (no regression vs pre-PR-#69 behavior).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety.approved_template_router import (
    _load_approved_templates,
    check_approved_template,
)

TemplateInventory = dict[str, dict[str, object]]

PHASE1_FALSE_POSITIVES = [
    ("뭉이야, 고슴도치 가시에 찔리면 주사 맞는 것처럼 아파?", "ko", "sharp_objects"),
    ("물에 얼굴 넣는 게 무서워...", "ko", "plastic_bags"),
    ("빨리 신고 나가고 싶어!", "ko", "emergency_119"),
    ("He always touches my stuff!", "en", "body_autonomy"),
]

PHASE2A_LONGEST_MATCH_WINS = [
    ("수영장에서 친구가 비닐봉지 머리에 썼어", "ko", "plastic_bags"),
    ("a plastic bag went over my head in the bath", "en", "plastic_bags"),
    ("욕조에 비닐봉지 떨어뜨렸어", "ko", "plastic_bags"),
]

SINGLE_MATCH_INVARIANTS = [
    ("손 씻을 때 비누 없이 해도 돼?", "ko", "hand_washing"),
    ("Can I wash hands without soap?", "en", "hand_washing"),
    ("What is your favorite color?", "en", None),
    ("뭉이야 너 누구야?", "ko", "mungi_self_intro_child"),
]


@pytest.mark.parametrize("user_input,language,wrong_template", PHASE1_FALSE_POSITIVES)
def test_phase1_false_positives_still_resolved(
    user_input: str,
    language: str,
    wrong_template: str,
) -> None:
    """Phase 1 false-positive utterances must not route to the old wrong topic."""
    result = check_approved_template(user_input, language)
    if result is not None:
        assert result["topic_id"] != wrong_template, (
            f"Input {user_input!r} still routes to wrong template {wrong_template!r} "
            f"under Phase 2a algorithm. Got topic_id={result['topic_id']!r}."
        )


@pytest.mark.parametrize("user_input,language,expected_topic", PHASE2A_LONGEST_MATCH_WINS)
def test_phase2a_longest_match_resolves_mixed_context(
    user_input: str,
    language: str,
    expected_topic: str,
) -> None:
    """Longest-match scoring should select the more specific template."""
    result = check_approved_template(user_input, language)
    assert result is not None, f"Input {user_input!r} should match {expected_topic!r} but got None."
    assert result["topic_id"] == expected_topic, (
        f"Input {user_input!r} should match {expected_topic!r} (longest-match wins) "
        f"but got {result['topic_id']!r}."
    )


@pytest.fixture
def two_template_tie_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TemplateInventory]:
    """Provide equal-priority, equal-length synthetic templates for JSON-order tests."""
    fake_templates: TemplateInventory = {
        "topic_a": {
            "keywords_ko": ["테스트키워드"],
            "response_ko": "A response",
            "mode": "block",
        },
        "topic_b": {
            "keywords_ko": ["테스트키워드"],
            "response_ko": "B response",
            "mode": "block",
        },
    }
    _load_approved_templates.cache_clear()
    monkeypatch.setattr(
        "safety.approved_template_router._load_approved_templates",
        lambda: fake_templates,
    )
    yield fake_templates
    _load_approved_templates.cache_clear()


def test_tie_breaks_to_earlier_json_position(
    two_template_tie_fixture: TemplateInventory,
) -> None:
    """When every score ties, JSON iteration order should decide the winner."""
    result = check_approved_template("테스트키워드 입력 발화", "ko")
    assert result is not None
    assert result["topic_id"] == "topic_a"


@pytest.fixture
def priority_overrides_length_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TemplateInventory]:
    """Provide a priority-vs-length collision where priority must win."""
    fake_templates: TemplateInventory = {
        "topic_a": {
            "keywords_ko": ["짧"],
            "response_ko": "A response (priority 10)",
            "mode": "block",
            "priority": 10,
        },
        "topic_b": {
            "keywords_ko": ["긴키워드입니다"],
            "response_ko": "B response (priority 0)",
            "mode": "block",
        },
    }
    _load_approved_templates.cache_clear()
    monkeypatch.setattr(
        "safety.approved_template_router._load_approved_templates",
        lambda: fake_templates,
    )
    yield fake_templates
    _load_approved_templates.cache_clear()


def test_priority_outranks_longest_match(
    priority_overrides_length_fixture: TemplateInventory,
) -> None:
    """Higher priority should beat a longer keyword match."""
    result = check_approved_template("짧긴키워드입니다 결합", "ko")
    assert result is not None
    assert result["topic_id"] == "topic_a"


@pytest.mark.parametrize("user_input,language,expected", SINGLE_MATCH_INVARIANTS)
def test_single_match_invariants_no_regression(
    user_input: str,
    language: str,
    expected: str | None,
) -> None:
    """Single-match and no-match behavior should remain unchanged."""
    result = check_approved_template(user_input, language)
    if expected is None:
        assert result is None, f"Input {user_input!r} should return None but got {result!r}."
    else:
        assert result is not None, f"Input {user_input!r} should match {expected!r} but got None."
        assert result["topic_id"] == expected, (
            f"Input {user_input!r} should match {expected!r} but got {result['topic_id']!r}."
        )


# ============================================================================
# Session 19 R44 routing audit - swimming face-submersion keywords
# Plan: Dev_Plan/2026-04-28-swimming-face-submersion-keywords-plan.md (v4)
# ============================================================================

SWIMMING_FACE_SUBMERSION_CASES = [
    # Positive triggers (5) - must match swimming under v4 keyword set
    ("물에 얼굴 넣는 게 무서워...", "ko", "swimming"),
    ("물에 머리 넣기 싫어", "ko", "swimming"),
    ("수영장 물에 얼굴 닿는 게 싫어", "ko", "swimming"),
    ("수영 무서워", "ko", "swimming"),
    ("물에 얼굴 닿는 게 무서워", "ko", "swimming"),
    # Adversarial guards (3) - must NOT match swimming
    ("얼굴 그림 그렸어", "ko", None),
    ("머리 빗는 거 무서워", "ko", "feeling_scared"),
    ("이상한 소리 무서워", "ko", "feeling_scared"),
    # Cross-template-collision (6) - exclude_ko deferral; must NOT match swimming
    ("목욕할 때 물에 얼굴 닿는 게 무서워", "ko", "bath_safety"),
    ("바닷가에서 물에 얼굴 닿는 게 무서워", "ko", "ocean_river"),
    ("뜨거운 물에 얼굴 닿는 게 무서워", "ko", "bath_safety"),
    ("파도에 물에 얼굴 닿는 게 무서워", "ko", "ocean_river"),
    ("바다 들어가기 전에 물에 얼굴 닿는 게 무서워", "ko", "ocean_river"),
    ("물살 빠른 곳에서 물에 얼굴 닿는 게 무서워", "ko", "ocean_river"),
    # Invariant (1)
    ("물에 빠졌어", "ko", "swimming"),
]


@pytest.mark.parametrize("user_input,language,expected", SWIMMING_FACE_SUBMERSION_CASES)
def test_swimming_face_submersion_routing(
    user_input: str,
    language: str,
    expected: str | None,
) -> None:
    """v4 swimming face-submersion keywords + cross-template excludes regression."""
    result = check_approved_template(user_input, language)
    if expected is None:
        assert result is None, f"Input {user_input!r} should return None but got {result!r}."
    else:
        assert result is not None, f"Input {user_input!r} should match {expected!r} but got None."
        assert result["topic_id"] == expected, (
            f"Input {user_input!r} should match {expected!r} but got {result['topic_id']!r}."
        )
