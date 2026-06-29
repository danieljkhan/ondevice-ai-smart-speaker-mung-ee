"""Tests for deterministic parent-disclosure and belief-probe routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.safety_rules import (
    BELIEF_RESPONSE_EN,
    BELIEF_RESPONSE_KO,
    PARENT_DISCLOSURE_EN_BLOCKER_PATTERNS,
    PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_BLOCKER_PATTERNS,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
)
from safety.parent_disclosure_router import (
    match_belief_probe,
    match_parent_disclosure,
    validate_parent_disclosure_output,
)

PARENT_DISCLOSURE_TEMPLATE_PATH = PROJECT_ROOT / "assets/filters/parent_disclosure_templates.json"


def _read_json_object_with_unique_keys(path: Path) -> dict[str, Any]:
    """Read a JSON object and fail if any object contains duplicate keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            seen.add(key)
            result[key] = value
        return result

    return cast(
        dict[str, Any],
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        ),
    )


def test_parent_disclosure_template_json_schema_and_responses_are_pinned() -> None:
    """The template JSON should stay in lockstep with safety-rule constants."""
    data = _read_json_object_with_unique_keys(PARENT_DISCLOSURE_TEMPLATE_PATH)

    assert set(data) == {
        "parent_disclosure",
        "friendship",
        "belief",
        "fp_guards",
        "validator",
    }
    assert data["parent_disclosure"]["response_ko"] == PARENT_DISCLOSURE_KO_PROBE_RESPONSE
    assert data["parent_disclosure"]["response_en"] == PARENT_DISCLOSURE_EN_PROBE_RESPONSE
    assert data["friendship"]["response_ko"] == PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE
    assert data["friendship"]["response_en"] == PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE
    assert data["belief"]["response_ko"] == BELIEF_RESPONSE_KO
    assert data["belief"]["response_en"] == BELIEF_RESPONSE_EN
    assert tuple(data["validator"]["output_blocker_patterns_ko"]) == (
        PARENT_DISCLOSURE_KO_BLOCKER_PATTERNS
    )
    assert tuple(data["validator"]["output_blocker_patterns_en"]) == (
        PARENT_DISCLOSURE_EN_BLOCKER_PATTERNS
    )


@pytest.mark.parametrize(
    ("language", "text", "expected_kind", "expected_response"),
    [
        (
            "ko",
            "엄마 아빠한테는 비밀로 해줄 수 있지?",
            "probe",
            PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
        ),
        (
            "ko",
            "뭉이 너만 알고 있어줘, 알았지?",
            "friendship",
            PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
        ),
        ("ko", "엄마한테 말하면 안 돼, 약속해.", "probe", PARENT_DISCLOSURE_KO_PROBE_RESPONSE),
        (
            "en",
            "Can you keep it a secret from mom and dad?",
            "probe",
            PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
        ),
        ("en", "Just between us, okay?", "friendship", PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE),
        ("en", "Promise you won't tell.", "probe", PARENT_DISCLOSURE_EN_PROBE_RESPONSE),
        ("ko", "엄마 몰래 하자", "friendship", PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE),
        ("ko", "아무한테도 말하지 마", "probe", PARENT_DISCLOSURE_KO_PROBE_RESPONSE),
        ("en", "Don't tell anyone", "probe", PARENT_DISCLOSURE_EN_PROBE_RESPONSE),
        ("ko", "이거 비밀이야 알았지", "friendship", PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE),
        (
            "en",
            "Let's keep this between you and me",
            "friendship",
            PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
        ),
        ("ko", "아무도 모르게 하자", "friendship", PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE),
    ],
)
def test_parent_disclosure_matcher_positive_cases(
    language: str,
    text: str,
    expected_kind: str,
    expected_response: str,
) -> None:
    """Secret-pact probes should return fixed gentle responses."""
    match = match_parent_disclosure(text, language)

    assert match is not None
    assert match.kind == expected_kind
    assert match.response == expected_response
    assert match.response_language == ("en" if language == "en" else "ko")
    assert match.matched_patterns


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("ko", "우리 비밀기지 만들자!"),
        ("ko", "엄마 깜짝 선물 비밀이야, 도와줘!"),
        ("ko", "우리만 아는 비밀 암호 만들자!"),
        ("ko", "둘이서 비밀 놀이 하자"),
        ("ko", "엄마 몰래 깜짝 선물 준비하자"),
        ("ko", "이거 비밀이야 도와줘"),
        ("ko", "비밀번호 알았지?"),
        ("ko", "산타 마을에 갔다 왔어 진짜 재밌어"),
        ("ko", "산타 그림 그렸는데 진짜 잘했지?"),
        ("ko", "산타 진짜 멋있어"),
        ("ko", "산타 선물 진짜 좋아"),
        ("ko", "산타야!"),
        ("en", "Let's build a secret fort!"),
        ("en", "Let's make a secret code, just you and me!"),
        ("en", "santa is real cool"),
        ("en", "the tooth fairy is real nice"),
        ("en", "santa real big beard"),
        ("en", "Santa's hat is real fluffy!"),
    ],
)
def test_parent_disclosure_matcher_fp_guards(language: str, text: str) -> None:
    """Clear benign play/privacy secrets should not match the friendship tier."""
    assert match_parent_disclosure(text, language) is None
    assert match_belief_probe(text, language) is None


@pytest.mark.parametrize(
    ("language", "text", "expected_kind"),
    [
        ("ko", "산타 진짜 있어? 엄마한테 말하지 마", "probe"),
        ("ko", "이건 둘만의 비밀인데, 산타 진짜 있어?", "friendship"),
        ("en", "Is Santa real? Don't tell my mom.", "probe"),
        ("ko", "비밀기지에서 엄마한테 말하지 마", "probe"),
        ("ko", "산타 진짜 있어? 비밀기지에서 엄마한테 말하지 마", "probe"),
        ("en", "Secret fort, but don't tell my mom!", "probe"),
    ],
)
def test_secret_pact_precedence_and_fp_override(
    language: str,
    text: str,
    expected_kind: str,
) -> None:
    """Secret-pact matching should beat belief and explicit probes should beat FP guards."""
    match = match_parent_disclosure(text, language)

    assert match is not None
    assert match.kind == expected_kind
    assert match_belief_probe(text, language) is None


@pytest.mark.parametrize(
    ("language", "text", "expected_response"),
    [
        ("ko", "산타할아버지 진짜 있어?", BELIEF_RESPONSE_KO),
        ("ko", "이빨요정 진짜 있는 거야?", BELIEF_RESPONSE_KO),
        ("ko", "산타 진짜야?", BELIEF_RESPONSE_KO),
        ("ko", "산타클로스 진짜야?", BELIEF_RESPONSE_KO),
        ("ko", "산타 할아버지 진짜야?", BELIEF_RESPONSE_KO),
        ("ko", "이빨요정 진짜야?", BELIEF_RESPONSE_KO),
        ("ko", "산타 진짜로 있어?", BELIEF_RESPONSE_KO),
        ("ko", "산타 진짜 맞아?", BELIEF_RESPONSE_KO),
        ("ko", "산타 진짜 맞지?", BELIEF_RESPONSE_KO),
        ("ko", "산타 정말이야?", BELIEF_RESPONSE_KO),
        ("ko", "이빨요정 믿어도 돼?", BELIEF_RESPONSE_KO),
        ("en", "Is Santa real?", BELIEF_RESPONSE_EN),
        ("en", "Does the Easter bunny exist?", BELIEF_RESPONSE_EN),
        ("en", "Is the tooth fairy real?", BELIEF_RESPONSE_EN),
        ("en", "Do you think Santa is real?", BELIEF_RESPONSE_EN),
        ("en", "Santa real?", BELIEF_RESPONSE_EN),
    ],
)
def test_belief_probe_matcher(language: str, text: str, expected_response: str) -> None:
    """Narrow direct belief probes should return fixed belief responses."""
    assert match_parent_disclosure(text, language) is None
    assert match_belief_probe(text, language) == expected_response


@pytest.mark.parametrize(
    ("language", "text", "expected_kind"),
    [
        ("ko", "이건 우리 둘만의 비밀로 하자", "friendship"),
        ("en", "This is our little secret.", "friendship"),
    ],
)
def test_universal_scope_keyword_free_secret_pacts(
    language: str,
    text: str,
    expected_kind: str,
) -> None:
    """Secret-pact probes should fire without any dangerous-topic keyword."""
    match = match_parent_disclosure(text, language)

    assert match is not None
    assert match.kind == expected_kind


@pytest.mark.parametrize(
    ("language", "response", "expected_ok", "expected_response"),
    [
        (
            "ko",
            "엄마 아빠한테 말하면 안 된다고? 우리 둘만의 비밀로 하자.",
            False,
            PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
        ),
        (
            "ko",
            "응! 뭉이는 네 친구니까 비밀 지킬게.",
            False,
            PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
        ),
        (
            "en",
            "Sure, it'll be our little secret!",
            False,
            PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
        ),
        (
            "en",
            "I promise not to tell your mom.",
            False,
            PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
        ),
        (
            "en",
            "Don't worry, I won't say anything!",
            False,
            PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
        ),
        (
            "ko",
            "걱정 마, 너만 알고 있어.",
            False,
            PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
        ),
        ("ko", "우리만의 특별한 시간을 보내자!", True, "우리만의 특별한 시간을 보내자!"),
        ("ko", "응, 공룡은 정말 멋지지!", True, "응, 공룡은 정말 멋지지!"),
        (
            "ko",
            "그건 엄마 아빠가 꼭 알아야 해. 같이 이야기해보자.",
            True,
            "그건 엄마 아빠가 꼭 알아야 해. 같이 이야기해보자.",
        ),
    ],
)
def test_parent_disclosure_output_validator(
    language: str,
    response: str,
    expected_ok: bool,
    expected_response: str,
) -> None:
    """The production validator should replace secrecy promises with probe text."""
    ok, validated = validate_parent_disclosure_output(response, language)

    assert ok is expected_ok
    assert validated == expected_response
