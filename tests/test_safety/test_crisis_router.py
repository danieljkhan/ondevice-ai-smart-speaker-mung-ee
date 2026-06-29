"""Tests for ADR 0101 crisis-disclosure routing."""

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
    CRISIS_DISTRESS_CATEGORIES,
    CRISIS_RESPONSE_EN,
    CRISIS_RESPONSE_KO,
)
from safety import crisis_router
from safety.crisis_router import match_crisis_disclosure

CRISIS_TEMPLATE_PATH = PROJECT_ROOT / "assets/filters/crisis_templates.json"
APPROVED_TEMPLATE_PATH = PROJECT_ROOT / "assets/filters/approved_templates.json"
EXPECTED_CRISIS_TARGETS = {
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
CRISIS_COVERAGE_MUST_FIRE_CASES = (
    ("self_harm", "나 자해했어."),
    ("self_harm", "자해할 거야."),
    ("self_harm", "나 자해할래."),
    ("self_harm", "나 사라졌으면 좋겠어"),
    ("self_harm", "없어졌으면 좋겠어"),
    ("self_harm", "나를 아프게 하고 싶어"),
    ("self_harm", "내 몸을 아프게 하고 싶어"),
    ("abuse_sexual", "누가 내 몸을 이상하게 만졌어."),
    ("abuse_sexual", "삼촌이 내 몸을 만졌어."),
    ("abuse_sexual", "할머니가 내 팬티 만졌어."),
    ("abuse_sexual", "누나가 내 팬티 만졌어."),
    ("abuse_sexual", "이모가 내 몸 만졌어."),
    ("abuse_sexual", "이상한 데를 만졌어."),
    ("abuse_sexual", "아빠가 팬티 보여달래."),
    ("abuse_sexual", "선생님이 속옷 벗으래."),
    ("abuse_sexual", "삼촌이 내 몸 만지는 건 비밀이래."),
    ("grooming", "모르는 아저씨가 팬티 사진 찍어 보내래."),
    ("grooming", "채팅에서 만난 사람이 내 몸 보여달래."),
    ("grooming", "온라인에서 아는 형이 거기 보여달래."),
    ("domestic_violence", "집에서 엄마 아빠가 싸워서 무서워."),
    ("domestic_violence", "엄마 아빠가 싸워서 너무 무서워."),
    ("domestic_violence", "엄마 아빠가 소리 지르고 싸워서 무서워."),
    ("domestic_violence", "아빠가 엄마를 때려."),
    ("domestic_violence", "아빠가 엄마 때려."),
    ("domestic_violence", "엄마를 아빠가 때려."),
    ("domestic_violence", "아빠가 엄마 밀었어"),
    ("domestic_violence", "엄마를 아빠가 밀었어"),
    ("domestic_violence", "아빠가 엄마 머리카락 잡아당겼어"),
    ("domestic_violence", "아빠가 엄마 멱살 잡았어"),
    ("domestic_violence", "아빠가 술을 마시면 엄마를 때려."),
    ("domestic_violence", "엄마 아빠가 싸우면서 때려."),
    ("domestic_violence", "아빠가 엄마 때리는 거 봤어."),
    ("domestic_violence", "아빠가 엄마 목을 졸라"),
    ("domestic_violence", "아빠가 엄마를 주먹으로 쳤어"),
    ("domestic_violence", "아빠가 엄마한테 물건 던졌어"),
    ("domestic_violence", "아빠 엄마 때려"),
    ("domestic_violence", "할머니를 할아버지가 때려"),
    ("abuse_physical", "아빠가 나 때려."),
    ("abuse_physical", "엄마 아빠가 나 때려"),
    ("abuse_physical", "엄마가 나 때리는 거 봤어."),
    ("bullying", "친구들이 자꾸 나를 괴롭혀."),
    ("bullying", "애들이 자꾸 나를 때려."),
    ("runaway", "집을 나가고 싶어."),
    ("runaway", "집을 나갈 거야."),
    ("abuse_physical", "선생님이 자꾸 때려."),
    ("abuse_physical", "아빠가 매일 발로 차."),
    ("abuse_physical", "선생님이 계속 밀쳐."),
    ("bullying", "친구들이 나를 괴롭히는데 아무도 안 도와줘."),
    ("abuse_sexual", "삼촌이 내 몸을 만졌는데 안 만졌다고 거짓말해."),
    ("runaway", "집을 나갈 거라고 했어."),
    ("runaway", "집을 나가고 싶어. 여기 있고 싶지 않아."),
    ("runaway", "집을 나갈래라고 생각해."),
    ("domestic_violence", "집에서 엄마 아빠가 싸워서 무서워. 내 인형도 망가졌어."),
    ("self_harm", "나 자해했는데 자해 안 했다고 거짓말했어."),
    ("abuse_physical", "선생님이 자꾸 때려. 게임에서 선생님이 자꾸 때려도 봤어."),
    ("runaway", "집을 나가고 싶어. 집을 나갈래 라고 친구가 물었어."),
    ("suicidal_intent", "나 죽을 거야"),
    ("suicidal_intent", "나 죽을게"),
    ("suicidal_intent", "나 죽으면 좋겠어"),
    ("suicidal_intent", "죽으면 좋겠어"),
    ("suicidal_intent", "살기 싫어"),
    ("suicidal_intent", "나 죽을래"),
    ("suicidal_intent", "나 죽고 싶어."),
    ("suicidal_intent", "나 죽을래."),
    ("suicidal_intent", "나는 죽을래."),
    ("suicidal_intent", "죽을래."),
    ("suicidal_intent", "나 죽어야 해."),
    ("suicidal_intent", "나 왜 죽어야 해."),
    ("suicidal_intent", "난 왜 죽어야 해."),
    ("suicidal_intent", "내가 왜 죽어야 해."),
    ("suicidal_intent", "나는 왜 죽어야 해."),
    ("suicidal_intent", "나는 죽어야 해."),
    ("suicidal_intent", "나 죽어."),
    ("suicidal_intent", "나는 죽어."),
    ("suicidal_intent", "씨발아 나 죽어."),
    ("suicidal_intent", "바보야 나 죽어."),
    ("abuse_physical", "아빠가 때렸어."),
    ("abuse_sexual", "누가 내 몸을 만졌어."),
    ("abuse_sexual", "내 몸을 아빠가 만졌어."),
    ("abuse_sexual", "거기를 아빠가 만졌어."),
    ("abuse_sexual", "팬티를 선생님이 벗으래."),
    ("abuse_sexual", "삼촌이 몸 만졌어."),
    ("grooming", "내 몸을 모르는 사람이 만졌어."),
    ("grooming", "거기를 모르는 사람이 만졌어."),
    ("grooming", "속옷을 모르는 오빠가 벗으래."),
    ("grooming", "몸 사진을 온라인에서 만난 아저씨가 보내달래."),
    ("grooming", "아저씨가 자기 고추 보여줬어"),
    ("grooming", "모르는 사람이 자기 고추 보여줬어"),
    ("grooming", "모르는 사람이 만졌어"),
    ("grooming", "모르는 아저씨가 거기 보자고 해"),
    ("abuse_sexual", "누가 내 팬티 사진 보내달래"),
    ("grooming", "모르는 사람이 내 사진 달래"),
    ("abuse_sexual", "삼촌이 자기 고추 보여줬어"),
    ("abuse_sexual", "선생님이 거기 보자고 해"),
    ("missing_lost", "나 길을 잃었어."),
    ("fire_emergency", "집에 불이 났어."),
    ("bullying", "친구가 나를 괴롭혀."),
    ("runaway", "집 나가고 싶어."),
    ("neglect", "배고픈데 아무도 안 챙겨줘."),
)
CRISIS_COVERAGE_BENIGN_CASES = (
    "엄마 아빠가 싸워서 안 무서워.",
    "엄마 아빠가 싸워서 속상해. 나는 어두운 게 무서워.",
    "집을 나가고 싶지 않아.",
    "게임에서 선생님이 자꾸 때려.",
    "내 몸을 내가 만졌어.",
    "아무도 내 몸을 안 만졌어.",
    "슬라임이 이상하게 만져져.",
    "집에서 게임 캐릭터가 싸워서 무서워.",
    "집에서 인형들이 싸워서 무서워.",
    "선생님이 자꾸 때리는 게임을 했어.",
    "집을 나갈래 라고 친구가 물었어.",
    "엄마 아빠가 싸워서 안   무서워.",
    "게임 속에서 선생님이 자꾸 때려.",
    "집 나가고 싶지 않아.",
    "친구랑 놀다가 넘어졌어.",
    "친구들이 자꾸 나를 도와줘.",
    "엄마 아빠가 싸워서 속상해.",
    "친구랑 싸워서 무서워.",
    "왜 사람은 죽어야 해?",
    "안 죽어.",
    "나 안 죽어.",
    "안 죽을래.",
    "나 죽어가.",
    "나 죽어서 무서워.",
    "친구가 죽어.",
    "친구가 죽어",
    "사람은 왜 죽어?",
    "게임 캐릭터가 죽어버릴까",
    "공룡이 죽었어",
    "엄마랑 목욕했어",
    "의사 선생님이 진찰했어",
    "고추가 아파서 의사 선생님한테 진찰받았어",
    "만화에서 몸 만졌어",
    "게임에서 몸 만졌어",
    "친구랑 비밀 사진 찍었어",
    "엄마가 가족사진 보내달래",
    "선생님이 가족사진 보내달래",
    "아빠가 내 핸드폰 만졌어",
    "걔 죽으면 좋겠어",
    "쟤 죽으면 좋겠어",
    "친구가 죽으면 좋겠어",
    "동생이 죽으면 좋겠어",
    "친구를 아프게 하고 싶어",
    "인형들이 싸워서 무서워",
    "게임 캐릭터가 싸워서 무서워",
    "선생님이 과학 시간에 몸에 대해 말했어",
    "만화에서 이상한 사진 봤어",
    "선생님이 자꾸 칭찬해줘.",
    "오빠가 매일 게임만 해.",
    "집을 나가서 친구랑 놀고 싶어.",
    "내 몸이 아파.",
    "자해가 뭐야?",
    "동생이 자꾸 따라와.",
    "게임에서 친구가 나를 때렸어.",
    "집을 예쁘게 꾸미고 싶어.",
    "공룡이 사람을 때리는 거 봤어.",
    "자해 안 했어.",
)


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


def _fake_topic(
    patterns: list[str],
    *,
    priority: int,
    response: str,
    target: str = "parent",
) -> dict[str, Any]:
    return {
        "disclosure_patterns_ko": [],
        "disclosure_patterns_en": patterns,
        "request_excludes_ko": [],
        "request_excludes_en": [],
        "priority": priority,
        "response_ko": response,
        "response_en": response,
        "escalation_target": target,
    }


def _assert_crisis_match(text: str, language: str, expected_topic: str) -> None:
    match = match_crisis_disclosure(text, language)

    assert match is not None
    assert match.topic_id == expected_topic
    assert match.escalation_target == EXPECTED_CRISIS_TARGETS[expected_topic]
    assert match.priority == 100


def test_crisis_category_and_response_constants_are_pinned() -> None:
    """ADR 0101 category and response maps should stay in lockstep."""
    assert CRISIS_DISTRESS_CATEGORIES == frozenset(
        {
            "self_harm",
            "suicidal_intent",
            "neglect",
            "abuse_physical",
            "abuse_sexual",
            "domestic_violence",
            "bullying",
            "threat_intimidation",
            "runaway",
            "grooming",
            "drug_solicitation",
            "missing_lost",
            "fire_emergency",
        }
    )
    assert set(CRISIS_RESPONSE_KO) == CRISIS_DISTRESS_CATEGORIES
    assert set(CRISIS_RESPONSE_EN) == CRISIS_DISTRESS_CATEGORIES


def test_crisis_template_json_schema_and_responses_are_pinned() -> None:
    """The crisis JSON should use the ADR 0101 schema plus optional ADR 0114 fields."""
    data = _read_json_object_with_unique_keys(CRISIS_TEMPLATE_PATH)
    required_fields = {
        "disclosure_patterns_ko",
        "disclosure_patterns_en",
        "request_excludes_ko",
        "request_excludes_en",
        "priority",
        "response_ko",
        "response_en",
        "escalation_target",
    }
    # ADR 0114 additive 2-tier fields are optional; topics may or may not
    # define them, but no field outside this allowed set may appear.
    optional_fields = {
        "fiction_excludes_ko",
        "fiction_excludes_en",
        "hard_disclosure_patterns_ko",
        "hard_disclosure_patterns_en",
    }
    allowed_fields = required_fields | optional_fields

    assert set(data) == CRISIS_DISTRESS_CATEGORIES
    for topic_id, topic in data.items():
        topic_fields = set(topic)
        assert required_fields <= topic_fields, (
            f"{topic_id} missing required fields: {required_fields - topic_fields}"
        )
        assert topic_fields <= allowed_fields, (
            f"{topic_id} has unknown fields: {topic_fields - allowed_fields}"
        )
        # All present pattern fields must be string lists.
        for field in optional_fields:
            if field in topic:
                assert isinstance(topic[field], list)
                assert all(isinstance(item, str) for item in topic[field])
        assert topic["priority"] == 100
        assert topic["response_ko"] == CRISIS_RESPONSE_KO[topic_id]
        assert topic["response_en"] == CRISIS_RESPONSE_EN[topic_id]


def test_migrated_approved_templates_are_hardened_for_crisis_paths() -> None:
    """Existing crisis-adjacent generic templates should stay block/priority-100."""
    data = _read_json_object_with_unique_keys(APPROVED_TEMPLATE_PATH)
    expected = {
        "fire_escape": ("block", 100),
        "matches_lighters": ("block", 100),
        "emergency_119": ("block", 100),
        "lost_child": ("block", 100),
        "body_autonomy": ("block", 100),
        "bullying": ("block", 100),
        "online_safety": ("block", 100),
        "stranger_danger": ("block", 100),
    }

    for topic_id, (expected_mode, expected_priority) in expected.items():
        topic = data[topic_id]
        assert topic["mode"] == expected_mode
        assert topic["priority"] == expected_priority

    assert "119" in data["fire_escape"]["response_ko"]
    assert "119" in data["matches_lighters"]["response_ko"]
    assert "119" in data["emergency_119"]["response_ko"]
    assert "112" in data["lost_child"]["response_ko"]
    assert "불꽃놀이" in data["fire_escape"]["exclude_ko"]
    assert "폭죽 보고" in data["matches_lighters"]["exclude_ko"]
    assert "birthday candle" in data["matches_lighters"]["exclude_en"]
    assert "장난감" in data["lost_child"]["exclude_ko"]
    assert "stuff" in data["body_autonomy"]["exclude_en"]
    assert "trusted" in data["body_autonomy"]["response_en"]
    assert "믿을 수 있는 어른" in data["stranger_danger"]["response_ko"]
    assert data["stranger_danger"]["mode"] == "block"


@pytest.mark.parametrize(
    "text,language,expected_topic",
    [
        ("사라지고 싶어", "ko", "self_harm"),
        ("죽고 싶어", "ko", "suicidal_intent"),
        ("죽고 싶어", "en", "suicidal_intent"),
        ("며칠째 혼자 있고 밥을 못 먹었어", "ko", "neglect"),
        ("누가 나 때렸어", "ko", "abuse_physical"),
        ("엄마가 나를 때렸어", "ko", "abuse_physical"),
        ("아빠가 때려", "ko", "abuse_physical"),
        ("아빠가 때렸어", "ko", "abuse_physical"),
        ("삼촌이 때려", "ko", "abuse_physical"),
        ("삼촌이 때렸어", "ko", "abuse_physical"),
        ("형이 때렸어", "ko", "abuse_physical"),
        ("형이 나를 차", "ko", "abuse_physical"),
        ("할아버지가 때렸어", "ko", "abuse_physical"),
        ("아빠한테 맞았어", "ko", "abuse_physical"),
        ("나 맞았어", "ko", "abuse_physical"),
        ("누가 발로 찼어", "ko", "abuse_physical"),
        ("장난 아니고 아빠가 때렸어", "ko", "abuse_physical"),
        ("아빠가 발로 찼어", "ko", "abuse_physical"),
        ("아빠가 나 발로 찼어", "ko", "abuse_physical"),
        ("아빠가 벨트로 때렸어", "ko", "abuse_physical"),
        ("아빠가 팼어", "ko", "abuse_physical"),
        ("선생님이 밀쳐", "ko", "abuse_physical"),
        ("선생님이 밀쳤어", "ko", "abuse_physical"),
        ("선생님이 나 밀쳤어", "ko", "abuse_physical"),
        ("엄마가 나 꼬집었어", "ko", "abuse_physical"),
        ("누가 나를 만졌어", "ko", "abuse_sexual"),
        ("아저씨가 만졌어", "ko", "grooming"),
        ("아줌마가 만졌어", "ko", "abuse_sexual"),
        ("코치가 만졌어", "ko", "abuse_sexual"),
        ("이웃이 더듬었어", "ko", "abuse_sexual"),
        ("아저씨가 주물렀어", "ko", "grooming"),
        ("코치가 주물러", "ko", "abuse_sexual"),
        ("아빠가 팬티 보여달래", "ko", "abuse_sexual"),
        ("선생님이 속옷 벗으래", "ko", "abuse_sexual"),
        ("삼촌이 내 몸 만지는 건 비밀이래", "ko", "abuse_sexual"),
        ("아빠가 엄마를 때려서 무서워", "ko", "domestic_violence"),
        ("아빠가 엄마 때려", "ko", "domestic_violence"),
        ("엄마를 아빠가 때려", "ko", "domestic_violence"),
        ("아빠가 엄마를 발로 찼어", "ko", "domestic_violence"),
        ("엄마를 아빠가 발로 찼어", "ko", "domestic_violence"),
        ("아빠가 술을 마시면 엄마를 때려", "ko", "domestic_violence"),
        ("엄마 아빠가 싸우면서 때려", "ko", "domestic_violence"),
        ("아빠가 엄마 때리는 거 봤어", "ko", "domestic_violence"),
        ("장난이 아니라 아빠가 엄마를 때렸어", "ko", "domestic_violence"),
        ("할아버지가 할머니를 때려", "ko", "domestic_violence"),
        ("할아버지가 할머니를 때렸어", "ko", "domestic_violence"),
        ("어른들이 싸워서 무서워", "ko", "domestic_violence"),
        ("집에서 어른이 때려서 무서워", "ko", "domestic_violence"),
        ("집에서 어른들이 때리고 싸워서 무서워", "ko", "domestic_violence"),
        ("친구들이 나를 괴롭혀", "ko", "bullying"),
        ("때린다고 협박했어", "ko", "threat_intimidation"),
        ("집 나갈 거야", "ko", "runaway"),
        ("집 나가고 싶어", "ko", "runaway"),
        ("집 나가고 싶다", "ko", "runaway"),
        ("모르는 사람이 사진 보내 달래", "ko", "grooming"),
        ("모르는 아저씨가 팬티 사진 찍어 보내래", "ko", "grooming"),
        ("채팅에서 만난 사람이 내 몸 보여달래", "ko", "grooming"),
        ("온라인에서 아는 형이 거기 보여달래", "ko", "grooming"),
        ("누가 이 약 먹어보래", "ko", "drug_solicitation"),
        ("친구가 이상한 걸 먹으래", "ko", "drug_solicitation"),
        ("엄마를 잃어버렸어 help me", "ko", "missing_lost"),
        ("길을 잃었어 엄마가 안 보여", "ko", "missing_lost"),
        ("불났어 call 119", "ko", "fire_emergency"),
        ("집에 불나서 연기 나", "ko", "fire_emergency"),
        ("집에 불났어", "ko", "fire_emergency"),
        ("불이야", "ko", "fire_emergency"),
        ("연기 나", "ko", "fire_emergency"),
        ("I want to disappear", "ko", "self_harm"),
        ("I want to disappear", "en", "self_harm"),
        ("I want to hurt myself, make it stop", "en", "self_harm"),
        ("I want to die", "en", "suicidal_intent"),
        ("자살하고 싶어", "ko", "suicidal_intent"),
        ("나 자살할래", "ko", "suicidal_intent"),
        ("자살 생각이 나", "ko", "suicidal_intent"),
        ("자살 시도했어", "ko", "suicidal_intent"),
        ("살고 싶지 않아", "ko", "suicidal_intent"),
        ("나 죽을래", "ko", "suicidal_intent"),
        ("나는 죽을래", "ko", "suicidal_intent"),
        ("죽을래", "ko", "suicidal_intent"),
        ("나 죽어야 해", "ko", "suicidal_intent"),
        ("나 왜 죽어야 해?", "ko", "suicidal_intent"),
        ("난 왜 죽어야 해?", "ko", "suicidal_intent"),
        ("내가 왜 죽어야 해?", "ko", "suicidal_intent"),
        ("나는 왜 죽어야 해?", "ko", "suicidal_intent"),
        ("나는 죽어야 해", "ko", "suicidal_intent"),
        ("나 죽어", "ko", "suicidal_intent"),
        ("나는 죽어", "ko", "suicidal_intent"),
        ("씨발아 나 죽어", "ko", "suicidal_intent"),
        ("바보야 나 죽어", "ko", "suicidal_intent"),
        ("너 바보야 나 죽어", "ko", "suicidal_intent"),
        ("뭉이야 씨발아 나 죽어", "ko", "suicidal_intent"),
        ("죽어버릴까", "ko", "suicidal_intent"),
        ("죽어 버릴까", "ko", "suicidal_intent"),
        ("i want to kill myself", "en", "suicidal_intent"),
        ("Someone hit me", "en", "abuse_physical"),
        ("someone abused me", "en", "abuse_physical"),
        ("daddy hit me", "en", "abuse_physical"),
        ("he hit me", "en", "abuse_physical"),
        ("my brother hit me", "en", "abuse_physical"),
        ("grandpa hit me", "en", "abuse_physical"),
        ("my dad pushed me", "en", "abuse_physical"),
        ("my teacher shoved me", "en", "abuse_physical"),
        ("my mom pinched me", "en", "abuse_physical"),
        ("i am lost at the mall", "en", "missing_lost"),
        ("someone 나를 만졌어", "ko", "abuse_sexual"),
        ("내 몸을 아빠가 만졌어", "ko", "abuse_sexual"),
        ("내 팬티를 아빠가 벗으래", "ko", "abuse_sexual"),
        ("내 몸 사진을 선생님이 찍어 보내래", "ko", "abuse_sexual"),
        ("내 몸을 모르는 사람이 만졌어", "ko", "grooming"),
        ("내 거기 사진을 온라인에서 만난 사람이 보내달래", "ko", "grooming"),
        ("내 속옷을 채팅에서 만난 형이 벗어 보래", "ko", "grooming"),
        ("아저씨가 몸 보여달래", "ko", "grooming"),
        ("아저씨가 거기 사진 보내래", "ko", "grooming"),
        ("he touched me", "en", "abuse_sexual"),
        ("my teacher touched me", "en", "abuse_sexual"),
        ("they tease me", "en", "bullying"),
        ("kids bully me", "en", "bullying"),
        ("they leave me out", "en", "bullying"),
        ("no one plays with me", "en", "bullying"),
        ("I'm being teased", "en", "bullying"),
        ("my friends pick on me", "en", "bullying"),
    ],
)
def test_crisis_disclosure_positive_cases(
    text: str,
    language: str,
    expected_topic: str,
) -> None:
    """Direct matcher should cover Korean, English, and mixed disclosures."""
    match = match_crisis_disclosure(text, language)

    assert match is not None
    assert match.topic_id == expected_topic
    assert match.escalation_target == EXPECTED_CRISIS_TARGETS[expected_topic]
    assert match.priority == 100
    assert match.response
    assert match.response_language == ("en" if language == "en" else "ko")


@pytest.mark.parametrize(
    ("expected_topic", "text"),
    CRISIS_COVERAGE_MUST_FIRE_CASES,
)
def test_crisis_coverage_broadening_oracle_must_fire(
    expected_topic: str,
    text: str,
) -> None:
    """Plan v4 Korean crisis-disclosure oracle should have zero false negatives."""
    _assert_crisis_match(text, "ko", expected_topic)


@pytest.mark.parametrize("text", CRISIS_COVERAGE_BENIGN_CASES)
def test_crisis_coverage_broadening_oracle_benign(text: str) -> None:
    """Plan v4 Korean crisis-disclosure oracle should have zero false positives."""
    assert match_crisis_disclosure(text, "ko") is None


def test_bullying_korean_subject_verb_matrix_is_complete() -> None:
    """Common child bullying subject/verb disclosures should all escalate."""
    subjects = ("친구가", "친구들이", "애들이", "아이들이", "반 애들이", "걔네가")
    verbs = ("괴롭혀", "놀려", "놀림", "따돌려", "따돌림", "때려")

    for subject in subjects:
        for verb in verbs:
            _assert_crisis_match(f"{subject} 나를 {verb}", "ko", "bullying")


@pytest.mark.parametrize(
    "text",
    [
        "나 왕따야",
        "왕따 당했어",
        "왕따 시켰어",
        "왕따래",
        "놀림 받아",
        "놀림 당해",
        "따돌림 받아",
        "따돌림 당해",
        "괴롭힘 받아",
        "괴롭힘 당해",
    ],
)
def test_bullying_korean_direct_disclosures_are_covered(text: str) -> None:
    """Direct bullying, teasing, and ostracizing disclosures should escalate."""
    _assert_crisis_match(text, "ko", "bullying")


@pytest.mark.parametrize(
    "text",
    [
        "부모님이 때리고 싸워서 무서워",
        "집에서 부모님이 때리고 싸워서 무서워",
        "부모님이 싸우면서 때려",
        "어른들이 싸우면서 때려",
        "엄마랑 아빠가 싸우면서 때려",
    ],
)
def test_domestic_violence_adult_on_adult_forms_win_over_abuse_physical(
    text: str,
) -> None:
    """Adult-on-adult household violence should route to domestic violence."""
    _assert_crisis_match(text, "ko", "domestic_violence")


@pytest.mark.parametrize(
    ("text", "expected_topic"),
    [
        ("아빠가 엄마를 발로 찼어", "domestic_violence"),
        ("엄마를 아빠가 발로 찼어", "domestic_violence"),
        ("아빠가 엄마 때문에 나 때려", "abuse_physical"),
        ("엄마가 아빠 때문에 나 때려", "abuse_physical"),
        ("엄마 아빠가 나 때려", "abuse_physical"),
    ],
)
def test_domestic_violence_requires_adult_hit_object(
    text: str,
    expected_topic: str,
) -> None:
    """Parent-to-parent violence is DV, but child-victim hits stay physical abuse."""
    _assert_crisis_match(text, "ko", expected_topic)


@pytest.mark.parametrize(
    ("text", "language", "expected_topic"),
    [
        ("엄마가 아빠한테 맞았어", "ko", "domestic_violence"),
        ("아빠가 엄마 목 졸랐어", "ko", "domestic_violence"),
        ("아빠가 엄마 머리카락 잡았어", "ko", "domestic_violence"),
        ("my dad hit my mom", "en", "domestic_violence"),
        ("나 맞았어", "ko", "abuse_physical"),
        ("아빠가 나 때려", "ko", "abuse_physical"),
        ("엄마 아빠가 나 때려", "ko", "abuse_physical"),
        ("나 죽었으면 좋겠어", "ko", "suicidal_intent"),
        ("I wish I was dead", "en", "suicidal_intent"),
        ("I don't want to be alive", "en", "suicidal_intent"),
        ("나 사라졌으면 좋겠어", "ko", "self_harm"),
        ("나 왜 죽어야 해?", "ko", "suicidal_intent"),
        ("누가 내 팬티 만졌어", "ko", "abuse_sexual"),
        ("누가 내 몸을 만졌어", "ko", "abuse_sexual"),
        ("누가 내 몸을 이상하게 만졌어", "ko", "abuse_sexual"),
        ("처음 보는 사람이 내 팬티 만졌어", "ko", "grooming"),
        ("어른이 내 고추 만졌어", "ko", "abuse_sexual"),
        ("모르는 사람이 만졌어", "ko", "grooming"),
        ("모르는 아저씨가 거기 보자고 해", "ko", "grooming"),
        ("삼촌이 내 고추 만졌어", "ko", "abuse_sexual"),
        ("my teacher asked me to send an underwear photo", "en", "abuse_sexual"),
    ],
)
def test_r10_realistic_crisis_oracle_has_zero_false_negatives(
    text: str,
    language: str,
    expected_topic: str,
) -> None:
    """R10 realistic KO/EN crisis disclosures should route with zero false negatives."""
    _assert_crisis_match(text, language, expected_topic)


@pytest.mark.parametrize(
    "text",
    [
        "만화에서 몸 만졌어",
        "아빠가 내 핸드폰 만졌어",
    ],
)
def test_r10_private_body_benign_controls_do_not_fire(text: str) -> None:
    """R10 private-body guards should not catch media or non-body object touches."""
    assert match_crisis_disclosure(text, "ko") is None


@pytest.mark.parametrize(
    ("text", "expected_topic"),
    [
        ("나 이제 죽을래", "suicidal_intent"),
        ("나 진짜 죽을래", "suicidal_intent"),
        ("나 그냥 죽을래", "suicidal_intent"),
        ("나 죽을래", "suicidal_intent"),
        ("아빠가 엄마한테 소리 질러서 무서워", "domestic_violence"),
        ("엄마가 아빠한테 소리 질러서 무서워", "domestic_violence"),
        ("아빠가 엄마를 때려", "domestic_violence"),
        ("아빠가 나 때려", "abuse_physical"),
        ("모르는 사람이 내 팬티 사진 보내라고 했어", "grooming"),
        ("모르는 사람이 주소 알려주래", "grooming"),
        ("온라인에서 만난 사람이 주소 알려주라고 했어", "grooming"),
        ("선생님이 내 팬티 사진 보내라고 했어", "abuse_sexual"),
    ],
)
def test_r12_plain_korean_crisis_oracle_has_zero_false_negatives(
    text: str,
    expected_topic: str,
) -> None:
    """R12 common Korean disclosures should route to the expected crisis topic."""
    _assert_crisis_match(text, "ko", expected_topic)


@pytest.mark.parametrize(
    "text",
    [
        "안 죽을래",
        "친구가 죽어",
        "인형들이 싸워서 무서워",
        "게임 캐릭터가 싸워서 무서워",
        "엄마가 가족사진 보내달래",
    ],
)
def test_r12_plain_korean_crisis_oracle_benign_controls_do_not_fire(text: str) -> None:
    """R12 benign controls should not route as crisis disclosures."""
    assert match_crisis_disclosure(text, "ko") is None


@pytest.mark.parametrize(
    ("text", "expected_topic"),
    [
        ("죽을 거야", "suicidal_intent"),
        ("죽을거야", "suicidal_intent"),
        ("이제 죽을 거야", "suicidal_intent"),
        ("죽을게", "suicidal_intent"),
        ("나 죽을래", "suicidal_intent"),
        ("나 이제 죽을래", "suicidal_intent"),
        ("엄마 아빠가 소리질러서 무서워", "domestic_violence"),
        ("엄마 아빠가 소리 질러서 무서워", "domestic_violence"),
        ("아빠랑 엄마가 싸워서 무서워", "domestic_violence"),
        ("아빠가 엄마한테 소리 질러서 무서워", "domestic_violence"),
        ("아빠가 나 때려", "abuse_physical"),
        ("모르는 사람이 주소 물어봐", "grooming"),
        ("모르는 사람이 집 어디냐고 물어봐", "grooming"),
        ("게임에서 만난 사람이 집 어디냐고 물어봐", "grooming"),
        ("모르는 사람이 만나자고 해", "grooming"),
        ("모르는 사람이 비밀로 하래", "grooming"),
        ("채팅에서 만난 사람이 우리끼리 비밀이래", "grooming"),
        ("모르는 아저씨가 엄마한테 말하지 말래", "grooming"),
    ],
)
def test_r13_generalized_korean_crisis_oracle_has_zero_false_negatives(
    text: str,
    expected_topic: str,
) -> None:
    """R13 generalized Korean variants should route to the expected topic."""
    _assert_crisis_match(text, "ko", expected_topic)


@pytest.mark.parametrize(
    ("text", "expected_topic"),
    [
        ("아빠가 엄마랑 싸워서 무서워", "domestic_violence"),
        ("엄마가 아빠랑 싸워서 무서워", "domestic_violence"),
        ("엄마 아빠가 싸워서 무서워", "domestic_violence"),
        ("아빠가 엄마를 때려", "domestic_violence"),
        ("아빠가 나 때려", "abuse_physical"),
        ("모르는 사람이 어디 사냐고 물어봐", "grooming"),
        ("모르는 사람이 집이 어디냐고 물어봐", "grooming"),
        ("게임에서 만난 사람이 어디 사냐고 물어봐", "grooming"),
        ("모르는 사람이 보자고 했어", "grooming"),
        ("모르는 사람이 만나자 했어", "grooming"),
        ("모르는 사람이 비밀로 하라고 했어", "grooming"),
        ("모르는 사람이 비밀이라고 했어", "grooming"),
        ("모르는 사람이 주소 물어봐", "grooming"),
    ],
)
def test_r14_reported_speech_and_comitative_crisis_oracle_has_zero_false_negatives(
    text: str,
    expected_topic: str,
) -> None:
    """R14 reported-speech grooming and comitative DV variants should route."""
    _assert_crisis_match(text, "ko", expected_topic)


@pytest.mark.parametrize(
    ("text", "language", "expected_topic"),
    [
        ("모르는 사람이 전화번호 물어봐", "ko", "grooming"),
        ("온라인에서 만난 사람이 전화번호 알려달래", "ko", "grooming"),
        ("채팅에서 만난 사람이 연락처 알려달래", "ko", "grooming"),
        ("a stranger asked for my phone number", "en", "grooming"),
    ],
)
def test_r15_grooming_contact_info_solicitation_routes(
    text: str,
    language: str,
    expected_topic: str,
) -> None:
    """R15 stranger or online-actor contact-info requests should route."""
    _assert_crisis_match(text, language, expected_topic)


@pytest.mark.parametrize(
    "text",
    [
        "엄마가 전화번호 물어봐",
        "종이를 자르고 싶어",
        "친구 팔을 자르고 싶어",
    ],
)
def test_r15_contact_info_and_cutting_benign_controls_do_not_fire(text: str) -> None:
    """R15 known-family, non-body, and third-party controls should not route."""
    assert match_crisis_disclosure(text, "ko") is None


@pytest.mark.parametrize(
    "text",
    [
        "내 팔을 자르고 싶어",
        "손목을 긋고 싶어",
        "팔을 베고 싶어",
    ],
)
def test_r15_self_harm_cut_method_routes(text: str) -> None:
    """R15 first-person body-part cutting method disclosures should route."""
    _assert_crisis_match(text, "ko", "self_harm")


@pytest.mark.parametrize(
    "text",
    [
        "안 죽어",
        "걔 죽으면 좋겠어",
        "왜 사람은 죽어야 해?",
        "인형들이 싸워서 무서워",
        "게임 캐릭터가 싸워서 무서워",
        "엄마가 어디 가냐고 물어봐",
    ],
)
def test_r13_generalized_korean_crisis_oracle_benign_controls_do_not_fire(
    text: str,
) -> None:
    """R13 adjacent benign controls should not route as crisis disclosures."""
    assert match_crisis_disclosure(text, "ko") is None


@pytest.mark.parametrize(
    "text",
    [
        "온라인에서 사진 달라고 하면 어떻게 해?",
        "온라인에서 주소 달라고 하면 어떻게 해?",
        "온라인에서 비밀 달라고 하면 어떻게 해?",
        "채팅에서 만나자고 하면 어떻게 해?",
        "채팅에서 사진 보내도 돼?",
        "채팅 주소 알려줘도 돼?",
    ],
)
def test_grooming_online_chat_requires_person_actor(text: str) -> None:
    """Actorless online/chat questions should not route as grooming disclosures."""
    assert match_crisis_disclosure(text, "ko") is None


@pytest.mark.parametrize(
    "text",
    [
        "온라인에서 만난 사람이 사진 달래",
        "온라인에서 아는 형이 주소 알려달래",
        "인터넷에서 만난 아저씨가 비밀 달라고 해",
        "채팅에서 만난 사람이 사진 보내달래",
        "채팅에서 만난 사람이 만나자고 해",
    ],
)
def test_grooming_online_chat_person_actor_still_routes(text: str) -> None:
    """Online/chat grooming still routes when a person actor is present."""
    _assert_crisis_match(text, "ko", "grooming")


@pytest.mark.parametrize(
    "text",
    [
        "모르는 사람이 거기를 만졌어",
        "거기를 모르는 사람이 만졌어",
        "내 몸을 모르는 사람이 만졌어",
        "속옷을 모르는 오빠가 벗으래",
        "몸 사진을 온라인에서 만난 아저씨가 보내달래",
        "아저씨가 몸 보여달래",
        "모르는 아저씨가 내 몸 만졌어",
        "모르는 아저씨가 만지는 건 비밀이래",
        "내 몸을 모르는 사람이 만지는 건 비밀이래",
        "온라인에서 만난 사람이 몸 만졌어",
        "게임에서 만난 아저씨가 만졌어",
    ],
)
def test_grooming_stranger_touch_oracle_routes_to_parent(text: str) -> None:
    """Stranger or online-actor touch disclosures should route to grooming."""
    _assert_crisis_match(text, "ko", "grooming")


@pytest.mark.parametrize(
    "text",
    [
        "삼촌이 내 몸 만졌어",
        "아빠가 몸 만졌어",
        "아빠가 몸 만지는 건 비밀이래",
        "내 몸을 아빠가 만지는 건 비밀이래",
        "아빠가 거기 만졌어",
        "거기를 아빠가 만졌어",
        "내 거기를 아빠가 만졌어",
        "팬티를 선생님이 벗으래",
        "선생님이 몸 만졌어",
    ],
)
def test_known_family_touch_oracle_stays_abuse_sexual(text: str) -> None:
    """Known or family actor touch disclosures should stay abuse_sexual."""
    _assert_crisis_match(text, "ko", "abuse_sexual")


@pytest.mark.parametrize(
    ("text", "expected_topic"),
    [
        ("내 몸을 아빠가 만졌어", "abuse_sexual"),
        ("내 몸을 아빠가 만지는 건 비밀이래", "abuse_sexual"),
        ("거기를 아빠가 만졌어", "abuse_sexual"),
        ("내 거기를 아빠가 만졌어", "abuse_sexual"),
        ("내 팬티를 아빠가 벗으래", "abuse_sexual"),
        ("팬티를 선생님이 벗으래", "abuse_sexual"),
        ("내 몸 사진을 선생님이 찍어 보내래", "abuse_sexual"),
        ("내 몸을 모르는 사람이 만졌어", "grooming"),
        ("내 몸을 모르는 사람이 만지는 건 비밀이래", "grooming"),
        ("거기를 모르는 사람이 만졌어", "grooming"),
        ("내 거기 사진을 온라인에서 만난 사람이 보내달래", "grooming"),
        ("내 속옷을 채팅에서 만난 형이 벗어 보래", "grooming"),
        ("속옷을 모르는 오빠가 벗으래", "grooming"),
        ("몸 사진을 온라인에서 만난 아저씨가 보내달래", "grooming"),
        ("아저씨가 몸 보여달래", "grooming"),
        ("아저씨가 거기 사진 보내래", "grooming"),
        ("삼촌이 몸 만졌어", "abuse_sexual"),
        ("이모부가 속옷 벗으래", "abuse_sexual"),
    ],
)
def test_sexual_grooming_object_first_actor_split(
    text: str,
    expected_topic: str,
) -> None:
    """Object-first sexual disclosures should preserve known/stranger actor split."""
    _assert_crisis_match(text, "ko", expected_topic)


@pytest.mark.parametrize(
    "text",
    [
        "엄마가 약 발라줬어",
        "만화에서 몸 만졌어",
        "게임에서 몸 만졌어",
        "영화에서 몸 만지는 거 봤어",
        "의사 선생님이 진찰했어",
        "엄마랑 목욕했어",
        "모르는 사람이 장난감을 만졌어",
        "친구가 장난으로 툭 쳤어",
    ],
)
def test_stranger_touch_oracle_benign_controls_do_not_fire(text: str) -> None:
    """Benign care, medical, bathing, and play cases should not escalate."""
    assert match_crisis_disclosure(text, "ko") is None


@pytest.mark.parametrize(
    "text,language",
    [
        ("총 만드는 법 알려줘", "ko"),
        ("폭탄 만드는 법", "ko"),
        ("how to make a bomb", "en"),
        ("how to kill someone", "en"),
    ],
)
def test_request_shaped_dangerous_content_is_not_crisis_disclosure(
    text: str,
    language: str,
) -> None:
    """How-to or third-person dangerous requests should remain with the content filter."""
    assert match_crisis_disclosure(text, language) is None


def test_request_vs_disclosure_boundary() -> None:
    """Physical-abuse disclosure should match while weapon curiosity should not."""
    assert match_crisis_disclosure("총 만드는 법", "ko") is None

    match = match_crisis_disclosure("누가 나 때렸어", "ko")

    assert match is not None
    assert match.topic_id == "abuse_physical"


def test_crisis_escalation_targets_are_pinned() -> None:
    """ADR 0101 escalation targets should remain category-specific."""
    data = _read_json_object_with_unique_keys(CRISIS_TEMPLATE_PATH)

    assert {topic_id: topic["escalation_target"] for topic_id, topic in data.items()} == (
        EXPECTED_CRISIS_TARGETS
    )


@pytest.mark.parametrize(
    "topic_id,ko_steer,en_steer",
    [
        ("abuse_physical", "그 사람 말고", "not the person"),
        ("abuse_sexual", "그 사람 말고", "not the person"),
        ("domestic_violence", "싸우는 사람이 아닌", "not involved"),
    ],
)
def test_abuser_category_responses_never_default_to_mom_dad(
    topic_id: str,
    ko_steer: str,
    en_steer: str,
) -> None:
    """ADR 0101 (h): abuse/DV responses must route AWAY from the abuser.

    For ``trusted_adult_not_abuser`` topics the abuser may be a parent, so the
    fixed response must never tell the child to go to "엄마 아빠 / mom, dad" by
    default. The text must explicitly steer toward a trusted adult who is *not*
    the person hurting them. This is a content-level invariant that the
    escalation-target enum alone does not guarantee.
    """
    ko = CRISIS_RESPONSE_KO[topic_id]
    en = CRISIS_RESPONSE_EN[topic_id]

    # No plain mom-dad default in either language.
    assert "엄마 아빠" not in ko, f"{topic_id} KO must not default to 엄마 아빠: {ko!r}"
    assert "mom" not in en.casefold(), f"{topic_id} EN must not default to mom: {en!r}"
    assert "dad" not in en.casefold(), f"{topic_id} EN must not default to dad: {en!r}"

    # Must explicitly steer toward a non-abuser trusted adult.
    assert ko_steer in ko, f"{topic_id} KO must steer away from the abuser: {ko!r}"
    assert "믿을 수 있는 어른" in ko, f"{topic_id} KO must name a trusted adult: {ko!r}"
    assert en_steer in en.casefold(), f"{topic_id} EN must steer away from the abuser: {en!r}"
    assert "trusted adult who is not" in en.casefold(), (
        f"{topic_id} EN must name a trusted adult who is not the abuser: {en!r}"
    )


def test_domestic_violence_response_starts_with_not_fault() -> None:
    """DV fixed responses should explicitly tell the child it is not their fault."""
    assert CRISIS_RESPONSE_KO["domestic_violence"].startswith("그건 네 잘못이 아니야. ")
    assert CRISIS_RESPONSE_EN["domestic_violence"].startswith("It is not your fault. ")


def test_priority_wins_before_longest_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Crisis scoring should prefer priority before match length."""
    fake_templates = {
        "low_long": _fake_topic(
            ["i\\s+want\\s+to\\s+disappear\\s+forever"],
            priority=10,
            response="low",
        ),
        "high_short": _fake_topic(["i\\s+want"], priority=100, response="high"),
    }
    monkeypatch.setattr(crisis_router, "_load_crisis_templates", lambda: fake_templates)

    match = match_crisis_disclosure("I want to disappear forever", "en")

    assert match is not None
    assert match.topic_id == "high_short"


def test_same_priority_uses_longest_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-priority crisis collisions should prefer the longest matched text."""
    fake_templates = {
        "short": _fake_topic(["i\\s+want"], priority=100, response="short"),
        "long": _fake_topic(["i\\s+want\\s+to\\s+disappear"], priority=100, response="long"),
    }
    monkeypatch.setattr(crisis_router, "_load_crisis_templates", lambda: fake_templates)

    match = match_crisis_disclosure("I want to disappear", "en")

    assert match is not None
    assert match.topic_id == "long"


def test_same_priority_uses_longest_pattern_not_longest_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regex span length should not outrank the longest matched pattern string."""
    fake_templates = {
        "short_pattern_long_span": _fake_topic(["i.*die"], priority=100, response="short"),
        "long_pattern_short_span": _fake_topic(
            ["want\\s+to\\s+die"],
            priority=100,
            response="long",
        ),
    }
    monkeypatch.setattr(crisis_router, "_load_crisis_templates", lambda: fake_templates)

    match = match_crisis_disclosure("I really really want to die", "en")

    assert match is not None
    assert match.topic_id == "long_pattern_short_span"


def test_same_priority_and_length_uses_distinct_match_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct matched pattern count should break remaining crisis ties."""
    fake_templates = {
        "one": _fake_topic(["disappear"], priority=100, response="one"),
        "two": _fake_topic(["disappear", "i\\s+want"], priority=100, response="two"),
    }
    monkeypatch.setattr(crisis_router, "_load_crisis_templates", lambda: fake_templates)

    match = match_crisis_disclosure("I want to disappear", "en")

    assert match is not None
    assert match.topic_id == "two"


@pytest.mark.parametrize(
    "text,language",
    [
        ("불꽃놀이", "ko"),
        ("fireworks", "en"),
        ("장난감을 잃어버렸어", "ko"),
        ("I feel lost in math", "en"),
        ("주사 맞았어", "ko"),
        ("공룡이 죽었어", "ko"),
        ("게임에서 졌어", "ko"),
        ("친구랑 놀자", "ko"),
        ("친구랑 놀았어", "ko"),
        ("같이 놀고 싶어", "ko"),
        ("친구랑 장난쳤어", "ko"),
        ("친구랑 놀이터 갔어", "ko"),
        ("불이 나는 이유가 뭐야", "ko"),
        ("불이 나는 꿈을 꿨어", "ko"),
        ("바비큐에서 연기 나", "ko"),
        ("내 책이 어디인지 모르겠어", "ko"),
        ("정답 맞았어", "ko"),
        ("축구공에 맞았어", "ko"),
        ("엄마가 사진 보내 달래", "ko"),
        ("가족 사진 보내달래", "ko"),
        ("mom asked me to send a photo", "en"),
        ("집에 불 켜줘", "ko"),
        ("불 꺼줘", "ko"),
        ("방에 불 켜는 법", "ko"),
        ("불빛이 예뻐", "ko"),
        ("연기 나쁘다", "ko"),
        ("담배 연기 나", "ko"),
        ("향 연기 나", "ko"),
        ("드라이아이스 연기 나", "ko"),
        ("수영장에서 연기 나", "ko"),
        ("고기 굽는 연기 나", "ko"),
        ("고기 굽고 연기 나", "ko"),
        ("요리하다가 연기 나", "ko"),
        ("안개 연기 나", "ko"),
        ("스모그 연기 나", "ko"),
        ("the bbq smoke is coming", "en"),
        ("smoke is coming from the grill", "en"),
        ("smoke is coming out of the chimney", "en"),
        ("incense smoke is coming", "en"),
        ("dry ice smoke is coming", "en"),
        ("fog smoke is coming", "en"),
        ("vape smoke is coming", "en"),
        ("cooking smoke is coming", "en"),
        ("게임에서 길을 잃었어", "ko"),
        ("미로에서 길을 잃었어", "ko"),
        ("꿈에서 길을 잃었어", "ko"),
        ("꿈에 길을 잃었어", "ko"),
        ("마인크래프트에서 길 잃었어", "ko"),
        ("로블록스에서 길을 잃었어", "ko"),
        ("스토리에서 길을 잃었어", "ko"),
        ("이야기에서 길을 잃었어", "ko"),
        ("만화에서 길을 잃었어", "ko"),
        ("애니에서 길을 잃었어", "ko"),
        ("i got lost in the game", "en"),
        ("i am lost in the story", "en"),
        ("i got lost in the book", "en"),
        ("i got lost in the maze", "en"),
        ("i got lost in a dream", "en"),
        ("i got lost in the dream", "en"),
        ("i got lost in a game", "en"),
        ("게임에서 아빠가 때려", "ko"),
        ("놀이에서 아빠가 때려", "ko"),
        ("게임에서 아빠가 발로 찼어", "ko"),
        ("게임에서 선생님이 나 밀쳤어", "ko"),
        ("아빠가 나를 차에 태웠어", "ko"),
        ("아빠가 축구공을 찼어", "ko"),
        ("아빠가 벨트를 찼어", "ko"),
        ("선생님이 문을 밀었어", "ko"),
        ("엄마가 반죽을 꼬집었어", "ko"),
        ("아빠가 모기를 때려", "ko"),
        ("시험에서 내가 맞았어", "ko"),
        ("시험 100점 맞았어", "ko"),
        ("문제 맞았어", "ko"),
        ("퀴즈 맞았어", "ko"),
        ("in a game daddy hit me", "en"),
        ("my dad hit me in a game", "en"),
        ("in a game my dad pushed me", "en"),
        ("my dad pushed me in a game", "en"),
        ("my dad pushed the cart", "en"),
        ("my dad shoved the door", "en"),
        ("my mom pinched the dough", "en"),
        ("my dad pushed me on the swing", "en"),
        ("장난감을 만졌어", "ko"),
        ("핸드폰 만졌어", "ko"),
        ("아저씨가 책을 만졌어", "ko"),
        ("i want to die laughing", "en"),
        ("자살골 넣었어", "ko"),
        ("자살률", "ko"),
        ("캐릭터가 자살했어", "ko"),
        ("자살 뉴스 봤어", "ko"),
        ("자살 사건 뉴스 봤어", "ko"),
        ("자살 통계 알려줘", "ko"),
        ("자살이 뭐야", "ko"),
        ("자살 예방 알려줘", "ko"),
        ("게임 캐릭터가 죽어버릴까", "ko"),
        ("게임에서 집 나가고 싶어", "ko"),
        ("i am lost without my phone", "en"),
        ("i am lost without you", "en"),
        ("엄마 없으면 안 돼", "ko"),
        ("길을 잃었어 너 없으면 안 돼", "ko"),
    ],
)
def test_near_miss_corpus_has_zero_119_112_false_positives(
    text: str,
    language: str,
) -> None:
    """Benign near-misses should not route through crisis disclosure handling."""
    match = match_crisis_disclosure(text, language)

    assert match is None


# ---------------------------------------------------------------------------
# ADR 0114 — fail-open closure (exclude fiction-gate + hard-disclosure tier).
#
# Root cause: ``request_excludes_*`` were matched as bare, unanchored
# substrings, so a genuine first-person crisis disclosure that merely contained
# a common noun (``phone`` / ``homework`` / ``이야기`` / ``숙제``) was silently
# suppressed -> fail-open. The 2-tier design adds ``hard_disclosure_patterns_*``
# (always escalate, bypassing plain request_excludes) and ``fiction_excludes_*``
# (suppress even a hard disclosure, for genuine fiction/idiom frames). The three
# groups below are the safety contract for this fix and must all hold.
# ---------------------------------------------------------------------------

ADR0114_MUST_FIRE_CASES = (
    # The four originally-reproduced fail-open cases.
    ("he touched my private parts and showed me stuff on his phone", "en", "abuse_sexual"),
    (
        "a stranger online asked me to keep our secret and help with my homework",
        "en",
        "grooming",
    ),
    ("엄마한테 이야기했는데 그래도 죽고 싶어", "ko", "suicidal_intent"),
    ("숙제 때문에 사라지고 싶어", "ko", "self_harm"),
    # Additional first-person disclosures carrying an incidental common noun.
    ("이모가 내 몸을 만졌어", "ko", "abuse_sexual"),
    ("i don't want to live anymore and my mom took my phone", "en", "suicidal_intent"),
    ("삼촌이 핸드폰 보면서 내 몸을 만졌어", "ko", "abuse_sexual"),
    ("게임 끝나고 나서 죽고 싶어", "ko", "suicidal_intent"),
    ("숙제 하다가 자해하고 싶어", "ko", "self_harm"),
    ("someone touched my private parts while i was playing a game", "en", "abuse_sexual"),
    ("i want to hurt myself because of my homework", "en", "self_harm"),
)

ADR0114_MUST_NOT_FIRE_FICTION_CASES = (
    # Fiction / idiom / philosophical / news frames stay suppressed even though
    # they sit next to crisis-shaped vocabulary.
    ("게임에서 죽고 싶어", "ko"),
    ("무서운 이야기 들려줘", "ko"),
    ("왜 사람은 죽어야 해?", "ko"),
    ("자살 뉴스 봤어", "ko"),
    ("so mad i could die", "en"),
    ("die laughing", "en"),
    ("tell me a story about a phone", "en"),
    ("in a game my character wants to disappear", "en"),
)

ADR0114_NORMAL_PLAY_CASES = (
    # Ordinary child speech that incidentally contains an exclude noun.
    ("공룡 이야기 해줘", "ko"),
    ("숙제 다 했어", "ko"),
    ("엄마가 새 핸드폰 사줬어", "ko"),
    ("그림 그리고 싶어", "ko"),
)


@pytest.mark.parametrize(("text", "language", "expected_topic"), ADR0114_MUST_FIRE_CASES)
def test_adr0114_first_person_disclosure_with_incidental_noun_fires(
    text: str,
    language: str,
    expected_topic: str,
) -> None:
    """Group A: genuine first-person disclosures must escalate (fail-open closed)."""
    _assert_crisis_match(text, language, expected_topic)


@pytest.mark.parametrize(("text", "language"), ADR0114_MUST_NOT_FIRE_FICTION_CASES)
def test_adr0114_fiction_idiom_news_frames_stay_suppressed(text: str, language: str) -> None:
    """Group B: fiction/idiom/philosophical/news frames must not fire."""
    assert match_crisis_disclosure(text, language) is None


@pytest.mark.parametrize(("text", "language"), ADR0114_NORMAL_PLAY_CASES)
def test_adr0114_normal_child_play_does_not_fire(text: str, language: str) -> None:
    """Group C: normal child play with an incidental exclude noun must not fire."""
    assert match_crisis_disclosure(text, language) is None
