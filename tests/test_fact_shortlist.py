"""Tests for Phase 0 confirmable-fact shortlist grounding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import core.fact_shortlist as fact_shortlist
from core.fact_shortlist import FactEntry, FactMatch, get_fact_entries, match_fact
from core.llm_backend_config import LLMBackendConfig
from scripts.run_phase0_ab_harness import (
    MockBackend,
    determine_placement_winner,
    load_shortlist_topics,
    parse_placements,
    run_phase0_ab_experiment,
    validate_holdout_against_shortlist,
)
from scripts.score_fact_holdout import (
    AgeBand,
    Axis,
    HoldoutRow,
    ResponseRow,
    classify_response,
    load_holdout_rows,
    load_response_rows,
    score_response_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHORTLIST_PATH = REPO_ROOT / "assets" / "prompts" / "confirmable_facts.json"
HOLDOUT_PATH = REPO_ROOT / "assets" / "prompts" / "confirmable_facts_holdout.jsonl"
EXPECTED_TOTAL_SHORTLIST_ENTRIES = 522
EXPECTED_DEFAULT_BAND_ENTRY_COUNT = 412
EXPECTED_HOLDOUT_TOTAL_ROWS = 430
EXPECTED_HOLDOUT_AXIS_COUNTS = {"matched": 328, "unmatched": 102}
EXPECTED_CATEGORIES = {
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
REMOVED_LEGACY_CATEGORIES = (
    "body",
    "space",
    "korean_history",
    "geography",
    "science_basic",
    "general",
)


def _legacy_backend_config() -> LLMBackendConfig:
    return LLMBackendConfig(
        backend="qwen3_legacy",
        model_path="/models/qwen.gguf",
        n_ctx=2048,
        max_tokens=256,
        temperature=0.4,
        n_gpu_layers=99,
    )


def _make_pipeline(monkeypatch: pytest.MonkeyPatch, mode: str | None) -> Any:
    from core.pipeline import ConversationPipeline, PipelineConfig

    if mode is None:
        monkeypatch.delenv("MUNGI_FACT_SHORTLIST", raising=False)
    else:
        monkeypatch.setenv("MUNGI_FACT_SHORTLIST", mode)

    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        return ConversationPipeline(MagicMock(), PipelineConfig())


def _sample_holdout_row(
    *,
    topic: str = "bone_count_adult",
    question: str = "뼈가 몇 개야?",
    gold_answer: str = "206개",
    acceptable_variants: tuple[str, ...] = ("206개",),
    numeric_tolerance: int | None = 0,
    axis: Axis = "matched",
    age_band: AgeBand = "under_10",
) -> HoldoutRow:
    return HoldoutRow(
        topic=topic,
        question=question,
        category="body_health",
        axis=axis,
        gold_answer=gold_answer,
        acceptable_variants=acceptable_variants,
        numeric_tolerance=numeric_tolerance,
        age_band=age_band,
    )


def _sample_shortlist_entry(**overrides: Any) -> dict[str, Any]:
    entry = {
        "topic": "sample_topic",
        "category": "body_health",
        "triggers_ko": ["테스트 트리거"],
        "triggers_en": [],
        "fact_ko": "테스트 사실",
        "fact_en": None,
        "source_pm": "test",
        "numeric_tolerance": 0,
        "age_band": "under_10",
        "confidence_source": "pm_audited",
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    ("topic", "trigger"),
    [(entry.topic, trigger) for entry in get_fact_entries() for trigger in entry.triggers_ko],
)
def test_match_fact_matches_every_shortlist_trigger(topic: str, trigger: str) -> None:
    match = match_fact(trigger, "ko")

    assert match is not None
    assert match.topic == topic
    assert match.matched_trigger in {
        candidate
        for entry in get_fact_entries()
        if entry.topic == topic
        for candidate in entry.triggers_ko
    }


def test_match_fact_prefers_longest_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = fact_shortlist._parse_fact_entries(
        [
            {
                "topic": "short_match",
                "category": "body_health",
                "triggers_ko": ["뼈가 몇 개"],
                "triggers_en": [],
                "fact_ko": "짧은 트리거",
                "fact_en": None,
                "source_pm": "test",
                "numeric_tolerance": 0,
            },
            {
                "topic": "long_match",
                "category": "body_health",
                "triggers_ko": ["사람 몸에서 뼈가 몇 개"],
                "triggers_en": [],
                "fact_ko": "긴 트리거",
                "fact_en": None,
                "source_pm": "test",
                "numeric_tolerance": 0,
            },
        ]
    )
    monkeypatch.setattr(fact_shortlist, "_FACT_ENTRIES", entries)

    match = match_fact("사람 몸에서 뼈가 몇 개 있는지 궁금해.", "ko")

    assert match == FactMatch(
        topic="long_match",
        fact_ko="긴 트리거",
        fact_en=None,
        matched_trigger="사람 몸에서 뼈가 몇 개",
    )


def test_match_fact_breaks_ties_lexicographically(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = fact_shortlist._parse_fact_entries(
        [
            {
                "topic": "alpha_topic",
                "category": "body_health",
                "triggers_ko": ["같은 길이 트리거"],
                "triggers_en": [],
                "fact_ko": "alpha",
                "fact_en": None,
                "source_pm": "test",
                "numeric_tolerance": 0,
            },
            {
                "topic": "zeta_topic",
                "category": "body_health",
                "triggers_ko": ["같은 길이 트리거"],
                "triggers_en": [],
                "fact_ko": "zeta",
                "fact_en": None,
                "source_pm": "test",
                "numeric_tolerance": 0,
            },
        ]
    )
    monkeypatch.setattr(fact_shortlist, "_FACT_ENTRIES", entries)

    match = match_fact("이거 같은 길이 트리거 맞지?", "ko")

    assert match is not None
    assert match.topic == "alpha_topic"


def test_match_fact_is_case_insensitive_and_normalizes_whitespace() -> None:
    match = match_fact("  미국의   수도는   어디  야?  ", "ko")

    assert match is not None
    assert match.topic == "usa_capital"


def test_match_fact_returns_none_for_no_match() -> None:
    assert match_fact("오늘은 무슨 놀이를 할까?", "ko") is None


@pytest.mark.parametrize(
    ("question", "expected_topic", "expected_fragment"),
    [
        ("측우기 뭐야", "cheugugi", "비의 양"),
        ("임진왜란 설명해줘", "imjin_war_about", "1592"),
        ("임진왜란 몇 년", "imjin_war_start", "1592"),
        ("조선 첫 왕이 누구야", "joseon_first_king", "태조 이성계"),
        ("이순신 누구야", "yi_sunsin", "이순신"),
        ("세종대왕 누구야", "sejong_who", "조선의 네 번째 임금"),
        ("훈민정음 뭐야", "hunminjeongeum", "1443"),
        ("거북선 뭐야", "geobukseon", "거북선"),
        ("광개토대왕 누구야", "gwanggaeto", "고구려"),
    ],
)
def test_match_fact_matches_curated_korean_history_facts(
    question: str,
    expected_topic: str,
    expected_fragment: str,
) -> None:
    match = match_fact(question, "ko")

    assert match is not None
    assert match.topic == expected_topic
    assert expected_fragment in match.fact_ko


def test_cheugugi_fact_describes_rainfall_amount_not_depth() -> None:
    match = match_fact("측우기 뭐야", "ko")

    assert match is not None
    assert "비의 양" in match.fact_ko
    assert "깊이" not in match.fact_ko


def test_ungrounded_korean_history_query_returns_no_shortlist_match() -> None:
    assert match_fact("갑오개혁 뭐야", "ko") is None


def test_match_fact_en_returns_none_when_triggers_are_empty() -> None:
    assert match_fact("What is the capital of the United States?", "en") is None


def test_get_fact_entries_loads_all_topics() -> None:
    entries = get_fact_entries()

    assert len(entries) == EXPECTED_DEFAULT_BAND_ENTRY_COUNT
    assert len({entry.topic for entry in entries}) == EXPECTED_DEFAULT_BAND_ENTRY_COUNT
    assert {entry.age_band for entry in entries} == {"under_10"}
    assert {entry.confidence_source for entry in entries} == {"pm_audited"}


def test_shortlist_schema_and_duplicates_are_valid() -> None:
    payload = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    entries = fact_shortlist._parse_fact_entries(payload)

    assert len(payload) == EXPECTED_TOTAL_SHORTLIST_ENTRIES
    assert len(entries) == EXPECTED_TOTAL_SHORTLIST_ENTRIES
    assert all(row["age_band"] in {"under_10", "under_15"} for row in payload)
    assert all(row["confidence_source"] == "pm_audited" for row in payload)
    assert all(isinstance(entry, FactEntry) for entry in entries)
    assert {entry.category for entry in entries} <= EXPECTED_CATEGORIES
    assert all(entry.fact_ko for entry in entries)
    assert all(entry.triggers_ko for entry in entries)


def test_shortlist_payload_topics_are_unique() -> None:
    payload = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    topics = [row["topic"] for row in payload]

    assert len(topics) == len(set(topics)) == EXPECTED_TOTAL_SHORTLIST_ENTRIES


def test_shortlist_payload_covers_all_category_age_band_cells() -> None:
    payload = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    covered_cells = {(row["category"], row["age_band"]) for row in payload}

    assert len(covered_cells) == len(EXPECTED_CATEGORIES) * 2
    assert {category for category, _age_band in covered_cells} == EXPECTED_CATEGORIES


def test_shortlist_payload_band_distribution_stays_within_target_range() -> None:
    payload = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    band_counts = Counter(row["age_band"] for row in payload)
    under_10_ratio = band_counts["under_10"] / len(payload)

    assert band_counts == {"under_10": 412, "under_15": 110}
    assert 0.70 <= under_10_ratio <= 0.90


def test_shortlist_payload_numeric_tolerance_values_are_int_or_none() -> None:
    payload = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))

    assert all(
        row["numeric_tolerance"] is None or isinstance(row["numeric_tolerance"], int)
        for row in payload
    )


def test_duplicate_topic_detection_raises() -> None:
    payload = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    duplicated_payload = [payload[0], payload[0]]

    with pytest.raises(ValueError, match="Duplicate shortlist topic"):
        fact_shortlist._parse_fact_entries(duplicated_payload)


def test_empty_trigger_array_raises() -> None:
    payload: list[dict[str, Any]] = [
        {
            "topic": "empty_trigger",
            "category": "body_health",
            "triggers_ko": [],
            "triggers_en": [],
            "fact_ko": "테스트",
            "fact_en": None,
            "source_pm": "test",
            "numeric_tolerance": 0,
        }
    ]

    with pytest.raises(ValueError, match="must not be empty"):
        fact_shortlist._parse_fact_entries(payload)


def test_parse_fact_entries_accepts_all_new_categories() -> None:
    payload = [
        _sample_shortlist_entry(topic=f"topic_{index}", category=category)
        for index, category in enumerate(sorted(EXPECTED_CATEGORIES), start=1)
    ]

    entries = fact_shortlist._parse_fact_entries(payload)

    assert {entry.category for entry in entries} == EXPECTED_CATEGORIES


@pytest.mark.parametrize("category", REMOVED_LEGACY_CATEGORIES)
def test_parse_fact_entries_rejects_removed_legacy_categories(category: str) -> None:
    payload = [_sample_shortlist_entry(category=category)]

    with pytest.raises(ValueError, match="unsupported category"):
        fact_shortlist._parse_fact_entries(payload)


def test_parse_fact_entries_accepts_new_age_bands() -> None:
    entries = fact_shortlist._parse_fact_entries(
        [
            _sample_shortlist_entry(topic="under_10_topic", age_band="under_10"),
            _sample_shortlist_entry(topic="under_15_topic", age_band="under_15"),
        ]
    )

    assert [entry.age_band for entry in entries] == ["under_10", "under_15"]


def test_parse_fact_entries_accepts_valid_confidence_sources() -> None:
    entries = fact_shortlist._parse_fact_entries(
        [
            _sample_shortlist_entry(topic="pm_audited_topic", confidence_source="pm_audited"),
            _sample_shortlist_entry(
                topic="wikidata_verified_topic",
                confidence_source="wikidata_verified",
            ),
            _sample_shortlist_entry(
                topic="keep_pool_seeded_topic",
                confidence_source="keep_pool_seeded",
            ),
        ]
    )

    assert [entry.confidence_source for entry in entries] == [
        "pm_audited",
        "wikidata_verified",
        "keep_pool_seeded",
    ]


def test_parse_fact_entries_defaults_confidence_source_to_pm_audited() -> None:
    payload = _sample_shortlist_entry()
    payload.pop("confidence_source")

    entries = fact_shortlist._parse_fact_entries([payload])

    assert entries[0].confidence_source == "pm_audited"


def test_parse_fact_entries_rejects_invalid_confidence_source() -> None:
    payload = [_sample_shortlist_entry(confidence_source="unverified")]

    with pytest.raises(ValueError, match="field confidence_source must be one of"):
        fact_shortlist._parse_fact_entries(payload)


def test_holdout_rows_all_match_their_topic() -> None:
    rows = load_holdout_rows(HOLDOUT_PATH)
    raw_rows = [
        json.loads(line)
        for line in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    axis_counts = Counter(row.axis for row in rows)

    assert len(rows) == EXPECTED_HOLDOUT_TOTAL_ROWS
    assert len(raw_rows) == EXPECTED_HOLDOUT_TOTAL_ROWS
    assert dict(axis_counts) == EXPECTED_HOLDOUT_AXIS_COUNTS
    assert all(row["age_band"] in {"under_10", "under_15"} for row in raw_rows)
    for row in rows:
        if row.axis != "matched" or row.age_band != "under_10":
            continue
        match = match_fact(row.question, "ko")
        assert match is not None
        assert match.topic == row.topic


def test_holdout_rows_cover_all_shortlist_cells_via_matched_axis() -> None:
    shortlist_topics = load_shortlist_topics(SHORTLIST_PATH)
    holdout_rows = load_holdout_rows(HOLDOUT_PATH)
    matched_cells = {(row.category, row.age_band) for row in holdout_rows if row.axis == "matched"}
    shortlist_cells = {(topic.category, topic.age_band) for topic in shortlist_topics.values()}

    assert matched_cells == shortlist_cells


def test_holdout_unmatched_rows_use_topics_outside_shortlist() -> None:
    shortlist_topics = load_shortlist_topics(SHORTLIST_PATH)
    holdout_rows = load_holdout_rows(HOLDOUT_PATH)

    assert all(row.topic not in shortlist_topics for row in holdout_rows if row.axis == "unmatched")


def test_validate_holdout_against_shortlist_allows_unmatched_topics() -> None:
    shortlist_topics = {
        "matched_topic": load_shortlist_topics(SHORTLIST_PATH)["bone_count_adult"],
    }
    holdout_rows = [
        _sample_holdout_row(topic="matched_topic"),
        _sample_holdout_row(
            topic="outside_topic",
            question="이건 목록 밖 질문이야?",
            axis="unmatched",
            numeric_tolerance=None,
        ),
    ]

    validate_holdout_against_shortlist(holdout_rows, shortlist_topics)


def test_validate_holdout_against_shortlist_rejects_missing_matched_cell() -> None:
    shortlist_topics = {
        "u10_topic": load_shortlist_topics(SHORTLIST_PATH)["bone_count_adult"],
        "u15_topic": load_shortlist_topics(SHORTLIST_PATH)["vertebrate_classes"],
    }
    holdout_rows = [_sample_holdout_row(topic="u10_topic")]

    with pytest.raises(ValueError, match="Matched holdout missing shortlist cells"):
        validate_holdout_against_shortlist(holdout_rows, shortlist_topics)


def test_validate_holdout_against_shortlist_rejects_age_band_mismatch() -> None:
    shortlist_topics = {
        "topic": load_shortlist_topics(SHORTLIST_PATH)["bone_count_adult"],
        "other": load_shortlist_topics(SHORTLIST_PATH)["tooth_count_adult"],
    }
    holdout_rows = [
        _sample_holdout_row(topic="other", question="영구치는 몇 개?", age_band="under_10"),
        _sample_holdout_row(topic="topic", age_band="under_15"),
    ]

    with pytest.raises(ValueError, match="Holdout age_band mismatch"):
        validate_holdout_against_shortlist(holdout_rows, shortlist_topics)


def test_disabled_mode_keeps_prompt_unchanged_vs_explicit_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _make_pipeline(monkeypatch, "disabled")

    messages = pipeline._build_messages("뼈가 몇 개야?", detected_language="ko")

    assert all(not message["content"].startswith("[참고 정보] ") for message in messages[1:])
    assert messages[-1] == {"role": "user", "content": "뼈가 몇 개야?"}


def test_default_mode_is_p2(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch, None)

    messages = pipeline._build_messages("미국 수도가 어디야?", detected_language="ko")

    assert messages[-2]["role"] == "user"
    assert messages[-2]["content"].startswith("[참고 정보] ")
    assert messages[-1] == {"role": "user", "content": "미국 수도가 어디야?"}


def test_p1_injects_fact_after_safety_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch, "p1")
    pipeline._pending_safety_guide = "물가에서는 어른과 함께 있어."

    messages = pipeline._build_messages("뼈가 몇 개야?", detected_language="ko")
    system_prompt = messages[0]["content"]

    safety_index = system_prompt.rfind("[안전 가이드]")
    fact_index = system_prompt.rfind("[참고 정보] 성인의 뼈")

    assert safety_index != -1
    assert fact_index != -1
    assert safety_index < fact_index


def test_p1_unmatched_turn_does_not_emit_fact_block(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch, "p1")

    messages = pipeline._build_messages("오늘은 뭐 하고 놀까?", detected_language="ko")

    assert "[참고 정보] 성인의 뼈" not in messages[0]["content"]


def test_p2_injects_synthetic_user_message_before_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch, "p2")

    messages = pipeline._build_messages("미국 수도가 어디야?", detected_language="ko")

    assert messages[-2]["role"] == "user"
    assert messages[-2]["content"].startswith("[참고 정보] ")
    assert messages[-1] == {"role": "user", "content": "미국 수도가 어디야?"}


def test_p2_unmatched_turn_does_not_emit_synthetic_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _make_pipeline(monkeypatch, "p2")

    messages = pipeline._build_messages("오늘 기분이 어때?", detected_language="ko")

    assert (
        len([message for message in messages[1:] if message["content"].startswith("[참고 정보] ")])
        == 0
    )


def test_invalid_fact_shortlist_flag_raises_at_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.pipeline import ConversationPipeline, PipelineConfig

    monkeypatch.setenv("MUNGI_FACT_SHORTLIST", "p3")
    with patch("core.pipeline.LLMBackendConfig.load", return_value=_legacy_backend_config()):
        with pytest.raises(ValueError, match="Unsupported MUNGI_FACT_SHORTLIST value"):
            ConversationPipeline(MagicMock(), PipelineConfig())


def test_classify_response_matches_textual_variant() -> None:
    row = _sample_holdout_row(
        gold_answer="정답값",
        acceptable_variants=("성인의 뼈는 206개",),
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(row, "성인의 뼈는 206개야.", injection_state="on")

    assert verdict == "correct"
    assert matched_variant == "성인의 뼈는 206개"


def test_classify_response_accepts_paraphrase_content_overlap() -> None:
    row = _sample_holdout_row(
        topic="day_night_rotation",
        question="낮과 밤은 왜 생겨?",
        gold_answer="낮과 밤은 지구가 스스로 도는 자전 때문에 생겨",
        acceptable_variants=(),
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(
        row,
        "지구가 스스로 돌기 때문에 낮과 밤이 생기는 거야.",
        injection_state="on",
    )

    assert verdict == "correct"
    assert matched_variant == "낮과 밤은 지구가 스스로 도는 자전 때문에 생겨"


@pytest.mark.parametrize(
    ("gold_answer", "response"),
    [
        (
            "원자는 물질을 이루는 가장 기본이 되는 작은 알갱이야",
            "원자는 물질을 이루는 아주 작은 알갱이야.",
        ),
        (
            "나침반은 자석의 성질을 이용해 방향을 찾는 도구야",
            "나침반은 자석이 자기 힘을 이용해서 북쪽 방향을 찾아주는 거야.",
        ),
        (
            "산업혁명은 18세기 무렵 기계로 물건을 대량 생산하기 시작한 큰 변화야",
            "산업혁명은 기계로 물건을 많이 만들게 된 큰 변화야.",
        ),
    ],
)
def test_classify_response_accepts_phase_a3_content_overlap_regressions(
    gold_answer: str,
    response: str,
) -> None:
    row = _sample_holdout_row(
        gold_answer=gold_answer,
        acceptable_variants=(),
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(row, response, injection_state="on")

    assert verdict == "correct"
    assert matched_variant == gold_answer


@pytest.mark.parametrize(
    ("gold_answer", "response", "numeric_tolerance"),
    [
        (
            "원자는 물질을 이루는 가장 기본이 되는 작은 알갱이야.",
            "원자는 물질을 이루는 아주 작은 알갱이야.",
            None,
        ),
        (
            "옛 우리나라에는 고구려·백제·신라 세 나라가 있던 삼국시대가 있었어.",
            "삼국시대는 고구려, 백제, 신라 세 나라가 있던 아주 오래된 시대야!",
            0,
        ),
    ],
)
def test_classify_response_accepts_phase_a3_corrective_examples(
    gold_answer: str,
    response: str,
    numeric_tolerance: int | None,
) -> None:
    row = _sample_holdout_row(
        gold_answer=gold_answer,
        acceptable_variants=(),
        numeric_tolerance=numeric_tolerance,
    )

    verdict, matched_variant = classify_response(row, response, injection_state="on")

    assert verdict == "correct"
    assert matched_variant == gold_answer


@pytest.mark.parametrize(
    ("gold_answer", "response", "numeric_tolerance"),
    [
        (
            "지구의 중력은 물건을 아래쪽으로 끌어당겨.",
            "중력은 물건을 아래로 잡아당기는 힘이야! 지구는 우리를 꽉 붙잡아 주는 거지.",
            None,
        ),
        (
            "자동차는 기름으로 가는 차와 전기로 가는 전기차로 나눌 수 있어.",
            "전기차는 전기로 움직여서 기름을 쓰지 않아. "
            "기름 차는 휘발유나 경유 같은 기름으로 힘을 얻어 움직이는 거야!",
            None,
        ),
        (
            "동화는 어린이를 위해 지은 이야기야.",
            "동화는 어린이를 위한 재미있는 이야기야. 상상력을 키우고 좋은 것을 배우게 해주는 거지!",
            None,
        ),
        (
            "전래동화는 옛날부터 입에서 입으로 전해 내려온 이야기야.",
            "전래동화는 옛날 사람들이 입에서 입으로 이야기해 온 재미있는 이야기야!",
            None,
        ),
        (
            "한글(훈민정음)은 조선의 세종대왕이 만들었어.",
            "세종대왕이 한글을 만들었어. 세종대왕은 정말 멋진 분이야!",
            None,
        ),
        (
            "인터넷은 전 세계의 컴퓨터를 서로 연결한 큰 망이야.",
            "인터넷은 전 세계 컴퓨터들을 연결해주는 아주 큰 길이야. "
            "거기서 많은 정보를 주고받을 수 있어!",
            None,
        ),
        (
            "아이가 부모를 닮는 건 부모에게서 특징을 물려받기 때문이야.",
            "부모 닮아서 특징이 나타나는 거야. 부모한테서 멋진 점을 물려받은 거지!",
            None,
        ),
        (
            "전기는 전등·텔레비전 같은 물건을 움직이게 해.",
            "전기는 전등이나 텔레비전을 움직이게 하는 거야. "
            "전기 기술자는 이런 걸 만들고 고치는 일을 해!",
            None,
        ),
        (
            "전기는 잘못 만지면 위험하니 콘센트를 함부로 만지면 안 돼.",
            "전기는 뜨거울 수 있고 감전 위험이 있어. 콘센트는 절대 함부로 만지지 마야.",
            None,
        ),
        (
            "전시회는 작품을 사람들이 볼 수 있게 모아 보여 주는 행사야.",
            "전시회는 사람들이 멋진 작품들을 구경할 수 있게 모아 놓은 거야.",
            None,
        ),
    ],
)
def test_classify_response_documents_phase_a3_known_false_negatives(
    gold_answer: str,
    response: str,
    numeric_tolerance: int | None,
) -> None:
    row = _sample_holdout_row(
        gold_answer=gold_answer,
        acceptable_variants=(),
        numeric_tolerance=numeric_tolerance,
    )

    verdict, matched_variant = classify_response(row, response, injection_state="on")

    assert verdict == "confident-fabrication"
    assert matched_variant is None


def test_classify_response_honors_numeric_tolerance() -> None:
    row = _sample_holdout_row(gold_answer="206개", acceptable_variants=(), numeric_tolerance=5)

    in_band = classify_response(row, "약 203개 정도야.", injection_state="on")
    out_of_band = classify_response(row, "200개야.", injection_state="on")

    assert in_band == ("correct", "206개")
    assert out_of_band == ("confident-fabrication", None)


@pytest.mark.parametrize(
    ("gold_answer", "response"),
    [
        ("사람의 콩팥(신장)은 2개야", "사람 콩팥은 두 개야."),
        ("벌은 날개가 6개야", "벌 날개는 여섯 개야."),
        ("거미 다리는 8개야", "거미 다리는 여덟이야."),
        ("사람 목뼈는 7개, 등뼈는 12개, 허리뼈는 5개야", "등뼈는 열두 개야."),
        ("척추뼈는 27개 정도야", "척추뼈는 스물일곱 개야."),
    ],
)
def test_classify_response_accepts_korean_numeral_words(
    gold_answer: str,
    response: str,
) -> None:
    row = _sample_holdout_row(
        gold_answer=gold_answer,
        acceptable_variants=(),
        numeric_tolerance=0,
    )

    verdict, matched_variant = classify_response(row, response, injection_state="on")

    assert verdict == "correct"
    assert matched_variant == gold_answer


@pytest.mark.parametrize(
    "response",
    [
        "모르겠어.",
        "잘 모르겠어요.",
        "확실하지 않아서 말하기 어려워.",
        "어른에게 물어보자.",
        "확실히 모르겠어.",
        "기억이 안 나.",
    ],
)
def test_classify_response_detects_deference_phrases(response: str) -> None:
    row = _sample_holdout_row()

    assert classify_response(row, response, injection_state="off")[0] == "appropriate-deference"
    assert classify_response(row, response, injection_state="on")[0] == "inappropriate-deference"


def test_classify_response_handles_mixed_numeric_text() -> None:
    row = _sample_holdout_row(gold_answer="206개", acceptable_variants=(), numeric_tolerance=10)

    verdict, matched_variant = classify_response(
        row,
        "약 200개 정도예요.",
        injection_state="on",
    )

    assert verdict == "correct"
    assert matched_variant == "206개"


def test_classify_response_returns_other_for_non_answer() -> None:
    row = _sample_holdout_row()

    verdict, matched_variant = classify_response(
        row,
        "같이 찾아보자.",
        injection_state="on",
    )

    assert verdict == "other"
    assert matched_variant is None


def test_classify_response_treats_unmatched_deference_as_correct() -> None:
    row = _sample_holdout_row(
        topic="outside_topic",
        question="숏리스트 밖 질문",
        axis="unmatched",
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(
        row,
        "잘 모르겠어.",
        injection_state="on",
    )

    assert verdict == "correct"
    assert matched_variant is None


def test_classify_response_treats_unmatched_assertion_as_false_match() -> None:
    row = _sample_holdout_row(
        topic="outside_topic",
        question="숏리스트 밖 질문",
        axis="unmatched",
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(
        row,
        "정답은 42개야.",
        injection_state="on",
    )

    assert verdict == "confident-fabrication"
    assert matched_variant is None


def test_classify_response_rejects_canberra_sydney_false_positive() -> None:
    row = _sample_holdout_row(
        gold_answer="캔버라 (시드니·멜버른은 수도 아님)",
        acceptable_variants=(),
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(
        row,
        "호주 수도는 시드니야. 거기 정말 멋진 곳이야!",
        injection_state="on",
    )

    assert verdict == "confident-fabrication"
    assert matched_variant is None


@pytest.mark.parametrize(
    ("gold_answer", "response"),
    [
        (
            "미술에는 시대마다 인상주의처럼 서로 다른 흐름(사조)이 있었어.",
            "미술 사조가 뭐야? 뭉이가 재미있는 걸 알려줄게. 인상주의 같은 거 알아볼까?",
        ),
        (
            "전래동화는 옛날부터 입에서 입으로 전해 내려온 이야기야.",
            "어떤 전래동화 말이야? 어떤 이야기인지 알려주면 뭉이가 설명해줄게!",
        ),
    ],
)
def test_classify_response_rejects_deflection_overlap_false_positives(
    gold_answer: str,
    response: str,
) -> None:
    row = _sample_holdout_row(
        gold_answer=gold_answer,
        acceptable_variants=(),
        numeric_tolerance=None,
    )

    verdict, matched_variant = classify_response(row, response, injection_state="on")

    assert verdict != "correct"
    assert matched_variant is None


@pytest.mark.parametrize(
    ("gold_answer", "response"),
    [
        ("사람의 콩팥(신장)은 2개야", "사람 콩팥은 3개야."),
        ("고래는 포유류야", "고래는 물고기야."),
        ("벌은 날개가 6개야", "벌은 다리가 8개야."),
    ],
)
def test_classify_response_rejects_low_overlap_fabrications(
    gold_answer: str,
    response: str,
) -> None:
    row = _sample_holdout_row(
        gold_answer=gold_answer,
        acceptable_variants=(),
        numeric_tolerance=0,
    )

    verdict, matched_variant = classify_response(row, response, injection_state="on")

    assert verdict == "confident-fabrication"
    assert matched_variant is None


def test_score_response_rows_rejects_missing_response() -> None:
    holdout_rows = [_sample_holdout_row()]

    with pytest.raises(ValueError, match="Missing response row"):
        score_response_rows(holdout_rows, [], injection_state="off")


def test_score_response_rows_summarizes_all_verdicts() -> None:
    holdout_rows = [
        _sample_holdout_row(topic="t1", question="q1", gold_answer="206개"),
        _sample_holdout_row(topic="t2", question="q2", gold_answer="206개"),
        _sample_holdout_row(topic="t3", question="q3", gold_answer="206개"),
        _sample_holdout_row(topic="t4", question="q4", gold_answer="206개"),
        _sample_holdout_row(topic="t5", question="q5", gold_answer="206개"),
    ]
    response_rows = [
        ResponseRow(topic="t1", question="q1", response="206개야."),
        ResponseRow(topic="t2", question="q2", response="모르겠어."),
        ResponseRow(topic="t3", question="q3", response="200개야."),
        ResponseRow(topic="t4", question="q4", response="같이 찾아보자."),
        ResponseRow(topic="t5", question="q5", response="206개 정도야."),
    ]

    scored_rows, summary = score_response_rows(holdout_rows, response_rows, injection_state="off")

    assert len(scored_rows) == 5
    assert summary["verdict_counts"] == {
        "correct": 2,
        "appropriate-deference": 1,
        "inappropriate-deference": 0,
        "confident-fabrication": 1,
        "other": 1,
    }


def test_score_response_rows_adds_age_band_breakdown() -> None:
    holdout_rows = [
        _sample_holdout_row(topic="t1", question="q1", age_band="under_10"),
        _sample_holdout_row(
            topic="t2",
            question="q2",
            age_band="under_15",
            gold_answer="정답값",
            acceptable_variants=("정답값",),
            numeric_tolerance=None,
        ),
    ]
    response_rows = [
        ResponseRow(topic="t1", question="q1", response="206개야."),
        ResponseRow(topic="t2", question="q2", response="모르겠어."),
    ]

    _scored_rows, summary = score_response_rows(holdout_rows, response_rows, injection_state="off")

    assert summary["age_band_breakdown"]["under_10"]["verdict_counts"]["correct"] == 1
    assert summary["age_band_breakdown"]["under_15"]["verdict_counts"]["appropriate-deference"] == 1


def test_load_response_rows_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        load_response_rows(path)


def test_load_holdout_rows_rejects_invalid_age_band(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    path.write_text(
        json.dumps(
            {
                "topic": "outside_topic",
                "question": "질문",
                "category": "body_health",
                "axis": "matched",
                "gold_answer": "정답",
                "acceptable_variants": [],
                "numeric_tolerance": None,
                "age_band": "college",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field age_band must be one of"):
        load_holdout_rows(path)


def test_parse_placements_deduplicates_in_order() -> None:
    assert parse_placements("p2,p1,p2") == ["p2", "p1"]


def test_validate_holdout_against_shortlist_accepts_repo_artifacts() -> None:
    shortlist_topics = load_shortlist_topics(SHORTLIST_PATH)
    holdout_rows = load_holdout_rows(HOLDOUT_PATH)

    validate_holdout_against_shortlist(holdout_rows, shortlist_topics)


def test_run_phase0_ab_experiment_with_mock_backend(tmp_path: Path) -> None:
    holdout_rows = [
        HoldoutRow(
            topic="bone_count_adult",
            question="궁금한데 뼈가 몇 개야?",
            category="body_health",
            axis="matched",
            gold_answer="206개 (성인 기준; 신생아는 약 270개)",
            acceptable_variants=("206개", "성인의 뼈는 206개"),
            numeric_tolerance=0,
            age_band="under_10",
        ),
        HoldoutRow(
            topic="usa_capital",
            question="미국 공부 중인데 미국 수도가 어디야?",
            category="world_geography",
            axis="matched",
            gold_answer="워싱턴 D.C. (정식 명칭: District of Columbia)",
            acceptable_variants=("워싱턴 d.c.", "워싱턴 dc"),
            numeric_tolerance=None,
            age_band="under_10",
        ),
    ]
    answers = {
        "궁금한데 뼈가 몇 개야?": "206개야.",
        "미국 공부 중인데 미국 수도가 어디야?": "워싱턴 D.C.야.",
    }
    wrong_answers = {
        "궁금한데 뼈가 몇 개야?": "200개야.",
        "미국 공부 중인데 미국 수도가 어디야?": "시드니야.",
    }

    def responder(messages: list[dict[str, str]]) -> str:
        user_question = messages[-1]["content"]
        has_fact_context = any(
            "성인의 뼈는 206개" in message["content"] or "워싱턴 D.C." in message["content"]
            for message in messages
        )
        return answers[user_question] if has_fact_context else wrong_answers[user_question]

    summary = run_phase0_ab_experiment(
        holdout_rows=holdout_rows,
        backend=MockBackend(responder),
        backend_config=_legacy_backend_config(),
        placements=["p1", "p2"],
        output_dir=tmp_path,
    )

    assert summary["winner"]["verdict"] == "GO"
    assert summary["winner"]["winner"] == "p1"
    assert summary["cells"]["p1_off"]["summary"]["verdict_counts"]["confident-fabrication"] == 2
    assert summary["cells"]["p1_on"]["summary"]["verdict_counts"]["correct"] == 2
    assert summary["cells"]["p2_on"]["summary"]["verdict_counts"]["correct"] == 2
    assert (tmp_path / "phase0_summary.json").exists()
    assert (tmp_path / "placement_winner.json").exists()

    for placement in ("p1", "p2"):
        for injection_state in ("off", "on"):
            response_path = tmp_path / f"responses_{placement}_{injection_state}.jsonl"
            response_rows = [
                json.loads(line)
                for line in response_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            assert response_rows
            assert all("ttft_s" in row for row in response_rows)
            assert all("gen_time_s" in row for row in response_rows)
            assert all(row["ttft_s"] is None for row in response_rows)
            assert all(row["gen_time_s"] is None for row in response_rows)

    scored_rows, cell_summary = score_response_rows(
        holdout_rows,
        load_response_rows(tmp_path / "responses_p1_on.jsonl"),
        injection_state="on",
    )
    assert len(scored_rows) == 2
    assert cell_summary["verdict_counts"]["correct"] == 2


def test_determine_placement_winner_returns_stop_when_none_clear() -> None:
    winner = determine_placement_winner(
        {
            "p1": {"relative_failure_reduction": 0.25, "verdict": "stop"},
            "p2": {"relative_failure_reduction": 0.49, "verdict": "stop"},
        }
    )

    assert winner["verdict"] == "STOP"
    assert winner["winner"] is None
