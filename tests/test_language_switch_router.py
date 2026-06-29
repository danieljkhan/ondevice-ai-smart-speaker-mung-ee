"""Tests for deterministic Korean-English language switch routing."""

from __future__ import annotations

import pytest

from safety.language_switch_router import get_switch_confirmation, match_language_switch

KO_TO_EN_CONFIRMATION = (
    "좋아! 이제 영어로 말할게. 한국어로 돌아오고 싶으면 "
    "한국어로 말해줘 라고 하거나, 오른쪽 위에 있는 한영 전환 단추를 눌러줘!"
)
EN_TO_KO_CONFIRMATION = (
    "좋아! 이제 한국어로 얘기하자! 영어로 하고 싶으면 "
    "영어로 말해줘 라고 하거나, 오른쪽 위에 있는 한영 전환 단추를 눌러줘!"
)


def test_get_switch_confirmation_returns_template_strings() -> None:
    """Confirmation lookup returns the validated template text by target language."""
    assert get_switch_confirmation("en") == (KO_TO_EN_CONFIRMATION, "ko")
    assert get_switch_confirmation("ko") == (EN_TO_KO_CONFIRMATION, "ko")


@pytest.mark.parametrize(
    "text",
    [
        "영어로 말해줘",
        "영어로 말해 줘!",
        "영어로 이야기하자",
        "영어로 얘기하자",
        "뭉이야 영어로 말해줘",
        "영어로 말하자",
        "뭉이야 영어로 말하자",
        "영어로 말해",
        "영어로 대화하자",
        "영어로 하자",
    ],
)
def test_ko_to_en_switch_matches_whole_turn(text: str) -> None:
    """Korean-to-English triggers switch only as whole-turn commands."""
    match = match_language_switch(text, "ko")

    assert match is not None
    assert match.target_language == "en"
    assert match.confirmation_language == "ko"
    assert match.confirmation_text == KO_TO_EN_CONFIRMATION
    assert match.matched_patterns


@pytest.mark.parametrize(
    "text",
    [
        "한국어로 말해줘",
        "우리말로 말해 줘!",
        "한글로 이야기하자",
        "한국어",
        "한국말",
        "한국말 해줘",
        "한국말로 해줘",
        "한국말로 하자",
        "한국어로 해",
        "우리말로 해줘",
        "한국어로 말하자",
        "한국어로 말해",
        "한국으로 말하자",
        "한국으로 말해",
        "한국으로 말해줘",
        "한국으로 얘기하자",
        "한국으로 이야기하자",
        "우리나라말",
        "우리나라말로 말하자",
        "우리말",
        "우리말로 말하자",
        "한글",
        "한글로 말하자",
        "한글 하자",
        "say it in korean",
        "SAY IT IN KOREAN!",
        "let's talk in korean",
    ],
)
def test_en_to_ko_switch_matches_whole_turn(text: str) -> None:
    """English-to-Korean triggers switch only as whole-turn commands."""
    match = match_language_switch(text, "en")

    assert match is not None
    assert match.target_language == "ko"
    assert match.confirmation_language == "ko"
    assert match.confirmation_text == EN_TO_KO_CONFIRMATION
    assert match.matched_patterns


@pytest.mark.parametrize(
    ("text", "current_language"),
    [
        ("영어로 말해줘", "en"),
        ("한국어로 말해줘", "ko"),
        ("say it in korean", "ko"),
    ],
)
def test_directionality_guard_blocks_current_language_repeats(
    text: str,
    current_language: str,
) -> None:
    """Triggers are inert unless they target the opposite session language."""
    assert match_language_switch(text, current_language) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "영어로 손씻기 알려줘",
        "영어로 약 먹어도 돼?",
        "영어로 손 씻는 법 말해줘",
        "영어로 번역해줘",
        "영어 공부",
        "영어 단어 알려줘",
        "영어 시험",
        "what is English",
        "teach me English",
        "한국 어디야?",
        "한국 음식 알려줘",
        "한글 공부",
        "한글 가르쳐줘",
        "우리말 책",
        "i like korean food",
    ],
)
def test_object_bearing_and_study_utterances_do_not_switch(text: str) -> None:
    """Object-bearing and English-study utterances must never switch language."""
    assert match_language_switch(text, "ko") is None
    assert match_language_switch(text, "en") is None


@pytest.mark.parametrize(
    "text",
    [
        "우리나라로 가자",
        "한국으로 가자",
        "한국으로 가고 싶어",
        "한국으로 여행 가자",
    ],
)
def test_korea_location_utterances_do_not_switch(text: str) -> None:
    """Korea-location utterances are not language switch commands."""
    assert match_language_switch(text, "ko") is None
    assert match_language_switch(text, "en") is None


@pytest.mark.parametrize(
    "text",
    [
        "영어로 말해줘",
        "영어로 말해 줘!",
        "영어로 이야기하자",
        "영어로 얘기하자",
        "뭉이야 영어로 말해줘",
    ],
)
def test_ko_to_en_entry_regression_still_matches_valid_forms(text: str) -> None:
    """The stricter Korean-to-English entry command remains unchanged."""
    match = match_language_switch(text, "ko")

    assert match is not None
    assert match.target_language == "en"


@pytest.mark.parametrize(
    "text",
    [
        "영어로 손씻기 알려줘",
        "영어로 약 먹어도 돼?",
        "영어로 손 씻는 법 말해줘",
        "영어로 번역해줘",
        "영어 공부",
        "영어 단어 알려줘",
        "영어 시험",
        "what is English",
        "teach me English",
    ],
)
def test_ko_to_en_entry_regression_preserves_fp_zero(text: str) -> None:
    """The Korean-to-English entry command still rejects its false-positive corpus."""
    assert match_language_switch(text, "ko") is None
