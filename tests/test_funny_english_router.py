"""Tests for the deterministic Funny English entry router."""

from __future__ import annotations

import pytest

from safety.funny_english_router import match_funny_english


@pytest.mark.parametrize(
    "text",
    [
        "Funny English",
        "Mungi, funny english!",
        "honey english",
        "퍼니 잉글리시",
        "퍼니 인글리시 하자",
        "퍼니 잉글리시 하자",
        "퍼니 잉글리시 시작해줘",
        "퍼니 잉글리시 시작해 줘",
        "퍼니 잉글리시 해줘",
        "퍼니 영어 하자",
        "퍼니 잉글리 씨 공부하자",
        "뭉이야 퍼니 잉글리시 할래",
        "퍼니 잉글리쉬",
        "퍼니 잉글리 씨",
        "퍼니 잉글리쉬 하자",
        "잉글리쉬",
        "잉글리 씨",
        "뭉이야 퍼니 잉글리쉬 할래",
        "허니잉글리시아자.",
        "허니잉글리시하자.",
        "잉글리시",
        "잉글리시 하자",
        "허니 잉글리시",
        "뭉이야 허니 잉글리시 하자",
        "영어 공부",
        "영어 공부 하자",
        "영어 공부 할래",
        "영어 공부 해 보자",
        "뭉이야 영어공부 시작하자",
        "영어 공부 하자!",
        "여공부하자.",
        "서공부하자.",
        "영아 공부하자",
        "여 공부 할래",
        "뭉이야 영어 읽기 하자",
        "영어 읽기 하자",
        "영어 게임",
    ],
)
def test_funny_english_entry_triggers(text: str) -> None:
    """Whole-turn entry phrases match Funny English mode."""
    match = match_funny_english(text)

    assert match is not None
    assert match.confirmation_language == "ko"


@pytest.mark.parametrize(
    "text",
    [
        "영어 숙제 알려줘",
        "영어 읽기 숙제 알려줘",
        "영어 공부 도와줘",
        "영어 공부 방법 알려줘",
        "영어 공부 어떻게 해",
        "영어 공부 하자 그리고 공룡 얘기해줘",
        "영어 단어 알려줘",
        "영어 숙제 하자",
        "영어 수업 하자",
        "여공부",
        "여공부 방법 알려줘",
        "여공부 도와줘",
        "잉글리시 숙제 알려줘",
        "인글리시 숙제 알려줘",
        "잉글리시 단어 알려줘",
        "잉글리쉬 숙제 알려줘",
        "잉글리 씨 단어 알려줘",
        "잉글리쉬 시험",
        "어서 공부하자",
        "독서 공부하자",
        "국어 공부하자",
        "수학 공부하자",
        "역사 공부하자",
        "영어로 말해줘",
        "브리티시 잉글리시 알려줘",
        "teach me english words",
        "english reading homework please",
        "퍼니 잉글리시 책 뭐야",
        "퍼니 잉글리시 뭐야",
        "퍼니 영어 숙제 알려줘",
    ],
)
def test_funny_english_false_positive_guards(text: str) -> None:
    """Object-bearing English-study requests are not stolen by the mode trigger."""
    assert match_funny_english(text) is None
