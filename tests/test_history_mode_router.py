"""Tests for deterministic Korean-history mode entry routing."""

from __future__ import annotations

import time

import pytest

from safety.history_mode_router import match_history_mode

HISTORY_CONFIRMATION = "좋아! 재미있는 우리역사를 시작할게!"


@pytest.mark.parametrize(
    "text",
    [
        "재미있는 우리역사",
        "재미 있는 우리 역사",
        "재밌는 우리 역사",
        "이 있는 우리 역사",
        "뭉이야 재미있는 우리역사!",
        "옛날 이야기 들려줘",
        "옛날 이야기 들려 줘!",
        "우리 역사 들려줘",
        "우리역사 들려 줘",
        "역사공부 하자",
        "역사 공부 하자",
        "역사 공부 하자니까.",
        "역사 공부하자니까",
        "역사 공부 하자니깐",
        "역사 알아보자",
        "역사 알려줘",
        "한국사 공부하자",
        "한국사 알려줘",
        "한국역사 알아보자",
        "한국 역사 알아보자",
        "역사 들려줘",
        "역사 배우자",
        "국사 공부하자",
        "보리 역사 공부하자.",
        "보리 역사 알려줘",
        "역사 공부하자. 역사 공부.",
        "역사 공부하자, 역사 공부.",
        "역사 공부하다.",
        "역사를 공부하자",
        "역사 공부 좀 하자",
        "역사 공부하고 싶어",
        "역사 공부할까",
        "역사 배우고 싶어",
        "한국사 공부하다",
        "역사를 배우자",
        "음 역사 알려줘",
        "역사 알려줘 역사",
        "그 우리 역사 들려줘",
        "그 우리 역사 알려줘",
    ],
)
def test_history_mode_entry_matches_short_bounded_turns(text: str) -> None:
    """History-mode triggers match as short bounded commands."""
    match = match_history_mode(text)

    assert match is not None
    assert match.confirmation_language == "ko"
    assert match.confirmation_text == HISTORY_CONFIRMATION
    assert match.matched_patterns


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "안녕",
        "뭉이야 안녕",
        "역사 숙제 알려줘",
        "역사 시험 문제 줘",
        "역사 문제 풀어줘",
        "역사 퀴즈 내줘",
        "역사 요약 해줘",
        "역사 연대표 알려줘",
        "역사 번역 해줘",
        "우리 역사 숙제 알려줘",
        "음 우리 역사 숙제 알려줘",
        "재미있는 우리 역사 숙제",
        "우리 역사 시험",
        "우리 역사에서 고조선 알려줘",
        "옛날 이야기 호랑이 들려줘",
        "고조선 이야기 들려줘",
        "임진왜란 알려줘",
        "임진왜란 역사 알려줘",
        "임진왜란 역사를 알려줘",
        "임진왜란 한국사 알려줘",
        "보리 임진왜란 역사 알려줘",
        "보리 임진왜란 역사를 알려줘",
        "세종대왕 알려줘",
        "세종대왕 역사를 알려줘",
        "역사 공부하다 말고 게임하자",
        "역사 공부하다 말자",
        "역사 공부하다 싫어",
        "역사 공부 안 할래",
        "역사 게임 하자",
        "역사 만화 보고 싶어",
        "역사 공부하자 말고 게임하자",
        "역사 공부하자 응 어 음 그",
        "역사 공부하자 해 줘",
        "역사 공부하자 그 해",
        "역사 알려줘 해줘",
        "영어 공부하자",
        "수학 공부하자",
        "숫자 공부하자",
        "영어로 말해줘",
        "옛날에 우리나라에서 가장 강했던 시대가 언제인지 자세히 알려줘",
    ],
)
def test_history_mode_entry_rejects_false_positives(text: str) -> None:
    """Object-bearing history requests remain normal conversation turns."""
    assert match_history_mode(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "역사 공부하자" + " " * 20000 + "말고",
        "역사 공부하다 " + "," * 20000 + " 말고",
    ],
)
def test_history_mode_entry_is_linear_on_adversarial_padding(text: str) -> None:
    """The entry matcher must not exhibit super-linear backtracking (ReDoS)."""
    start = time.perf_counter()
    result = match_history_mode(text)
    elapsed = time.perf_counter() - start
    assert result is None
    assert elapsed < 0.5
