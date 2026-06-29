"""Tests for deterministic spoken time / day / date query routing."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from core.conversation_memory_schema import KST
from core.datetime_router import match_datetime_query

# 2026-06-18 is a Thursday (목요일) at 15:30 KST.
_NOW = datetime(2026, 6, 18, 15, 30, tzinfo=KST)


@pytest.mark.parametrize(
    ("text", "expected_kind", "expected_response"),
    [
        ("지금 몇 시야?", "time", "지금 오후 세 시 삼십 분이야"),
        ("뭉이야 지금 몇 시야", "time", "지금 오후 세 시 삼십 분이야"),
        ("지금 시간 알려줘", "time", "지금 오후 세 시 삼십 분이야"),
        ("오늘 무슨 요일이야?", "day", "오늘은 목요일이야"),
        ("오늘 요일 알려줘", "day", "오늘은 목요일이야"),
        ("오늘 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘 날짜 알려줘", "date", "오늘은 유월, 십팔일이야"),
        ("오늘 몇 월 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘 몇월 몇일이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘 몇 일이야?", "date", "오늘은 유월, 십팔일이야"),
        # Bare "몇월" (month-only, no 며칠) must answer month+day like 며칠 does
        # (regression: child asked "오늘 몇월이야?" and got "can't tell").
        ("오늘 몇월이야?", "date", "오늘은 유월, 십팔일이야"),
        ("지금 몇월이야?", "date", "오늘은 유월, 십팔일이야"),
        ("몇월이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘 몇 월이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘은 몇월이야?", "date", "오늘은 유월, 십팔일이야"),
        ("지금은 몇월이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘 날짜랑 요일 알려줘", "date_day", "오늘은 유월, 십팔일 목요일이야"),
        ("오늘 요일이랑 날짜 알려줘", "date_day", "오늘은 유월, 십팔일 목요일이야"),
        # "지금" prefix on date/date_day queries must work like "오늘"
        # (regression: child asked "지금 몇월 며칠이야?" and got "can't tell").
        ("지금 몇월 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("지금 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("지금 날짜 뭐야?", "date", "오늘은 유월, 십팔일이야"),
        ("지금 날짜 알려줘", "date", "오늘은 유월, 십팔일이야"),
        ("지금 날짜랑 요일 뭐야?", "date_day", "오늘은 유월, 십팔일 목요일이야"),
        ("지금 요일이랑 날짜 알려줘", "date_day", "오늘은 유월, 십팔일 목요일이야"),
        # Topic/subject particle after 오늘/지금 must still match (regression:
        # child asked "오늘은 무슨 요일이야?" and got a hallucinated weekday
        # because the bare 오늘 anchor rejected the trailing 은/는/이 particle).
        ("오늘은 무슨 요일이야?", "day", "오늘은 목요일이야"),
        ("오늘는 무슨 요일이야?", "day", "오늘은 목요일이야"),
        ("오늘이 무슨 요일이야?", "day", "오늘은 목요일이야"),
        ("오늘은 요일 알려줘", "day", "오늘은 목요일이야"),
        ("지금은 몇 시야?", "time", "지금 오후 세 시 삼십 분이야"),
        ("지금은 시간 알려줘", "time", "지금 오후 세 시 삼십 분이야"),
        ("오늘은 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘은 몇월 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘은 날짜 알려줘", "date", "오늘은 유월, 십팔일이야"),
        ("지금은 며칠이야?", "date", "오늘은 유월, 십팔일이야"),
        ("오늘은 날짜랑 요일 뭐야?", "date_day", "오늘은 유월, 십팔일 목요일이야"),
        ("오늘은 요일이랑 날짜 알려줘", "date_day", "오늘은 유월, 십팔일 목요일이야"),
    ],
)
def test_match_datetime_query_positives(
    text: str, expected_kind: str, expected_response: str
) -> None:
    """Whole-turn clock/calendar questions return a spelled-out answer."""
    match = match_datetime_query(text, now=_NOW)

    assert match is not None
    assert match.kind == expected_kind
    assert match.response_text == expected_response
    assert match.matched_patterns


def test_match_datetime_query_noon_reads_as_word() -> None:
    """Noon is read as the bare word 정오 with a vowel-final copula."""
    noon = datetime(2026, 6, 18, 12, 0, tzinfo=KST)
    match = match_datetime_query("지금 몇 시야?", now=noon)

    assert match is not None
    assert match.response_text == "지금 정오야"


def test_match_datetime_query_midnight_reads_as_word() -> None:
    """Midnight is read as 자정 with a batchim-final copula."""
    midnight = datetime(2026, 6, 18, 0, 0, tzinfo=KST)
    match = match_datetime_query("지금 시간 알려줘", now=midnight)

    assert match is not None
    assert match.response_text == "지금 자정이야"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "안녕",
        "세 시에 만나자",
        "내일 며칠이야",
        "어제 무슨 요일이었어",
        "소풍 무슨 요일이야",
        "며칠 남았어",
        "다음 주 무슨 요일이야",
        "몇 시에 자야 해",
        # The optional topic/subject particle after 오늘/지금 must not widen the
        # match into ordinary conversation that merely starts with 오늘은/지금은.
        "오늘은 기분이 어때?",
        "지금은 뭐해?",
        "오늘은 학교 갔어",
        "지금은 졸려",
    ],
)
def test_match_datetime_query_false_positives(text: str) -> None:
    """Non-clock turns and future/past references stay normal conversation."""
    assert match_datetime_query(text, now=_NOW) is None


@pytest.mark.parametrize(
    "text",
    [
        # Exceeds the >=3 surrounding-word bound, so the turn cannot match even
        # while forcing the engine over a long whitespace/comma run.
        "가 나 다 라" + " " * 20000 + "지금 몇 시야 마 바 사",
        "오늘 며칠 가 나 다 라 " + "," * 20000 + " 마 바 사",
    ],
)
def test_match_datetime_query_is_linear_on_adversarial_padding(text: str) -> None:
    """The matcher must not exhibit super-linear backtracking (ReDoS)."""
    start = time.perf_counter()
    result = match_datetime_query(text, now=_NOW)
    elapsed = time.perf_counter() - start
    assert result is None
    assert elapsed < 0.5
