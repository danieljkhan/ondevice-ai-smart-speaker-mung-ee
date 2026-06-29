"""Tests for child-friendly Korean spelling of clock time, day, and date."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.conversation_memory_schema import KST
from core.korean_datetime import (
    append_copula,
    format_date_ko,
    format_day_ko,
    format_time_ko,
)


def _at(hour: int, minute: int = 0, *, year: int = 2026, month: int = 6, day: int = 18) -> datetime:
    """Build a KST-aware datetime for time-of-day cases."""
    return datetime(year, month, day, hour, minute, tzinfo=KST)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (_at(15, 30), "오후 세 시 삼십 분"),
        (_at(9, 0), "오전 아홉 시"),
        (_at(13, 5), "오후 한 시 오 분"),
        (_at(0, 0), "자정"),
        (_at(12, 0), "정오"),
        (_at(0, 30), "밤 열두 시 삼십 분"),
        (_at(12, 45), "오후 열두 시 사십오 분"),
        (_at(1, 0), "오전 한 시"),
        (_at(11, 59), "오전 열한 시 오십구 분"),
        (_at(23, 1), "오후 열한 시 일 분"),
    ],
)
def test_format_time_ko(now: datetime, expected: str) -> None:
    """Time spelling follows native-hour / Sino-minute rules with omissions."""
    assert format_time_ko(now) == expected


@pytest.mark.parametrize(
    ("month", "day", "expected"),
    [
        (6, 18, "유월, 십팔일"),
        (10, 1, "시월, 일일"),
        (1, 1, "일월, 일일"),
        (12, 31, "십이월, 삼십일일"),
        (7, 25, "칠월, 이십오일"),
    ],
)
def test_format_date_ko(month: int, day: int, expected: str) -> None:
    """Date spelling uses irregular months (유월/시월) and Sino days."""
    assert format_date_ko(datetime(2026, month, day, 12, 0, tzinfo=KST)) == expected


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (15, "월요일"),
        (16, "화요일"),
        (17, "수요일"),
        (18, "목요일"),
        (19, "금요일"),
        (20, "토요일"),
        (21, "일요일"),
    ],
)
def test_format_day_ko(day: int, expected: str) -> None:
    """Weekday sweep over a Mon-Sun span in June 2026."""
    assert format_day_ko(datetime(2026, 6, day, 12, 0, tzinfo=KST)) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("세 시", "세 시야"),
        ("유월, 십팔일", "유월, 십팔일이야"),
        ("월요일", "월요일이야"),
        ("정오", "정오야"),
        ("자정", "자정이야"),
        ("오전 아홉 시", "오전 아홉 시야"),
    ],
)
def test_append_copula(phrase: str, expected: str) -> None:
    """Copula agrees with the final syllable's batchim."""
    assert append_copula(phrase) == expected


def test_append_copula_empty_falls_back() -> None:
    """An empty phrase falls back to the vowel-final copula."""
    assert append_copula("") == "야"
