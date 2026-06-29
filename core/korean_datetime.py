"""Child-friendly Korean spelling of local clock time, day-of-week, and date.

Pure standard-library helpers that turn a :class:`datetime.datetime` into a
fully spelled-out Korean phrase suitable for text-to-speech. Hours use native
Korean numerals with their dedicated counter ``시``; minutes, months, and days
use Sino-Korean numerals. Irregular month readings (``유월`` for June, ``시월``
for October) are hard-coded because the regular Sino-Korean speller would
mispronounce them.

No I/O is performed; every function is deterministic given its ``now`` argument.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

# Native Korean hour numerals (1-12) used with the counter 시.
_NATIVE_HOURS: Final[dict[int, str]] = {
    1: "한",
    2: "두",
    3: "세",
    4: "네",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
    10: "열",
    11: "열한",
    12: "열두",
}

# Sino-Korean digit and unit syllables for the 1-59 speller (minutes/days).
_SINO_ONES: Final[tuple[str, ...]] = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_SINO_TENS: Final[tuple[str, ...]] = ("", "십", "이십", "삼십", "사십", "오십")

# Sino-Korean month readings with the irregular 유월 (6) and 시월 (10).
_SINO_MONTHS: Final[dict[int, str]] = {
    1: "일월",
    2: "이월",
    3: "삼월",
    4: "사월",
    5: "오월",
    6: "유월",
    7: "칠월",
    8: "팔월",
    9: "구월",
    10: "시월",
    11: "십일월",
    12: "십이월",
}

_HANGUL_BASE: Final[int] = 0xAC00
_JONGSEONG_COUNT: Final[int] = 28


def _spell_sino_1_59(value: int) -> str:
    """Spell an integer in ``1..59`` using Sino-Korean numerals.

    Raises:
        ValueError: If ``value`` is outside the supported ``1..59`` range.
    """
    if not 1 <= value <= 59:
        raise ValueError(f"Sino-Korean speller supports 1-59, got {value}")
    tens, ones = divmod(value, 10)
    return f"{_SINO_TENS[tens]}{_SINO_ONES[ones]}"


def format_time_ko(now: datetime) -> str:
    """Spell the clock time of ``now`` in child-friendly Korean.

    Hours read with native Korean numerals plus 시; minutes read with
    Sino-Korean numerals plus 분 and are omitted entirely when zero. Midnight
    (00:00) reads ``자정`` and noon (12:00) reads ``정오``.
    """
    hour = now.hour
    minute = now.minute

    if hour == 0 and minute == 0:
        return "자정"
    if hour == 12 and minute == 0:
        return "정오"

    if hour == 0:
        # Past-midnight hours read as "밤 열두 시".
        head = "밤 열두 시"
    elif hour == 12:
        head = "오후 열두 시"
    elif hour < 12:
        head = f"오전 {_NATIVE_HOURS[hour]} 시"
    else:
        head = f"오후 {_NATIVE_HOURS[hour - 12]} 시"

    if minute == 0:
        return head
    return f"{head} {_spell_sino_1_59(minute)} 분"


def format_day_ko(now: datetime) -> str:
    """Spell the weekday of ``now`` as a Korean ``요일`` phrase."""
    day_names = ("월", "화", "수", "목", "금", "토", "일")
    return f"{day_names[now.weekday()]}요일"


def format_date_ko(now: datetime) -> str:
    """Spell the calendar date of ``now`` as ``{month}, {day}일`` in Korean.

    The comma between month and day inserts a prosodic boundary for the TTS
    engine. Without it, Supertonic phrases the date as one breath and lets the
    short, irregular month reading (e.g. ``유월`` for June) trail off and get
    swallowed before the day, so listeners hear only the day. The comma makes
    the month a phrase-final, emphasized unit so it stays intelligible.
    """
    month = _SINO_MONTHS[now.month]
    day = f"{_spell_sino_1_59(now.day)}일"
    return f"{month}, {day}"


def append_copula(phrase: str) -> str:
    """Append the Korean copula (``이야``/``야``) agreeing with the final syllable.

    A syllable that ends in a final consonant (batchim) takes ``이야``; a
    vowel-final syllable takes ``야``. Phrases not ending in a composed Hangul
    syllable fall back to ``야``.
    """
    if not phrase:
        return "야"
    last = phrase[-1]
    code = ord(last) - _HANGUL_BASE
    if 0 <= code < 11172 and code % _JONGSEONG_COUNT != 0:
        return f"{phrase}이야"
    return f"{phrase}야"


__all__ = [
    "append_copula",
    "format_date_ko",
    "format_day_ko",
    "format_time_ko",
]
