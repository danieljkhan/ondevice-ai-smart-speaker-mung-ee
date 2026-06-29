"""Deterministic expression classifier for spoken Mungi responses."""

from __future__ import annotations

from core.character_expression import CharacterExpression

_CONCERNED_TERMS: tuple[str, ...] = (
    "슬픔",
    "속상",
    "그랬구나",
    "눈물",
    "울어",
    "울음",
    "울보",
    "울었",
    "울고",
    "힘들",
    "무서",
    "걱정",
    "조심",
    "위험",
)
_ENCOURAGING_TERMS: tuple[str, ...] = (
    "배울",
    "배워",
    "공부하자",
    "같이 생각",
    "해보자",
    "할 수 있어",
    "도전",
    "알아보자",
)
_HAPPY_TERMS: tuple[str, ...] = ("우와", "멋지다", "대단", "축하", "신난")
_AFFECTIONATE_TERMS: tuple[str, ...] = ("좋아", "사랑", "고마워")
_SURPRISED_TERMS: tuple[str, ...] = ("진짜?", "놀라", "세상에")
_GREETING_TERMS: tuple[str, ...] = ("안녕", "반가")


def classify_expression(text: str) -> CharacterExpression:
    """Return the best character expression for a genuine conversational response."""
    if any(term in text for term in _CONCERNED_TERMS):
        return CharacterExpression.CONCERNED
    if any(term in text for term in _ENCOURAGING_TERMS):
        return CharacterExpression.HAPPY
    if any(term in text for term in _HAPPY_TERMS):
        return CharacterExpression.HAPPY
    if any(term in text for term in _AFFECTIONATE_TERMS):
        return CharacterExpression.AFFECTIONATE
    if any(term in text for term in _SURPRISED_TERMS):
        return CharacterExpression.SURPRISED
    if any(term in text for term in _GREETING_TERMS):
        return CharacterExpression.GREETING
    return CharacterExpression.SPEAKING
