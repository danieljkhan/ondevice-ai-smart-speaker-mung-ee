"""Tests for deterministic response-expression classification."""

from __future__ import annotations

import pytest

from core.character_expression import CharacterExpression
from core.expression_classifier import classify_expression


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("우와, 정말 멋지다!", CharacterExpression.HAPPY),
        ("그랬구나, 많이 속상했겠다.", CharacterExpression.CONCERNED),
        ("진짜? 세상에!", CharacterExpression.SURPRISED),
        ("무서웠겠다. 조심하자.", CharacterExpression.CONCERNED),
        ("뭉이를 좋아해 줘서 고마워.", CharacterExpression.AFFECTIONATE),
        ("안녕! 반가워.", CharacterExpression.GREETING),
        ("오늘은 구름이 많아.", CharacterExpression.SPEAKING),
        ("That sounds good.", CharacterExpression.SPEAKING),
        ("이번주에 뭘 배울지 같이 생각해볼까", CharacterExpression.HAPPY),
        ("오늘은 뭘 배울까?", CharacterExpression.HAPPY),
        ("서울 구경 가자", CharacterExpression.SPEAKING),
        ("거울을 보렴", CharacterExpression.SPEAKING),
        ("겨울이 왔어", CharacterExpression.SPEAKING),
        ("눈물이 났구나", CharacterExpression.CONCERNED),
        ("엉엉 울었어", CharacterExpression.CONCERNED),
        ("자꾸 울고 있어", CharacterExpression.CONCERNED),
        ("공부하다 울었어", CharacterExpression.CONCERNED),
        ("같이 해보자!", CharacterExpression.HAPPY),
        ("너는 할 수 있어", CharacterExpression.HAPPY),
    ],
)
def test_classify_expression_maps_korean_lexicon(
    text: str,
    expected: CharacterExpression,
) -> None:
    """Korean tone terms map to expressions while uncertain text stays SPEAKING."""
    assert classify_expression(text) is expected
