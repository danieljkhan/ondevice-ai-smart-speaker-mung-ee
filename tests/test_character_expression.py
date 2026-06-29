"""Tests for the Phase 2 character expression values."""

from __future__ import annotations

from core.character_expression import CharacterExpression


def test_neutral_expression_exists() -> None:
    """The Phase 1 enum exposes the neutral expression."""
    assert CharacterExpression.NEUTRAL.value == "neutral"
    assert CharacterExpression("neutral") is CharacterExpression.NEUTRAL


def test_expression_values_are_strings() -> None:
    """Enum values stay string-backed for Phase 2 extension."""
    assert all(isinstance(member.value, str) for member in CharacterExpression)


def test_all_character_expression_members_exist() -> None:
    """The Phase 2 enum exposes the complete expression set."""
    assert {member.name for member in CharacterExpression} == {
        "NEUTRAL",
        "IDLE",
        "LISTENING",
        "THINKING",
        "SPEAKING",
        "HAPPY",
        "SAD",
        "SURPRISED",
        "CONCERNED",
        "JOYFUL",
        "GREETING",
        "EXCITED",
        "ANGRY",
        "SULKY",
        "SLEEPY",
        "TIRED",
        "SHY",
        "WINKING",
        "AFFECTIONATE",
    }


def test_character_expression_values_are_unique() -> None:
    """Each expression has one unique serialized value."""
    assert len({member.value for member in CharacterExpression}) == 19


def test_character_expression_values_match_lowercase_names() -> None:
    """Enum values match lowercase member names for asset filenames."""
    assert all(member.value == member.name.lower() for member in CharacterExpression)
