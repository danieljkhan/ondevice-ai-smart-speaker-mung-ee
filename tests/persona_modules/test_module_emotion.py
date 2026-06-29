from __future__ import annotations

from pathlib import Path

from core.persona_modules import IntentSignals, assemble_persona_prompt

FRAGMENT_DIR = Path("assets/prompts/persona_modules")


def _fragment(name: str) -> str:
    return (FRAGMENT_DIR / name).read_text(encoding="utf-8")


def _assemble(language: str) -> str:
    return assemble_persona_prompt(
        language=language,  # type: ignore[arg-type]
        backend="gemma4_text",
        intent_signals=IntentSignals.all_true(),
    ).text


def test_ko_emotion_surface_reaction_rules_are_loaded() -> None:
    fragment = _fragment("m_emotion.ko.txt")

    assert "EMOTION RESPONSE RULES" in fragment
    assert fragment in _assemble("ko")


def test_en_emotion_slot_is_reserved_empty() -> None:
    en_slot = _fragment("m_emotion.en.txt")
    ko_fragment = _fragment("m_emotion.ko.txt")

    assert en_slot == ""
    assert ko_fragment.strip() not in _assemble("en")


def test_rule_six_protocol_is_not_duplicated_in_emotion_module() -> None:
    fragment = _fragment("m_emotion.ko.txt")

    assert "3-step protocol" not in fragment
    assert "Step 1:" not in fragment
    assert "Step 2:" not in fragment
    assert "Step 3:" not in fragment
