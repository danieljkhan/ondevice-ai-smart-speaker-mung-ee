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


def test_ko_reference_rules_are_preserved_for_p1_byte_identity() -> None:
    fragment = _fragment("m_reference.ko.txt")

    assert "REFERENCE INFORMATION RULES" in fragment
    assert fragment in _assemble("ko")


def test_en_reference_rules_are_preserved_for_p1_byte_identity() -> None:
    fragment = _fragment("m_reference.en.txt")

    assert "REFERENCE INFORMATION RULES" in fragment
    assert fragment in _assemble("en")
