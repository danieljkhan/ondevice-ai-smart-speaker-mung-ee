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


def test_ko_response_constraints_preserve_scope_and_length_cap() -> None:
    fragment = _fragment("m_response_constraints.ko.txt")

    assert "CRITICAL RULES" in fragment
    assert "maximum 60" in fragment
    assert fragment in _assemble("ko")


def test_en_response_constraints_preserve_length_cap() -> None:
    fragment = _fragment("m_response_constraints.en.txt")

    assert "Maximum 80 characters" in fragment
    assert fragment in _assemble("en")
