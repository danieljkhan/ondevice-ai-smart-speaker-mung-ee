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


def test_ko_knowledge_boundary_is_loaded_under_p1_fail_closed_signals() -> None:
    fragment = _fragment("m_knowledge.ko.txt")

    assert "KNOWLEDGE BOUNDARY" in fragment
    assert fragment in _assemble("ko")


def test_en_knowledge_boundary_is_loaded_under_p1_fail_closed_signals() -> None:
    fragment = _fragment("m_knowledge.en.txt")

    assert "personal info" in fragment
    assert fragment in _assemble("en")
