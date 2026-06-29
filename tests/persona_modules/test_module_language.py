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


def test_language_module_preserves_ko_processing_rules() -> None:
    fragment = _fragment("m_language.ko.txt")

    assert "LANGUAGE PROCESSING RULES" in fragment
    assert fragment in _assemble("ko")


def test_language_module_preserves_en_language_rules() -> None:
    fragment = _fragment("m_language.en.txt")
    assembled = _assemble("en")

    assert "LANGUAGE RULES" in fragment
    for line in fragment.splitlines():
        if line:
            assert line in assembled
