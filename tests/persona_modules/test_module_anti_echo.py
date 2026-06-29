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


def test_ko_anti_echo_module_is_loaded() -> None:
    fragment = _fragment("m_anti_echo.ko.txt")

    assert "ANTI-ECHO RULE" in fragment
    assert fragment in _assemble("ko")


def test_en_anti_echo_slot_is_reserved_empty() -> None:
    en_slot = _fragment("m_anti_echo.en.txt")
    ko_fragment = _fragment("m_anti_echo.ko.txt")

    assert en_slot == ""
    assert ko_fragment.strip() not in _assemble("en")
