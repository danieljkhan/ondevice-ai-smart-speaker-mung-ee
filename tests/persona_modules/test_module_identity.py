from __future__ import annotations

from pathlib import Path

from core.persona_modules import IntentSignals, assemble_persona_prompt

FRAGMENT_DIR = Path("assets/prompts/persona_modules")


def _fragment(name: str) -> str:
    return (FRAGMENT_DIR / name).read_text(encoding="utf-8")


def _assemble(language: str, backend: str = "gemma4_text") -> str:
    return assemble_persona_prompt(
        language=language,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        intent_signals=IntentSignals.all_true(),
    ).text


def test_identity_fragments_are_loaded_for_both_languages() -> None:
    ko_identity = _fragment("m_identity.ko.txt")
    en_identity = _fragment("m_identity.en.txt")

    assert ko_identity.strip()
    assert en_identity.strip()
    assert ko_identity in _assemble("ko")
    assert en_identity in _assemble("en")


def test_personality_trailer_stays_in_ko_assembled_prompt() -> None:
    trailer = _fragment("personality_trailer.txt")

    assert trailer.strip()
    assert trailer in _assemble("ko")
