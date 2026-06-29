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


def test_speech_module_is_loaded_for_ko() -> None:
    speech = _fragment("m_speech.ko.txt")

    assert "SPEECH RULES" in speech
    assert speech in _assemble("ko")


def test_speech_module_is_not_loaded_for_en() -> None:
    speech = _fragment("m_speech.ko.txt")
    marker_line = next(line for line in speech.splitlines() if "informal casual speech" in line)
    casual_marker = marker_line.partition("(")[2].partition(")")[0]

    assembled = _assemble("en")

    assert speech.strip() not in assembled
    assert casual_marker
    assert casual_marker not in assembled
