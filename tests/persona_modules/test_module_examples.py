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


def test_ko_examples_preserve_five_dialogue_lines() -> None:
    fragment = _fragment("m_examples.ko.txt")

    assert fragment.count("- Child:") == 5
    assert fragment in _assemble("ko")


def test_en_examples_preserve_vocab_helper_example_only() -> None:
    fragment = _fragment("m_examples.en.txt")

    assert fragment.strip().startswith("Example:")
    assert fragment.count("Example:") == 1
    assert fragment in _assemble("en")
