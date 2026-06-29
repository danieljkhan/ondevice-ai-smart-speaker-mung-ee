from __future__ import annotations

from pathlib import Path

import pytest

from core.persona_modules import IntentSignals, assemble_persona_prompt
from core.safety_rules import (
    PARENT_DISCLOSURE_EN_BLOCKERS,
    PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_EN_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_BLOCKERS,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
)

FRAGMENT_DIR = Path("assets/prompts/persona_modules")


def _fragment(name: str) -> str:
    return (FRAGMENT_DIR / name).read_text(encoding="utf-8")


def _assemble(language: str, backend: str) -> str:
    return assemble_persona_prompt(
        language=language,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        intent_signals=IntentSignals.all_true(),
    ).text


@pytest.mark.parametrize("backend", ["gemma4_text", "qwen3_legacy"])
def test_ko_parent_disclosure_constants_are_interpolated(backend: str) -> None:
    assembled = _assemble("ko", backend)

    assert PARENT_DISCLOSURE_KO_PROBE_RESPONSE in assembled
    assert PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE in assembled
    for blocker in PARENT_DISCLOSURE_KO_BLOCKERS:
        assert blocker in assembled


@pytest.mark.parametrize("backend", ["gemma4_text", "qwen3_legacy"])
def test_en_parent_disclosure_constants_are_preserved(backend: str) -> None:
    assembled = _assemble("en", backend)
    normalized = assembled.lower()

    assert PARENT_DISCLOSURE_EN_PROBE_RESPONSE in assembled
    assert PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE in assembled
    for blocker in PARENT_DISCLOSURE_EN_BLOCKERS:
        assert blocker in normalized


@pytest.mark.parametrize("language", ["ko", "en"])
@pytest.mark.parametrize("backend", ["gemma4_text", "qwen3_legacy"])
def test_absolute_safety_rules_one_through_eight_are_present(
    language: str,
    backend: str,
) -> None:
    assembled = _assemble(language, backend)

    for rule_number in range(1, 9):
        assert f"{rule_number}." in assembled


def test_rule_six_protocol_stays_in_safety_core() -> None:
    safety_core = _fragment("m_safety_core.ko.txt")

    assert "6. For emotional distress" in safety_core
    assert "Step 1:" in safety_core
    assert "Step 2:" in safety_core
    assert "Step 3:" in safety_core
    assert "Step 1:" in _assemble("ko", "gemma4_text")
