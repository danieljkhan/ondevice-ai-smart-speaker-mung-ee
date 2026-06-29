from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from core.persona_modules import IntentSignals, assemble_persona_prompt

FIXTURE_DIR = Path("tests/persona_modules/fixtures/byte_identity")
TRUSTED_OVERRIDE = "<<<TRUSTED_OVERRIDE_TEST_STRING>>>"


def _fixture(row: int) -> str:
    return (FIXTURE_DIR / f"row_{row:02d}.txt").read_text(encoding="utf-8")


def _guide(topic: str, language: str) -> str:
    templates = json.loads(
        Path("assets/filters/approved_templates.json").read_text(encoding="utf-8")
    )
    key = "response_en" if language == "en" else "response_ko"
    return str(templates[topic][key])


def _assemble(
    *,
    language: str,
    backend: str,
    safety_guide: str | None = None,
    trusted_full_prompt_override: str | None = None,
) -> str:
    return assemble_persona_prompt(
        language=language,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        intent_signals=IntentSignals.all_true(),
        safety_guide=safety_guide,
        examples_budget=2,
        trusted_full_prompt_override=trusted_full_prompt_override,
    ).text


def _patch_overlay_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def fake_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self.name in {"m_persona_overlay.txt", "persona.md"}:
            raise FileNotFoundError(self)
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


def _patch_overlay_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def fake_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self.name in {"m_persona_overlay.txt", "persona.md"}:
            return "   \n\t"
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


def _patch_intent_rules(monkeypatch: pytest.MonkeyPatch, payload: str | None) -> None:
    original = Path.read_text

    def fake_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self.name == "intent_rules.json":
            if payload is None:
                raise FileNotFoundError(self)
            return payload
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


def _patch_legacy_overlay_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    persona_modules = importlib.import_module("core.persona_modules")
    for flag_name in (
        "_FORCE_PERSONA_OVERLAY_FOR_TESTS",
        "_TEST_FORCE_PERSONA_OVERLAY",
        "_FORCE_LEGACY_OVERLAY_GUARD",
    ):
        monkeypatch.setattr(persona_modules, flag_name, True, raising=False)
    for function_name in (
        "_should_load_persona_overlay",
        "_loads_persona_overlay",
        "_should_include_persona_overlay",
    ):
        if hasattr(persona_modules, function_name):
            monkeypatch.setattr(persona_modules, function_name, lambda *args, **kwargs: True)


def test_row_01() -> None:
    assert _assemble(language="ko", backend="gemma4_text") == _fixture(1)


def test_row_02() -> None:
    assert _assemble(
        language="ko", backend="gemma4_text", safety_guide=_guide("volcano", "ko")
    ) == _fixture(2)


def test_row_03() -> None:
    assert _assemble(
        language="ko", backend="gemma4_text", safety_guide=_guide("earthquake", "ko")
    ) == _fixture(3)


def test_row_04() -> None:
    assert _assemble(
        language="ko", backend="gemma4_text", safety_guide=_guide("flood", "ko")
    ) == _fixture(4)


def test_row_05() -> None:
    assert _assemble(language="en", backend="gemma4_text") == _fixture(5)


def test_row_06() -> None:
    assert _assemble(
        language="en", backend="gemma4_text", safety_guide=_guide("volcano", "en")
    ) == _fixture(6)


def test_row_07(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_legacy_overlay_guard(monkeypatch)
    with pytest.raises(ValueError, match="qwen3_legacy backend never loads M-PERSONA-OVERLAY"):
        _assemble(language="ko", backend="qwen3_legacy")


def test_row_08() -> None:
    assert _assemble(language="ko", backend="qwen3_legacy") == _fixture(8)


def test_row_09() -> None:
    assert _assemble(
        language="ko", backend="qwen3_legacy", safety_guide=_guide("volcano", "ko")
    ) == _fixture(9)


def test_row_10() -> None:
    assert _assemble(language="en", backend="qwen3_legacy") == _fixture(10)


def test_row_11() -> None:
    assert _assemble(
        language="en", backend="qwen3_legacy", safety_guide=_guide("volcano", "en")
    ) == _fixture(11)


def test_row_12() -> None:
    assert _assemble(language="ko", backend="gemma4_text") == _fixture(12)


def test_row_13() -> None:
    assert _assemble(
        language="ko", backend="gemma4_text", safety_guide=_guide("volcano", "ko")
    ) == _fixture(13)


def test_row_14() -> None:
    assert _assemble(language="ko", backend="gemma4_text") == _fixture(14)


def test_row_15() -> None:
    assert _assemble(language="ko", backend="gemma4_text") == _fixture(15)


def test_row_16() -> None:
    assert _assemble(language="ko", backend="gemma4_text") == _fixture(16)


def test_row_17(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _fixture(17)
    _patch_overlay_missing(monkeypatch)
    assert _assemble(language="ko", backend="gemma4_text") == expected


def test_row_18(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _fixture(18)
    _patch_overlay_empty(monkeypatch)
    assert _assemble(language="ko", backend="gemma4_text") == expected


def test_row_19(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _fixture(19)
    _patch_overlay_missing(monkeypatch)
    assert (
        _assemble(language="ko", backend="gemma4_text", safety_guide=_guide("volcano", "ko"))
        == expected
    )


def test_row_20() -> None:
    assert _assemble(
        language="ko",
        backend="gemma4_text",
        trusted_full_prompt_override=TRUSTED_OVERRIDE,
    ) == _fixture(20)


def test_row_21() -> None:
    assert _assemble(
        language="en",
        backend="gemma4_text",
        trusted_full_prompt_override=TRUSTED_OVERRIDE,
    ) == _fixture(21)


def test_row_22(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _fixture(22)
    _patch_intent_rules(monkeypatch, None)
    assert _assemble(language="ko", backend="gemma4_text") == expected


def test_row_23(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _fixture(23)
    _patch_intent_rules(monkeypatch, "{malformed")
    assert _assemble(language="ko", backend="gemma4_text") == expected


def test_row_24(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _fixture(24)
    _patch_intent_rules(monkeypatch, '{"schema_version": "1.0.0"}')
    assert _assemble(language="ko", backend="gemma4_text") == expected
