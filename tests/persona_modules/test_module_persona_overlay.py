from __future__ import annotations

from pathlib import Path

import pytest

from core.persona_modules import IntentSignals, assemble_persona_prompt

FRAGMENT_DIR = Path("assets/prompts/persona_modules")


def _fragment(name: str) -> str:
    return (FRAGMENT_DIR / name).read_text(encoding="utf-8")


def _assemble(language: str, backend: str = "gemma4_text", safety_guide: str | None = None) -> str:
    return assemble_persona_prompt(
        language=language,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        intent_signals=IntentSignals.all_true(),
        safety_guide=safety_guide,
    ).text


def _patch_overlay_read_text(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    original = Path.read_text

    def fake_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self.name in {"m_persona_overlay.txt", "persona.md"}:
            if value is None:
                raise FileNotFoundError(self)
            return value
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


def test_gemma_ko_loads_persona_overlay_after_separator() -> None:
    overlay = _fragment("m_persona_overlay.txt")
    assembled = _assemble("ko")

    assert overlay.strip()
    assert "### §CAPABILITY" in overlay
    assert "\n\n---\n\n" in assembled
    assert assembled.endswith(overlay)


def test_en_path_bypasses_persona_overlay() -> None:
    overlay = _fragment("m_persona_overlay.txt")
    assembled = _assemble("en")

    assert overlay.strip() not in assembled
    assert "\n\n---\n\n" not in assembled


def test_qwen_legacy_path_bypasses_persona_overlay() -> None:
    overlay = _fragment("m_persona_overlay.txt")
    assembled = _assemble("ko", backend="qwen3_legacy")

    assert overlay.strip() not in assembled
    assert "\n\n---\n\n" not in assembled


def test_missing_overlay_falls_back_to_base_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_overlay_read_text(monkeypatch, None)

    assert _assemble("ko") == _assemble("ko", backend="qwen3_legacy")


def test_empty_overlay_falls_back_to_base_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_overlay_read_text(monkeypatch, "  \n\t")

    assert _assemble("ko") == _assemble("ko", backend="qwen3_legacy")
