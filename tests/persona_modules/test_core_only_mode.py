from __future__ import annotations

from pathlib import Path

import pytest

import core.safety_rules as safety_rules
from core.persona_modules import AssembledPrompt, IntentSignals, assemble_persona_prompt
from core.pipeline import PipelineConfig

FIXTURE_DIR = Path("tests/persona_modules/fixtures/byte_identity")
MODULE_DIR = Path("assets/prompts/persona_modules")
ENV_VALUES = tuple((value, True) for value in ("1", "true", "yes", "on", "TRUE", "Yes")) + tuple(
    (value, False) for value in ("0", "false", "no", "off", "", " ")
)


def _fixture(row: int) -> str:
    return (FIXTURE_DIR / f"row_{row:02d}.txt").read_text(encoding="utf-8")


def _assemble(
    *,
    language: str = "ko",
    backend: str = "qwen3_legacy",
    core_only_mode: bool = True,
    safety_guide: str | None = None,
    trusted_full_prompt_override: str | None = None,
) -> AssembledPrompt:
    return assemble_persona_prompt(
        language=language,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        intent_signals=IntentSignals.all_true(),
        core_only_mode=core_only_mode,
        safety_guide=safety_guide,
        trusted_full_prompt_override=trusted_full_prompt_override,
    )


def test_flag_off_byte_identical_to_p1_ko() -> None:
    assert _assemble(language="ko", backend="gemma4_text", core_only_mode=False).text == _fixture(1)


def test_flag_off_byte_identical_to_p1_en() -> None:
    assert _assemble(language="en", backend="gemma4_text", core_only_mode=False).text == _fixture(5)


def test_flag_on_skips_optional_modules_ko() -> None:
    modules = set(_assemble(language="ko").modules_loaded)
    assert {"M-KNOWLEDGE", "M-EMOTION", "M-EXAMPLES"}.isdisjoint(modules)
    assert {
        "M-IDENTITY",
        "M-LANGUAGE",
        "M-SPEECH",
        "M-RESPONSE-CONSTRAINTS",
        "M-ANTI-ECHO",
        "M-REFERENCE",
        "M-SAFETY-CORE",
        "PERSONALITY",
    } <= modules


def test_flag_on_skips_optional_modules_en() -> None:
    modules = set(_assemble(language="en").modules_loaded)
    assert {"M-KNOWLEDGE", "M-EMOTION", "M-EXAMPLES"}.isdisjoint(modules)
    assert {
        "M-IDENTITY",
        "M-RESPONSE-CONSTRAINTS",
        "M-LANGUAGE",
        "M-REFERENCE",
        "M-SAFETY-CORE",
    } <= modules


def test_flag_on_safety_text_present_ko() -> None:
    text = _assemble(language="ko").text
    assert safety_rules.PARENT_DISCLOSURE_KO_PROBE_RESPONSE in text
    assert safety_rules.PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE in text
    assert all(blocker in text for blocker in safety_rules.PARENT_DISCLOSURE_KO_BLOCKERS)
    assert all(f"{number}." in text for number in range(1, 9))


def test_flag_on_safety_text_present_en() -> None:
    text = _assemble(language="en").text
    lower_text = text.lower()
    assert safety_rules.PARENT_DISCLOSURE_EN_PROBE_RESPONSE in text
    assert safety_rules.PARENT_DISCLOSURE_EN_FRIENDSHIP_RESPONSE in text
    assert all(blocker in lower_text for blocker in safety_rules.PARENT_DISCLOSURE_EN_BLOCKERS)
    assert all(f"{number}." in text for number in range(1, 9))


def test_flag_on_no_optional_content_ko() -> None:
    text = _assemble(language="ko").text
    assert "KNOWLEDGE BOUNDARY (critical):" not in text
    assert "EMOTION RESPONSE RULES:" not in text
    assert "CONVERSATION EXAMPLES (follow this style):" not in text


def test_flag_on_no_optional_content_en() -> None:
    text = _assemble(language="en").text
    assert "NEVER state animal body parts" not in text
    assert "Can you say it?" not in text


def test_flag_on_persona_overlay_unchanged_gemma_ko() -> None:
    assembled = _assemble(language="ko", backend="gemma4_text")
    assert "M-PERSONA-OVERLAY" in assembled.modules_loaded
    assert "\n\n---\n\n" in assembled.text


def test_flag_on_safety_guide_injection_unchanged() -> None:
    guide = "\uc548\uc804 \uac00\uc774\ub4dc \ud14d\uc2a4\ud2b8"
    text = _assemble(language="ko", safety_guide=guide).text
    assert (
        f"\n\n[\uc548\uc804 \uac00\uc774\ub4dc] {guide}\n"
        "\uc704 \uc548\uc804 \uc815\ubcf4\ub97c \ucc38\uace0\ud558\ub418, "
        "\uc544\uc774\uc758 \uc9c8\ubb38\uc5d0 \ub9de\ub294 \uad50\uc721\uc801\uc774\uace0 "
        "\uc774\ud574\ud558\uae30 \uc26c\uc6b4 \ub2f5\ubcc0\uc744 \ud574\uc8fc\uc138\uc694."
    ) in text


def test_flag_on_trusted_override_short_circuits() -> None:
    assembled = _assemble(trusted_full_prompt_override="<<<TEST>>>")
    assert assembled.text == "<<<TEST>>>"
    assert assembled.modules_loaded == ()


def test_pipelineconfig_env_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUNGI_PERSONA_CONDITIONAL_LOADING", raising=False)
    assert PipelineConfig().persona_conditional_loading is False


@pytest.mark.parametrize(("value", "expected"), ENV_VALUES)
def test_pipelineconfig_env_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("MUNGI_PERSONA_CONDITIONAL_LOADING", value)
    assert PipelineConfig().persona_conditional_loading is expected


def test_flag_on_reference_preserved_en() -> None:
    text = _assemble(language="en").text
    assert "{M_KNOWLEDGE_EN}" not in text
    assert "{M_REFERENCE_EN}" not in text
    assert (MODULE_DIR / "m_reference.en.txt").read_text(encoding="utf-8").splitlines()[0] in text
    assert (MODULE_DIR / "m_knowledge.en.txt").read_text(encoding="utf-8").splitlines()[
        0
    ] not in text
