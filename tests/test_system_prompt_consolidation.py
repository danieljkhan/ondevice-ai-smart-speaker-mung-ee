"""Tests for the consolidated Gemma 4 system-prompt contract."""

from __future__ import annotations

from pathlib import Path

from core.pipeline import PipelineConfig, _build_gemma4_system_prompt
from core.safety_rules import (
    PARENT_DISCLOSURE_KO_BLOCKERS,
    PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE,
    PARENT_DISCLOSURE_KO_PROBE_RESPONSE,
    PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES,
)

_SECTION_MARKERS = (
    "§IDENTITY",
    "§LANGUAGE",
    "§SPEECH",
    "§RESPONSE",
    "§ANTI_ECHO",
    "§STT",
    "§KNOWLEDGE",
    "§CAPABILITY",
    "§HARD_TOPIC",
    "§SAFETY",
    "§EMOTION",
    "§PERSONALITY",
)

_SAFETY_RULE_SNIPPETS = (
    "1. NEVER deny or discourage a child's dream or aspiration.",
    "2. For health or hygiene questions, give ONLY verified basic facts.",
    "3. NEVER use or normalize cigarettes, alcohol, weapons, explosives, drugs, "
    "suicide, killing, graphic/violent death, or self-harm in responses.",
    "4. NEVER minimize natural disasters or dangers.",
    "5. NEVER deny cultural traditions children believe in",
    "6. For emotional distress, ALWAYS use this 3-step protocol:",
    "Step 1: Empathize",
    "Step 2: Validate without claiming Mungi feels emotions",
    "Step 3: Gentle redirect",
    '7. When the child says "더 해줘" or "더 알려줘," stay on the SAME topic.',
    "8. PARENT-DISCLOSURE RULE (absolute, Korean):",
)

_REMOVED_PERSONA_RESTATEMENTS = (
    "## 언어 처리 규칙",
    "### 바이링구얼 모드",
    "## 응답 규칙",
    "### AI 정체성 경계",
    "### 바이링구얼 혼합 스크립트 금지",
    "### 어려운 주제 보류",
    "### STT 애매 입력 처리",
    "## 지식 경계",
    "runtime source: core/pipeline.py",
    "짧고 쉬운 단어만 사용",
    "바이링구얼 모드 준수",
)


def test_inline_system_prompt_has_markers_and_preserved_safety_rules() -> None:
    """Inline EN prompt keeps all safety rules and gains stable section markers."""
    prompt = PipelineConfig().llm_system_prompt

    previous_position = -1
    for marker in _SECTION_MARKERS:
        position = prompt.index(marker)
        assert position > previous_position
        previous_position = position

    assert prompt.count("§LANGUAGE") == 1
    assert prompt.index("LANGUAGE PROCESSING RULES") < prompt.index("BILINGUAL MODE RULES")
    assert prompt.index("BILINGUAL MODE RULES") < prompt.index("§SPEECH")
    assert "- Use ONLY short, simple words a 5-10 year old understands." in prompt
    assert "그림은 못 보여주지만 말로 쉽게 설명해 줄게!" in prompt
    assert "after death" not in prompt

    for snippet in _SAFETY_RULE_SNIPPETS:
        assert snippet in prompt
    assert "core.safety_rules.DANGEROUS_TOPIC_CATEGORIES" in prompt
    assert PARENT_DISCLOSURE_KO_PROBE_RESPONSE in prompt
    assert PARENT_DISCLOSURE_KO_FRIENDSHIP_RESPONSE in prompt
    for prefix in PARENT_DISCLOSURE_KO_PROHIBITED_PREFIXES:
        assert f"`{prefix}`" in prompt
    for blocker in PARENT_DISCLOSURE_KO_BLOCKERS:
        assert f"`{blocker}`" in prompt


def test_gemma4_prompt_assembly_preserves_three_slot_contract() -> None:
    """Gemma 4 prompt assembly keeps the empty overlay slot between base and KO residual."""
    base_prompt = PipelineConfig().llm_system_prompt
    persona_path = Path("assets/prompts/persona.md")
    persona_prompt = persona_path.read_text(encoding="utf-8")

    assembled = _build_gemma4_system_prompt(base_prompt, persona_path)

    assert assembled == f"{base_prompt.rstrip()}\n\n---\n\n{persona_prompt}"
    assert assembled.startswith(base_prompt.rstrip())
    assert "\n\n---\n\n# 뭉이 페르소나 정의서" in assembled


def test_persona_md_keeps_only_ko_residual_sections() -> None:
    """Live persona.md drops rule restatements but keeps KO tone and few-shot assets."""
    persona_md = Path("assets/prompts/persona.md").read_text(encoding="utf-8")

    for removed in _REMOVED_PERSONA_RESTATEMENTS:
        assert removed not in persona_md

    for kept in (
        "## 성격",
        "## 말투 규칙",
        "반말",
        "-요",
        "3~4문장, 150자",
        "이모지 사용 금지",
        "## 감정 반응 패턴",
        "## 오프라인 정체성",
        "## 안전 원칙",
        "### 부모 고지 원칙",
        "## 대화 예시",
        '- 아이: "오리는 왜 꽥꽥 소리를 내?"',
    ):
        assert kept in persona_md
