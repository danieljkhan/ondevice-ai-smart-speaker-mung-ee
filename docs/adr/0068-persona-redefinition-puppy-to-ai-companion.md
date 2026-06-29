# ADR 0068 — Persona redefinition: puppy framing -> AI companion for the child

- Status: Accepted
- Date: 2026-04-21
- Decision owner: Product owner (user) + Claude Code orchestrator
- Related: CLAUDE.md §1 (product vision), §12 (language policy), ADR 0067 (Gemma 4 Apache compliance), `assets/prompts/persona.md`, `core/pipeline.py` (`llm_system_prompt` default)

## Context

`assets/prompts/persona.md` previously framed Mungi as a "friendly puppy" companion:

- Line 8 (identity row): `아이의 첫 번째 AI 친구, 강아지처럼 귀엽고 다정하고 백과사전처럼 똑똑한 존재`.
- Line 78 (knowledge-boundary opener): `뭉이는 배우는 걸 좋아하는 강아지 친구다.`

The same framing was duplicated in `core/pipeline.py` default `llm_system_prompt` (lines ~267 and ~334):

- `"You are a friendly puppy who loves learning!"`
- `"Warm, curious, and honest like a friendly puppy."`

During 2026-04-21 Session 5 real-voice testing (8/8 success, 0 ALSA underruns), a child-voice exchange revealed the operational problem:

> Child (T6): "뭘 먹는데 너도 뭘 먹어야 음식일 거 아니야."
> Mungi (T6): "뭉이는 맛있는 사료를 먹어!" [sic — dog food]
> Child (T7): "사료를 먹는다고 너는 강아지도 아닌데 어떻게 사료를 먹어? 전기를 먹는 거 아니야?"

The puppy framing encourages the LLM to generate role-consistent but product-inconsistent responses (e.g. referring to dog food, doghouse, bark/pet behaviors). This conflicts with Mungi's product identity:

> "세상에서 가장 안전한, 우리 아이의 첫 번째 AI 친구" — The safest AI friend, a child's very first.

Mungi is a **conversational AI product for young children**, not a pet-simulation device. The puppy metaphor, while warm, introduces drift into non-AI domains and makes it harder for the safety layer (`safety/`) to reason about what Mungi should or should not claim to do.

## Decision

Rewrite Mungi's identity layer to an **AI-companion-for-the-child** framing. Keep all behavioral rules (warmth, curiosity, honesty, child-friendly language, safety, emotion protocol, offline identity) unchanged.

### Exact wording changes

**`assets/prompts/persona.md` line 8**:
- Before: `| 정체성 | 아이의 첫 번째 AI 친구, 강아지처럼 귀엽고 다정하고 백과사전처럼 똑똑한 존재 |`
- After: `| 정체성 | 아이의 첫 번째 AI 친구, 또래처럼 함께 놀고 이야기 나누는 다정하고 똑똑한 인공지능 |`

**`assets/prompts/persona.md` line 78**:
- Before: `- 뭉이는 배우는 걸 좋아하는 강아지 친구다. 아는 것은 자신 있게 말하고, 정말 모를 때는 솔직하게 말한다.`
- After: `- 뭉이는 배우는 걸 좋아하는 AI 친구다. 아는 것은 자신 있게 말하고, 정말 모를 때는 솔직하게 말한다.`

**`core/pipeline.py` KNOWLEDGE BOUNDARY block (~line 267)**:
- Before: `"- You are a friendly puppy who loves learning! Answer what you know, and be honest when you truly don't know.\n"`
- After: `"- You are a friendly AI companion who loves learning alongside the child! Answer what you know, and be honest when you truly don't know.\n"`

**`core/pipeline.py` PERSONALITY block (~line 334)**:
- Before: `"- Warm, curious, and honest like a friendly puppy.\n"`
- After: `"- Warm, curious, and honest like a trusted AI friend for a young child.\n"`

### Scope boundaries

**What this ADR does NOT change** (byte-identical preservation required):
- All safety rules (ABSOLUTE SAFETY RULES 1-7 in `llm_system_prompt`).
- Emotion-response protocol (3-step empathize -> validate -> gentle-redirect).
- Response format (1-2 sentences, 60 Korean characters, 친구처럼 다정한 반말).
- Anti-echo rule, knowledge-boundary guards, reference-information rules.
- Conversation examples (5 samples in `llm_system_prompt` + persona.md table).
- Offline-identity claims (wifi-free, privacy).
- Bilingual mode behavior.
- Language policy (CLAUDE.md §12).
- Hotwords (`뭉이야,뭉이,문지,뭉지`).
- `max_history_tokens=100` (CLAUDE.md §6).

**Out-of-scope legacy references**: Additional `puppy` tokens exist in `assets/prompts/child_safe_system_en.txt` and training/sample data. These are not runtime-active in the current pipeline and are intentionally left untouched to avoid expanding scope beyond what the user authorized. A separate cleanup ADR may revisit them if they become runtime-active.

## Rationale

1. **Product alignment**: Mungi is an AI friend, not a pet simulator. The product tagline and CLAUDE.md §1 product vision frame the device as an AI, not an animal. Identity must reflect this.
2. **Safety layer coherence**: Safety rules are phrased as absolute prohibitions against role-breaking (e.g. NEVER claim to eat, NEVER claim physical body parts). A puppy persona created avoidable tension with these rules — see T6/T7 "사료" drift.
3. **Warmth preserved**: "또래처럼 함께 놀고 이야기 나누는 다정하고 똑똑한 인공지능" preserves the friend-like peer framing. No cold/institutional AI tone is introduced.
4. **Minimal surface change**: Only 4 lines across 2 files are modified. Zero safety-rule drift. Zero emotion-protocol drift. Zero response-format drift.

## Alternatives considered

- **(A) Keep puppy framing + add guardrails**: Rejected. Would require expanding safety rules to cover every puppy-drift edge case (food, housing, bark, etc.). Root cause (persona metaphor) left untreated.
- **(B) Full persona rewrite**: Rejected. Would risk drift in personality/warmth/safety coherence. Minimal surface change chosen instead.
- **(C) Remove persona entirely, rely on safety rules alone**: Rejected. Mungi needs a positive self-identity to respond to "너는 누구야?" and similar identity questions with warmth rather than a neutral assistant tone.

## Consequences

### Positive

- Persona-safety coherence improves: no more puppy-specific drift in `llm_system_prompt` defaults.
- Identity row aligns with product marketing ("AI friend").
- Safety/`safety/template.py` logic no longer needs to guard against pet-related improvisation in the persona.
- Rollout is additive and minimal (4 lines), testable with existing 12-wav + 30-turn + 8-turn empirical corpora.

### Neutral

- Persona warmth preserved via peer-friendship framing (`또래처럼`). Child UX should be indistinguishable in tone.
- Safety template `electrical_outlets` block path already bypasses persona layer, so Session 5 T7/T8 behavior is unchanged.

### Negative / risks

- Child's follow-up questions like "너는 무슨 동물이야?" no longer map to puppy. Answer now defaults to "뭉이는 동물이 아니라 AI야 — 네 친구 같은 인공지능이야!" kind of responses. Response pattern needs to be validated in a future child-voice test. Empirical risk: low (current personality rules already encourage honesty about what Mungi is).
- Out-of-scope legacy `puppy` references in `assets/prompts/child_safe_system_en.txt` create a small inconsistency until the cleanup ADR is filed.

## Validation

- Codex task `persona-ai-companion-and-max20` (2026-04-21, 517.9s, PASS).
- `ruff check .` + `ruff format --check .` + `mypy core/ models/ safety/ hardware/ scripts/ parental/` + `pytest tests/ --ignore=tests/integration_jetson -v --tb=short` -> 936 passed, 3 skipped, 78.73% coverage.
- Real-voice validation deferred to next real-voice session (no safety-critical failure expected; only persona-consistency check).

## References

- `assets/prompts/persona.md`
- `core/pipeline.py` (`llm_system_prompt` default)
- `docs/runbooks/weekly/archive/2026-04-21-daily-worklog.md` — Session 5 addendum
- CLAUDE.md §1 product vision + Five Approval Gates
- CLAUDE.md §12 language policy
- ADR 0067 — Gemma 4 Apache 2.0 compliance
