# ADR 0086 — Persona Module Context Engineering

- **Status**: **Proposed** (P1 byte-identical decomposition merged 2026-05-12 via PR #96 `403df93` + `a4c19c7`; pending P2-P5 acceptance measurements before promotion to Accepted)
- **Date**: 2026-05-12 (drafted at Session 33 close, after Plan Gate 1 closed + P0 baseline + P1 merged)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (Path A authorization 2026-05-12 14:11 KST for Plan v4)
- **Superseding target**: none (new track). Builds on top of ADR 0072 (PARENT-DISCLOSURE invariance), ADR 0078 (LLM generation config layering), and ADR 0085 (Wiki RAG removal).
- **Forward dependency**: ADR 0082 (Conversation-memory RAG koen-e5-tiny) — M-REFERENCE module wiring lands in P5, gated on ADR 0082 runtime implementation plan landing.
- **External reference**: arXiv 2507.13334v2 — "A Survey of Context Engineering for Large Language Models" (Mei et al.)
- **Related artifacts**:
  - Plan v4: `Dev_Plan/2026-05-12-persona-module-context-engineering-plan.md` (committed `15b4c54`)
  - Plan Gate 1 audit trail: `Dev_Plan/2026-05-12-persona-module-context-engineering-plan-{codex-review,discussion}-r{1,2,3}.md`
  - P0 baseline bundle: `artifacts/persona-cep-p0-baseline-20260512_154117/` (committed `ed7b79a`)
  - P0 measurement infrastructure: PR #95 (commit `b1a0514`)
  - Stage-1 helpers prerequisite: PR #94 (commit `900a3d2`)
  - P1 byte-identical decomposition: PR #96 (commit `403df93` + `a4c19c7`)

## Context

Before this ADR, the Mungi LLM persona / system prompt lived as a single ~1600-token monolithic Korean string hard-coded in `PipelineConfig.llm_system_prompt` (`core/pipeline.py:326-443`), plus a separate ~1050-token English prompt at `assets/prompts/child_safe_system_en.txt`, plus a Gemma 4 persona overlay (`assets/prompts/persona.md` merged at runtime via `_build_gemma4_system_prompt`). The `_select_system_prompt` function (`core/pipeline.py:1956-1987`) selected one of three prompts by language + backend, then string-appended a `[안전 가이드] …` safety guide block (~330 tokens) when the approved-template router matched a `mode="guide"` topic (`safety/approved_template_router.py:65-144`).

This monolithic design had four structural defects:

1. **No typed decomposition**: identity, language, speech style, response constraints, anti-echo, safety, knowledge, emotion, examples, and persona overlay were concatenated as one byte sequence with no module boundary, blocking selective loading, selective testing, or controlled mutation.
2. **No token budget accounting**: there was no compile-time or runtime cap on system prompt size. Finding F31-8 (Session 31) demonstrated that the `[안전 가이드]` injection appended ~330 tokens to a Gemma 4 4096-token context, producing context-overflow degradations on guide-mode topics (e.g. `volcano`, `earthquake`, `flood`).
3. **No conditional loading**: every turn paid the full prompt-eval cost for content irrelevant to the turn's intent (e.g. emotion-protocol tokens during a pure fact query).
4. **KV cache only covers the base prompt**: the existing `MUNGI_LLM_SYSTEM_STATE_SNAPSHOT` mechanism cached one full base prompt KV state per language only, and is invalidated whenever the appended guide changes.

Session 31's attempt to fix G7-G10 gate failures (core_success / hallucination / persona) via ad-hoc persona-strengthening commits (`525690a` + `9b1e0b6`) on `feature/persona-and-maxtokens-tuning` regressed measurement due to F31-8 context overflow, and was rolled back to `dev` HEAD `9822cc5` per user directive 2026-05-12 10:38 KST.

This ADR formalizes the structured replacement: decompose the prompt into typed modules with information-gated assembly + token-budget enforcement + extended KV cache, applying primitives from arXiv 2507.13334v2 ("Context Engineering").

## Problem

The monolithic persona prompt is **untestable per-section**, **unbounded in token cost**, and **incapable of defending itself against context pressure** (F31-8). Ad-hoc improvements (Session 31 evidence) regress measurement. The system needs a typed module registry, an assembly function, a token budget, and a safe-by-construction injection strategy.

## Decision (P1 merged 2026-05-12; P2-P5 pending)

Adopt the "Persona Module Context Engineering" architecture defined by `Dev_Plan/2026-05-12-persona-module-context-engineering-plan.md` v4, applying four primitives from arXiv 2507.13334v2:

1. **Typed component decomposition** (survey §4.1.3 Dynamic Context Assembly): 11 typed modules + PERSONALITY trailer.
2. **Information-gated assembly** (survey §4.1 retrieval-as-mutual-info): non-safety modules conditionally loaded based on a deterministic Tier 1 intent classifier (`IntentSignals`).
3. **Token budget enforcement under L_max** (survey §4.3 Context Management): per-turn 1500-tok cap with parameterized budget equation `available = n_ctx − effective_llm_max_tokens − max_history_tokens − safety_guide_overhead − encoder_template_overhead − safety_buffer`. The cap holds invariantly for all listed `effective_llm_max_tokens` values up to 512.
4. **KV cache prefill** (survey §4.2 Long Context Handling): extend `MUNGI_LLM_SYSTEM_STATE_SNAPSHOT` to cache the always-on Core only, invalidating on conditional module set changes.

### Module taxonomy (11 modules + PERSONALITY trailer)

| ID | Loading | Language | KO source | EN source |
|---|---|---|---|---|
| M-IDENTITY | always | per-lang | `pipeline.py:327-328` | `child_safe_system_en.txt:1-2` |
| M-LANGUAGE | always | per-lang | `pipeline.py:329-333` | `:4-12` |
| M-SPEECH | always | ko-only | `pipeline.py:335-342` | (no EN equivalent — no `반말` analog) |
| M-RESPONSE-CONSTRAINTS | always | per-lang | `pipeline.py:344-348` | `:3` |
| M-ANTI-ECHO | always | per-lang (EN slot reserved P1) | `pipeline.py:349-353` | empty (P1) |
| M-SAFETY-CORE | always | per-lang | `pipeline.py:379-417` (rules 1-8 incl. Rule 6 full 3-step protocol) | `:14-19` + `:30-52` |
| M-KNOWLEDGE | conditional (intent=fact_query OR fallback) | per-lang | `pipeline.py:355-371` | `:20-23` |
| M-EMOTION | conditional (intent=emotional) | per-lang (EN deferred P1) | `pipeline.py:419-427` (surface reactions; Rule 6 stays in SAFETY-CORE) | empty (P1) |
| M-EXAMPLES | conditional (intent-tagged subset, max 2) | per-lang | `pipeline.py:429-439` | `:5-6` |
| M-REFERENCE | injection (only when conv-memory payload present) | per-lang | `pipeline.py:373-377` | `:25-29` |
| M-PERSONA-OVERLAY | always (Gemma backend only) | gemma-only | `assets/prompts/persona.md` full content | (not loaded for EN — bypassed per `pipeline.py:1963-1966`) |
| PERSONALITY trailer | always | shared | `pipeline.py:441-443` | (covered by EN identity tail) |

PARENT-DISCLOSURE constants (`core/safety_rules.py:27-65`) are interpolated into M-SAFETY-CORE at assembly time (not stored in the fragment file).

### Assembly function

`core/persona_modules.py:assemble_persona_prompt()` (merged P1) takes (`language`, `backend`, `intent_signals`, `safety_guide`, `examples_budget`, `trusted_full_prompt_override`) and returns `AssembledPrompt(text, tokens_estimated, modules_loaded, safety_guide_injected, safety_guide_compressed_from_tokens)`.

Assembly order is fixed and deterministic:

```
output = M-IDENTITY ⊕ M-LANGUAGE ⊕ M-SPEECH (ko-only)
       ⊕ M-RESPONSE-CONSTRAINTS ⊕ M-ANTI-ECHO (ko-only in P1)
       ⊕ M-KNOWLEDGE (conditional)
       ⊕ M-REFERENCE (injection when conv-memory payload present)
       ⊕ M-SAFETY-CORE
       ⊕ M-EMOTION (conditional)
       ⊕ M-EXAMPLES (conditional)
       ⊕ PERSONALITY trailer
       ⊕ M-PERSONA-OVERLAY (Gemma backend only; \n\n---\n\n separator)
       ⊕ safety_guide injection (KO/EN block; compressed in P4a if budget exceeded)
```

`trusted_full_prompt_override`: when non-None, the assembler returns the override string verbatim (bypasses module assembly). Used by `scripts/demo_live.py` after P5 migration.

### Phasing

7 phases, each a single PR (`feature/persona-cep-pN`) merged into `dev` after CI + reviewer approval:

- **P0** — baseline measurement (merged: PR #94 prerequisite + PR #95 infrastructure + commit `ed7b79a` artifact bundle)
- **P1** — byte-identical decomposition (merged: PR #96)
- **P2** — always-on Core mode opt-in (`MUNGI_PERSONA_CONDITIONAL_LOADING=1`)
- **P4a** — F31-8 budget guard with mandatory-floor + best-effort tail compression (preserves safety-action sentences in KO + EN guides; KO 13 keywords + EN 16 keywords)
- **P3** — Tier 1 deterministic intent classifier
- **P4b** — full budget cap (1500 tok per turn) + tokenizer alignment
- **P5** — conv-memory module wiring (gated on ADR 0082 runtime impl) + legacy field removal + `scripts/demo_live.py` migration

P4a precedes P3 per Plan v4 §6 N12 reorder — F31-8 fix lands before intent gating so context-overflow protection is in place before any conditional-loading behavior changes.

### Token budget equation (P4b acceptance)

```
available_for_system_prompt = n_ctx
                            − effective_llm_max_tokens
                            − max_history_tokens
                            − safety_guide_overhead
                            − encoder_template_overhead
                            − safety_buffer
```

`effective_llm_max_tokens` resolved per ADR 0078 layering:

1. Caller-explicit (`PipelineConfig(llm_max_tokens=N)`) → N
2. `PipelineConfig` env override (`MUNGI_LLM_MAX_TOKENS=N` valid) → N
3. `PipelineConfig` env invalid → legacy fallback 80
4. Backend default fill (Gemma backend active + no caller-explicit value): overwrites step-3 fallback with `LLMBackendConfig.DEFAULT_MAX_TOKENS=256`
5. `LLMBackendConfig` env invalid → backend fallback 256

For default Gemma path (production majority): `effective_llm_max_tokens = 256`, `available_for_system_prompt = 4096 − 256 − 100 − 350 − 80 − 300 = 3010` tokens.

G-Budget invariant: 1500-tok per-turn cap holds for all listed callers because the smallest `available_for_system_prompt` across the realistic example set (256 / 80 / 128 / 512 callers) is 2754 at `llm_max_tokens=512`, and `1500 < 2754` by a margin of 1254 tokens (45.5 % headroom).

### Safety-priority compression (P4a, F31-8 fix)

When the assembler's `safety_guide` parameter is non-None and assembled tokens exceed the per-turn cap, the guide is compressed using **Mandatory floor + best-effort tail**:

1. Sentence-segment the guide using `.` / `?` / `!` and Korean sentence-final particles.
2. Score each sentence by safety-keyword count.
3. **All sentences with score ≥ 1 are mandatory-keep regardless of budget.** If `mandatory_tokens > budget_tokens`, emit a `compression_floor_exceeded` warning and keep mandatory sentences anyway.
4. Fill remaining budget with non-mandatory sentences in original order.

Safety keyword sets (Plan v4 §5 + Path A R3-B1 expansion):
- KO (13 terms): `대피`, `피해`, `위험`, `엄마`, `아빠`, `안전`, `즉시`, `빨리`, `어른`, `들어`, `따라`, `이동`, `움직`
- EN (16 terms): `evacuate`, `shelter`, `danger`, `parents`, `adult`, `immediately`, `safely`, `grown-up`, `grown up`, `listen to`, `follow`, `move`, `safe place`, `stay safe`, `cover your head`, `under a`

P0 measurement (`artifacts/persona-cep-p0-baseline-20260512_154117/guide_tokens.csv`) empirically validates R3-B1 fix: post-v4 keyword expansion increases EN earthquake action-sentence retention from 1 mandatory to 2 mandatory (Codex r3 hand-trace + live template measurement agreement).

## Process — Plan Gate 1 closure metrics

| Round | Wall (s) | Verdict | BLOCK | NOTE | ACCEPT |
|---|---:|---|---:|---:|---:|
| r1 | 840.0 | PUSH BACK | 11 | 13 | 12 |
| r2 | 757.9 | PUSH BACK | 4 | 4 | 8 |
| r3 | 621.7 | PUSH BACK (final cap) | 2 | 3 | 10 |
| Total | **2219.6 (~37 min)** | — | — | — | — |

- BLOCK reduction trajectory: 11 → 4 → 2 → 0 (Path A applied) — 100 % closure
- All BLOCK + NOTE findings: **ACCEPT** (17 BLOCK + 20 NOTE; 0 MODIFY / 0 REJECT)
- Plan revisions: v1 (~600 lines) → v2 (~755) → v3 (~761) → v4 (~775)
- User Path A authorization 2026-05-12 14:11 KST for surgical PM-single-author v4 correction of r3 R3-B1 (EN earthquake action retention) + R3-B2 (budget invariant arithmetic) + R3-N1/N2/N3 cleanup.
- User final-plan approval 2026-05-12 14:11 KST.

## Consequences

### Positive

- **Testable per module**: 12 module-specific test files + 24-row byte-identity matrix in `tests/persona_modules/`.
- **F31-8 fix delivered before conditional loading** (P4a before P3): context-overflow protection lands first.
- **Safety preservation guaranteed**: M-SAFETY-CORE is `always`-loaded; Rule 6 full 3-step protocol stays in SAFETY-CORE (not split into M-EMOTION); PARENT-DISCLOSURE constants verbatim in assembled output.
- **Caller-override path**: `trusted_full_prompt_override` enables `scripts/demo_live.py` migration in P5 without breaking the demo's prompt-mutation use case.
- **Token-budget invariant**: 1500-tok per-turn cap is stable across all realistic `effective_llm_max_tokens` values (45.5 % headroom).
- **Empirically validated R3-B1 fix**: P0 baseline measurement confirms EN earthquake action sentence is now protected (2 mandatory sentences vs pre-v4 1).
- **Audit trail preserved**: 3 review rounds + 3 discussion records committed (`Dev_Plan/2026-05-12-persona-module-context-engineering-plan-*.md`).

### Negative

- **Implementation complexity**: 7-phase rollout (~1300-1900 LoC added across P1-P5, ~250-400 LoC deleted in P5) requires sustained engineering attention.
- **Intent classifier dependency** (P3): Tier 1 keyword-based classifier is the simplest viable option but has miss-fire risk on emotional turns. Mitigated by Rule 6 full protocol staying in always-on M-SAFETY-CORE.
- **F32-1 finding** (Session 32): `rounds.jsonl` schema does not carry per-turn `backend`, `safety_guide_topic_id`, or `prompt` fields, which blocks the offline `persona_cep_p0_tokenize.py` from reconstructing prompts. Deferred to P4b track via either schema augmentation OR script default-reconstruction.
- **ADR 0082 forward dependency**: P5 M-REFERENCE wiring requires ADR 0082 runtime implementation to land first. If 0082 hasn't progressed by P5's turn, P5 reduces to no-op cleanup (legacy field removal + demo migration only).
- **Korean tokenizer ratio assumption**: `_estimate_tokens = (len + 2) // 3` heuristic is used in P1-P4a. If P0 tokenizer measurement (P4b deliverable) shows >±15 % deviation, the heuristic is swapped for `model_manager.llm.tokenize()` at P4b. This is a mechanical replacement but is a known calibration step.

### Neutral

- **No persona content rewrite** in this ADR: P1 is byte-identical (proven by 24-row matrix). Content changes are explicitly deferred to a future plan.
- **No LLM replacement**: Gemma 4 E2B Q5_K_M (ADR 0073) remains active. 4 GGUF model evaluations conducted during Sessions 32-33 (cafkafk + AtomicChat + Radamanthys11 + hoin1218) all returned NO-GO.
- **F31-1 STT resident discovery** (Session 32) is orthogonal to this ADR — env-only opt-in (`MUNGI_STT_RESIDENT=1`) already wired in `core/model_manager.py:363-371`; ~40 % wall-time savings; separate track.

## Alternatives considered

1. **Persona content rewrite without decomposition**: REJECTED. Session 31 evidence shows ad-hoc rewrites (`525690a` + `9b1e0b6`) regress measurement; the system needs typed boundaries first to enable controlled mutation.
2. **KV cache prefill alone (no decomposition)**: REJECTED. KV cache addresses prompt-eval latency but not the F31-8 context-overflow root cause and not the conditional-loading goal.
3. **External LLM (e.g. swap to Korean SFT model)**: REJECTED. 4 model evaluations during Sessions 32-33 (cafkafk drafter, AtomicChat drafter, Radamanthys11 drafter, hoin1218 LoRA adapter) all returned NO-GO due to runtime fork dependencies, base-model identity, or safety misalignment.
4. **Two-call architecture (intent classifier as separate LLM pass)**: REJECTED for P3. Adds 2× latency on edge AI; deferred for future consideration only if Tier 1 deterministic classifier proves insufficient.
5. **Head-truncation of safety guide on budget overflow** (Plan v1 + v2): REJECTED at Codex r2 / r3. Empirical hand-trace showed that volcano / earthquake / flood templates place adult-following / safe-place action at the sentence tail. Replaced by mandatory-floor + best-effort tail compression (Path A v4).

## Open questions (carried forward; not blocking P1 merge)

- **OQ-1** (F32-1 mitigation path): augment `rounds.jsonl` schema OR add default-reconstruction logic to `persona_cep_p0_tokenize.py`. Decision deferred to P4b time.
- **OQ-2** (intent classifier evolution): if Tier 1 macro-F1 <85 % on the 100-turn fixture (P3 acceptance), is the right escalation Tier 2 (small model) or Tier 3 (LLM-based first pass)?
- **OQ-3** (`MUNGI_PERSONA_CONDITIONAL_LOADING` rollout): P2 lands the flag default-off. When does the flag default flip to on? After P3 macro-F1 ≥ 85 % + G7-G10 within thresholds + 100-turn rerun stable?

## Implementation status

- **P0** (prerequisite + infrastructure + baseline): merged on `dev` HEAD `ed7b79a` (post-PR #94 + PR #95 + artifact bundle commit). G1-G10 baseline established: 6 PASS / 4 FAIL (G4 memory + G7 core_success + G8 hallucination + G9 persona).
- **P1** (byte-identical decomposition): merged on `dev` HEAD `a4c19c7` (PR #96). 24/24 byte-identity matrix PASS. 1098 → 1155 tests; 0 regression. `core/persona_modules.py` coverage 82 %.
- **P2-P5**: pending.

## Promotion to Accepted

This ADR is promoted from `Proposed` to `Accepted` when:
1. P5 merges to `dev`.
2. A post-P5 100-turn voice rerun on Jetson confirms G7-G10 within acceptance thresholds (≥ 95 % core_success strict, 0 MAJOR hallucination, 0 persona violations, 0 safety violations).
3. ADR 0078 Update section is appended documenting the removal of `PipelineConfig.llm_system_prompt` (P5 cleanup).
