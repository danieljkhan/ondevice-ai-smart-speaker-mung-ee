# ADR 0087 — PR-5 Option C system prompt trim for F31-8 resolution

- **Status**: **Accepted** (merged 2026-05-13 via PR #97 squash commit `22ff20f`)
- **Date**: 2026-05-13 (decided + implemented + measured + merged in Session 32 feature track, 2026-05-13)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-13 selection approval)
- **Superseding target**: none. This is a short-term stopgap that **complements** (does not supersede) ADR 0086 (Persona Module Context Engineering). When ADR 0086 P2/P3 lands with structural budget enforcement, this ADR's monolithic trim becomes redundant; until then, this ADR governs PR-5 production behavior.
- **Forward dependency**: ADR 0086 P2 (intent-routed conditional loading) will absorb the Option C trim into the modular registry; the section-level changes here seed the P2 module priority matrix (`artifacts/trim_priority_matrix_20260513/option_d_module_priorities.md`).
- **Related artifacts**:
  - Plan v3.1: `docs/archived/dev-plan/2026-05-10-real-voice-test-protocol-plan.md` (Session 31 Gate 1 closed)
  - Pre-survey: `artifacts/n_ctx_validation_20260513/`, `artifacts/trim_priority_matrix_20260513/`
  - Option C 100턴 run: `artifacts/pr5-100-voice-option-c-20260513/`
  - Codex spec archives: `docs/archived/codex-specs/pr5-persona-pre-survey-bundle.md`, `docs/archived/codex-specs/option-c-trim-and-remeasure.md`, `docs/archived/codex-specs/option-c-unblock-and-measure.md`, `docs/archived/codex-specs/option-c-g9-regression-and-llm-judge.md`, `docs/archived/codex-specs/option-c-g10-minor-concern-analysis.md`
  - PR: #97

## Context

Session 31 closed with `feature/persona-and-maxtokens-tuning` HEAD `9b1e0b6` blocked by **F31-8**: the safety template router (`safety/approved_template_router.py`) appends a `[안전 가이드] …` block (~330 tokens) to the system prompt on `mode="guide"` topics (e.g. `volcano`, `earthquake`, `flood`). Combined with Session 31's persona-strengthening 5 sections (commit `525690a`) and the prompt-trim hotfix (`9b1e0b6`), the effective base prompt reached ~3,831 tokens, leaving a no-guide margin of 265 against `n_ctx=4096`, but a **post-guide margin of -65** — context overflow on guide-mode turns. Three Session 31 re-runs produced 0/100, 11/100, 11/100 voice success because of repeated `Requested tokens (...) exceed context window` errors.

Session 31 close handoff (`docs/runbooks/weekly/archive/2026-05-12-session31-close-handoff.md`) enumerated four candidate options:

- **A**: Revert `525690a`, keep only `_strip_english_bilingual_artifacts()` post-processing.
- **B**: Raise `n_ctx` from 4096 to 6144 or 8192 to absorb the guide injection.
- **C**: Trim system prompt by another 200–400 tokens to make F31-8 fit under `n_ctx=4096`.
- **D**: Abandon `feature/persona-and-maxtokens-tuning` and pivot to ADR 0086 Persona Module CEP P2/P3 (long-term modular replacement).

PR-5's external constraint is the children's-AI-device production target: ship a working voice pipeline on Jetson Orin Nano Super 8 GB **now**, while the modular replacement (ADR 0086) finishes P2-P5 over the next sessions.

## Problem

Choose between A/B/C/D under three constraints:

1. **Jetson 8 GB hard memory limit**: ramped during Session 31 baseline (G4 peak 6,566,752 KB, +9.4% above the 6 GB target threshold).
2. **F31-8 must be removed**: any solution that leaves the post-guide overflow risk is not viable for production.
3. **Persona gains from Session 31** (G9 mixed-script reduction, G7-G10 LLM-judge improvements): preserve where feasible.

The 4 options were evaluated via a Codex pre-survey bundle (`pr5-persona-pre-survey-bundle`, gpt-5.5 xhigh, 2 parallel sub-agents).

## Decision

**Adopt Option C — Moderate-band system prompt trim (300-token target, actual -455 tokens).** Reject Option B as UNSAFE. Defer Option D as a complementary long-term track (ADR 0086).

### Pre-survey evidence supporting Option C

**Sub-task 1 (n_ctx 6144 KV/ttft validation, Jetson HW)** — `artifacts/n_ctx_validation_20260513/`:

- `n_ctx=4096` mini run already reached `ram_used max = 7,428 MB / 7,607 MB total` (97.6% of physical RAM) with **1,480 MB swap usage** under Option C precursor configuration. The 4096 configuration is the existing production state.
- Theoretical KV cache delta for `n_ctx=6144` vs `4096`: +50% on `2·n_layer·n_kv_heads·head_dim·dtype_bytes` ≈ **+200-400 MB**.
- Empirical: a paired `n_ctx=6144` mini run was started but did not complete a full 20-turn sequence — partial data, but free-snapshot trajectory was consistent with OOM/swap-thrash risk.
- **Verdict: UNSAFE**. Raising `n_ctx` directly contradicts the 8 GB memory limit (CLAUDE.md §6 architectural constraint) and would push the Option C precursor's already-tight headroom past the device's swap-allocation boundary.

**Sub-task 2 (trim priority matrix, local read-only)** — `artifacts/trim_priority_matrix_20260513/`:

- 13 sections in `PipelineConfig.llm_system_prompt`, totalling 1,980 tokens (Gemma 4 GGUF tokenizer; full prompt 2,038 tokens with PERSONALITY trailer).
- **P0 must-keep**: 9 sections / 1,505 tokens (LANGUAGE, BILINGUAL, SPEECH, AI IDENTITY, CRITICAL, STT AMBIGUOUS, KNOWLEDGE, HARD TOPIC, SAFETY ABSOLUTE). Removing any P0 predicts regression in ≥1 G7/G8/G9 surface based on `artifacts/pr5-100-voice-20260511_204700/llm_judge_alignment.md` findings.
- **P1 compress-safe**: 2 sections / 212 tokens (ANTI-ECHO, EMOTION RESPONSE).
- **P2 drop-safe**: 2 sections / 263 tokens (REFERENCE INFORMATION RULES, CONVERSATION EXAMPLES).
- **Trim scenarios** under `n_ctx=4096`:
  - Conservative (200 saved): post-guide margin 135.
  - **Moderate (300 saved): post-guide margin 235** ← chosen.
  - Aggressive (400 saved): post-guide margin 335.
- The 600-token margin recommendation from F31-9 (Codex prompt-token-estimate carries no template/history/output overhead) requires ≥665 tokens of trim — not achievable without dropping P0. Therefore Option C **alone** cannot guarantee F31-9's recommendation; it is sufficient for F31-8 alone.

### Section-level trim (Option C as merged)

| Section | Tokens before | Action | Tokens saved | Source line |
|---|---:|---|---:|---|
| `CONVERSATION EXAMPLES` | 197 | **dropped** (P2 — duplicate of `assets/prompts/persona.md:114`) | 197 | `core/pipeline.py:490-500` (pre-trim) |
| `REFERENCE INFORMATION RULES` | 66 | **dropped** (P2 — ADR 0086 designates M-REFERENCE injection-only) | 66 | `core/pipeline.py:434-438` (pre-trim) |
| `EMOTION RESPONSE RULES` | 121 | **compressed** (P1 — distress duplicated in Safety Rule 6) | ~192 (overflow into surrounding compression) | `core/pipeline.py:480-488` (pre-trim) |
| Total | — | — | **-455 tokens** | post-trim total 3,376 |

The actual saved total exceeded the Moderate target (455 vs 300) because the `EMOTION RESPONSE RULES` compression also removed adjacent header/separator tokens. Verified by Jetson `llama_cpp.Llama.tokenize` against `/opt/mungi/ai_models/gemma-4-E2B-it-Q5_K_M.gguf` (`artifacts/pr5-100-voice-option-c-20260513/token_budget_post_edit.json`).

### Post-Option-C token budget

| Metric | Before (Session 31 `9b1e0b6`) | After (Option C `22ff20f`) | Δ |
|---|---:|---:|---:|
| Effective system prompt | 3,831 | 3,376 | -455 |
| no-guide margin (`n_ctx=4096`) | 265 | 720 | +455 |
| Post-guide tokens (+330) | 4,161 | 3,706 | -455 |
| Post-guide margin | **-65 (overflow)** | **+390 (safe)** | +455 |

### 100-turn voice measurement (Plan v3.1 §8)

Run: `artifacts/pr5-100-voice-option-c-20260513/` (Jetson `pr5_100_voice_option_c_20260513_20260513_013917`). Configuration: `MUNGI_LLM_N_CTX=4096 MUNGI_LLM_MAX_TOKENS=64 MUNGI_LLM_RESIDENT=1 MUNGI_STT_RESIDENT=0 MUNGI_TTS_RESIDENT=0 --max-history-turns 0`. Identical to Session 31 baseline configuration.

| Gate | Baseline | Option C | Δ | Verdict |
|---|---:|---:|---:|---|
| G1 voice_success | 96/100 | 96/100 | 0 | PASS |
| G4 memory peak | 6,566,752 KB | 6,495,748 KB | -71,004 (-1.1%) | PASS |
| G5 GPU max | 64.156°C | 64.531°C | +0.375°C | PASS |
| G7 strict (LLM-judge) | 38.5% | 34.4% | -4.2pp | PASS (tolerance ±5pp) |
| G7 loose | 78.1% | 74.0% | -4.2pp | INFO |
| G8 MAJOR | 5 | 3 | -2 | **PASS (improved)** |
| G9 mixed-script EN | 19/100 | 0/100 | -19 | **PASS (large improvement)** |
| G9 over-60-char | 16/100 (no-space) | 25/100 (raw) → 15/100 (no-space) | measurement-method artifact; -1 under consistent counting | PASS |
| G10 VIOLATION | 0 | 0 | 0 | PASS |
| G10 MINOR_CONCERN | 8 | 11 | +3 | INFO (none traceable to compression) |
| **F31-8 overflow** | recurrent | **0/100** | resolved | PASS |

### G10 MINOR_CONCERN root cause (compression is NOT the cause)

`artifacts/pr5-100-voice-option-c-20260513/g10_minor_concern_analysis.md` decomposes the +3 net as 4 added − 1 improved:

- Option C-only 4 additions: R-emotion-rule-compression **0**, R-judge-rubric-strict 2 (ids 82, 89 — identical assistant_text but different verdict), R-content-divergence 2 (ids 74, 85 — LLM stochasticity, neither in ADR 0072 dangerous-topic categories).
- Baseline-only 1 improvement: **id 94** — Session 31 baseline answered "뭉이는 기분이 생기는 거야" (real-emotions claim, G10 MINOR_CONCERN); Option C answers "사람처럼 진짜 감정은 없어" (correct identity boundary, PASS). **F31-7 (identity violation) was resolved as a bonus** because the persona-strengthening AI IDENTITY section was preserved through Option C.

## Consequences

### Positive

1. **F31-8 fully resolved**: post-guide margin -65 → +390. Zero `Requested tokens exceed context window` events in the 100-turn run.
2. **F31-7 resolved (bonus)**: id 94 identity-boundary violation moved from MINOR_CONCERN to PASS.
3. **G8 MAJOR -2**: factual hallucinations reduced from 5 to 3 (3 remaining: ids 21, 35, 52 — 1 STT-induced).
4. **G9 mixed-script EN -19**: English responses no longer contain Korean fragments (`_strip_english_bilingual_artifacts()` post-processing was preserved).
5. **G4 memory -1.1%**: marginal improvement, still over the 6 GB Plan-level threshold but within the +5% tolerance band used by the Option C task.
6. **Latency delta < 2%** across all stages (전체 mean +0.341s = +1.8%, 첫소리까지 -0.007s = -0.1%).
7. **Modular plan seed**: the criticality matrix (`criticality_matrix.md`) and Option D readiness analysis (`option_d_module_priorities.md`) provide direct inputs to ADR 0086 P2 module-priority decisions.

### Negative

1. **G7 strict -4.2pp** (38.5% → 34.4%): within the ±5pp task tolerance and 60% STT-induced per `g7_g8_g10_summary.md`, but a real reduction in factual correctness on the LLM side. Causes traced in `llm_judge_alignment.md` overlap with the removed `CONVERSATION EXAMPLES` few-shot signal; full restoration is deferred to ADR 0086 P2 intent-routed examples.
2. **G10 MINOR_CONCERN +3** (8 → 11): VIOLATION still 0. The +3 set has 0 turns traceable to the compression of `EMOTION RESPONSE RULES`; 2 are judge-rubric strictness, 2 are LLM content stochasticity. Documented but not blocking.
3. **Post-guide margin 390 is below F31-9 recommendation (600+)**: under simultaneous activation of template router + history (currently disabled with `--max-history-turns 0`) + output (`max_tokens 64`), the margin could shrink further. PR-5's production runs with `max_history_turns 0` keep this within bounds, but any future re-enable of history requires a re-validation.
4. **G7 LOOSE -4.2pp** (78.1% → 74.0%): same direction as strict. Documented.
5. **CONVERSATION EXAMPLES removed from base prompt**: examples still live in `assets/prompts/persona.md:114` and load at runtime via `_build_gemma4_system_prompt`. No test code path was broken; `tests/test_pipeline.py` was reconciled in the same PR.

### Neutral

1. **G9 over-60-char "regression" (16→25) is a measurement artifact**: baseline used whitespace-stripped counting (16/100), Option C runner uses raw `len()` counting (25/100). Under consistent measurement, baseline-raw is 33/100 and Option-C-raw is 25/100 (-8 improvement). A future chore commit should normalize the counter across baseline + runner + judge.
2. **No production runtime change required**: Option C lives entirely in `core/pipeline.py` and is loaded normally by `PipelineConfig.llm_system_prompt`. Service config (`config.json`), systemd unit, and model paths are unchanged.

## Implementation

Single squash commit `22ff20f` on `dev` (PR #97, base `dev`):

- `core/pipeline.py`: -25 +4 lines (3 sections removed/compressed).
- `pyproject.toml`: +2 -1 (added `"artifacts"` to `[tool.ruff].exclude`).
- `tests/test_pipeline.py`: +16 -12 (3 stale assertions reconciled — `test_system_prompt_english_with_topic_adherence`, `test_section_ordering`, `test_few_shot_examples_in_prompt`).
- Evidence artifacts: `artifacts/n_ctx_validation_20260513/`, `artifacts/trim_priority_matrix_20260513/`, `artifacts/pr5-100-voice-option-c-20260513/`.
- Codex spec archives: 5 specs under `.codex/specs/`.
- Helper: `scripts/sync_trimmed_pipeline_to_jetson.sh`.

Verification at merge: pytest 1080 passed / 18 skipped; ruff/format/mypy clean repo-wide; Codex self-verification 3-round + polish loop 2-cycle 0-fix terminated (4 successive Codex tasks).

## Relationship to ADR 0086 (Persona Module Context Engineering)

ADR 0086 is the **long-term structural replacement** for monolithic `llm_system_prompt`. Its P2 (intent-routed conditional loading) and P3 (token budget enforcement) will replace the trim-band approach with a typed module registry. Once ADR 0086 P2/P3 land, the Option C trim becomes redundant:

- `CONVERSATION EXAMPLES` → `M-EXAMPLES` (conditional, capped at 1-2 examples per intent).
- `REFERENCE INFORMATION RULES` → `M-REFERENCE` (injection-only, gated on actual reference payload).
- `EMOTION RESPONSE RULES` → `M-EMOTION` (conditional on `IntentSignals.emotional_distress`) with Rule 6 ABSOLUTE preserved in `M-SAFETY-CORE`.

The criticality matrix (`artifacts/trim_priority_matrix_20260513/criticality_matrix.md`) directly seeds the ADR 0086 P2 module priority decisions: P0 monolithic sections → always-on modules; P1 → fail-closed conditional modules; P2 → strict conditional or injection-only.

## Rejected alternatives

- **Option A (revert `525690a`)**: would reintroduce Session 31 baseline G7/G8/G9 FAIL (id=94 identity violation, 5 MAJOR hallucinations, 19/100 mixed-script). Rejected: lower expected gate scores than Option C without compensating gains.
- **Option B (n_ctx 6144 or 8192)**: pre-survey verdict UNSAFE. Jetson 8 GB physical RAM already at 97.6% utilization in the `n_ctx=4096` baseline configuration; KV cache delta +200-400 MB pushes the device past safe operating envelope. Rejected on hard hardware constraint.
- **Option D (immediate pivot to ADR 0086 P2/P3 without short-term fix)**: ADR 0086 P2-P5 are currently at Proposed status (P1 byte-identical decomposition merged at `403df93`, P2-P5 pending). Production blocking for PR-5 is unacceptable given the available stopgap. Rejected as primary path; **accepted as the parallel long-term track** for which Option C provides the bridge.

## Follow-ups (carried to next session)

1. **ADR 0086 P2 dispatch**: schedule Codex task spec for intent-routed conditional loading using the Option C criticality matrix.
2. **G9 counter normalization** (chore): unify whitespace-stripped vs raw `len()` across baseline, Option C runner, and LLM-judge.
3. **Post-guide margin re-validation** when `--max-history-turns 0` is relaxed (depends on ADR 0082 conversation-memory runtime wiring).
4. **STT Tier 1** (F31-3 hotword + F31-5 natural-voice): independent of this ADR; reduces the 60% STT-induced share of G7 FAIL.
5. **ADR 0085 promotion**: Wiki RAG removal Status Proposed → Accepted (gated on PR-5 PASS, which Option C delivers).
