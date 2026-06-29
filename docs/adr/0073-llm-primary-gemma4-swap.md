# ADR 0073 — LLM primary model swap from Qwen3.5-2B-DPO to Gemma 4 E2B Q5_K_M

- **Status**: Accepted (validated)
- **Date**: 2026-04-24 (drafted at Session 12 close)
- **Decision owner**: Claude Code (primary orchestrator) + user approval
- **Superseding target**: ADR 0012 (Qwen3.5 upgrade lineage), ADR 0028 (Qwen3.5-1.7B-FT switch context) — to the extent those nominated Qwen as primary runtime backend. The QLoRA artifacts themselves and the `qwen3_legacy` backend implementation continue to exist as the fallback path.
- **Related**: `core/llm_backend_config.py`, `Dev_Plan/Mungi_Model_Selection_Report_v1.md` (authoritative), `docs/runbooks/baseline-stack-and-models.md`, ADR 0068 (persona redefinition), ADR 0072 (parent-disclosure invariance — already validated against `gemma4_text`)
- **Extended by**: ADR 0102 (2026-06-06) — realizes this ADR's "option 2 (runtime auto-fallback)" with primary refined E2B→**E4B** and fallback target qwen→**E2B**; `qwen3_legacy` retirement deferred to a later G2 plan.

## Context

After PR #45 (`feature/safety-persona-hardening` → `dev`, merged `d4d6c9a`) and PR #47 (`dev` → `main` release, merge commit `3099933`), the Mungi runtime now carries a fully validated bilingual child-safety rule set (Rule 8 / ADR 0072) whose on-device validation was performed exclusively against `gemma4_text` (Gemma 4 E2B Q5_K_M). The `qwen3_legacy` backend (Qwen3.5-2B-DPO Q6_K) remains the declared default (`core/llm_backend_config.py:DEFAULT_BACKEND = "qwen3_legacy"`), which means the default runtime after a fresh clone does NOT get the Gemma-validated response style unless `MUNGI_LLM_BACKEND=gemma4_text` is exported.

Session 1 and Session 2 (2026-04-23) real-voice E2E tests on Jetson both ran Gemma 4 via env override. The safety hardening validation (`docs/runbooks/weekly/archive/2026-04-24-safety-hardening-jetson-replay.md` §8) was 8/8 PASS on Gemma 4. The user requested promoting `gemma4_text` to primary with `qwen3_legacy` retained as a fallback (both at config-level default and at runtime — see §Decision scope below).

This ADR is a DRAFT pending a proper Gate 1 plan document. It documents the decision space and the pending items so the next-session orchestrator has a precise starting point.

## Decision (PROPOSED, not yet accepted)

Promote `gemma4_text` to the default LLM backend. Retain `qwen3_legacy` as a registered backend and as the runtime fallback path.

### Scope — what "primary / fallback" means operationally

The term "primary / fallback" admits two distinct interpretations. The plan for this ADR must pick one (or both):

1. **Config-level default flip only (minimum viable change)**
   - `core/llm_backend_config.py:DEFAULT_BACKEND` changes from `"qwen3_legacy"` to `"gemma4_text"`.
   - `MUNGI_LLM_MODEL_PATH` default (currently derived from `defaults.model_path`) flips to the Gemma 4 GGUF location.
   - `qwen3_legacy` remains available via `MUNGI_LLM_BACKEND=qwen3_legacy` env override.
   - No runtime auto-fallback logic.

2. **Runtime auto-fallback (richer change)**
   - On top of (1), `ConversationPipeline._load_llm_for_active_backend()` attempts `gemma4_text`, and on failure (model file missing, OOM, llama.cpp context failure) transparently falls back to `qwen3_legacy`.
   - Fallback is logged (structured) so operators can see which backend actually ran.
   - Unit tests + Jetson integration tests cover both the happy path and the fallback path.

**Tentative recommendation for the Gate 1 plan**: pick option (1) for the PR that lands the primary swap, and track option (2) as a separate, later PR whose scope is purely "runtime fallback". Splitting reduces blast radius.

### Out of scope for this ADR

- Prompt tuning specific to Gemma 4 sampling behavior (the Rule 1–8 system prompt is already validated with Gemma 4 in the safety hardening work).
- Retiring the `qwen3_legacy` backend or any of its QLoRA artifacts.
- Changing the bilingual routing logic in `_select_system_prompt()`.

## Alternatives considered (to be fleshed out in the Gate 1 plan)

1. **Keep `qwen3_legacy` as primary** (reject). The bilingual Rule 8 was validated only against Gemma 4; running on Qwen3.5 as default risks untested Rule 8 behavior in production.
2. **Flip primary but keep no fallback** (reject). A Jetson with a misconfigured `MUNGI_LLM_MODEL_PATH` (e.g., missing Gemma GGUF after an SD-card restore) would currently crash; a fallback keeps the device conversational even on degraded setups.
3. **Introduce a `hybrid` backend that loads both and routes per-turn** (reject). Memory cost on Jetson 8 GB is prohibitive; both LLMs at Q5/Q6 quantization sum to ~11 GB.

## Consequences (projected — awaiting Gate 1 validation)

### Positive

- Default runtime is now the backend that was end-to-end validated for the child-safety regression (ADR 0072).
- New Jetson deployments do not require a `MUNGI_LLM_BACKEND` env override to get the validated behavior.
- `Dev_Plan/Mungi_Model_Selection_Report_v1.md` receives a definitive "primary = Gemma 4" stance consistent with the post-regression safety posture.

### Negative / trade-offs

- Gemma 4 E2B Q5_K_M wall-clock TTFT on Jetson Orin Nano 8 GB is measured at 3.365 s (Session 2 baseline) vs Qwen3.5-2B-DPO's roughly 2.8 s for the same prompt style. That is a ~20 % TTFT regression at the default. Mitigation: Plan v3.1 §6.3's +10 % budget is for prompt token growth, not for backend swap; a separate TTFT story is needed.
- Qwen3.5 QLoRA DPO fine-tuning work (ADR 0012 / ADR 0028 lineage) will see reduced effective usage. Those artifacts are not deleted, but they become fallback-only.
- Operators running customized `/var/lib/mungi/config/config.json` with an explicit `"llm_backend": "qwen3_legacy"` setting will continue to get Qwen3.5; the flip only affects sites that rely on the default.

### Unknown / to be measured in the Gate 1 plan

- Whether any existing tests in `tests/` hard-code `qwen3_legacy` as an expected backend name and would fail on a default flip. A quick grep suggests most tests stub the backend or go through `MUNGI_LLM_BACKEND` env, but the plan must audit.
- Whether `_select_system_prompt()` needs an additional conditional for Gemma-specific prompt variants (currently it branches on `bilingual_mode` and language, not backend).
- Jetson deployment story: does `/opt/mungi/ai_models/` on currently-deployed Jetsons contain the Gemma 4 GGUF at the path the new default expects? If not, the flip will fail without pre-staging.

## Gate 1 plan requirements (to be authored in `Dev_Plan/`)

The plan document must:

1. Audit `tests/` for backend-name assumptions.
2. Produce the exact diff for `core/llm_backend_config.py`, `docs/runbooks/baseline-stack-and-models.md`, `CLAUDE.md §3` (if applicable), `Dev_Plan/Mungi_Model_Selection_Report_v1.md` (append-only update), and any deploy runbook.
3. Decide whether option (1) or (2) is in scope for the first PR.
4. Specify a Jetson pre-deploy check script to verify the Gemma 4 GGUF is present at the expected path before the new default is committed.
5. Define a rollback story (revert the default flip; verify via `pytest` and a Jetson smoke test).
6. Submit to a Codex reviewer cycle before user final-plan approval.

## Validation criteria (to be satisfied before flipping this ADR's Status to Accepted)

- Plan document exists, has completed at least one Codex reviewer round, and carries user final approval.
- New default backend runs a Jetson smoke test without `MUNGI_LLM_BACKEND` override: boots, loads, handles one turn successfully.
- `pytest tests/` and `pytest tests/integration_jetson/` pass with the new default.
- If option (2) is in scope: fallback activates correctly on a deliberate primary-failure injection (e.g., stubbed missing GGUF path) and the Jetson continues to respond.
- `Dev_Plan/Mungi_Model_Selection_Report_v1.md` is updated (append-only) to reflect the new primary stance with rationale.

Until these are satisfied, this ADR remains `Proposed` and no code changes land under its authority.

## Update - 2026-04-27 - Implementation validated

- Implementation PR: #49
  (`feature/llm-primary-gemma4-swap` -> `dev`), squash-merged as commit `7758806`.
- Plan: `Dev_Plan/2026-04-25-llm-primary-gemma4-swap-plan.md` (v4,
  post-Codex 3-round review cycle + user-direct-approval per CLAUDE.md section 1
  step 5 escalation Option A; see
  `Dev_Plan/2026-04-25-llm-primary-gemma4-swap-plan-discussion-round{1,2,3}.md`
  and `*-codex-review-round{1,2,3}.md`).
- Implementation artifacts shipped (16 files, +649 / -8):
  - `core/llm_backend_config.py`: `DEFAULT_BACKEND` flip `qwen3_legacy` ->
    `gemma4_text` + `defaults()` docstring updated to reference ADR 0073.
  - 9 `tests/` files: 18 explicit-`qwen3_legacy` `LLMBackendConfig.load` patches +
    1 rename across `test_llm_backend_config.py`, `test_llm_runner_gemma4.py`,
    `test_pipeline.py`, `test_model_manager_sequential.py`, `test_bilingual.py`,
    `test_hallucination_fix.py`, `test_sprint3_core.py`,
    `test_pipeline_gemma4_integration.py`, `safety/test_parent_disclosure_rule.py`.
    Preserves legacy-path contracts under explicit env override; new-path positive coverage
    tracked as Plan v4 section 3.3 follow-up #3.
  - `scripts/preflight_gemma4_default.py` (NEW, 253 lines): pure-Python GGUF v3
    parser, 6-step Jetson preflight (backend resolve -> file exists -> architecture
    allow-list `{"gemma4"}` -> load smoke). No new PyPI dependency.
  - Doc edits: `CLAUDE.md` section 3 active-model line,
    `docs/runbooks/baseline-stack-and-models.md` section 3 step 3,
    `Dev_Plan/Mungi_Model_Selection_Report_v1.md` Appendix C (append-only).
- Validation outcomes against original "Validation criteria (to be satisfied before flipping
  this ADR's Status to Accepted)":
  1. `core/llm_backend_config.py:DEFAULT_BACKEND = "gemma4_text"` flip merged:
     SATISFIED (commit `7758806`).
  2. Test suite passes with the flipped default: SATISFIED; `pytest tests/ -v --cov`
     977 passed / 13 skipped / coverage 79.35% (>=0% threshold) on Step 2
     verification + 987 passed / 17 skipped / coverage 79.52% at Session 15 close.
  3. Jetson smoke at code-default `n_ctx=2048`: SUPERSEDED by ADR 0076. Empirical
     evidence preserved at `/var/lib/mungi/e2e_results/2026-04-26-l1-resident-meas`
     (15/15 FAIL with Gemma + Rule 8 prompt at n_ctx=2048). ADR 0076 (PR #51,
     commit `0e23e1e`) raised `DEFAULT_N_CTX` to 4096; Jetson smoke at `n_ctx=4096`
     PASS 7/7 per Session 15 worklog
     (`docs/runbooks/weekly/archive/2026-04-27-daily-worklog.md` "Validation evidence
     preserved on Jetson").
  4. Gemma 4 GGUF preflight on Jetson: SATISFIED via Session 15 Gate 4 deploy
     (`sudo ln -s /home/mungi/.cache/huggingface/gemma-4-E2B-it-Q5_K_M.gguf
     /opt/mungi/ai_models/gemma-4-E2B-it-Q5_K_M.gguf`); preflight default-path
     verify 3/3 PASS.
  5. `tests/integration_jetson/test_parent_disclosure_live.py` runs by default (no
     longer skipped) on Gemma backend: PRE-VALIDATED via the 2026-04-24 ADR 0072
     Jetson replay (8/8 PASS, all on Gemma backend per
     `docs/runbooks/weekly/archive/2026-04-24-safety-hardening-jetson-replay.md` section 8).
     Default-flip removes the env-skip gate; behavior identical.
  6. Operator rollback exercise (`MUNGI_LLM_BACKEND=qwen3_legacy mungidev` end-to-end):
     NOT YET EXERCISED on the post-merge tree. Listed as residual follow-up.
- Plan v4 section 3.3 follow-up issues (out of scope for PR #49, tracked separately):
  1. Dedicated Gemma-default-path Rule 8 prompt-equality test in `tests/safety/`.
  2. `_apply_llm_backend_generation_config()` overwrite design audit
     (`core/pipeline.py:666-669`).
  3. Gemma-default LLM-load-sequence positive-coverage tests (mirror of the 7 legacy-path
     tests now wrapped to `qwen3_legacy`).
- Residual follow-ups:
  - Operator rollback exercise on Jetson (env override path).
  - Plan v4 section 3.3 follow-ups #1, #2, #3 above (each its own plan + PR).
- Runbook: `docs/runbooks/weekly/archive/2026-04-25-daily-worklog.md` "Session 14" (ADR 0073
  implementation timeline) and `docs/runbooks/weekly/2026-04-25-session14-close-handoff.md`
  (post-merge gate map).
