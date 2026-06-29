# ADR 0076: L1 LLM resident default mode

- **Status**: Accepted (validated)
- **Date**: 2026-04-26

## Context

Session 12 and the L1 rollout measurements showed that the default Gemma 4 text path spends most
user-visible latency on repeated LLM load/unload work. Text-input resident measurements on Jetson
Orin Nano Super with no RAG showed stable system RAM headroom and hot-turn latency improvement
across seed=1 and seed=42 runs. The same investigation showed n_ctx=2048 is unusable after the Gemma
4 swap because the production prompt, persona, and early conversation history overflowed the context
window at turn 1 in preserved pre-bump evidence.

## Decision

Make L1 resident mode the default by setting ManagerConfig.llm_resident=True. Operators can disable
the default with MUNGI_LLM_RESIDENT=0 or the text E2E runner's --no-llm-resident option. Bundle the
DEFAULT_N_CTX=4096 bump for the Gemma 4 backend defaults and llm_runner default loaders because the
prior 2048-token default is empirically too small for the current prompt stack.

## Validation

- Unit coverage asserts the new `ManagerConfig.llm_resident` default, the
  `MUNGI_LLM_RESIDENT` truthy/falsy/invalid semantics, and the `DEFAULT_N_CTX=4096`
  backend default.
- Jetson-only gates validate resident text-input memory, hot-turn latency, Korean TTFT,
  and a 27-turn long-conversation KV-cache regression path. The long gate parses
  `rounds.jsonl` for per-turn success/error data and `tegrastats.log` for system RAM.
- ADR status remains `Proposed` until the Jetson smoke gates pass with peak system RAM
  below 5500 MB and no crossing of the existing 6000 MB critical guard.

## Consequences

The default path keeps the LLM loaded between turns, improving hot-turn latency while retaining the
existing ManagerConfig.memory_limit_mb=6000 critical guard. This L1 default remains valid only while
Jetson smoke peak system RAM stays below 5500 MB; if RAG, STT residency, or full-audio residency
pushes peak RAM above that invariant, the default must be revisited and operators should roll back
with MUNGI_LLM_RESIDENT=0. models.llm_runner.run_chat_generation() does not call llm.reset(), so
bounded KV-cache behavior is validated empirically by the long-conversation regression gate rather
than by a new reset hook. Invalid MUNGI_LLM_RESIDENT values follow the existing shared boolean
helper behavior: they warn and disable resident mode instead of preserving the default. The Gemma 4
GGUF symlink at the default model path remains a runtime prerequisite.

## Operator notes

- Accepted truthy values for `MUNGI_LLM_RESIDENT` are `1`, `true`, and `yes`.
- Accepted falsy values are `0`, `false`, and `no`.
- Empty or unset preserves the default. Invalid values such as `on`, `off`, `truee`, or
  `2` emit `UserWarning` and disable resident mode.
- Stage 2 must measure full-audio plus RAG and optional STT residency before removing the
  environment-variable rollback path.

**Korean TTFT threshold (revised 2026-04-27):** The original Session 2 baseline of 3.365 s from ADR 0073 plan v3.1 section 6.3 is invalidated as an L1 regression target by cumulative changes since then: PR #45 Rule 8 prompts, PR #49 Gemma 4 default, and the n_ctx 2048->4096 increase. During the L1 implementation Jetson smoke, empirical hot-turn mean TTFT was 3.94 s, consistent across two short-gate runs and a long-conversation gate run at 3.89 s. The Korean TTFT gate threshold is now 4.50 s, which is the empirical 3.94 s value plus roughly 14% cushion and treats L1 with the current Gemma 4 default and Rule 8 prompt stack as the new TTFT baseline. A separate follow-up plan tracks TTFT-rise root-cause attribution across prompt growth, KV path, and tokenizer/sampling; it is not a blocker for L1.

## Related ADRs

- ADR 0073 - LLM primary model swap to Gemma 4
- ADR 0009 - Sequential GPU loading

## References

- Dev_Plan/2026-04-26-l1-llm-resident-rollout-plan.md
- core/model_manager.py ManagerConfig.llm_resident
- scripts/e2e_60rounds_text_tts.py resident-mode Jetson gates

## Update - 2026-04-27 - Implementation validated

- Implementation PR: #51
  (`feature/l1-llm-resident-default` -> `dev`), squash-merged as commit `0e23e1e`.
- Plan: Plan v3 (post-Codex 3-round review cycle: R1 PUSH BACK -> R2 PUSH BACK ->
  R3 APPROVE WITH NOTES) + 1 BLOCK-during-implementation resolved via user-decided
  Option A (Korean TTFT threshold revision). See
  `docs/runbooks/weekly/archive/2026-04-27-daily-worklog.md` "Plan Gate 1 metrics" +
  "Codex implementation metrics" for the full cycle (30.6 min review wall + 46.5 min
  implementation wall).
- Implementation artifacts shipped (19 files, +1467 / -8):
  - Source (4 modifications): `core/llm_backend_config.py`, `core/model_manager.py`,
    `models/llm_runner.py`, `scripts/e2e_60rounds_text_tts.py` (new
    `--no-llm-resident` flag and `MUNGI_LLM_RESIDENT` env wiring).
  - Runbook: `docs/runbooks/baseline-stack-and-models.md`.
  - ADR (this file).
  - Tests (5 modifications + 1 Jetson modification + 1 NEW Jetson): unit-test updates
    for the new defaults + `MUNGI_LLM_RESIDENT` truthy/falsy/invalid semantics + 4 new
    Jetson L1 gates + 1 memory-gate bug fix.
  - Plan trail: 6 NEW files under `Dev_Plan/` (Plan v3 + 2 discussions + 3 Codex
    reviews).
- Validation outcomes against original Validation criteria:
  1. Unit coverage of `ManagerConfig.llm_resident` default + `MUNGI_LLM_RESIDENT`
     semantics + `DEFAULT_N_CTX=4096`: SATISFIED; local QC at Session 15 close:
     987 passed / 17 skipped / coverage 79.52%.
  2. Jetson smoke gates (resident text-input memory + hot-turn latency + Korean TTFT +
     27-turn long-conversation KV-cache regression): 7/7 PASS per Session 15 close
     worklog (`docs/runbooks/weekly/archive/2026-04-27-daily-worklog.md` "Validation evidence
     preserved on Jetson"). Long-gate parses both `rounds.jsonl` per-turn success and
     `tegrastats.log` system RAM as required.
  3. Peak system RAM `<5500 MB` and no crossing of the existing `6000 MB` critical guard:
     SATISFIED for the L1-text-default scope. Per-gate peaks: text-resident-short
     4280 MiB, long-conv KV gate 4324 MiB, resident seed=1 4232 MiB, resident seed=42
     4249 MiB, env-disable smoke (baseline) 3830 MiB. All <5500 MB. Audio memory gate
     measured peak 6079 MiB but is OUT OF SCOPE for the L1-text-default invariant
     (audio path is a separate residency decision, listed in Consequences as requiring
     revisit if RAG / STT / full-audio residency pushes RAM above the invariant).
- Residual follow-ups (per Session 15 worklog "Outstanding follow-ups"):
  1. TTFT regression investigation (Session 15 section 10 #1): RESOLVED; Session 16
     dropped per Option Y (acknowledged as comparison artifact, not confirmed
     regression; non-revertible contributors; user-impact dominated by L1's ~2%
     full-turn improvement). See `docs/runbooks/weekly/archive/2026-04-27-daily-worklog.md`
     "Session 16" for rationale.
  2. Stage-2 measurements: full-audio + STT/TTS-resident memory; RAG-enabled memory;
     30+ min sustained thermal gate.
- Runbook: `docs/runbooks/weekly/archive/2026-04-27-daily-worklog.md` "Daily worklog -
  2026-04-27 (Session 15 close)".
