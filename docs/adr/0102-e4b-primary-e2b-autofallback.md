# ADR 0102 — E4B primary LLM + E2B automatic load-failure fallback (G1)

- **Status**: Accepted (G1 implemented + verified 2026-06-06; pending merge to `dev`)
- **Date**: 2026-06-06
- **Decision owner**: Claude Code (primary orchestrator) + user approval
- **Extends**: ADR 0073 (realizes its "option 2 — runtime auto-fallback"; refines primary E2B→**E4B** and changes the fallback target qwen→**E2B**). The `qwen3_legacy` retirement is **deferred to a follow-up (G2) plan/ADR**.
- **Related**: `core/llm_backend_config.py`, `core/model_manager.py` (`load_gemma_with_fallback`), `core/pipeline.py`, `models/llm_runner.py`, `Dev_Plan/2026-06-06-e4b-primary-e2b-fallback-qwen-retire-plan.md` (v3 — G1 scope), `docs/runbooks/baseline-stack-and-models.md`.

## Context

Phase 1 made **E4B the live main model** via `/var/lib/mungi/config/config.json` `llm_model_path`, but the repo CODE default for the Gemma backend was still **E2B** (`models/llm_runner.py:DEFAULT_GEMMA4_TEXT_MODEL_PATH`). Two gaps followed:

1. A fresh clone / SD-restore / config-absent boot would resolve the **E2B** default, not E4B — so the live-validated primary was not pinned in the repo contract.
2. ADR 0073's "option 2" auto-fallback was never built; its sketch named **qwen** as the fallback target. With E4B now primary, the natural fallback is the smaller, already-validated **E2B**, not qwen.

We need: (a) pin E4B as the primary in the repo contract (not only via the runtime config file); (b) a real **load-failure** auto-fallback so a missing/broken primary GGUF keeps the device conversational; (c) telemetry for which model actually ran. Retiring `qwen3_legacy` is a larger, higher-blast-radius change (model-family/discovery coupling, ~15 tests) and is intentionally deferred so qwen remains a rollback path while the new E4B/E2B fallback is proven.

## Decision

1. **Pin E4B primary in code**: `DEFAULT_GEMMA4_TEXT_MODEL_PATH` → `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`; add `DEFAULT_GEMMA4_FALLBACK_MODEL_PATH` → `gemma-4-E2B-it-Q5_K_M.gguf`; add `llm_fallback_model_path` resolution (env `MUNGI_LLM_FALLBACK_MODEL_PATH` > config `llm_fallback_model_path` > E2B default), mirroring `model_path`. **Config-absent ⇒ E4B primary, E2B fallback** (resolved primary ≠ fallback, so the fallback is not a no-op). The fallback attempt is skipped only when resolved primary path == resolved fallback path.
2. **Manager-owned auto-fallback**: `ModelManager.load_gemma_with_fallback(primary, fallback, …)` runs inside the `_do_load("llm", …)` callable. It force-clears the GPU `llm` slot (calls the LLM **release hooks before** clearing — not the resident-skipping `_unload_current_gpu()`), attempts the **primary (E4B)**; on **load** failure it captures only a string reason (no traceback/frame refs → no pinned CUDA context), calls `_recover_cuda_memory_after_oom()`, then attempts the **fallback (E2B)**; if both fail it restores consistent manager state (`_current_gpu_model = NONE`) and raises. Fallback engages on **load failure only — never on bad model output**. The primary is released before the fallback loads, so E4B + E2B are **never co-resident** (Jetson 8 GB).
3. **Telemetry (distinct fields)**: `TurnMetrics` gains `llm_model_fallback_used`, `llm_model_path_actual`, `llm_model_fallback_reason` — distinct from the existing n_gpu-layer `fallback_used` diagnostic — propagated through `TurnMetrics.to_dict()` and every flattened E2E/demo record builder.

## Scope / non-goals

- **In scope (G1)**: the three decisions above. `qwen3_legacy` is **UNCHANGED** and remains a registered backend (rollback safety net).
- **Deferred to G2** (separate plan/ADR, after one stable E4B/E2B cycle): retiring the `qwen3_legacy` backend / dispatch branch / default model-family + stop-sequence discovery coupling; a warning/fail-fast for the retired `MUNGI_LLM_BACKEND=qwen3_legacy`; GGUF de-link/quarantine + discoverability exclusion; the ~15 qwen-asserting tests; and the broad qwen-fallback documentation cleanup (incl. ADR 0073, `CLAUDE.md`, runbooks).

## Alternatives considered

1. **Config-only E4B (no code pin)** — reject: a config-absent boot would silently run E2B and the fallback would be a no-op (primary == fallback).
2. **Keep qwen as the fallback target** — reject: E4B→E2B keeps a single validated Gemma family and avoids holding the qwen lineage on the critical path; qwen stays only as a manual rollback until G2.
3. **Hybrid load-both / per-turn routing** — reject: two LLMs co-resident exceed the Jetson 8 GB budget; force-clear-before-each-attempt is the memory-safe design.

## Consequences

### Positive
- Fresh / config-absent deploys get E4B primary with a working E2B fallback; the device stays conversational on a missing or broken primary GGUF.
- Operators see the actually-loaded model and the fallback reason via telemetry.
- qwen3_legacy retained as a rollback path while the new fallback is proven.

### Negative / trade-offs
- Two Gemma GGUFs (E4B + E2B) must be present on the runtime for the fallback to engage (additive disk; reversible).
- E4B TTFT > E2B (E4B is the larger model); accepted per the live-validation track.

## Verification

- `pytest` 4304 passed / 0 failed; `ruff check .` + `ruff format --check .` clean; `mypy core/ models/ safety/ hardware/ scripts/ parental/` clean except 2 PRE-EXISTING unrelated errors in `scripts/convert_emoji_to_character.py` (commit #141).
- Independent adversarial review: **PASS** (fallback state machine, load-only trigger, traceback-safe reason + CUDA recovery, no double-residency, E4B/E2B contract + consumer audit, telemetry completeness, qwen untouched).
- ACs asserted: config-absent contract (primary≠fallback), fallback-injection, **bad-output-does-NOT-trigger-fallback**, force-clear of a resident LLM, CUDA-recovery-before-fallback, skip-when-paths-equal, metric serialization in flattened records.
- **Pending (Gate 4)**: Jetson runtime smoke — E4B primary turn + injected primary-failure → E2B fallback.
