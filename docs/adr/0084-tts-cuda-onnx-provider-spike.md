# ADR 0084 — TTS CUDA ONNX Runtime Provider — Spike + Decision

- **Status**: Proposed (DRAFT — accept/defer pending Session 28 PR Z spike measurement)
- **Date**: 2026-05-07 (drafted); decision date pending PR Z execution
- **Author**: Claude Code orchestrator (PM)
- **Related plan**: `docs/archived/dev-plan/2026-05-07-session28-3findings-synthesis-plan.md` v3 §6.2 PR Z detail
- **Related ADRs**: 0058 (Wave 2 T2.1 TTS resident deferred — memory cost), 0065 (Wave 3 T3.1 GPU STT abort — concurrent contention), 0076 (L1 LLM resident default — memory invariant), 0046 (Supertonic primary TTS engine), 0083 (Stage-2 measurement architectural decisions)

## Context

Stage-2 Phase 1 measurement (2026-05-07, Configuration A and A2) surfaced that:

1. **Supertonic 1.1.2 is ONNX-Runtime-based** (4 .onnx models under `/opt/mungi/ai_models/supertonic-2/onnx/`: `vocoder.onnx`, `duration_predictor.onnx`, `vector_estimator.onnx`, `text_encoder.onnx`).
2. **`onnxruntime-gpu 1.23.0` is already installed** in the Jetson venv. `onnxruntime.get_device()` returns `'GPU'`. Available providers include `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`.
3. **Supertonic library hardcodes CPU-only execution**: `supertonic/config.py:128` defines `DEFAULT_ONNX_PROVIDERS = ["CPUExecutionProvider"]` with the comment `# GPU support can be added by extending this list`. Line 127 has a TODO `# Add parsing of SUPERTONIC_ONNX_PROVIDERS environment variable` indicating upstream awareness.
4. **Live load test (this session)**: Loading `supertonic.TTS(model_dir="/opt/mungi/ai_models/supertonic-2", auto_download=False)` on Jetson logs `Using ONNX providers: ['CPUExecutionProvider']` — confirms the hardcoded default in production.
5. ADR 0065 abort decision for STT GPU does NOT apply to TTS due to **pipeline sequencing**: STT runs concurrently with LLM offload (which ADR 0065 measured as TTFT 1.7×); TTS runs AFTER LLM completes per turn (sequential), so the concurrent-memory-contention pattern that drove ADR 0065 reject is absent for TTS.
6. ADR 0058 (TTS-resident deferred) was a memory-cost decision (450 MB resident TTS adds LLM partial-offload pressure). ADR 0058 did NOT examine GPU-vs-CPU TTS execution; it only addressed the resident-vs-per-turn-load axis.

Stage-2 Configuration A2 (production-default LLM-resident) measured:
- TTS load_count = 37 (loads per turn, not resident — consistent with ADR 0058 default-False)
- Average per-turn `total_ms = 21,637 ms`; portion attributable to TTS load+synth pending observability bundle (Session 28 PR 1)
- Peak `system_ram_mb = 5,707` (G2a 6,000 PASS, G1 5,500 FAIL by 207 MB — RAG-on context per F27-13)

A TTS GPU spike could plausibly:
- Reduce per-turn TTS load (CPU model load → GPU model load is typically faster after first GPU transfer)
- Reduce per-turn TTS synth latency (vocoder + vector estimator are common GPU-friendly ops)
- Add resident GPU memory cost (CUDA context + ORT session memory)

These tradeoffs cannot be predicted from precedent alone; they require direct measurement on this exact platform (JetPack 6.2 / CUDA 12.6 / Orin Nano Super 8 GB unified memory / sherpa-onnx 1.12.38+cuda12.6 build of `onnxruntime-gpu`).

## Decision (PROPOSED — pending PR Z spike)

This ADR proposes a measurement-driven decision procedure with binary outcomes:

### Procedure

1. **PR Z spike implementation** (per synthesis plan §6.2 step Z.2):
   - Insert ~5-line monkey-patch in `models/tts_runner.py:SupertonicEngine.load()` (or `core/model_manager.py:load_tts()` upstream) gated behind env var `MUNGI_TTS_ONNX_PROVIDER`:
     ```python
     # Pseudocode — actual diff in PR Z
     if os.getenv("MUNGI_TTS_ONNX_PROVIDER", "").lower() == "cuda":
         import supertonic.config
         supertonic.config.DEFAULT_ONNX_PROVIDERS = [
             "CUDAExecutionProvider",
             "CPUExecutionProvider",
         ]
     ```
   - The monkey-patch must execute BEFORE `supertonic.TTS(...)` is first imported with hardcoded defaults active.
2. **Measurement** (synthesis plan §6.2 step Z.4-Z.5):
   - Stage-2 Phase 1 Configuration A2 rerun with `MUNGI_TTS_ONNX_PROVIDER=cuda` set.
   - Compare to Config A2 baseline (CPU-only TTS, Stage-2 run `qwen3_mix_20260507_141439`):
     - `tts_load_ms` (per `summary.json` aggregate or per-turn from `rounds.jsonl`)
     - `tts_ms` (per-turn synthesis time)
     - `tts_first_chunk_ms`
     - Peak `system_ram_mb` and per-turn `process_rss_mb`
     - LLM `llm_ttft_ms` (regression check — should not be affected since TTS is post-LLM, but verify)
     - New TTS errors (`tts_load_error_count`, `tts_synth_error_count`)
3. **ADR finalization** (synthesis plan §6.2 step Z.6 — PM authors final ADR per outcome).

### Decision criteria

#### **ACCEPT** (status flips to "Accepted") — ALL of the following must hold:

| Criterion | Threshold |
|---|---|
| TTS load latency reduction | `tts_load_ms` average ≥ 30 % faster vs CPU baseline |
| TTS synthesis latency reduction | `tts_ms` average ≥ 20 % faster vs CPU baseline |
| Memory increase | Peak `system_ram_mb` increase ≤ 300 MB vs Config A2 baseline (5,707 MB) → new peak ≤ 6,007 MB; G2a 6,000 MB invariant tolerance ±50 MB acceptable |
| LLM regression | `llm_ttft_ms` average regression ≤ 100 ms (or 0 — TTS is post-LLM) |
| TTS errors | `tts_load_error_count == 0 AND tts_synth_error_count == 0` (no new error class introduced) |

If ACCEPT: ADR 0084 status → **Accepted**. Implementation:
- Make the env-var monkey-patch a permanent feature (orchestrator may upstream `SUPERTONIC_ONNX_PROVIDERS` to supertonic library per the existing TODO at `supertonic/config.py:127`).
- Production default decision (env-var default vs. hardcoded CUDA) deferred to follow-up PR with operator readiness review.
- Update CLAUDE.md §3 baseline stack section if production default flips.

#### **DEFER** (status: "Deferred to <future>") — ANY of the following:

| Criterion | Trigger |
|---|---|
| Memory increase | Peak `system_ram_mb` increase > 500 MB (G2a 6,000 MB invariant breach) |
| LLM regression | `llm_ttft_ms` average regression > 200 ms |
| Latency gain | TTS latency improvement < 10 % (insufficient ROI for ongoing complexity) |
| New TTS errors | Any `tts_*_error_count > 0` |

If DEFER: ADR 0084 status → **Deferred** with documented reason. Possible future revisit triggers:
- Hardware upgrade (Orin NX 16 GB unified memory)
- Supertonic library version bump that changes ONNX session memory characteristics
- Stage-3 budget allowing alternate TTS engine evaluation (Piper TTS comparative measurement)

## Rationale

### Why this is worth a spike, not just a defer-by-precedent

- ADR 0065 reject was specific to STT GPU (concurrent with LLM). TTS is sequential. **PM must not blanket-apply ADR 0065 across all GPU-acceleration questions.**
- ADR 0058 deferred TTS-resident, not TTS-GPU. The two are orthogonal axes. **Fresh measurement is needed.**
- Cost of spike: 30-45 min Jetson run + 30 min ADR finalization. Low.
- Cost of mis-deferring (rejecting without measurement): perpetuates per-turn TTS CPU load (currently 37 loads × ~hundreds of ms each = significant share of 21.6 s avg total_ms).
- Cost of mis-accepting (would surface in measurement): G2a memory budget breach surfaces in observability bundle (Session 28 PR 1).

### Why monkey-patch gated by env var (not direct library mutation)

- Preserves library default for safety (any other code path that imports `supertonic` without env-var set retains current behavior).
- Reversible: unsetting env var restores CPU-only.
- Upstream-friendly: aligns with the existing `supertonic/config.py:127` TODO (`SUPERTONIC_ONNX_PROVIDERS` env var) — Mungi's `MUNGI_TTS_ONNX_PROVIDER` is a near-mirror that maps cleanly to the eventual upstream.

### Why CUDAExecutionProvider only (not TensorRT, not the full chain)

- TensorRT requires per-model engine compilation (not just session creation) — significant additional setup cost; deferred to separate spike per synthesis plan §10.
- `[CUDAExecutionProvider, CPUExecutionProvider]` chain provides automatic CPU fallback if any model fails CUDA load — defensive default.

## Validation plan

(Identical to synthesis plan §6.2 step Z.4-Z.5; replicated here for ADR completeness.)

Stage-2 Phase 1 Configuration A2 rerun under PR Z workflow:
1. Pre-run: confirm `MUNGI_TTS_ONNX_PROVIDER=cuda` exported; confirm Jetson logs show `Using ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']` on TTS load.
2. Run with same 38-WAV pool (`/home/mungi/qwen3_test/20260507_stage2_cfgA_{ko,en}/`), same `--rag --repeat-passes 1 --max-rounds 0 --llm-max-tokens 128 --llm-n-gpu-layers 99`.
3. Generate `report.md` + `conversation.md` per `scripts/generate_e2e_report.py` + custom mix-runner conversation generator.
4. Tabulate decision-criterion metrics; compare vs Config A2 baseline (`qwen3_mix_20260507_141439`).
5. PM finalizes ADR 0084 status and decision rationale based on tabulated comparison.

## Consequences

### If ACCEPTED

- TTS execution path adds CUDA dependency on Jetson runtime; if `onnxruntime-gpu` is uninstalled or CUDA driver mismatch occurs, fall-back to CPU is automatic (provider chain).
- Memory budget shifts: peak `system_ram_mb` baseline increases by spike-measured delta; F27-13 G1 invariant revisit becomes more urgent.
- Operator runbooks need updated env-var documentation (`docs/runbooks/baseline-stack-and-models.md` + `docs/runbooks/jetson-deployment-operations.md`).
- Stage-3 latency planning may need to re-baseline against TTS-GPU-enabled measurement.

### If DEFERRED

- Stage-2 measurement TTS path remains CPU-only (current behavior).
- Per-turn TTS CPU load cost remains as observed (~300-500 ms per ADR 0058 baseline).
- ADR 0084 stays in "Deferred" status; revisit triggers documented in §Decision criteria DEFER section.
- Synthesis plan PR Z scope completes at the spike measurement phase; no production code changes ship from this work.

## Out of scope

- TensorRT execution provider — separate spike if PR Z accepts but latency gain marginal.
- Piper TTS comparative measurement — separate housekeeping if Supertonic CUDA defers.
- TTS-resident toggle (ADR 0058 deferred) — orthogonal axis; not changed by this ADR.
- Production default change (env-var default → CUDA hardcode) — deferred to post-ACCEPT follow-up PR with operator readiness review.
- Supertonic upstream `SUPERTONIC_ONNX_PROVIDERS` PR — only after ACCEPT validates value.
- `MUNGI_TTS_PROVIDER` (separate from `MUNGI_TTS_ONNX_PROVIDER`) — not introduced; env var name collision risk evaluated as low (TTS provider has only Supertonic vs Piper, not GPU-vs-CPU).

## Update (2026-05-13)

References to "Piper TTS comparative measurement" (lines 90, 143) and the
Supertonic-vs-Piper enumeration in the `MUNGI_TTS_PROVIDER` discussion (line 147)
are superseded by ADR 0088. Comparative measurement against Piper is no longer planned.
