# ADR 0062: Wave 2 final verdict — baseline recovered, 8.0 s target deferred to Wave 3

- **Status**: Accepted
- **Date**: 2026-04-15

## Context

Wave 2 (E2E Bottleneck Improvement Plan, `Dev_Plan/2026-04-14-
E2E-Bottleneck-Improvement-Plan.md` §3) targeted `avg_first_sound_ms
≤ 8,000` on the 24-round Qwen3-ASR bilingual mix benchmark (Jetson
Orin Nano Super 8 GB, streaming TTS, resident STT+LLM, q8_0 KV,
Option B cap). The plan decomposed the 11,004 ms Wave 1 baseline
into five tracks:

| Track | Expected Δ | Landed |
|---|---:|---|
| T2.0 reporting infrastructure | 0 (enabler) | ✅ commit `fe59a98` |
| T2.1 TTS resident | −450 | ⏸ deferred (ADR 0058) |
| T2.2 sentence chunking | −621 (perceived) | ✅ B-fix `adec921` + metric fix `0d6875a` |
| T2.3 LLM prefix caching | −547 | ⏸ deferred (ADR 0061) |
| Option B max_tokens 60 | −300 | ✅ commit `372b49f` |

The Wave 2 work ran from morning of 2026-04-15 (T2.0/T2.1 planning)
through the evening crash-recovery chain (memory guards, layered
eviction, build-infra alignment, KV quantization) into the late-night
Option B measurement. 15 commits total.

## Decision

Close Wave 2 with the following final metrics and verdict.

### Final measurement (2026-04-15 22:06 KST, commit `372b49f`)

Config: `MUNGI_LLM_RESIDENT=1 MUNGI_STT_RESIDENT=1
MUNGI_TTS_STREAMING=1 MUNGI_LLM_KV_TYPE=q8_0 MUNGI_LLM_MAX_TOKENS=60`
(Phase 2 snapshot and LlamaCache both OFF after ADR 0058/0061
deferrals).

| Metric | Wave 1 baseline | Wave 2 target | **Wave 2 final** | Δ vs baseline | Gate |
|---|---:|---:|---:|---:|:---:|
| `avg_first_sound_ms` | 11,004 | ≤ 8,000 | **10,912** | **−92** | ❌ gap 2,912 |
| `avg_llm_ttft_ms_after_first` | 1,377 | ~830 | 1,367 | −10 | ✅ |
| `avg_llm_ms` | 3,635 | — | 3,532 | −103 | ✅ |
| `avg_tts_first_chunk_ms` | N/A | ≤ 400 | 717 | N/A | ❌ |
| CRITICAL memory events | N/A | 0 | 0 | — | ✅ |
| STT force unloads | N/A | 0 | 0 | — | ✅ |
| Pipeline errors | 0 | 0 | 0 | — | ✅ |

### Verdict

- **Baseline recovered** (−92 ms vs 2026-04-14 baseline; no
  regression on any track).
- **Primary target MISSED** (10,912 ms vs 8,000 ms target; 2.9 s
  gap).
- **Stability improvements permanent** (VAD protection, layered
  eviction, KV quantization opt-in, Option B operator control,
  build-infra alignment, metric-methodology fix).

Wave 2 is closed as "partial success": baseline preserved,
infrastructure hardened, but the latency target remains unmet.
The gap is attributable to the two deferred tracks (T2.1 −450 ms
and T2.3 −547 ms hypothetical) plus second-order effects from
their memory pressure.

### What actually delivered the improvements

| Delivered | Contribution |
|---|---|
| Option B (60-token cap as memory-pressure trigger) | −3,664 ms from Phase 1 baseline via secondary warm-state preservation |
| T2.2 B-fix (pipeline TTS lifecycle externalized) | −938 ms from the T2.2 A broken state |
| Metric fix (first_chunk semantic) | −746 ms reporting correction |
| Fix-D layered eviction + VAD protection | pipeline stability (prevents multi-second cascading regressions) |
| Phase 1 q8_0 KV quantization | ~200 MB memory headroom (enables the above) |

The **Option B measurement itself** shows the biggest latency
delta, but the design intent of "cap response length" only
explains a fraction of it. The dominant mechanism is that the
60-token cap (and the cumulative headroom from q8_0) keeps the
process RSS below the CRITICAL threshold (6,500 MB), which
prevents the cascade of `guard_stt_resident_memory` STT evictions
that were worth ~11 s × 7 occurrences in mid-session runs.

### Why the 8.0 s target could not be reached

1. **T2.1 TTS resident** gives −450 ms on paper but introduces LLM
   memory contention under the current ADR 0013 memory budget (see
   ADR 0058 Update). Wave 3 must either enlarge the budget (Orin
   NX 16 GB) or restructure the pipeline so TTS residence does not
   compete with LLM full-offload.
2. **T2.3 LLM prefix caching** is blocked by llama-cpp-python
   0.3.17's `create_chat_completion` ignoring pre-loaded KV state
   (see ADR 0061 feasibility test). Wave 3 needs a lower-level
   code path or the HTTP-server mode.
3. **STT** remains the dominant per-turn cost (~5.8 s on CPU).
   Wave 3 must pursue sherpa-onnx GPU build (−2.8 s estimated).
4. The Wave 2 plan's `first_sound` target was set against a
   hypothetical sum of best-case deltas that assumed all tracks
   landed simultaneously. The observed measurements on Jetson 8 GB
   show those tracks interact via memory pressure in ways the plan
   did not model.

## Consequences

### Positive

- **Wave 1 performance preserved** — no regression.
- **Infrastructure debt paid down**:
  - VAD CRITICAL protection (ADR 0060) — eliminates 20-error
    cascade on memory pressure.
  - Layered eviction — STT preserved across memory spikes.
  - Build-infra 0.3.17 alignment — fresh-clone reproducibility
    restored.
  - KV quantization opt-in — 200 MB Jetson headroom available on
    demand.
  - Metric methodology — `first_chunk` semantic fixes a ~700 ms
    measurement bias.
- **Direct API testing discipline established** — the 3-line
  Jetson Python test that invalidated Phase 2 is now a required
  check for any llama-cpp-python API assumption (captured in ADR
  0061).
- **opt-in defaults** — Phase 1 q8_0, Phase 2 snapshot, Option B
  60-cap all default to OFF. The default production path is
  unchanged; operators opt in per measurement.

### Negative / Risks

- **2.9 s gap to 8.0 s target** persists. Wave 3 must close this.
- **LlamaCache default ON** after T2.3 deferral — measured 0.4 %
  hit rate; candidate for follow-up disable commit to simplify
  the default path. Deferred to a single-commit cleanup.
- **Option B "secondary warm-state" mechanism** is fragile — it
  works because the 60-token cap reduces RSS growth just enough.
  If future changes increase RSS (e.g., a new resident model), the
  gain could disappear. Measurement should re-validate after any
  resident-feature addition.
- **3 ADRs deferred** (0058, 0061, this one) reflect an
  accumulating plan/reality gap. Wave 3 planning must revise the
  first_sound budget model to account for memory-pressure
  coupling between tracks.

## Wave 3 seed

Priority-ordered follow-up tracks:

1. **STT GPU build** (sherpa-onnx `-DSHERPA_ONNX_ENABLE_GPU=ON`,
   source build ~1–2 h). Estimated STT 5.8 → 2.5–3.0 s (−2.8 s).
   The single biggest remaining lever for `first_sound`.
2. **T2.1 retry under Fix-D** — measure TTS resident again with
   the layered eviction and VAD protection already in place.
   Memory contention profile may differ from the ADR 0058 test.
3. **T2.3 Phase 2 retry via low-level `Llama.__call__()`** —
   bypass `create_chat_completion` and manually tokenize +
   generate incremental user tokens after `load_state()`.
   Requires reimplementing the chat-template renderer inline.
4. **`conversation_history` per-turn clear policy** — currently
   `clear_history()` runs every turn, which defeats prefix-cache
   hits. Redesign so system-prompt KV is durable across turns.
5. **Orin NX 16 GB migration feasibility study** — memory
   contention is the underlying blocker for both T2.1 retry and
   Phase 2 retry; doubling RAM eliminates most guards.
6. **LlamaCache deprecation** — flip
   `PipelineConfig.llm_cache_prompt` default to False (hit rate
   0.4 % not worth the overhead) as a clean-up commit.

First-action for the next Wave 3 planning session:

- Draft `Dev_Plan/2026-04-<next>-Wave3-Plan.md` with the 6 tracks
  above plus a revised first_sound budget model that accounts for
  Jetson 8 GB memory-pressure coupling.
- Apply Plan Gate v2 (CLAUDE.md §1/§8): Codex deep review +
  mutual-discussion + user approval before any implementation.

## References

- `docs/runbooks/weekly/archive/2026-04-15-daily-worklog.md` §13 (evening
  session) and §14 (late-night Option B + Wave 2 close).
- `docs/runbooks/weekly/archive/2026-04-15-next-session-handoff.md` — end-
  of-session handoff updated with Wave 3 seed.
- ADR 0058 — T2.1 deferred (memory contention).
- ADR 0060 — VAD protection + layered eviction (permanent).
- ADR 0061 — T2.3 Phase 2 API feasibility deferred.
- docs/archived/dev-plan/2026-04-15-Wave2-Plan.md — original plan (v2, Plan Gate
  v2 approved).
- Final measurement: `/var/lib/mungi/conversations/
  qwen3_mix_20260415_220648/summary.json`.
- Commit chain (15 commits) listed in worklog §13.1 and §13.15.
