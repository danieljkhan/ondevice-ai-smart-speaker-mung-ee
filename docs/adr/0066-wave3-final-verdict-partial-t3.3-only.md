# ADR 0066: Wave 3 final verdict — PARTIAL, T3.3 low-level LLM only

- **Status**: Accepted
- **Date**: 2026-04-20

## Context

Wave 3 (Plan `docs/archived/dev-plan/2026-04-16-Wave3-Plan.md`) targeted
`avg_first_sound_ms <= 8,000` on the 24-round Qwen3-ASR bilingual mix
benchmark (Jetson Orin Nano Super 8 GB). The plan decomposed the
10,912 ms Wave 2 baseline into five software-only tracks:

| Track | Expected delta | Landed |
|---|---:|---|
| T3.0 stability telemetry | 0 (enabler) | PR #18 |
| T3.1 sherpa-onnx GPU STT | -3,100 to -3,600 | ABORT (ADR 0065 — unified memory limit) |
| T3.3 low-level Llama chat | -524 (TTFT) | PR #35 — partial effect measured |
| T3.4 history-window CLI | 0 (enabler for T3.3) | PR #20 |
| T3.5 LlamaCache removal | 0 (tech debt) | PR #19 |

T3.1 abort (ADR 0065) removed the primary performance lever (-3,100
to -3,600 ms). The 8,000 ms target became unreachable with software
changes alone. T3.3 was the sole remaining performance lever.

## Decision

Close Wave 3 as **PARTIAL** with the following measurements and verdict.

### M3 measurement (2026-04-20 13:26 KST, commit `a572da9`)

Config: `MUNGI_LLM_LOW_LEVEL_CHAT=1` + `--tts-streaming` + page cache
drop active (sudoers configured).

| Metric | Wave 2 final | Wave 3 target | **M3 result** | delta vs W2 | Gate |
|---|---:|---:|---:|---:|:---:|
| avg_first_sound_ms | 10,912 | <= 8,000 | **17,246** | +6,334 | FAIL |
| avg_llm_ttft_ms | 1,297 | - | **1,075** | -222 | - |
| avg_llm_ttft_ms_after_first | 1,367 | <= 1,000 | **1,133** | -234 | FAIL |
| avg_llm_ms | 3,532 | - | **2,884** | -648 | - |
| avg_stt_total_ms | 6,171 | - | **13,105** | +6,934 | REGRESS |
| avg_tts_load_ms | 0.009 | - | **0.009** | 0 | PASS |
| avg_tts_first_chunk_ms | 717 | - | **683** | -34 | PASS |
| critical_memory_events | 0 | 0 | **0** | 0 | PASS |

### first_sound regression root cause

The +6,334 ms regression in `avg_first_sound_ms` is NOT caused by T3.3.
It is caused by STT model loading becoming slow (+6,934 ms) after the
sudoers fix enabled proper page cache drops between turns. In Wave 2,
`sudo -n true` failed silently, page cache was never dropped, and the
STT model data remained in kernel page cache for fast reloading (~354 ms).
With page cache drops properly working, each STT load reads from NVMe
(~7,500 ms).

This is a pre-existing architectural issue: the `model_manager.py`
`_drop_page_cache()` is called after STT unload, which evicts the STT
model data before the next turn needs to reload it. Wave 2 measurements
were taken under a broken sudoers configuration that accidentally
avoided this penalty.

### T3.3 isolated effect

The T3.3 low-level Llama chat path delivered:
- `avg_llm_ttft_ms_after_first`: 1,367 -> 1,133 ms (**-234 ms**, ~48% of expected -495 ms)
- `avg_llm_ms` total: 3,532 -> 2,884 ms (**-648 ms**)

The shortfall vs. the expected -495 ms is because the current
implementation does `llm.reset()` + full prompt `eval()` each call —
it does not implement prefix reuse (`load_state` + incremental `eval`).
The observed improvement comes from bypassing `create_chat_completion`
overhead only.

### Verdict

**PARTIAL** — T3.3 delivered measurable LLM TTFT improvement but the
8,000 ms first_sound target is unreachable without:
1. STT acceleration (GPU STT blocked by unified memory — ADR 0065)
2. STT loading optimization (page cache strategy or resident STT mode)
3. Prefix reuse in the low-level LLM path (future enhancement)

## Consequences

- Wave 3 closes. No further implementation tracks.
- The low-level LLM path (`MUNGI_LLM_LOW_LEVEL_CHAT=1`) remains gated
  and opt-in. Default path is unchanged.
- STT loading optimization (page cache strategy) is the #1 performance
  opportunity for Wave 4.
- Prefix reuse in `run_chat_generation_lowlevel()` is a Wave 4 candidate.
- The sudoers fix (`/usr/bin/true` added to NOPASSWD) is a permanent
  infrastructure improvement. The page cache drop behavior is correct;
  the STT loading penalty it reveals is the real issue to solve.

## References

- Wave 3 Plan: `docs/archived/dev-plan/2026-04-16-Wave3-Plan.md`
- T3.1 abort: `docs/adr/0065-wave3-t3.1-gpu-stt-abort-unified-memory-limit.md`
- T3.3 implementation: PR #35
- Wave 2 close: `docs/adr/0062-wave2-final-verdict-baseline-recovery.md`
- M3 summary: `/var/lib/mungi/conversations/qwen3_mix_20260420_132647/summary.json`
