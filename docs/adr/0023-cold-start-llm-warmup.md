# ADR 0023: Cold-Start LLM Warmup Strategy

- **Status**: Accepted
- **Date**: 2026-03-25
- **Context**: Eliminating OOM failures during early conversation turns on Jetson

## Context

Jetson Orin Nano Super 8GB exhibits cold-start instability when loading
Qwen3-4B-Q4_K_M with full GPU offload (`n_gpu_layers=-1`) for the first
time after boot or after prolonged idle.

Symptoms observed in 2026-03-23 E2E test:
- Turns 5–9 experienced a 12-turn consecutive failure streak
- CUDA memory allocation failed during LLM load due to fragmented
  unified memory
- After the first successful load, subsequent turns ran reliably

Root cause: the CUDA runtime, page cache, and unified memory allocator
require an initial allocation cycle to establish stable memory mappings.
Without warmup, the first real user turn bears this initialization cost
and frequently fails under tight 8GB budget.

## Decision

Add an optional LLM warmup cycle at pipeline startup, executed before the
first user turn.

### Warmup Implementation

`ConversationPipeline.warmup_llm()` in `core/pipeline.py` (line 296):

1. Load LLM model (full GPU offload)
2. Generate exactly 1 token with a minimal prompt:
   ```
   <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n
   ```
3. Unload LLM model immediately after generation
4. Idempotent cleanup on both success and failure paths

### Configuration

| Parameter | Default | Demo Override | Description |
|-----------|---------|---------------|-------------|
| `enable_warmup` | `False` | `True` | Enable warmup at startup |

The flag is `False` by default to avoid unnecessary overhead in test
scripts that do not need cold-start protection. Live demo scripts
(`scripts/demo_live.py`) set it to `True`.

### Combined with Page Cache Drop

The warmup cycle benefits from the existing page cache drop mechanism
(ADR 0013): `unload_llm()` triggers `drop_caches` after warmup, ensuring
the subsequent real LLM load has maximum available memory.

## Consequences

- **100% success rate** achieved after warmup deployment (eliminated the
  12-turn failure streak in turns 5–9)
- Adds ~3–5 seconds startup latency (one LLM load/unload cycle)
- No impact on per-turn performance after warmup completes
- Warmup is a pragmatic solution — the underlying CUDA memory fragmentation
  issue on Jetson unified memory remains. If Jetson firmware or JetPack
  updates improve memory allocator behavior, warmup may become unnecessary.

## References

- ADR 0009: Sequential GPU loading (load/unload lifecycle)
- ADR 0013: Page cache drop (cache reclamation after warmup unload)
- ADR 0016: Unified memory stage unload (memory budget context)
- `core/pipeline.py:296`: `warmup_llm()` implementation
- `scripts/demo_live.py:91,238`: Warmup configuration and invocation
- `docs/runbooks/weekly/archive/2026-03-24-daily-worklog.md`: Warmup deployment
