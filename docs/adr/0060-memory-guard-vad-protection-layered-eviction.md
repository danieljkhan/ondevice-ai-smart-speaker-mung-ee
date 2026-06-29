# ADR 0060: Memory guard hardening — VAD protection + layered eviction

- **Status**: Accepted
- **Date**: 2026-04-15

## Context

Wave 2 T2.3 prefix-caching work (commits `9905b97` / `12bb1cb`) attached
a bounded `LlamaCache` (200 MB) on every resident-mode turn. The extra
200 MB pushed `usage_mb` past the `check_memory_health()` CRITICAL
threshold (6,500 MB) at round 5 of the first 24-round Jetson
measurement (`/tmp/t2.3_bounded_run.log`, 2026-04-15 16:31 KST).

The CRITICAL path at `core/model_manager.py:798-803` called
`unload_all()`, which iterates every entry of `_MODEL_NAMES` — including
the permanent-resident `vad` slot. `core/pipeline.py::_run_vad`
dereferences `self._mm.vad` with no None-guard, so rounds 5–24 crashed
with `AttributeError: 'NoneType' object has no attribute 'reset_states'`
20 consecutive times. The pipeline had no reload path for VAD after an
`unload_all()`.

Separately, `guard_stt_resident_memory()` at
`core/model_manager.py:821-848` forced STT unload whenever
`MemAvailable < 1024 MB`. The T2.3 cache nudged `MemAvailable` just
below that threshold every few turns, and each force-unload cost an
~11 s cold STT reload on the next turn. A 2026-04-15 18:06 run
(`a4bca7c`) showed 7 STT force-unloads in 24 rounds, contributing
most of a +9,000 ms `first_sound` regression versus the T2.2 B-fix
baseline.

The T2.3 feature itself is questionable (hit rate 0.4 % because
`clear_history()` runs between turns and `create_chat_completion`
re-tokenizes the whole messages list), but the VAD-unload crash and
the STT-eviction cycle are independent defects that also trigger for
other causes (any future resident feature that touches MemAvailable
around the 1,024 MB floor, any CRITICAL-memory recovery, etc.).

## Decision

Harden the memory guard in two layers that are always on, regardless
of whether T2.3 prefix caching is enabled:

### Layer A — VAD protection in CRITICAL recovery

`core/model_manager.py::check_memory_health()` CRITICAL branch
replaces `unload_all()` with per-type unloads:

```python
if usage_mb > 6500:
    logger.critical(
        "CRITICAL memory: %d MB — unloading transient models "
        "(VAD retained as permanent resident)",
        usage_mb,
    )
    self.unload_stt(force=True)
    self.unload_llm()
    self.unload_tts(force=True)
    self._drop_page_cache()
    self._gc_collect()
    self._current_gpu_model = ModelType.NONE
    return MemoryHealth.CRITICAL
```

The VAD slot (`_models["vad"]`) stays populated so
`core/pipeline.py::_run_vad` can safely dereference it on the next
turn. Transient models (STT/LLM/TTS) still get unloaded — the next
turn will lazy-load them via their normal `load_*` paths.

`unload_all()` is retained as a public method for legitimate full-
reset callers (shutdown, tests, explicit operator requests).

### Layer D — layered eviction in STT resident guard

`core/model_manager.py::guard_stt_resident_memory()` tries a cheaper
recovery before forcing STT unload:

```python
if available_mb < threshold_mb:
    # 1. Cheap: flush LLM prompt cache (~200 ms, ~200 MB reclaim)
    if self.flush_llm_prompt_cache():
        self._drop_page_cache()
        self._gc_collect()
        recovered_mb = self._get_available_memory_mb()
        if recovered_mb >= threshold_mb:
            # Memory recovered; STT stays resident
            return True
    # 2. Fallback: force STT unload (the original expensive path)
    self.unload_stt(force=True)
    return False
```

`flush_llm_prompt_cache()` is a new ModelManager method that
delegates to `models/llm_runner.py::flush_prompt_cache(llm)`. The
llm_runner helper clears the attached cache in this order (real API
surface, not MagicMock):

1. `cache.cache_state.clear()` — the llama-cpp-python 0.3.17
   `LlamaRAMCache` internal `OrderedDict`.
2. `cache.cache_clear()` / `cache.clear()` — future or alternate
   cache implementations.
3. `llm.cache = None` (or `set_cache(None)`) — detach entirely as a
   last resort so the next `_enable_prefix_cache()` call builds a
   fresh cache.

Each path is wrapped in its own `try/except` so one failing path does
not abort subsequent paths. **Never** writes to `cache.cache_size`;
that is a read-only `@property` on `LlamaRAMCache` and raises
`AttributeError`. A session-traceable bug on 2026-04-15 18:35 (`grep
cache_size = 0 in models/llm_runner.py`) confirmed the pre-fix
implementation did attempt that write and would have raised on the
real Jetson binding.

## Consequences

Positive:

- **VAD crash eliminated**: `AttributeError 'NoneType' ... reset_
  states` pattern (20 consecutive failures per incident) no longer
  reproducible from the CRITICAL path. Jetson measurement on
  2026-04-15 17:47 onward confirms 0 pipeline errors under CRITICAL
  guard activation.
- **STT resident stability**: Before Fix-D, 7 STT force-unloads per
  24-round run; after Fix-D, 0 STT force-unloads (5 cache flushes
  absorbed the pressure) in the 2026-04-15 18:34 run. `avg_stt_total_
  ms` recovered from 9,810 ms to 6,535 ms.
- **Decoupled from T2.3 verdict**: both layers are beneficial even
  when T2.3 prefix caching is off or deferred. Any future resident
  feature that adds ~100-200 MB of persistent footprint (e.g., a
  future TTS resident retry, sentence-chunking streaming cache
  enlargement, extra LLM state) benefits from the same layered
  eviction path.

Negative / Risks:

- **LLM prompt cache is flushed aggressively** under MemAvailable
  pressure. When T2.3 cache is active, frequent flushes defeat its
  purpose. Acceptable for now because T2.3 hit rate is already 0.4 %
  (see ADR 0058 Update); Phase 2 (save_state/load_state) bypasses
  LlamaCache entirely so this risk becomes moot.
- **`unload_all()` behavior preserved** for callers that still need
  a full reset (shutdown paths, tests). Contributors must not add new
  CRITICAL-memory callers of `unload_all()` expecting the old
  all-models-including-VAD semantics — the shutdown and memory-
  pressure paths now diverge deliberately.

## Tests

- `tests/test_model_manager_sequential.py::test_critical_memory_
  preserves_vad` — CRITICAL branch unloads STT/LLM/TTS and keeps
  VAD populated.
- `tests/test_model_manager_sequential.py::test_guard_stt_flushes_
  cache_before_unload` — both branches (flush recovers vs flush
  insufficient → STT unload).
- `tests/test_llm_runner.py::test_flush_prompt_cache_real_api_
  shape` — simulates a LlamaRAMCache-like object (read-only
  `cache_size` property, no public `clear`, `cache_state` as
  `OrderedDict`) and asserts Path 1 succeeds without
  `AttributeError`.

## References

- Commits:
  - `454133f [fix] wave2 memguard: preserve VAD + per-turn drop_caches`
    (Layer A introduction; per-turn drop was subsequently deferred
    via env flag — see §13.3 of the 2026-04-15 daily worklog for why)
  - `a4bca7c [fix] wave2 t2.3: layered eviction` (Layer D)
  - `6d18661 [fix] wave2 t2.3: flush_prompt_cache uses real LlamaRAMCache API`
    (real-API correctness; cache_size=0 bug removed)
- Incident logs:
  - `/tmp/t2.3_bounded_run.log` (2026-04-15 16:31 CRITICAL →
    unload_all → VAD ejected → 20 errors)
  - `/tmp/run2_no_drop.log` (2026-04-15 18:06 STT eviction cycle
    baseline)
  - `/tmp/run3_retry.log` (2026-04-15 18:45 Fix-D 5 flushes, 0 STT
    unloads, first_sound 14,911 ms)
- `docs/runbooks/weekly/archive/2026-04-15-daily-worklog.md` §13 — full
  narrative for the evening session.
- ADR 0013 — `drop_caches` doctrine. Layer D honors the same "drop
  caches at boundaries, not continuously" principle by calling
  `_drop_page_cache()` only after a successful cache flush.
- ADR 0058 — T2.1 TTS resident deferred; its Update section
  explains why T2.1 is now redundant under T2.2 B-fix and any
  future retry should presuppose ADR 0060's layered eviction.
- ADR 0059 — `_ACTIVE_SUPERTONIC_ENGINE` cross-module contract.

## Update — 2026-04-16 — Layer D clarification

Wave 3 T3.5 removed the T2.3 Attempt 1 `LlamaCache` attach path. After
that cleanup, `ModelManager.flush_llm_prompt_cache()` no longer interacts
with a `LlamaCache` or any external prompt-cache object.

The method name and Layer D role are preserved for callsite continuity in
the resident STT memory guard. The body now performs only the memory
reclaim that always followed a successful cache clear: `_drop_page_cache()`
plus `_gc_collect()`, then returns `True`.

The memory-reclaim behavior is unchanged. The removed step was the
cache-clear attempt itself, and there is no attached `LlamaCache` left to
clear after T3.5.
