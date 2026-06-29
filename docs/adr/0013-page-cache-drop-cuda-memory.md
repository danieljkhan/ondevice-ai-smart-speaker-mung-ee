# ADR 0013: Page Cache Drop for CUDA Memory Reclamation

- **Status**: Accepted
- **Date**: 2026-03-18
- **Context**: CUDA OOM during sequential model transitions on Jetson

## Context

On the Jetson Orin Nano Super (7.4 GB unified memory), CUDA's `cudaMalloc`
can only allocate from kernel `MemFree` pages. It cannot reclaim the kernel
page cache, even when `MemAvailable` shows sufficient memory.

After STT inference, the ctranslate2 model weights remain in page cache.
This blocks LLM full GPU offload (`n_gpu_layers=-1`), forcing a fallback
to 10 layers and degrading performance from 11 tok/s to 3.5 tok/s.

### Root cause

```
MemAvailable = MemFree + Reclaimable cache   (what Linux reports)
CUDA usable  = MemFree only                  (what cudaMalloc can use)
```

The `ModelManager._select_gpu_layers()` previously used `MemAvailable`,
incorrectly concluding that full offload was possible. CUDA then failed
with `cudaMalloc: out of memory`.

## Decision

1. **Drop page cache** before LLM GPU loading via
   `echo 1 > /proc/sys/vm/drop_caches` (page cache only, preserves
   dentry/inode cache).

2. **Use `MemFree`** instead of `MemAvailable` for GPU layer decisions.

3. **Auto-trigger**: `ModelManager._ensure_cuda_memory()` checks `MemFree`
   and drops cache only when below the required threshold (4000 MB).

4. **Sudoers rule**: `/etc/sudoers.d/mungi-drop-caches` grants passwordless
   `sudo tee /proc/sys/vm/drop_caches` to the `mungi` user. In production,
   the systemd service runs as root, making this unnecessary.

### Safety analysis

- **Data loss risk**: None. Only read-cache is dropped. Dirty pages are
  flushed before drop. Process heap memory (conversation history, model
  state) is unaffected.
- **Performance cost**: ~1 s additional model load time (NVMe re-read at
  1.8 GB/s vs cached 4.0 GB/s). Net gain: 8.8 s inference improvement.
- **Fallback**: If cache drop fails (no permission), the system falls back
  to partial GPU offload (20 or 10 layers) — degraded but functional.

### GPU layer selection thresholds (MemFree based)

| MemFree | n_gpu_layers | Performance |
|---------|-------------|-------------|
| >= 4000 MB | -1 (all 36) | ~11 tok/s |
| >= 3000 MB | 20 | ~7 tok/s |
| < 3000 MB | 10 (fallback) | ~3.5 tok/s |

## Consequences

- E2E pipeline improved from 20.4 s to 12.5 s (39% reduction).
- LLM consistently loads with full GPU offload after STT unload.
- `ModelManager` now imports `subprocess` for the sudo call.
- New methods: `_drop_page_cache()`, `_ensure_cuda_memory()`,
  `_get_meminfo_mb()`.
- `ManagerConfig.llm_n_gpu_layers` default changed from 10 to -1.

## Update (2026-03-18)

Extended automatic cache drop to model unload methods:

- `unload_stt()`: drops page cache after STT unload (commit `eeb0ebb`)
- `unload_llm()`: drops page cache after LLM unload (commit `70c5581`)

Previously cache drop only occurred in `_ensure_cuda_memory()` before LLM
load. Memory profiling confirmed STT leaves ~400MB page cache that blocks
subsequent LLM GPU allocation without explicit drop.

## References

- ADR 0016 (extends unified-memory discipline to CPU STT/TTS stage unload)

- ADR 0009 (sequential GPU loading — this ADR extends its memory management)
- ADR 0012 (LLM upgrade — requires cache drop for 4B model full offload)
- ADR 0015 (response sanitization — operates after LLM generation)
