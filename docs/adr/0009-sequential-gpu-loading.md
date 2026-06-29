# ADR 0009: Sequential GPU Loading for Jetson 8GB

- **Status**: Accepted
- **Date**: 2026-03-16
- **Context**: ModelManager GPU memory management

## Context

The Jetson Orin Nano Super has 8 GB of unified memory (UM) shared
between CPU and GPU. Loading all four AI models (VAD, STT, LLM, TTS)
simultaneously exceeds available GPU memory and triggers ENOMEM
errors. The development specification (§4.2) mandates sequential
GPU loading where at most one large model occupies the GPU at a time.

## Decision

Implement a sequential GPU loading protocol in `ModelManager`:

1. **VAD** — CPU-resident, always loaded via `initialize()`.
2. **STT / LLM** — GPU models loaded one at a time via `load(ModelType)`.
   Switching models follows: unload → GC → VRAM double verification →
   new load.
3. **TTS** — CPU-first execution, no GPU contention.
4. **LLM full offload** — When loaded via the sequential path, LLM
   uses `n_gpu_layers=-1` for maximum GPU utilization (exclusive
   access guaranteed).
5. **VRAM double verification** — After unloading a GPU model, poll
   until `torch.cuda.memory_allocated() <= 256 MB` AND
   `/proc/meminfo MemAvailable >= 1.0 GB` (50 ms interval, 3 s
   timeout). Gracefully skipped on non-Linux dev environments.
6. **STT preloading** — `preload_stt()` launches a background thread
   on VAD `speech_start` to hide STT load latency.
7. **Memory health** — Three-level classification: NORMAL (< 4.5 GB),
   WARNING (4.5–6.5 GB, forces GC), CRITICAL (> 6.5 GB, full unload).

### Backward compatibility

- `load_all()` is preserved but emits `DeprecationWarning`.
- Model properties (`mm.vad`, `mm.stt`, `mm.llm`, `mm.tts`) unchanged.
- `ManagerConfig.llm_n_gpu_layers` default (10) is preserved for the
  legacy `load_llm()` path.

### Pipeline integration

`ConversationPipeline.run_turn()` now calls `mm.load(ModelType.STT)`
before transcription and `mm.load(ModelType.LLM)` before generation,
ensuring automatic model switching. Content filter guards are applied
pre-LLM (input) and post-LLM (output).

## Consequences

- **Positive**: Eliminates GPU OOM on Jetson 8 GB; enables full GPU
  offload (`n_gpu_layers=-1`) for each model when it has exclusive
  access; adds measurable model-switch latency to `TurnMetrics`.
- **Negative**: Each turn incurs model-switch overhead (~1–3 s per
  transition). STT preloading partially mitigates this for the
  STT→LLM transition.
- **Risks**: VRAM verification timeout could delay turns under memory
  pressure. The 3 s timeout is chosen to fail fast rather than block
  indefinitely.

## Related

- Development specification §4.2 (GPU memory management)
- ADR 0006: Models layer architecture
- ADR 0007: Content filter architecture

## Later References

- ADR 0013: Page cache drop extends memory management with CUDA-aware reclamation.
- ADR 0012: LLM upgrade requires cache drop for 4B model full offload.
- ADR 0016: CPU STT/TTS are still transient stages because Jetson unified memory is shared.
