# ADR 0006: Models Layer Architecture

- Status: Accepted
- Date: 2026-03-13
- Updated: 2026-03-16 (sequential GPU loading, TTS unload)
- Decision makers: Daniel (user), project PM
- Related: ADR 0002 (Phase 0 Baseline), ADR 0005 (Jetson CUDA Policy)

## Context

Sprint 3 moved model-loading and inference logic out of ad hoc scripts and into a dedicated
`models/` layer so that the runtime pipeline could depend on stable runner interfaces instead of
CLI utilities. That change fixed the original architecture violation where higher-level runtime code
depended on `scripts/`.

Sprint 3 Day 3 added an additional contract on top of that architecture: model-adjacent execution
paths must treat empty input as a safe no-op instead of letting ambiguous behavior leak into the
pipeline. This update was triggered by supervisor feedback on three cases:

- STT received `None` or empty audio bytes.
- TTS received `None`, empty text, or whitespace-only text.
- LLM generation received `None`, empty text, or whitespace-only prompts.

These cases are not exceptional user behavior in an always-on child device. They are normal edge
conditions that can happen after silence detection, interrupted recordings, or upstream guardrails.

## Decision

### 1. The `models/` layer remains the single owner of model-runner contracts

The `models/` package owns the reusable execution contracts for:

- `models/vad_runner.py`
- `models/stt_runner.py`
- `models/llm_runner.py`
- `models/tts_runner.py`

Runtime orchestration code in `core/` may coordinate these runners, but it must not reintroduce
dependencies on `scripts/` for inference behavior.

### 2. Dependency direction remains one way

The allowed dependency direction is:

```text
core/ -> models/ -> external libraries
core/ -> safety/
```

The following directions remain prohibited:

- `models/ -> core/`
- `safety/ -> core/`
- `core/ -> scripts/`
- circular imports across model runners

### 3. Empty-input handling is part of the runner contract

The runtime must treat missing or empty input as a safe skip, not as an implicit error.

Required behavior:

- STT paths return early on `None` or empty audio bytes and log a warning.
- TTS paths return early on `None`, empty text, or whitespace-only text and log a warning.
- LLM generation returns a skipped result on `None`, empty text, or whitespace-only prompts and
  uses `ttft = -1.0` as the sentinel value for "generation did not run".

This contract applies both to model-runner implementations and to nearby pipeline entry points that
validate input before delegation.

### 4. Sequential GPU loading is part of the ModelManager contract

Jetson 8 GB unified memory requires at most one large model on the GPU at a time.
`ModelManager` exposes a sequential loading API (ADR 0009):

- `initialize()` — loads VAD (CPU-resident) only.
- `load(ModelType)` — loads the requested GPU model (STT or LLM), unloading the
  previous one first with VRAM double verification.
- `load_all()` — deprecated; emits `DeprecationWarning`.

The pipeline calls `mm.load(ModelType.STT)` before transcription and
`mm.load(ModelType.LLM)` before generation, ensuring automatic model switching.

### 5. TTS engines must implement `unload()`

`TTSEngine` (ABC) requires an `unload()` method that releases model resources.
This enables `ModelManager._unload_current_gpu()` to call `model.unload()` on any
GPU model that supports it. Both `SupertonicEngine` and `PiperEngine` implement this.

### 6. Ownership remains scope-aligned

| Layer | Owner | Responsibility |
|------|-------|----------------|
| `core/` | feature sub-agent | runtime orchestration and state flow |
| `models/` | feature sub-agent | model loading, inference, and runner contracts |
| `scripts/` | platform sub-agent | CLI utilities, benchmarks, and environment tooling |

> Note: Ownership migrated from lane-based agents (Lane A, Lane B)
> to sub-agent model per ADR 0008.

## Consequences

### Positive

- The architecture remains aligned with the one-way dependency rule in `AGENTS.md`.
- Empty-input behavior is now explicit and testable instead of implicit and fragile.
- Pipeline callers can distinguish skipped LLM generation from real generation via `ttft = -1.0`.
- Future model swaps should stay isolated to the `models/` layer instead of leaking into `core/`.
- Sequential GPU loading eliminates ENOMEM on Jetson 8 GB while enabling full GPU offload
  (`n_gpu_layers=-1`) for each model when it has exclusive access.

### Negative

- Callers must consistently honor the skip contract; treating `ttft = -1.0` as a normal latency
  value would be a bug.
- Additional guard code slightly increases branching around the pipeline, so regressions should keep
  coverage on empty-input paths.

## Related Documents

- `AGENTS.md`
- `core/pipeline.py`
- `models/llm_runner.py`
- `models/tts_runner.py`
- `docs/runbooks/weekly/archive/2026-03-13-sprint3-day2-worklog.md`
- `docs/runbooks/weekly/archive/2026-03-16-sprint3-day3-worklog.md`
- `docs/adr/0009-sequential-gpu-loading.md`

## Later References

- ADR 0016: Clarifies that CPU STT/TTS still require stage unload on Jetson unified memory.

## Update (2026-05-13)

The `PiperEngine` class referenced in line 88 of this ADR was removed from
`models/tts_runner.py` per ADR 0088. Only `SupertonicEngine` implements the TTS
engine interface going forward.
