# ADR 0088: Piper TTS Fallback Retirement

- **Status**: Accepted (2026-05-13)
- **Authority**: `Dev_Plan/2026-05-13-piper-tts-fallback-retirement-plan.md` (v3 approved)

## Context

Piper TTS was carried as a fallback option since the Phase 0 baseline in ADR 0002.
The 2026-05-13 verification found that this fallback no longer matched runtime reality:

1. The Piper voice model was absent from the Jetson runtime path
   `/opt/mungi/ai_models/`, as verified by Session 36 Jetson Action 5 pre-check.
2. `core/model_manager.load_tts()` already loaded only Supertonic and raised
   `ValueError` for any other `tts_engine` value. The Piper fallback branch never
   existed in executable `load_tts()` code; only stale docstrings still described it.
3. `models/tts_runner.py` still contained an unused `PiperEngine` class, but no active
   caller used it.
4. Documentation drift between docstrings, runbooks, and runtime behavior created
   avoidable confusion for new contributors.

This ADR records a documentation and dead-code cleanup, not a new runtime behavior change.
The executable TTS load path was already Supertonic-only before this retirement.

## Decision

Formally retire Piper TTS from the Mungi runtime and active documentation.

1. Remove the unused `PiperEngine` class from `models/tts_runner.py`.
2. Remove Piper-related tests, scripts, mypy ignore entries, and active documentation
   references.
3. Treat Supertonic TTS 2 as the sole TTS engine.
4. Preserve the existing fail-fast contract in `core/model_manager.load_tts()`: any
   non-Supertonic `tts_engine` value raises
   `ValueError("Unsupported TTS engine: ...")`.
5. Preserve the current pipeline-level mapping from that exception to
   `TurnResult.error=str(exc)`. No new `tts_unavailable` literal is introduced.

Historical plans and ADRs that described Piper as a fallback remain part of the audit trail,
but active docs now point to this ADR as the superseding decision.

## Consequences

### Positive

- Approximately 140 lines of unused `PiperEngine` code are removed.
- Five test files and two scripts are simplified.
- Active runbooks, project status docs, and model-stack references align with runtime behavior.
- The mypy optional-dependency ignore list is reduced.
- Contributors now see one TTS contract: Supertonic-only with explicit fail-fast behavior.

### Negative

- There is no in-process TTS fallback. This is formalized rather than newly introduced,
  because `load_tts()` already failed fast for non-Supertonic engine values.
- If Supertonic load fails, `load_tts()` raises and the pipeline records
  `TurnResult.error=str(exc)`, which is the same path used before this ADR.

### Neutral

- User-facing impact is zero. Piper voice assets were not deployed in production builds.
- The Piper code path was not reachable through the production `load_tts()` path.

## Verification

- Dispatch A removes the unused code, tests, scripts, and mypy ignore entries.
- Dispatch B creates this ADR, appends supersession updates to ADR 0002, ADR 0006,
  ADR 0054, and ADR 0084, and reconciles active documentation.
- `core/model_manager.load_tts()` retains the existing unsupported-engine `ValueError`
  contract for non-Supertonic values.
- Living model-stack documents describe Supertonic TTS 2 as the sole TTS engine.

## Related

- ADR 0002: Phase 0 baseline that originally listed Piper as a fallback.
- ADR 0006: Models layer architecture that referenced `PiperEngine`.
- ADR 0054: Gemma 4 extended pilot audio strategy that assumed Supertonic plus Piper.
- ADR 0081: openWakeWord retirement precedent.
- ADR 0084: TTS CUDA ONNX provider spike that still listed Piper comparison work.
- `Dev_Plan/2026-05-13-piper-tts-fallback-retirement-plan.md`
