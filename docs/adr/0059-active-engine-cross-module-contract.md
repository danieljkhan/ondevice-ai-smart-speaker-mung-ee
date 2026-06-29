# ADR 0059: Active Supertonic engine cross-module registration contract

- **Status**: Accepted
- **Date**: 2026-04-15

## Context

Wave 2 T2.2 addressed a Jetson crash path that surfaced as a
`FileNotFoundError` while sentence-level Supertonic synthesis tried to
construct a fallback engine during streaming playback. The root cause had two
layers.

First, `models/tts_runner.py` resolved the fallback streaming model directory
from a CWD-relative `"ai_models"` default. Under the Jetson runtime working
directory `/opt/mungi-repo`, that relative path pointed at the wrong location
instead of the deployed Supertonic assets under `/opt/mungi/ai_models/`.

Second, the sentence streaming consumer expected a producer-side registration
step that never happened. `models/tts_runner._resolve_sentence_engine()` looked
for `_ACTIVE_SUPERTONIC_ENGINE` before constructing a fallback engine, but
`ModelManager.load_tts()` did not publish the successfully created
`SupertonicEngine` instance into that module-level slot. As a result, the
consumer always missed the already-loaded engine and fell through to the broken
filesystem fallback.

The T2.2 bug-fix corrected both sides of the failure chain: the fallback model
path became the absolute Jetson default `/opt/mungi/ai_models/supertonic-2`,
and the producer now explicitly registers and clears the active engine used by
the streaming consumer.

## Decision

Adopt an explicit active-engine cross-module registration contract for
Supertonic sentence streaming.

1. Producer success path: `ModelManager.load_tts()` calls
   `models.tts_runner._set_active_supertonic_engine(engine)` after creating the
   `SupertonicEngine` for `cfg.tts_engine == "supertonic"`, and it clears the
   slot with `_set_active_supertonic_engine(None)` if the load path raises.
2. Producer unload path: `ModelManager.unload_tts()` clears the slot with
   `_set_active_supertonic_engine(None)` after engine teardown so no stale
   engine survives across unload boundaries.
3. Consumer resolution path: `synthesize_to_speaker_by_sentence()` reuses the
   active engine only when `model_dir` is `None` and the requested
   `voice_style` matches the registered engine. An explicit `model_dir` keyword
   argument always bypasses the active-engine fallback.
4. Fallback path default: `models/tts_runner._resolve_streaming_model_dir()`
   uses the absolute Jetson default `/opt/mungi/ai_models/supertonic-2` only
   when no environment override is configured and no caller supplied a
   `model_dir`, and it emits a warning so operators can see the fallback in
   logs.

## Consequences

Positive:

- The producer/consumer link is now explicit, so future refactors cannot break
  sentence streaming by silently dropping engine registration.
- Failure-path cleanup prevents stale module-level state from surviving a
  partially failed `load_tts()` call.
- Coverage now spans the success path, failure-path cleanup, explicit-model-dir
  resolution, active-engine reuse, and absolute-default fallback behavior.

Negative / Risks:

- Module-level global state still exists; this ADR codifies the contract rather
  than removing the shared mutable slot.
- If a future TTS backend replaces or supplements Supertonic, this single-slot
  contract must be generalized, for example to `_ACTIVE_<ENGINE>_ENGINE` names
  or a backend registry. That follow-up remains a Wave 3+ cleanup candidate.

## References

- `docs/runbooks/weekly/archive/2026-04-15-wave2-t2.1-report.md`
- ADR 0058: `docs/adr/0058-wave2-t2.1-tts-resident-deferred.md`
- Task specs: `.codex/current-task.md` and the prior
  `wave2-t2.2-bugfix-engine-sharing` task
- `core/model_manager.py` (`load_tts()`, `unload_tts()`)
- `models/tts_runner.py` (`_set_active_supertonic_engine()`,
  `_resolve_sentence_engine()`, `_resolve_streaming_model_dir()`,
  `synthesize_to_speaker_by_sentence()`)
- Tests covering the contract paths in `tests/test_model_manager_sequential.py`,
  `tests/test_tts_runner.py`, and `tests/test_e2e_qwen3_asr_mix.py`
