# ADR 0053 — Voice Runner v2 Architecture (sd.rec Blocking + State Machine)

- **Status**: Accepted
- **Date**: 2026-04-12
- **Author**: Claude Code (PM)
- **Supersedes**: Task 5.3 runner (deleted in `f48dbd4`, see ADR 0051)
- **Related ADRs**: 0049 (E2E Voice Pilot Methodology), 0051 (Runner Rollback)
- **Plan doc**: `docs/archived/dev-plan/e2e_voice_runner_redesign_v1.md`

---

## 1. Context

The Task 5.3 voice runner (630 lines) failed across 4 debugging iterations
(~150 min) and was reverted (ADR 0051). Three root causes were identified:

1. **Silero VAD streaming misuse**: `run_vad()` calls `model.reset_states()`
   on every invocation. Short chunks never accumulate enough speech
   probability to cross the detection threshold. The function is a batch
   API for complete utterance buffers, not a streaming API.

2. **Callback-based capture corruption**: `sd.InputStream` + callback +
   queue + seal produced audio with 10 dB lower overall RMS than direct
   `sd.rec()` capture. Sherpa STT rejected the output as "nospeech".
   Control-group test (same source, `sd.rec()`) produced valid STT.

3. **No state machine**: The runner began capturing immediately on stream
   open. No ARMED gate for environment hygiene. No coordinated
   capture/process transitions.

## 2. Decision

Build a new runner using **only components proven in prior tests**:

### Audio capture: `sd.rec()` blocking (NOT `sd.InputStream`)

`demo_live.py` uses `sd.rec()` and works reliably in live demos. The
runner adopts the same blocking pattern:

- **ARMED probe**: `sd.rec(PROBE_WINDOW_S * sr, ...)` + `sd.wait()`
- **CAPTURING segment**: `sd.rec(MAX_CAPTURE_S * sr, ...)` with a
  monitoring thread that calls `sd.stop()` on silence gap detection

`sd.InputStream` with callbacks is **forbidden** — per ADR 0051 §4.4,
it produced fundamentally corrupted audio that could not be attributed
to any single bug.

### Speech detection: RMS energy (NOT Silero VAD streaming)

For ARMED → CAPTURING onset detection, use `rms_db()` (simple RMS
energy in dB) instead of per-chunk Silero VAD. Silero VAD remains
used correctly inside `pipeline.run_turn()` on complete buffers.

### State machine: 5 explicit states

```
STARTUP → ARMED → CAPTURING → PROCESS → ARMED → ... → SHUTDOWN
```

| State | Audio stream | Purpose |
|-------|-------------|---------|
| STARTUP | Off | Preflight, model warmup, device discovery |
| ARMED | Short probes | Wait for speech onset via RMS energy |
| CAPTURING | Continuous record | Record until silence gap or max duration |
| PROCESS | Off | Pipeline processing (VAD+STT+LLM+TTS) |
| SHUTDOWN | Off | Cleanup, final summary |

### Pipeline: `run_turn()` only

The runner calls `pipeline.run_turn(audio, sample_rate)` which handles
VAD internally on the complete buffer. No direct `_run_vad()` calls.

## 3. Consequences

### Positive

1. **Proven capture path**: `sd.rec()` is identical to `demo_live.py`
   which works in production demos.
2. **No audio corruption risk**: No callback threads, no queues, no
   seal logic — the exact patterns that caused 10 dB RMS loss.
3. **Observable states**: Every transition is logged. Debugging is
   straightforward.
4. **Mandatory input saving**: `wavs_in/segment_NN_input.wav` saved
   before every `run_turn()` call — post-hoc analysis always possible.

### Negative

1. **No audio capture during processing**: While `run_turn()` executes
   (15-25s on first segment), the runner is not recording. Sony's
   8-second gaps are designed for this, but if processing exceeds the
   gap, the next message may be partially missed.
2. **RMS energy is less precise than VAD**: Energy onset may trigger on
   non-speech noise (fan, ambient). Mitigated by configurable threshold
   and user-controlled environment hygiene.

### Neutral

1. **~350 lines** vs original 630 lines — simpler architecture.
2. **8 unit tests** covering RMS math, state enum, CLI, device discovery.
3. **No changes to `core/pipeline.py`** or `models/vad_runner.py`.

## 4. Alternatives Considered

### A. Fix the Task 5.3 runner incrementally

Rejected. Four attempts (5.3.3, 5.3.4, 5.3.4.1, 5.3.6) accumulated
without commits, compounding bugs. The callback-based capture produced
fundamentally different audio than direct recording. Incremental
patching on a flawed architecture was proven ineffective.

### B. Use Silero VAD in streaming mode with state persistence

Would require modifying `models/vad_runner.py` to not call
`model.reset_states()` per invocation. Risk: changes to a shared
module could affect production pipeline behavior. The batch API usage
in `pipeline.run_turn()` is correct and proven — altering it for the
runner's convenience introduces regression risk.

### C. Use openwakeword for onset detection

Production wake word architecture. Premature for a pilot test runner.
Separate ADR needed when production wake word is implemented.

## 4.1 Post-deployment Discoveries (2026-04-12 smoke tests)

Three issues were found and fixed during Jetson smoke tests:

### Issue 1: Monitor thread deadlock (Task 5.4.1)

The original `_capture_segment()` used a monitor thread calling
`sd.stop()` while the main thread blocked on `sd.wait()`. This caused
a PortAudio/ALSA deadlock on Jetson. Fix: removed the monitor thread
entirely, using fixed-duration `sd.rec()` + `sd.wait()` + post-hoc
silence trimming.

### Issue 2: Speech onset loss (Task 5.4.2)

When ARMED probe detected speech, the transition to CAPTURING started
a new `sd.rec()` call, losing the probe audio containing the speech
onset. Fix: prepend probe audio to the capture buffer. Also lowered
VAD threshold (0.5 → 0.3) and min_speech_ms (250 → 100).

### Issue 3: Pipeline resampler degrading VAD input (Task 5.4.3)

The pipeline's `_prepare_input_audio()` resamples 48kHz → 16kHz using
basic linear interpolation, which introduces aliasing artifacts. VAD
rejected valid speech that direct STT (using Sherpa's own resampler)
recognized perfectly. Fix: pre-resample to 16kHz in the runner using
numpy FFT-based resampling before passing to `run_turn()`.

**Production implication**: The same resampler issue may affect
production runtime when using 48kHz USB microphone input. A separate
ADR should evaluate upgrading the pipeline's resampler.

## 5. Verification

- ruff check: PASS (0 violations)
- ruff format: PASS (already formatted)
- mypy: PASS (0 errors)
- pytest: PASS (8/8 tests)
- Codex self-verification: 3 rounds PASS
- Codex polish loop: 2 cycles, 20 iterations, 1 fix, terminated normally
- PM independent QC: 3/3 PASS

## 6. References

- Plan doc: `docs/archived/dev-plan/e2e_voice_runner_redesign_v1.md`
- ADR 0051: `docs/adr/0051-pilot-runner-debugging-rollback.md`
- Runner: `scripts/e2e_voice_runner.py`
- Tests: `tests/test_e2e_voice_runner.py`
- Proven reference: `scripts/demo_live.py` (sd.rec blocking pattern)
- Pipeline: `core/pipeline.py` (run_turn batch VAD)
