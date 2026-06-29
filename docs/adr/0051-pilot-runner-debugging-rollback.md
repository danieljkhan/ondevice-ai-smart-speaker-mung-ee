# ADR 0051 — Pilot Voice Runner Debugging Saga + Revert Decision

- **Status**: Accepted (revert executed)
- **Date**: 2026-04-12
- **Author**: Claude Code (PM)
- **Related ADRs**: 0049 (E2E Voice Pilot Methodology), 0050 (Hook Wrapper Relative Path)
- **Related worklog**: `docs/runbooks/weekly/archive/2026-04-11-daily-worklog.md` §13-22
- **Related post-mortem**: `docs/runbooks/weekly/archive/2026-04-11-ci-mypy-env-asymmetry-postmortem.md`

---

## 1. Context

Task 5.3 created `scripts/e2e_voice_runner.py` (630 lines) to drive the E2E
voice pilot on Jetson: live microphone capture → `ConversationPipeline.run_turn()`
→ STT/LLM/TTS → TTS playback. Committed in `f7e07ac` (Phase 2 bundle).
CI passed after `f78055a` (mypy fix). The runner was NEVER successfully
executed against real Sony recorder playback despite multiple attempts
across the 2026-04-11 evening/night session (~5 hours). This ADR records
the debugging history, failure modes, and the final decision to revert.

### Test methodology (for context)

The pilot test is an active interactive test:

```
Sony recorder plays pilot_ko_batch.wav (30 msgs × ~3s speech + 8s gap)
  → air ~30 cm
  → Jetson USB mic
  → runner captures each segment
  → runner feeds to ConversationPipeline.run_turn()
    → STT → LLM → TTS
  → TTS reply plays through Jetson USB audio out
  → Mungi responds during the 8-second silence gap
  → Sony plays next message
  → repeat 30 times
```

The 8-second gaps are the **design time window for Mungi's live response**.
Any "record the whole 5:51 and post-process" approach violates the test
methodology — the test is about real-time integrated performance, not
offline transcription accuracy.

## 2. Debugging Timeline (4 failed attempts + revert)

### Attempt 0 — Original Task 5.3 runner (Silero VAD streaming bug)

**Implementation** (committed in `f7e07ac`): Streaming VAD path

```python
# scripts/e2e_voice_runner.py, line ~283
prepared_audio = pipeline._prepare_input_audio(chunk, args.input_sample_rate)
has_speech = bool(prepared_audio) and bool(pipeline._run_vad(prepared_audio))
```

**Observed**: Runner starts, prints "Ready for KO batch. Press play", user
presses play on Sony recorder. Runner logs show **zero speech_segments**.
`runner.log` file is 0 bytes. `rounds.jsonl` is empty. tmux session stays
alive (not crashed). Runner silently waits forever for "speech" that never
arrives.

**Diagnosis**: Empirical test of `pipeline._run_vad()` on various chunk
sizes:

| Chunk | Result |
|-------|--------|
| 0.1 s (100 ms) | **0 segments** |
| 0.25 s | **0 segments** |
| 0.5 s | **0 segments** |
| 1.0 s | **0 segments** |
| 2.0 s | **0 segments** |
| **8.0 s (full buffer)** | **2 SpeechSegment detected** ✅ |

**Root cause**: `models/vad_runner.run_vad()` implementation calls
`model.reset_states()` on every invocation. Silero VAD is a stateful
streaming model requiring hidden-state continuity across frames to
accumulate speech probability. Per-call reset loses context → probability
never crosses threshold for short buffers. Pipeline's `_run_vad()` is a
**batch API** intended for full utterance buffers (used correctly inside
`run_turn()` on complete segment buffers), **NOT for streaming chunks**.

The runner misused this batch API as a streaming API. This was a pattern
directly copied from `scripts/e2e_live_test.py` (which uses
`VOICE_CHUNK_SECONDS = 0.25` in the same wrong way, but that script may
work by accident on real human voice due to different acoustic energy
profiles).

### Attempt 1 — Task 5.3.3 energy-based onset

**Change**: Replace streaming VAD with RMS energy threshold:

```python
# Before (broken)
has_speech = bool(prepared_audio) and bool(pipeline._run_vad(prepared_audio))

# After
has_speech = _chunk_is_loud(chunk, args.energy_threshold_db)  # default -40 dB
```

Added: `DEFAULT_ENERGY_THRESHOLD_DB = -40.0`, `--energy-threshold-db` CLI flag.

**Observed**: First test — runner detected speech and ran `run_turn()`
fully (18s processing). STT output was real Korean text: **"왜 프로젝트 행베리에서 태양이 죽어가고 있는 걸까요..."**. But this content is
completely unrelated to pilot_ko_batch.wav which should start with "뭉이야,
너는 바삭바삭한 치킨 좋아해?".

**PM claim (wrong)**: "Runner works perfectly. Problem is environment
contamination — YouTube audio was playing during test prep."

**User rebuttal**: Clarified environment was clean. Subsequent tests
repeated with explicit clean environment also failed.

**Actual root cause (partial)**:
1. The runner's InputStream opens immediately after "Ready" message is
   printed. There is NO state machine. No warmup. No ARMED state. The
   runner begins capturing on the very next chunk from the callback.
2. If any background audio exists in the room during this window (YouTube
   tab, another app, fan noise hitting threshold, Sony recorder click),
   the energy onset triggers on that audio, not on the intended Sony batch.
3. The user had no explicit "this is the capture window, clear all noise"
   protocol — the runner just started listening.
4. This is a **PM protocol gap**: the runner should not start capturing
   until the user confirms readiness, but there is no mechanism for that.

Subsequent tests in ostensibly clean environments also failed, suggesting
the environment contamination theory is incomplete. Other bugs (see
Attempt 3) also contributed.

**What "worked"**: The energy onset itself worked correctly (triggered on
loud signals), `run_turn()` executed end-to-end (STT + LLM + TTS all ran,
producing coherent Korean transcription and LLM response). The runner
infrastructure at this layer was functional. The failure was at a higher
level — **what** was being captured, not whether capture worked.

### Attempt 2 — Task 5.3.4 buffer tuning + pipeline VAD relaxation + input save

**Changes**:
- `DEFAULT_PRE_ROLL_MS`: 400 → 1000
- `DEFAULT_END_OF_SPEECH_S`: 1.0 → 2.0
- **NEW** `DEFAULT_MIN_CAPTURE_S = 3.0` (floor on captured buffer duration)
- **NEW** `DEFAULT_POST_PAD_MS = 500` (silence appended to sealed buffer)
- PipelineConfig VAD relaxation: `vad_threshold 0.5→0.2`, `vad_min_speech_ms 250→50`, `vad_min_silence_ms 100→30`, `vad_pad_ms 200→300`
- **NEW** `wavs_in/segment_NN_input.wav` saved before `run_turn()` for debugging

**Plus Task 5.3.4.1**: `tests/test_e2e_voice_runner.py` refactored to
reference `DEFAULT_*` constants directly (was hardcoded literal values
that broke with Task 5.3.4's new defaults).

**Observed**: Some segments DID execute the full pipeline (STT returned
non-empty Korean text), proving VAD relaxation partially worked. However:

```json
// Segment 1 (ref: msg 1 "뭉이야...치킨 좋아해?")
{"stt_text": "", "speech_segments": 0, "total_time_s": 0.44}

// Segment 2 (ref: msg 2 "나는 다리보다 날개가...")
{"stt_text": "눈이야 너는 바삭바삭한 치킨 좋아해",
 "speech_segments": 1, "wer": 0.852}
// ← Content is msg 1, not msg 2! Off-by-one.

// Segment 3 (ref: msg 3 "오늘 저녁에 엄마한테...")
{"stt_text": "다리하다 더 맛있더라 맛있더라더라 너 너는 너는 어느 부 어느 부위...",
 "speech_segments": 9, "wer": 1.750}
// ← Content is msg 2 (with VAD fragmentation causing word repetition)
```

**Root causes**:
1. **Segment 1 early seal**: min_capture=3s triggered seal before msg 1
   completed capture OR captured only partial speech → pipeline VAD saw
   insufficient speech evidence → 0 segments → early return.
2. **Segment index ↔ Sony message mismatch (off-by-one)**: Runner's
   `segment_index` increments per sealed segment, regardless of whether
   `run_turn()` succeeded. After seg 1 failed, segment_index became 2,
   but Sony's msg 1 was still playing → seg 2 captured msg 1 content.
   From that point, all segments were one behind the Sony timeline.
3. **Segment 3 fragmentation**: Pipeline VAD was relaxed to
   `vad_min_silence_ms=30` (very sensitive to silence) → split a single
   Sony message into 9 tiny segments based on intra-word pauses → STT
   transcribed each fragment separately producing word duplication.

**Lesson**: Segment numbering based on count, not on content matching,
is fragile under any failure condition. Also, over-relaxing VAD
parameters produces fragmentation artifacts that are as bad as
under-relaxing them.

### Attempt 3 — Task 5.3.6 audio_queue drain (worst failure)

**Theory**: `pipeline.run_turn()` takes 15-25 seconds on first segment
(lazy model loading). During this processing, the `sd.InputStream`
callback continues pushing 100ms chunks to `audio_queue`. After
`run_turn()` returns, the queue has ~150 chunks of stale audio (Sony's
subsequent messages + gaps that happened while the pipeline was busy).
The next iteration processes these stale chunks → segment N captures
audio from N+1 or N+2's timeframe → permanent off-by-one cascade.

**Change**: Added `_drain_audio_queue(audio_queue)` helper called
immediately after every `run_turn()` return, discarding all buffered
chunks before resuming the main loop.

**Observed (live test with explicit clean environment)**:

```
runner.log (only info-level entries):
  Speech detected for segment 1
  Segment 1 processed: success=True wer=1.000 total=0.478s
  Drained 1 stale audio chunks (0.1s)
  Speech detected for segment 2
  Segment 2 processed: success=True wer=1.000 total=0.727s
  Drained 1 stale audio chunks (0.1s)
  Speech detected for segment 3
  Segment 3 processed: success=True wer=1.000 total=0.363s
  Drained 3 stale audio chunks (0.3s)
  ERROR: ABORT consecutive WER threshold exceeded
```

All 3 segments returned in <1 second (early return, empty `user_text`,
`speech_segments=0`, `stt_text=""`).

**Saved wavs_in/ files for diagnosis**:

```
segment_01_input.wav: 6.4 s, Peak -19.76 dB, RMS avg -43.80, RMS Pk -30.53
segment_02_input.wav: 17.6 s, Peak -15.31 dB, RMS avg -49.53, RMS Pk -27.79
segment_03_input.wav: 8.9 s, Peak -15.11 dB, RMS avg -48.38, RMS Pk -29.24
```

**Direct Sherpa STT on saved files (bypassing pipeline VAD entirely)**:

```python
segs, info = run_stt(model, Path("segment_02_input.wav"), "ko")
# → lang='nospeech', text=(empty)
```

All 3 segment files are rejected by Sherpa STT as **non-speech audio**,
even though levels show clear Peak (speech-range -15 to -20 dB) and
RMS Pk (speech-range -27 to -30 dB). The contradiction is that overall
RMS is 10 dB lower than speech-range (-49 dB vs speech average -30 dB).

**Critical control group test** (same session, same environment, same
Sony playback, different capture method):

```
File                    Method                        Peak   RMS avg  RMS Pk  Sherpa STT
cmp_arecord.wav         arecord -D plughw:0,0 8s      -16.49 -35.97   -27.45  "나는 다리보다 나가 더 맛있더라..." ✅ (msg 2)
cmp_sdrec.wav           sd.rec(device=None) 8s        -15.93 -39.03   -28.65  "오늘 저녁에 엄마한테 치킨 사" ✅ (msg 3)
segment_02_input.wav    RUNNER (sd.InputStream+       -15.31 -49.53   -27.79  lang=nospeech, empty  ❌
                        callback+queue+seal)
```

**The runner's capture mechanism produces audio with 10 dB lower overall
RMS than direct sd.rec, despite identical Peak and RMS Pk values.**

Crest factor (Peak minus RMS avg):
- Normal speech: 10-15 dB
- cmp_sdrec.wav: 15.93 - (-39.03) = ~23 dB (normal for mixed speech + gaps)
- segment_02_input.wav: 15.31 - (-49.53) = **34 dB** (anomalously high)

A 34 dB crest factor means the audio is **mostly silence with brief
transients**, not speech + natural silence. Sherpa's "lang=nospeech"
verdict is consistent with this — the audio does not contain enough
sustained speech-level content to be classified as speech at all.

**Possible causes (NONE verified before revert)**:

1. **Stream stop/start ALSA glitches**: The runner calls `stream.stop()`
   before `run_turn()` and `stream.start()` after. Repeated stop/start
   cycles may cause PortAudio/ALSA driver-level discontinuities, dropping
   samples or resetting buffers. Direct `sd.rec()` has no such cycle.

2. **Callback thread timing**: The callback runs in a separate PortAudio
   thread and pushes 100ms chunks to `audio_queue`. Thread contention
   during `run_turn()` processing may cause chunks to be pushed out of
   order or with timing drift. The `audio_queue.put` is thread-safe but
   temporal ordering at the application level may not be.

3. **`_seal_captured_segment()` concatenation bug**: The sealing function
   calls `np.concatenate(sealed_chunks)` where `sealed_chunks` is a list
   of float32 arrays from the callback. If any chunks have inconsistent
   shapes or incorrect `copy=True` semantics, the result could be
   silently corrupted.

4. **`min_capture` + `post_pad` excessive silence injection**: When
   `min_capture=3.0s` forces capture to extend past actual speech, the
   runner keeps appending incoming chunks (which are silence after the
   message ended). Plus `post_pad_ms=500` adds explicit silence at the
   end. The total silence-to-speech ratio in the sealed buffer becomes
   dominated by silence, potentially triggering Sherpa's "nospeech"
   classification.

5. **Float32 → int16 conversion during `_save_input_audio`**: The helper
   writes float32 data to PCM_16 WAV via soundfile. If the float32 data
   has DC offset or values outside [-1, 1], the conversion could produce
   clipping artifacts that Sherpa can't interpret as speech.

6. **Compound of multiple above**: The most likely reality — several
   contributing factors, each small, but together producing fundamentally
   corrupted audio.

**Critical failing**: Because all 4 attempts (5.3.3, 5.3.4, 5.3.4.1,
5.3.6) were accumulated in the uncommitted working copy WITHOUT
individual commits or isolated deployments, none of the bugs could be
attributed to a specific change. Each Task built on the previous,
compounding complexity. The runner went from ~630 lines (Task 5.3
original) to ~830 lines (after 5.3.6), with none of the new paths
properly validated.

### Revert decision (user-directed)

User observed the "weird audio" captures from Task 5.3.6 and directed:

> "스모크 테스트 환경으로 원본하고 처음부터 다시 시작하자"
> (Translation: "Let's restart from the smoke test environment and the
> original. Give me your opinion.")

User later clarified "원본" was a typo/misspeak for "원복" (revert). They
wanted the runner reverted to the committed state `f78055a` via
`git checkout`, bringing back the original Task 5.3 runner (with the
known Silero VAD streaming bug) as the baseline for fresh debugging.

PM executed:
```bash
git checkout scripts/e2e_voice_runner.py
git checkout tests/test_e2e_voice_runner.py
scp scripts/e2e_voice_runner.py mungi@jetson:/opt/mungi-repo/scripts/
scp tests/test_e2e_voice_runner.py mungi@jetson:/opt/mungi-repo/tests/
```

**Post-revert state**:
- Runner: 630 lines, `f78055a` exactly
- Tests: 134 lines, `f78055a` exactly
- Contains: `pipeline._run_vad(chunk_100ms)` streaming bug (see Attempt 0)
- Jetson `/opt/mungi-repo/` synchronized

The reverted state has a KNOWN bug (Attempt 0's Silero VAD streaming
misuse), meaning fresh debugging must start by either fixing that bug
differently than Task 5.3.3 did, or by replacing the whole live-capture
architecture.

## 3. Decision

**Revert the runner to `f78055a` (committed state). Do NOT carry forward
any of the Task 5.3.3/5.3.4/5.3.4.1/5.3.6 changes.**

Rationale:
1. Those changes are unverifiable — each built on the previous without
   isolation, and the final state produced fundamentally corrupted audio
   that Sherpa rejects as non-speech.
2. None of the changes individually were proven to improve the runner's
   correctness against Sony recorder playback.
3. The PM cannot point to a single change that "works" — only to a
   sequence of changes that each introduced new problems.
4. User (correctly) lost confidence in the incremental debugging approach.
5. Starting from the known-committed `f78055a` baseline enables fresh
   analysis without contamination from failed experiments.

## 4. Consequences

### Immediate

1. **The pilot cannot run** in its current state. `f78055a` contains the
   Silero VAD streaming bug and will produce 0 segments on live Sony
   playback. Test execution requires a new fix or architectural rework.
2. **All time spent on Tasks 5.3.3/5.3.4/5.3.4.1/5.3.6 (~150 minutes)
   produced zero committed code**. The knowledge gained (failure modes,
   architecture insights, control-group comparison data) is captured in
   this ADR and the 2026-04-11 daily worklog §15-16.
3. **The `assets/voice_experiments/` directory contains multiple runner
   output directories** from failed runs, consuming disk space but
   gitignored. Safe to retain for offline analysis or delete.
4. **`~/.asoundrc` on Jetson is left in place** (ALSA default routing
   override for USB PnP audio). This is unrelated to the runner and
   remains a valid fix (see ADR 0052).

### For future debugging

1. **Incremental commits MANDATORY**: Any runner change must be a
   standalone commit. No accumulation of uncommitted modifications.
2. **Control group tests per iteration**: Every runner attempt must be
   compared to direct `arecord`/`sd.rec` capture using the same audio
   source, in the same session. If the runner produces different audio
   than the control, there is a bug in the runner's capture chain.
3. **Input segment saving from day 1**: Future runner designs must save
   captured audio to disk BEFORE calling any processing pipeline, so
   post-hoc analysis is always possible. Task 5.3.4 added this late —
   should have been in Task 5.3 original.
4. **Isolate state machine from capture logic**: The runner conflates
   "when to capture" (state machine) with "how to capture" (audio
   streaming). These should be independently testable.
5. **Direct STT preferred over pipeline VAD**: The smoke test
   (`load_stt_model` + `run_stt` on a WAV file) works reliably. Pipeline's
   `run_turn()` adds VAD which has its own issues. For the pilot test,
   bypassing pipeline VAD and calling STT directly may be more robust.
6. **Test environment hygiene protocol**: Before launching the runner,
   explicitly instruct the user to stop ALL background audio sources.
   The "ARMED" state mentioned in Section 2 Attempt 1 is the correct
   way to enforce this protocol.

### For production wake word

The runner's issues do NOT invalidate the production Mungi runtime.
`core/pipeline.py` uses `_run_vad()` correctly (on complete utterance
buffers inside `run_turn()`). The wake word architecture for production
will use `openwakeword` (already in project dependencies) for streaming
wake detection, NOT Silero VAD. The runner's misuse of `_run_vad()` for
streaming is orthogonal to production code quality.

### Documentation debt

1. **CLAUDE.md §3 needs correction**: Lists "Waveshare Audio Card for
   Jetson Nano (model 19491, SSS1629A5 codec + dual MEMS mics +
   PAM8403 amp)" — describes the I2S HAT variant. The actual installed
   product is a USB PnP variant (JMTek 0c76:1229). See ADR 0052.
2. **Plan doc §6 Hardware Setup**: Needs similar correction.
3. **`scripts/e2e_live_test.py` has same Silero VAD bug** as the original
   Task 5.3 runner. It may be currently unused but represents latent
   risk. Separate investigation needed.

## 5. References

- Commits: `f7e07ac`, `f78055a`, `ec22d22` (all committed this session)
- Uncommitted work (reverted): Task 5.3.3, 5.3.4, 5.3.4.1, 5.3.6
- Worklog: `docs/runbooks/weekly/archive/2026-04-11-daily-worklog.md` §13-22
- CI post-mortem: `docs/runbooks/weekly/archive/2026-04-11-ci-mypy-env-asymmetry-postmortem.md`
- ADR 0049: E2E Voice Pilot Methodology
- ADR 0052: Jetson ALSA Default Routing Override (next)
- `scripts/e2e_voice_runner.py` (current: reverted to `f78055a`)
- `models/vad_runner.py` (Silero VAD wrapper — batch API only)
- `core/pipeline.py` (pipeline's correct use of `_run_vad()` on full buffers)
- `scripts/simulate_ci_mypy.sh` (added this session for CI mypy validation)
