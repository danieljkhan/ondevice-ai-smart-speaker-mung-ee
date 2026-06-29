# ADR 0049: E2E Voice Pilot Methodology — Sony Recorder + Typecast Batch WAV

- **Status**: Accepted
- **Date**: 2026-04-10
- **Decision makers**: User + Claude Code (PM)

## Context

The Mungi text-input E2E test harness (`scripts/e2e_60rounds_text_tts.py`) bypasses the VAD and STT stages by feeding scripted text directly into `ConversationPipeline`. This validates LLM + TTS behavior but cannot catch any of the following:

- Microphone gain, clipping, or noise floor issues on the Waveshare audio card
- Silero VAD threshold/latency problems on live audio
- Sherpa-ONNX SenseVoice transcription accuracy under realistic acoustic conditions
- Integration bugs between mic input, VAD, STT, and downstream stages
- "Moong-ee" English normalization behavior in the acoustic path

The 2026-04-08 worklog §4 "next steps" item (2) explicitly called for a "production integration test — F2 voice + Moong-ee pronunciation through a real microphone and speaker". The Typecast AI MCP server was configured on 2026-04-08 19:30 KST, making high-quality bilingual (KO/EN) voice synthesis available.

Three methodology options were considered for driving the acoustic input:

1. **Windows PC speaker with `ffplay` automation.** Synthesize WAVs on Windows, drive playback through `ffplay` with fixed inter-utterance gaps, place the PC speaker ~30 cm from the Jetson microphone.

2. **Jetson-local self-playback.** Upload WAVs to Jetson, use `aplay` to play through Jetson's own PAM8403 amp + speaker, loop back into the Jetson MEMS mic.

3. **Sony portable voice recorder with user-triggered batch playback.** Synthesize all WAVs on Windows, pre-concatenate into two batch files (KO + EN) with silence padding, load onto a Sony voice recorder via USB mass-storage, user presses the recorder's play button once per batch.

## Decision

**Option 3: Sony voice recorder + pre-concatenated batch WAVs + user-triggered single play per batch.**

- Typecast synthesis and batch concatenation run on Windows (Codex Task 5.2).
- Batch format: **44.1 kHz stereo 16-bit Linear PCM WAV** (verified against Sony spec sheet).
- Two batch files: `pilot_ko_batch.wav`, `pilot_en_batch.wav`. Each starts with 2 s of silence and separates messages with 8 s of silence.
- User copies both batches to the Sony recorder via USB, positions the recorder ~30 cm from the Jetson mic, presses play once per batch.
- Jetson runner records continuously, detects VAD segments, calls `ConversationPipeline.run_turn()`, logs `rounds.jsonl`, and plays the TTS reply through the Jetson audio card's earphone/speaker output.
- KO and EN batches run sequentially (KO first) to avoid mid-run STT language config switching.

## Rationale

### Why a dedicated portable recorder beats Option 1 (Windows speaker)

1. **Removes the playback device as a variable.** Windows built-in speakers, USB speakers, and Bluetooth devices all have different frequency response and latency characteristics. The Sony recorder's spec sheet (verified 2026-04-10) documents a 50 Hz–20 kHz full-band Linear PCM path, which is a known-good playback profile.
2. **No network sync code.** Windows-to-Jetson coordinated playback would require timing protocols or RPC, adding complexity without diagnostic benefit.
3. **User trigger matches production intent.** A child speaking to Mungi is a discrete, user-initiated event. A manual play press better simulates that than an automated loop.

### Why this beats Option 2 (Jetson self-playback)

1. **Feedback risk.** Playing audio through Jetson's own speaker directly into its own microphones creates acoustic feedback that the pipeline was not designed to handle. Echo cancellation is not part of the current stack.
2. **Unrealistic geometry.** Self-playback places the "speaker source" at 0 cm from the mic and on the wrong axis, eliminating the ~30 cm child-at-arm's-length acoustic path that production will face.
3. **Shared hardware contention.** Using Jetson audio card for both playback and recording simultaneously can cause driver conflicts on the Waveshare SSS1629A5 codec.

### Why pre-concatenated batch WAVs beat per-message trigger

1. **Single play action per batch.** The user presses play once per batch (twice total for KO+EN). No per-message button presses.
2. **Fixed 8 s inter-utterance gap** gives the pipeline enough time to complete VAD→STT→LLM→TTS→reply playback before the next input, with margin.
3. **Silence padding (0.2 s pre + 0.2 s post per message, 2 s batch prefix)** ensures the VAD warmup and onset detection succeed on the first utterance.
4. **Manifest-based matching.** The k-th detected VAD segment maps to the k-th manifest entry by order, with live divergence logging as an early warning.

### Why Typecast for audio generation (vs Supertonic or pre-recorded samples)

1. **Typecast MCP is already configured** (2026-04-08 19:30 KST) with bilingual KO/EN support and emotion control.
2. **Quality parity.** Typecast produces natural-sounding speech adequate for testing STT acoustic path; Supertonic is the Mungi TTS engine under test (using it for input would be a self-reference loop).
3. **Child-voice availability.** Typecast has child-like voice presets; Supertonic's production voice F2 is a young female adult.
4. **Ethical alternative to real child recording.** No consent or privacy issues.

## Voice Selection Role Split

Typecast MCP is available to Claude Code only, not to Codex CLI (MCP servers connect to the Claude Code harness). This forces a two-step voice selection:

| Step | Agent | Tool | Output |
|------|-------|------|--------|
| 1. Voice catalog exploration | Claude Code (PM) | Typecast MCP interactive | Candidate voice list |
| 2. Sample auditioning | Claude Code (PM) | Typecast MCP `synthesize` | Chosen KO + EN voice IDs |
| 3. Task spec update | Claude Code (PM) | `.codex/current-task.md` | Voice IDs embedded in Task 5.2 spec |
| 4. Batch synthesis | Codex | Typecast HTTP REST API | ~60 individual WAVs + 2 batch WAVs |

Codex does not need MCP access; it consumes pre-decided voice IDs and uses the HTTP REST API directly, which is better for resume-safe batch automation anyway.

## Consequences

### Positive

- First full acoustic E2E test covering VAD and STT stages — closes a major testing gap
- Decouples playback device variability from Mungi pipeline variability
- Simple, low-automation setup — pilot can run in ~1 hour end-to-end once implementation is done
- Manual play trigger better simulates production user interaction pattern
- Results provide baseline WER (KO, EN) and latency budget for future regression tracking
- Phase 2 full run (60 rounds) uses the identical harness with a larger batch — no re-architecture needed

### Negative

- Requires physical presence during the test run (user to press play, switch batches, adjust volume)
- Cannot be run in CI/CD without significant rework (which is fine — this is pilot validation, not continuous regression)
- Sony recorder is a dependency on user-supplied hardware — not reproducible outside the user's setup without a similar recorder
- Echo loop possible if user chooses a loud speaker for Jetson audio-card output — mitigation is to prefer earphone or low-volume speaker

### Neutral

- Typecast API usage is well within typical free-tier quotas (~60 calls per pilot)
- Test results are captured in `rounds.jsonl` + `tegrastats.log` + reply WAVs, same format as existing text E2E reports

## Success Criteria

Binding acceptance metrics for the pilot (ADR records them to make regression obvious):

| Metric | Target | Scope |
|--------|--------|-------|
| VAD trigger rate | ≥ 95 % | Per batch |
| STT transcription rate | ≥ 95 % | Per batch |
| WER (KO) | ≤ 15 % | Character-level Levenshtein |
| WER (EN) | ≤ 20 % | Word-level Levenshtein |
| LLM response rate | ≥ 98 % | Per segment |
| Full-pipeline success | ≥ 90 % | Per round |
| Avg "첫소리까지" | ≤ 4.0 s | Per CLAUDE.md §9 definition |
| Safety template misfires | 0 | Unexpected guide/block activations |

Pilot is considered failed if ≥ 2 metrics miss target. Failure triggers root-cause diagnosis and a scoped re-run, not an abandonment.

## References

- `docs/archived/dev-plan/2026-04-10-E2E-Voice-Pilot-Plan.md` — Full v2 plan with phases, task specs, architecture
- `docs/runbooks/weekly/archive/2026-04-08-daily-worklog.md` — Prior session's §4 next steps that motivated this pilot
- `docs/runbooks/weekly/archive/2026-04-08-e2e-input-scripts.md` — Source 597-topic pool (Moong-ee normalized)
- `docs/adr/0046-tts-voice-selection-f2.md` — F2 production voice decision (reply-side TTS)
- `core/pipeline.py` — `ConversationPipeline.run_turn(audio_samples)` entry point reused by the new runner
- `scripts/e2e_60rounds_text_tts.py` — Text-input E2E runner whose preflight helper is reused
- CLAUDE.md §2 (Jetson E2E workflow), §8 (delegation rule), §9 (latency report format), §13 (language policy), §14 (autonomous operation)
