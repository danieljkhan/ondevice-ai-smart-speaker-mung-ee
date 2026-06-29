# ADR 0100: Mungi voice — mung-ee v2 custom KO voice + synthesis/playback spec (supersedes ADR 0046 selection)

- **Status**: Accepted (2026-06-03 — ratifies the shipped voice configuration. Reference spec:
  `docs/runbooks/voice-final-spec.md`, PR #158 `10bf1cc`.)
- **Date**: 2026-06-03
- **Authority**: `docs/runbooks/voice-final-spec.md` (code-grounded spec); project memory
  `project_tts_tobi_custom_voice`; PR #151 `1585d97` (playback silence padding).
- **Related**: ADR 0046 (TTS voice selection — F2 preset; **superseded in part** by this ADR), ADR 0088
  (Piper retirement — Supertonic is the sole engine), ADR 0053 (voice runner v2 architecture), ADR 0043
  (Supertonic voice blending), ADR 0099 (kiosk audio routing context).

## Context

ADR 0046 selected the Supertonic preset voice `F2` for the F2 milestone. The voice has since evolved to a
**custom Korean voice ("mung-ee")** with tuned synthesis parameters and a designed sound-cue set, but no ADR
ratified the current configuration — the values lived only in code, project memory, and (as of 2026-06-03) the
new reference doc `docs/runbooks/voice-final-spec.md`. Per CLAUDE.md §7 rule 4 the voice identity + synthesis
policy is an architecture-level decision; this ADR ratifies it and records ADR 0046's F2 selection as
superseded for production Korean output. All values below are grounded in current code (file:line in the
reference doc).

## Decision

### D1 — Active voice = `mung-ee` v2 (custom Supertonic KO voice)
Production Korean output uses the custom **mung-ee v2** Supertonic voice, not the ADR 0046 `F2` preset.
- Asset: a runtime voice JSON `/var/lib/mungi/voices/mung-ee.json` on the Jetson (gitignored / licensed — see
  `assets/voice/` in `.gitignore`; predecessor `tobi.json` also retained on device).
- Selection: `MUNGI_TTS_VOICE_STYLE_KO` env points to the mung-ee JSON path. `core/model_manager.py`
  `ModelConfig.tts_voice_style` keeps the `F2` preset as the **fallback** when no per-language voice is set;
  `tts_voice_style_ko` / `tts_voice_style_en` are the per-language selectors. English keeps a preset voice.
- Supertonic remains the sole engine (ADR 0088).

### D2 — Synthesis parameters: `speed=0.95`, runtime `total_steps=7`
`models/tts_runner.py::synthesize` applies `speed=0.95` unconditionally (ko/en) and a runtime default
`total_steps=7` (chosen via the user A/B/C cue + runtime latency tradeoff). **Offline** cue/asset generators
may pass a higher `total_steps` for cleaner pre-rendered audio.

### D3 — Playback silence padding: lead 0.30 s / trail 0.40 s
`hardware/audio_player.py` pads every playback (PR #151 `1585d97`) to fix USB-audio-card front/tail clipping
(the tail clip had eaten response endings and the ack ding-dong's 2nd note): lead `0.30 s`
(`MUNGI_AUDIO_LEAD_SILENCE`), trail `0.40 s` (`MUNGI_AUDIO_TRAIL_SILENCE`). Output is pinned to the PulseAudio
sink (`MUNGI_AUDIO_OUTPUT_DEVICE=pulse`, ADR 0099 D5).

### D4 — Sound-cue set (pre-rendered, RAM-loaded)
`core/sound_bank.py` loads immutable WAV cues from `assets/sounds/` (tracked in git): tap **ack** =
`feedback/ack.wav` (the **딩-동 / ding-dong softB** cue), long-press **chime** =
`feedback/long_press_chime.wav`, **wake** = time-bucketed `welcome_{morning|afternoon|evening|night}` on
first-of-day else `wake_ack` (wake-day boundary 05:00; buckets morning 05–10 / afternoon 11–16 / evening
17–20 / night otherwise), **error** = `error/{kind}/*.wav`, **sleep** = `sleep/*.wav`. Speech cues are
rendered offline with the mung-ee voice (D2 higher offline steps); padding (D3) is applied at playback, not
baked into the cue files.

## Consequences

- ADR 0046's `F2` selection is **superseded for production Korean output** (F2 remains the code-default
  fallback only). The mung-ee JSON is a runtime/licensed asset, so reproducing the device voice requires that
  file (not in git); the configuration and recipe are captured in `docs/runbooks/voice-final-spec.md`.
- Synthesis is pinned (`speed`, `total_steps`) in code; tuning requires a code change (or the documented env
  overrides for padding/output/voice selection).
- This decision is the default voice spec unless superseded by a later ADR; the reference doc
  `voice-final-spec.md` is kept in sync with code as the living spec.

## Related ADRs

- ADR 0046 — TTS voice selection (F2). Superseded in part (production KO voice).
- ADR 0088 — Piper retirement (Supertonic sole engine).
- ADR 0053 — Voice runner v2 architecture.
- ADR 0043 — Supertonic voice blending.
- ADR 0099 — Boot-persistent kiosk runtime (audio routing / pulse output pin).

## References

- `docs/runbooks/voice-final-spec.md` (authoritative, code-grounded).
- `models/tts_runner.py::synthesize`, `core/model_manager.py` `ModelConfig`, `hardware/audio_player.py`,
  `core/sound_bank.py`, `assets/sounds/`.
- PR #151 `1585d97` (playback silence padding); PR #158 `10bf1cc` (voice spec doc).
- Project memory `project_tts_tobi_custom_voice`, `project_runtime_config_gap` (R-8 env-override operation).
