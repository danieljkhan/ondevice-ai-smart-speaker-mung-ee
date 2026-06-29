# Mungi Voice — Final Spec (2026-06-03)

Authoritative reference for the 뭉이 (Mungi) voice: TTS engine, custom voice identity, synthesis parameters,
playback padding, and the pre-rendered sound-cue set. Values below are grounded in the current code/config
(file:line cited); update this doc when those change. This consolidates the long-running "voice final spec"
that previously lived only in memory and across several ADRs (it is **not** itself an ADR — the ADR 0046
successor is still open; see §8).

## 1. Engine & voice identity

- **Engine**: Supertonic TTS 2 — the **sole** TTS engine (Piper retired, ADR 0088). Config default
  `tts_engine = "supertonic"` (`core/model_manager.py` `ModelConfig`). Sample rate comes from the loaded model.
- **Active Korean voice**: **`mung-ee` v2** — a custom Supertonic KO voice. Runtime asset
  `/var/lib/mungi/voices/mung-ee.json` (Jetson). The predecessor `tobi.json` is also present on the device.
  Voice JSON assets are runtime-only / gitignored (licensed — see `assets/voice/` in `.gitignore`).
- **Selection mechanism** (`core/model_manager.py`):
  - `tts_voice_style` (default `"F2"`, a shipped preset) — used when no per-language voice is set.
  - `tts_voice_style_ko` / `tts_voice_style_en` — per-language selectors (preset name or absolute JSON path).
  - Env overrides: `MUNGI_TTS_VOICE_STYLE` (legacy, sets both), `MUNGI_TTS_VOICE_STYLE_KO`,
    `MUNGI_TTS_VOICE_STYLE_EN`. The KO voice is pinned to the mung-ee JSON via `MUNGI_TTS_VOICE_STYLE_KO`
    (the Jetson has no `config.json` — it runs on env overrides; see project memory R-8).
  - At synth time (`models/tts_runner.py::synthesize`), if a per-language style is configured the KO style is
    used for `lang="ko"` and the EN style for `lang="en"`; otherwise the single `tts_voice_style` is used.

## 2. Synthesis parameters

`models/tts_runner.py::synthesize(text, language="ko", total_steps=7)`:

| Param | Value | Notes |
|-------|-------|-------|
| `speed` | **0.95** | Applied unconditionally for both `ko` and `en` (`tts_runner.py:682`). |
| `total_steps` | **7** | Runtime default (`tts_runner.py:630`). Chosen from the user A/B/C cue + runtime decision. **Offline** cue/asset generators may pass a **higher** step count for cleaner pre-rendered audio (docstring `tts_runner.py:638`). |
| `lang` | `"ko"` (primary) / `"en"` | Unknown language → warns + defaults to `"en"`. |

Blank/None text → empty array (no synthesis). Older Supertonic APIs without `voice_style` fall back to a
plain `synthesize(text)` call.

## 3. Playback silence padding

`hardware/audio_player.py` (PR #151, dev `1585d97`) pads every playback to fix USB-audio-card front/tail
clipping (the tail clip had eaten response endings and the ack ding-dong's 2nd note):

| Pad | Default | Env override |
|-----|---------|--------------|
| **lead** (front) | **0.30 s** (`_DEFAULT_PLAYBACK_LEAD_SILENCE_S`) | `MUNGI_AUDIO_LEAD_SILENCE` |
| **trail** (tail) | **0.40 s** (`_DEFAULT_PLAYBACK_TRAIL_SILENCE_S`) | `MUNGI_AUDIO_TRAIL_SILENCE` |

- Output device on the kiosk is pinned to the PulseAudio sink: `MUNGI_AUDIO_OUTPUT_DEVICE=pulse`
  (Session 94 HDMI-steal fix — the touchscreen DP→HDMI adapter introduced an HDA HDMI sink that the blind
  fallback otherwise selected; the USB speaker is only reachable for output via pulse). See
  `scripts/mungi-kiosk-start.sh` and the touchscreen-display project memory.

## 4. Sound cues (`core/sound_bank.py`, RAM-loaded immutable at init)

Root: the runtime `sounds/` dir (repo `assets/sounds/`, tracked in git — 27 WAVs). All cues are **pre-rendered
WAVs** synthesized offline with the mung-ee voice (speech cues) plus designed tones (ack/chime).

| Cue | Asset | Picker | Notes |
|-----|-------|--------|-------|
| **Tap ack** | `feedback/ack.wav` (or a `feedback/ack/` dir of variants if present) | `pick_ack()` | The **딩-동 (ding-dong) softB** cue. Optional — `None` if absent. |
| **Long-press chime** | `feedback/long_press_chime.wav` | `chime()` | Parent long-press entry cue. |
| **Wake — first of day** | `wake/welcome_{morning,afternoon,evening,night}/*.wav` | `pick_wake()` | Time-bucketed welcome. |
| **Wake — same day** | `wake/wake_ack/*.wav` | `pick_wake()` | Returning-wake acknowledgement. |
| **Error** | `error/{kind}/*.wav` (e.g. `error/stt_load_fail/01..03.wav`) | `pick_error(kind)` | STT-load + other failure cues. |
| **Sleep** | `sleep/*.wav` | `pick_sleep()` | Idle→sleep cue. |

Wake-day / time-bucket logic (`sound_bank.py`):
- **Wake-day boundary 05:00** (`WAKE_DAY_HOUR_OFFSET = 5`): "first of day" resets at 05:00 local, so a wake
  after midnight but before 05:00 still counts as the previous day.
- **Time buckets**: morning `05–10`, afternoon `11–16`, evening `17–20`, night otherwise.

## 5. Cue generation (recipe)

Speech cues (wake/sleep/error/wake_ack) are rendered **offline** with the active **mung-ee** voice. Offline
renders may use a **higher `total_steps`** than the runtime `7` for cleaner pre-rendered assets (per
`tts_runner.py` docstring), then are saved as WAVs under `assets/sounds/`. The tap **ack** is the designed
**ding-dong softB** tone. (Regenerating cues = re-render with the mung-ee voice + commit the WAVs; the
silence padding in §3 is applied at **playback** time, not baked into the cue files.)

## 6. Path / source map

| Concern | Location |
|---------|----------|
| Synthesis params (speed/steps/lang) | `models/tts_runner.py::synthesize` (~626–702) |
| Voice selectors + env | `core/model_manager.py` `ModelConfig` (`tts_voice_style[_ko/_en]`, `__post_init__`) |
| Playback padding + output device | `hardware/audio_player.py` (`_DEFAULT_PLAYBACK_*_SILENCE_S`, ~29–99, 330) |
| Sound-cue loading/picking | `core/sound_bank.py` |
| Cue WAV assets (tracked) | `assets/sounds/{wake,error,sleep,feedback}/…` |
| Active voice JSON (runtime) | `/var/lib/mungi/voices/mung-ee.json` (Jetson; gitignored) |
| Kiosk env pins | `scripts/mungi-kiosk-start.sh` (`MUNGI_AUDIO_OUTPUT_DEVICE=pulse`, …) |

## 7. References

- ADR 0088 — Piper TTS fallback retirement (Supertonic is the sole engine).
- ADR 0053 — Voice runner v2 architecture.
- ADR 0046 — TTS voice selection (F2). **Superseded in practice** by mung-ee v2 + speed 0.95 + steps 7;
  no successor ADR yet (§8).
- ADR 0043 — Supertonic voice blending.
- PR #151 (`1585d97`) — playback silence padding (lead 0.30 s / trail 0.40 s).
- Project memory: `project_tts_tobi_custom_voice`, `project_touchscreen_display_solution` (Session 94 audio
  routing), `project_runtime_config_gap` (R-8: no `config.json` on Jetson; env-override operation).

## 8. Open items

- **ADR 0046 successor (unresolved)**: no ADR records the current voice decision (mung-ee v2 KO custom voice,
  `speed=0.95`, runtime `total_steps=7`, ack ding-dong softB, lead 0.30 s / trail 0.40 s). This doc captures
  the spec; a successor ADR should ratify it.
- **~11-second tail** (ADR 0058, Wave 3) — separate TTS-latency item, not addressed here.
