# ADR 0069 — Audio sample NaN/Inf sanitization (defense-in-depth at downmix + WAV write)

- Status: Accepted
- Date: 2026-04-22
- Decision owner: Product orchestrator + Claude Code
- Related: `core/pipeline.py` (`_finite_float`, `_downmix_to_mono`, `_write_temp_wav`), Session 7 real-voice test crash (2026-04-22 ~01:30 KST, session `2026-04-22_01-28-31/`), follow-up verified in `2026-04-22_01-53-38/`

## Context

During the 2026-04-22 Session 7 real-voice test on Jetson Orin Nano Super, a reproducible pipeline crash occurred when the user spoke a short low-energy filler (Korean nasal hum, ~0.3–0.5 s). The stack trace:

```
STT: '嗯。' (2.127s)
Pipeline error: cannot convert float NaN to integer
Traceback (most recent call last):
  File "/opt/mungi-repo/core/pipeline.py", line 815, in run_turn
    input_wav = self._save_conversation_audio(
  File "/opt/mungi-repo/core/pipeline.py", line 1875, in _save_conversation_audio
    self._write_temp_wav(target_path, mono_samples)
  File "/opt/mungi-repo/core/pipeline.py", line 1927, in _write_temp_wav
    pcm_data = struct.pack(
  File "/opt/mungi-repo/core/pipeline.py", line 1929, in <genexpr>
    *(max(-32768, min(32767, int(s * 32768))) for s in samples),
ValueError: cannot convert float NaN to integer
```

Root cause chain:

1. `sounddevice` captured audio from the USB PnP Audio Device (Solid State System `0c76:1229`). For short / low-energy captures (~0.3–0.5 s speech followed by mostly silence in a 20-second buffer), the ALSA path produced a buffer containing NaN / Inf float samples at boundary positions — most likely driver-level instability in the first few frames or uninitialized buffer tail.
2. The VAD and Qwen3-ASR STT stages silently tolerated the NaN samples (NumPy propagates NaN through math operations; Sherpa-ONNX returned `嗯。`).
3. The pipeline then called `_save_conversation_audio(audio_samples, sample_rate, ...)` to persist the input WAV. Internally:
   - `_downmix_to_mono` converted the NumPy array to a Python list of floats, preserving NaN values.
   - `_resample_audio` performed linear interpolation, which propagates NaN through `a * (1 - f) + b * f`.
   - `_write_temp_wav` called `int(sample * 32768)` per sample, which raises `ValueError` when `sample` is NaN.
4. The exception bubbled up to `run_turn`, was caught, and the turn was reported as an error state with the generic fallback TTS `"문제가 생겼어. 한 번 더 말해줄래?"`. Full turn was lost.

This bug was latent and predates Session 6. It was exposed by the combination of `MAX_SECONDS=20` recording window (Session 5 change) plus a very short low-energy filler utterance.

## Decision

Sanitize non-finite audio samples at two layers of the audio save path (defense-in-depth):

### Layer 1 — Crash-site defense in `_write_temp_wav`

Inline NaN/Inf check at the int-conversion boundary. Replace any non-finite sample with 0 (silence) before `int(sample * 32768)`.

### Layer 2 — Upstream defense in `_downmix_to_mono`

Introduce a module-local helper:

```python
@staticmethod
def _finite_float(value: Any) -> float:
    """Return ``float(value)``, substituting NaN/Inf with 0.0."""

    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f
```

Route every float append in `_downmix_to_mono` through `_finite_float`. This prevents NaN from reaching VAD / STT / resample / save in the first place, reducing the surface area of potential NaN-related bugs across the whole pipeline.

### What this ADR does NOT change

- `_resample_audio`: unchanged. Its linear-interpolation body does not generate NaN from finite inputs, and Layer 2 guarantees the inputs are finite.
- VAD / STT / LLM / TTS stages: unchanged. They already tolerated NaN silently in this incident, but going forward they receive only finite samples.
- Hotword hallucination guard (L1+L2 from Session 6): unchanged. Out of scope for this fix — the guard did not fire here because `嗯。` is a single Chinese token, not a repeated hotword pattern.
- Audio hardware / ALSA / PulseAudio configuration: unchanged. Driver-level NaN may continue to occur on specific low-energy captures; this fix makes the pipeline resilient rather than attempting to eliminate the upstream cause (which would require kernel / driver investigation out of scope for this project).

## Rationale

1. **Resilience over root-cause driver fix**: the USB PnP Audio Device is commodity hardware with third-party ALSA drivers. Pursuing a driver-level fix for NaN sample boundaries is high-effort, low-reward, and not in our control. Defensive sanitization at the application layer is predictable, low-risk, and test-covered.
2. **Defense-in-depth is cheap**: both layers together add ~40 lines of code and have negligible runtime cost (a single NaN check per sample; < 1 ms overhead on a 20-second 48 kHz capture).
3. **Silence-as-substitute is the safe choice**: replacing NaN with 0 produces silence at that sample, which is the correct physical meaning of "no signal". Alternatives (interpolation, discard) introduce more complexity without a clear UX benefit.
4. **Upstream Layer 2 protects future stages**: if future features (e.g. warm-resident LLM, streaming STT) introduce additional NaN-sensitive code paths, the upstream sanitization prevents regressions without requiring per-stage hardening.
5. **Matches Session 6 pattern**: identical to `_is_hotword_hallucination` + hotwords reduction — detect the driver/model edge case, sanitize at the pipeline boundary, keep downstream code simple and predictable.

## Alternatives considered

- **(A) Layer 1 only (minimal)**: sanitize only at `_write_temp_wav`. Pro: smallest change. Con: NaN still flows through VAD / STT / resample, creating implicit dependency on their NaN-tolerance — fragile if any of them changes or is replaced.
- **(B) Layer 2 only (upstream)**: sanitize only at `_downmix_to_mono`. Pro: single point of truth. Con: future code paths that bypass `_downmix_to_mono` (e.g. direct `sounddevice` access in tests, streaming buffers) would not be protected. Less resilient than A+B.
- **(C) Numpy-level `np.nan_to_num`**: replace the Python loop with a vectorized NumPy call. Pro: faster on large buffers. Con: pipeline currently uses plain Python lists throughout; introducing NumPy dependency in the hot path inconsistent with existing style. Revisit if performance profiling shows Layer 2 overhead is material.
- **(D) Driver-level ALSA debugging**: investigate USB PnP driver to eliminate NaN at source. Rejected — out of scope, high effort, low reward.

## Consequences

### Positive

- Real-voice pipeline is crash-free under low-energy / short input conditions. Verified empirically: session `2026-04-22_01-53-38` ran 20 turns with 0 crashes, 0 ALSA underruns, 20/20 success including bilingual, hotword guard firing at T18 (`"뭉이야 뭉이야"`), and guard false-positive prevention at T19 (`"뭉이야"`).
- Downstream code paths (VAD, STT, resample, WAV save, LLM, TTS) can assume finite floats throughout.
- Regression test in `tests/test_pipeline.py::TestAudioSanitization` covers `_finite_float` finite/non-finite cases, mono/stereo downmix with NaN, and WAV readback at NaN positions (0 PCM value).

### Neutral

- Negligible runtime overhead (< 1 ms per 20 s capture).
- No API changes. Callers of `_save_conversation_audio` / `_write_temp_wav` unaffected.

### Negative / risks

- If an upstream audio stage one day depends on a specific NaN pattern (e.g. for glitch detection), this fix would mask it. No current stage does this; reconsider if such a feature is added.
- ALSA driver edge cases that produce Inf instead of NaN are also silenced; future telemetry could expose how often this occurs, but for now the product-level answer is "recover and continue".

## Validation

- Codex task `nan-audio-sanitize` (2026-04-22, 348.4 s, PASS).
- `ruff check .` + `ruff format --check .` + `mypy core/ models/ safety/ hardware/ scripts/ parental/` + `pytest tests/ --ignore=tests/integration_jetson -v --tb=short` → 953 passed / 3 skipped / 79.16% coverage.
- Jetson real-voice 20-turn live session `2026-04-22_01-53-38`: 20/20 success, 0 crashes, mean TTFS 17.30 s (T18 hotword guard fire → TTFS 4.40 s / –76 %).
- Polish loop: 2 cycles × 10 iterations, 0 fixes each, terminated.

## References

- `core/pipeline.py::_finite_float`, `_downmix_to_mono`, `_write_temp_wav`
- `tests/test_pipeline.py::TestAudioSanitization`
- Session 7 daily worklog: `docs/runbooks/weekly/archive/2026-04-22-daily-worklog.md` Session 7 addendum
- Jetson artifacts: `/var/lib/mungi/conversations/2026-04-22_01-28-31/` (crash reproduction), `/var/lib/mungi/conversations/2026-04-22_01-53-38/` (post-fix 20-turn validation)
- ADR 0068 (persona redefinition) — prior Session 5 ADR, unrelated but same session family.
