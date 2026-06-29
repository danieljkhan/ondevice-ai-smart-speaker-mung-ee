# ADR 0043: Supertonic Voice Style Blending via Embedding Interpolation

**Status**: Experimental — Under Evaluation  
**Date**: 2026-04-08  
**Author**: Claude Code PM (Opus 4.6)  
**Related**:
- ADR 0002 — Phase 0 baseline (Supertonic TTS 2 as primary TTS)
- `Dev_Plan/voice_blend_experiment.py` — Experiment implementation
- `assets/voice_experiments/` — Generated audio samples (local, not committed)

## Context

Mungi uses Supertonic TTS 2 as its primary synthesis engine with 10 built-in voice styles
(M1–M5, F1–F5). The default production voice is **F2** (bright, cheerful female), chosen
for its child-friendly tone.

User research context: Mungi's product vision is "the world's safest first AI friend for
children." A voice that is slightly warmer, calmer, or more distinctive than any single
built-in style could strengthen the "AI friend" feeling without training a new model.

Constraints on Jetson Orin Nano:
- No GPU-accelerated voice encoder runtime (e.g., YourTTS, VALL-E, Tortoise) fits within 8 GB
- Full voice cloning requires a separate encoder model (~300–500 MB), reducing memory for LLM
- Any custom TTS model must support ARM64 and ONNX/CUDA deployment
- Development timeline does not allow custom model training before first field test

## Decision

Explore **linear interpolation of Supertonic built-in voice style embeddings** as a
zero-cost voice customization mechanism. This requires no additional model, no training,
and no extra memory.

### How Supertonic voice styles work

Each built-in voice is represented by two numpy arrays stored in the model's voice JSON:

| Array | Shape       | Meaning                          |
|-------|-------------|----------------------------------|
| `ttl` | `(1, 50, 256)` | Timbre/style latent (spectral character) |
| `dp`  | `(1, 8, 16)`  | Duration/prosody parameters      |

A blended style is computed entirely in numpy before passing to the synthesizer:

```python
blended_ttl = voice_A.ttl * w_A + voice_B.ttl * w_B   # w_A + w_B = 1.0
blended_dp  = voice_A.dp  * w_A + voice_B.dp  * w_B
custom = copy.deepcopy(voice_A)
custom.ttl[:] = blended_ttl
custom.dp[:]  = blended_dp
```

### Experiments (2026-04-08)

7 experiments × 2 languages (KO/EN) = 14 WAV files:

| # | Name | Recipe | Hypothesis |
|---|------|--------|------------|
| 1 | baseline_F2 | F2 原 | 기준선 |
| 2 | baseline_M4 | M4 原 | 남성 기준선 |
| 3 | blend_F2_M4_7030 | F2×0.7 + M4×0.3 | 밝음 + 편안함 |
| 4 | blend_F2_F5_5050 | F2×0.5 + F5×0.5 | 밝음 + 부드러움 |
| 5 | blend_F2_M4_F5_equal | (F2+M4+F5)/3 | 균형잡힌 중성 |
| 6 | sharp_F2 | F2.ttl + ∇ttl×1.5 | 스펙트럼 강조 |
| 7 | slow_F2 | F2, speed=0.8 | 느린 속도 |

### Custom style JSON export

The blend_F2_M4_7030 style is exported as:
```json
{"style_ttl": {"dims": [1,50,256], "data": [...]},
 "style_dp":  {"dims": [1,8,16],  "data": [...]}}
```
This format is compatible with `tts.get_voice_style_from_path(Path)`, enabling reuse
without re-computing the blend at runtime.

## Implementation

| File | Role |
|------|------|
| `Dev_Plan/voice_blend_experiment.py` | Standalone experiment runner |
| `assets/voice_experiments/*.wav` | Generated audio samples (gitignored) |
| `assets/voice_experiments/mungi_blend_F2_M4_7030.json` | Exportable custom style |

### Key implementation notes

1. **Shape handling**: `tts.synthesize()` returns `wav` with shape `(1, N)` (not `(N,)`).
   `scipy.io.wavfile.write` interprets `(rows, cols)` as `(n_samples, n_channels)`.
   Fix: `np.asarray(wav, dtype=np.float32).squeeze()` before WAV write.

2. **Style cloning**: `copy.deepcopy(base_style)` is the confirmed working pattern.
   In-place assignment (`custom.ttl[:] = blended`) avoids object reconstruction.

3. **Gradient sharpening**: `np.gradient(ttl, axis=1)` computes finite differences along
   the time axis (dim 1 of the 50-step style sequence). The result may be clipped
   by the synthesizer's internal normalization; audible effect varies.

## Consequences

### If a blended voice is selected for production

- Zero memory overhead (style arrays are ~50 KB total)
- Sub-millisecond blend computation at startup
- Style can be baked into a JSON file and loaded via existing API
- No changes to `core/pipeline.py` or `hardware/` layer

### If blending is insufficient

- Supertonic 2 supports custom voice training from audio (not tested on Jetson)
- Alternative: swap to a different built-in voice (F1, F3, F5) that better fits product tone
- Future option: integrate a lightweight voice encoder when memory headroom allows

## Decision Status

**Experimental.** A listening evaluation of the 14 generated WAV files is required before
promoting any blended style to production. The blend_F2_M4_7030 (F2 70% + M4 30%) is the
primary candidate based on hypothesis (warmth without losing brightness).

No production code changes until a specific blend is approved by product review.
