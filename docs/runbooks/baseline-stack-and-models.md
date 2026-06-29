# Baseline Technical Stack and Models

> Extracted from `CLAUDE.md §3` and `§11`. Rule authority: `CLAUDE.md`.
> Model selection rationale: `Dev_Plan/Mungi_Model_Selection_Report_v1.md` (authoritative).

## 3. Baseline Technical Stack

- Ubuntu 22.04
- JetPack 6.2
- CUDA 12.6
- Python 3.10

Default AI pipeline order:

1. Silero VAD
2. Qwen3-ASR STT (per ADR 0055 + 2026-04-29 Update; SenseVoice fallback retired by user direction 2026-04-28)
3. Gemma 4 E4B QAT Q4_K_XL via llama.cpp (primary; active model: `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`, code-pinned `DEFAULT_GEMMA4_TEXT_MODEL_PATH`).
   Load-failure auto-fallback: Gemma 4 E2B Q5_K_M (`gemma-4-E2B-it-Q5_K_M.gguf`) via `MUNGI_LLM_FALLBACK_MODEL_PATH` env / `llm_fallback_model_path` config / E2B default; engages on primary **load** failure only (ADR 0102).
   Manual rollback: Qwen3.5-2B DPO Q6_K (`Qwen3.5-2B-DPO.Q6_K.gguf`) via `MUNGI_LLM_BACKEND=qwen3_legacy` env override or `"llm_backend": "qwen3_legacy"` in `config.json` (ADR 0073; qwen retirement deferred to G2).
4. Supertonic TTS 2 (sole TTS engine; no in-process fallback — see ADR 0088)
5. Conversation Memory RAG (koen-e5-tiny ONNX shared + separate FAISS index)

## Runtime Memory Model

The runtime keeps sequential GPU-stage ownership from ADR 0009 while enabling L1 LLM residency by default: `ManagerConfig.llm_resident=True` keeps the Gemma 4 LLM loaded between turns, with `MUNGI_LLM_RESIDENT=0` or `--no-llm-resident` as the rollback path. This default is valid only while Jetson smoke evidence keeps peak system RAM below 5500 MB and the existing `ManagerConfig.memory_limit_mb=6000` critical guard remains uncrossed; if RAG, full-audio residency, or STT residency pushes memory beyond that invariant, disable L1 residency and re-measure before restoring the default.

Hardware interfaces:
- Audio: Waveshare Audio Card for Jetson Nano — **USB PnP variant** (JMTek chipset
  `0c76:1229`, external 3.5 mm mic/earphone jacks). Appears as ALSA `card 0` via
  `hw:0,0`. NOT the I2S HAT variant (the I2S version with SSS1629A5 codec / dual
  MEMS / PAM8403 amp was in an earlier plan but never acquired). **Requires a user-level
  `~/.asoundrc` override** because NVIDIA Jetson's default `/etc/asound.conf` routes
  `pcm.!default` to the Tegra APE (I2S HAT interface), leaving the USB card unused
  by any application that relies on the ALSA/PortAudio default device. See
  `docs/adr/0052-jetson-alsa-default-routing-override.md` for the override snippet
  and rationale.
- Display: Waveshare 4inch HDMI Capacitive Touch IPS LCD (C) 720×720 + DP→HDMI Active adapter + USB-C cable. Replaces retired WS2812B LED + RP2040 Pico hardware (per ADR-NEW-1 LED retirement, ADR-NEW-2 touchscreen adoption — both targeted for slots 0079/0080 at finalization).

Fallback tiering: tier 1 runtime auto-fallback (primary **load** failure) is **E2B** (`gemma-4-E2B-it-Q5_K_M.gguf`, ADR 0102). Manual rollback tier is `qwen3_legacy` (`Qwen3.5-2B-DPO.Q6_K.gguf`) via env/config override (ADR 0073).

> 2026-04-28: tier-2 manual backup (`Qwen3.5-2B-FT.Q6_K.gguf`) retired; the file was never deployed to Jetson and is removed from the active fallback hierarchy.

Removed models (deleted from Jetson 2026-04-05):
- `Qwen3-4B-Q4_K_M.gguf`: Replaced by the Qwen3.5-2B family for improved Korean quality and bilingual English retention.
- `Qwen3-8B-Q4_K_M.gguf`: Too large (4.7 GB) for Jetson 8 GB unified memory — GPU acceleration impossible, CPU-only ~2.6 tok/s.
- `Qwen3.5-2B-Q5_K_M.gguf`: Requires llama.cpp build b8233+ (Gated DeltaNet architecture); current llama-cpp-python bundles an incompatible earlier build.
- `Qwen3-1.7B-*.gguf` (all 6 Q3 variants: Q4_K_M, Q4_K_M-dpo, Q6_K, Q6_K-dpo, Q8_0, Q8_0-dpo): All eliminated per `Dev_Plan/Mungi_Model_Selection_Report_v1.md` section 2.1 — gibberish output, topic injection, factual errors, and hallucinations.
- `Qwen3.5-2B-DPO.Q4_K_M.gguf`, `Qwen3.5-2B-FT.Q4_K_M.gguf`: Q4_K_M quantization degrades English retention for 2B-class models (per Dev_Plan Report section 2.2).
- `Qwen3.5-2B-DPO.Q8_0.gguf`, `Qwen3.5-2B-FT.Q8_0.gguf`: Dangerous safety misinformation (e.g., "rubbing your body protects from lightning"); also Q8_0 SFT and DPO produced byte-identical outputs, suggesting a training error (per Dev_Plan Report section 2.2 and Appendix B).

Selection rationale source: `Dev_Plan/Mungi_Model_Selection_Report_v1.md` (Version 1.0, 2026-04-05).

## 11. Resolved Issues

- `onnxruntime-gpu` CUDAExecutionProvider: **Resolved** (ADR 0002, 2026-03-13). Root cause was a missing `LD_LIBRARY_PATH` entry. Fix: `mungidev` exports the correct CUDA library paths. Details: `docs/runbooks/2026-03-11-dev-env-setup.md`
