# ADR 0016: Unified Memory Stage Unload for CPU STT and TTS

- **Status**: Accepted
- **Date**: 2026-03-19
- **Context**: Jetson unified-memory stage lifecycle after CPU-first STT/TTS adoption

## Context

The Jetson Orin Nano Super uses unified memory shared by CPU and GPU. This
means "CPU-only" execution does not isolate STT or TTS from the LLM's memory
budget. Recent profiling established the current operating envelope:

- STT (SenseVoice int8): `CPU only`, about `+414 MB` RAM / `+493 MB` RSS
- LLM (Qwen3-4B full offload): about `+2133 MB` RAM / `+3309 MB` RSS
- TTS (Supertonic ONNX): `CPU only`, about `+241 MB` RAM / `+261 MB` RSS

After the STT runtime moved from `faster_whisper` to `sherpa_onnx`, the code
still needed to answer a policy question: should CPU STT/TTS stay loaded
because they do not directly allocate CUDA VRAM, or should they still be
treated as transient stages on Jetson?

The answer is that they remain part of the same memory budget. Native heaps,
page cache, and process RSS created by CPU STT/TTS reduce the physical memory
available for `Qwen3-4B-Q4_K_M` full GPU offload. GNOME vs TTY validation and
audio playback follow-up work both confirmed that headroom is tight enough
that stage cleanup must be explicit.

## Decision

Adopt a unified-memory stage-unload policy for the runtime pipeline:

1. **STT remains CPU by default**.
   - `ManagerConfig.stt_device` stays `cpu`.
   - `models/stt_runner.py` remains `sherpa_onnx`-based.

2. **TTS remains CPU-first**.
   - TTS does not compete for CUDA execution providers.
   - TTS still counts against the same unified-memory budget as the LLM.

3. **`ConversationPipeline.run_turn()` treats STT, LLM, and TTS as transient
   per-turn stages**.
   - `load(STT)` -> transcribe -> `unload_stt()`
   - `load(LLM)` -> generate -> `unload_llm()`
   - `load(TTS)` -> synthesize/play -> `unload_tts()`

4. **Direct unload helpers own real resource release**.
   - `unload_stt()` and `unload_tts()` must call model-specific cleanup hooks
     before clearing Python references.
   - The direct unload path must not rely on `_unload_current_gpu()` alone.

5. **LLM remains the priority consumer of Jetson memory headroom**.
   - The orchestration target is not "GPU-only purity".
   - The real target is preserving enough unified memory for stable
     `n_gpu_layers=-1` LLM loading.

6. **Page-cache reclamation remains part of the STT/LLM boundary**.
   - `unload_stt()` and `unload_llm()` continue to drop page cache so that the
     next LLM load can reclaim CUDA-usable memory.

## Consequences

- CPU STT is no longer mistaken for "free" memory-wise on Jetson.
- TTS no longer lingers in memory after playback on the direct-device path.
- `ModelManager` now centralizes explicit unload responsibility for STT/TTS
  rather than only nulling model references.
- Per-turn latency may increase slightly because TTS is reloaded each turn.
- The runtime policy is now aligned with the measured Jetson memory profile and
  with the 4B LLM full-offload objective.

## References

- ADR 0006: Models layer architecture
- ADR 0009: Sequential GPU loading for Jetson 8GB
- ADR 0012: LLM upgrade from Qwen3-1.7B to Qwen3-4B-Q4_K_M
- ADR 0013: Page cache drop for CUDA memory reclamation
- `docs/runbooks/weekly/archive/2026-03-18-prompt-tuning-e2e-report.md`
- `docs/runbooks/weekly/archive/2026-03-19-jetson-audio-tts-worklog.md`
- `docs/runbooks/weekly/archive/2026-03-19-unified-memory-stage-unload-worklog.md`

---

## Update — 2026-04-29

**Effective**: 2026-04-29
**Authority**: `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval)
**Disposition**: SenseVoice memory entry is **historical-only**.

The original memory accounting line "STT (SenseVoice int8): `CPU only`, about `+414 MB` RAM / `+493 MB` RSS" reflects the SenseVoice STT path that was active at ADR authorship. Per ADR 0055 + 2026-04-29 Update, SenseVoice has been retired as a runtime engine; STT is now Qwen3-ASR exclusively. Memory accounting for the active STT engine should be re-measured per the Qwen3-ASR runtime; the SenseVoice line above remains as a historical-record reference only.

The other memory accounting entries in this ADR (LLM, TTS, etc.) are likewise historical points-in-time anchored to the model set active at ADR authorship; consult `CLAUDE.md §3` and `docs/runbooks/baseline-stack-and-models.md` for the current active model set and re-measure as needed.

This Update annotates one entry as historical-only. The original Decision body above remains immutable per the ADR immutability rule.
