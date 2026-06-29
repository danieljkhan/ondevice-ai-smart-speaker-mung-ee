# ADR 0054: Gemma 4 Extended Pilot — Five-Option Audio Path Evaluation Strategy

- **Status**: Proposed
- **Date**: 2026-04-13

## Context

ADR 0048 established Gemma 4 pilot strategy but scoped only text-LLM comparison. Two questions arose
during Jetson preparation on 2026-04-13: (1) how to integrate TTS with the Gemma Docker path, (2)
whether Gemma's native ASR capability can eliminate Sherpa SenseVoice STT. Research confirmed Gemma
4 E2B/E4B has official audio input (30-sec limit, 40ms frame, 16kHz log-mel), but llama.cpp audio
support is experimental with community-reported failures (issue #21334 users cannot enable audio
even with mmproj flag). HuggingFace transformers path works officially but requires bitsandbytes NF4
+ int4 KV cache quantization + vision encoder skip to fit in 8 GB (~7.1 GB peak). ONNX Runtime Q4
offers smallest footprint (~6.5 GB peak) but onnxruntime-genai lacks Gemma 4 support (issue #2062,
PLE/variable head/KV sharing blockers) and CUDA EP has tensor core bug on Orin Nano (issue #24085,
7-8x slower than TensorRT). Silero VAD remains required in all paths due to Gemma's 30-second hard
limit on audio input.

## Decision

Adopt a staged evaluation approach for Gemma 4 audio pipeline on Jetson Orin Nano 8GB. Primary path:
Option A (text-only llama.cpp Docker) as safe baseline, immediately executable. Quick test: Option B
(llama.cpp audio with mmproj) — low-cost attempt before committing complexity. Conditional fallback:
Option D (HuggingFace transformers + bitsandbytes NF4 + int4 KV cache + vision encoder skip) if
Option B fails. Deferred: Option C (hybrid B+A fallback) is rejected due to complexity without
benefit; Option E (ONNX Runtime Q4) is deferred to Phase 7+ pending onnxruntime-genai Gemma 4
support (issue #2062) and CUDA EP tensor core fix. TTS integration is orthogonal — Supertonic TTS 2
(primary) + Piper (fallback) pipeline remains unchanged regardless of audio path choice; only the
LLM invocation contract changes (llama-cpp-python direct call -> HTTP POST to localhost:8080 for
Options A/B, or in-process transformers generate() for Option D). VAD (Silero) is retained in all
paths due to Gemma's 30-second audio input limit and turn-taking requirements. This ADR supersedes
the scope of ADR 0048 (which scoped only text comparison) to include full voice-pipeline integration
and multi-option audio evaluation.

## Consequences

Positive: (a) Pipeline is viable regardless of audio path outcome — Option A is the guaranteed
fallback. (b) Memory budget is known to fit on 8 GB Jetson for all active options (A: 7.3 GB, B: 7.5
GB, D: 7.1 GB with aggressive tuning). (c) TTS integration cost is minimized — no change to
Supertonic/Piper code, only LLM invocation contract changes. Negative: (a) Evaluation scope expands
from 3-4 hours to 8-13 hours due to multi-option testing. (b) Option D requires strict adherence to
all three optimization techniques (NF4 double-quant, int4 KV cache, vision encoder skip) — any slip
causes OOM. (c) If Option B succeeds and Option A is discarded, rollback to Option A mid-production
may be non-trivial. Neutral: (a) Option C (hybrid) rejection simplifies future decision tree. (b)
Option E deferral preserves the path without committing engineering budget — monitor upstream fixes.
(c) TTS streaming optimization is deferred to post-launch regardless of audio path.

## Related ADRs

- ADR 0048 (Gemma 4 Pilot — Model First, Infra Second) — superseded by this ADR's expanded scope
- ADR 0047 (JetPack 6.2.2 NVMe SSD Boot) — NVMe migration now complete, prerequisite for this pilot
- ADR 0053 (Voice Runner v2 Architecture) — voice pipeline architecture this ADR builds on

## References

- docs/archived/dev-plan/2026-04-13-Gemma4-TTS-Extended-Plan-v1.md — full implementation plan
- docs/archived/dev-plan/2026-04-09-Gemma4-Installation-Plan.md — original text-only pilot plan
- Gemma 4 E2B model card — https://huggingface.co/google/gemma-4-E2B-it
- llama.cpp audio input discussion — https://github.com/ggml-org/llama.cpp/discussions/21334
- onnxruntime-genai Gemma 4 feature request — https://github.com/microsoft/onnxruntime-genai/issues/2062
- onnxruntime CUDA EP tensor core bug — https://github.com/microsoft/onnxruntime/issues/24085
- bitsandbytes aarch64 wheels — https://github.com/bitsandbytes-foundation/bitsandbytes
- onnx-community Gemma 4 E2B-it ONNX — https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX

## Update (2026-05-13)

The "Supertonic (primary) + Piper (fallback)" pipeline assumption in lines 29 and
41 is superseded by ADR 0088. Supertonic is the sole TTS engine; no in-process
fallback exists.
