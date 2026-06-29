# ADR 0012: LLM Upgrade from Qwen3-1.7B to Qwen3-4B-Q4_K_M

- **Status**: Accepted
- **Date**: 2026-03-18
- **Context**: LLM model selection for child conversation quality
- **Supersedes**: Partial — updates model references in ADR 0002, ADR 0009

## Context

The Mungi product vision ("the safest first AI friend for children") requires
natural, empathetic Korean dialogue. The initial Qwen3-1.7B-Q8_0 (1.8 GB) was
functional but produced shallow, repetitive responses lacking contextual
understanding — unsuitable for sustained child conversation.

The Jetson Orin Nano Super has 7.4 GB usable unified memory. Any replacement
model must fit within GPU memory alongside the sequential loading pipeline
(VAD → STT → LLM → TTS).

## Decision

Adopt **Qwen3-4B-Q4_K_M** (2.5 GB) as the active LLM model.

### Models evaluated

| Model | Size | GPU Full Offload | tok/s | Verdict |
|-------|------|-----------------|-------|---------|
| Qwen3-1.7B-Q8_0 | 1.8 GB | Yes | 17.4 | Low quality |
| Qwen3-4B-Q4_K_M | 2.5 GB | Yes | 11.4 | **Selected** |
| Qwen3-8B-Q4_K_M | 4.7 GB | No (CUDA OOM) | 2.6 (CPU) | Rejected |

### Configuration

- `n_gpu_layers=-1` (full GPU offload, 36 layers)
- `n_ctx=2048`
- `flash_attn=True` (Flash Attention enabled)
- Qwen3 chat template (`<|im_start|>/<|im_end|>`)
- Non-thinking mode (`/no_think`) for real-time dialogue
- Sampling: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.2

### Retired models

- `Qwen3-1.7B-Q8_0.gguf`: Retained on Jetson as fallback.
- `Qwen3-8B-Q4_K_M.gguf`: Deleted. Too large for Jetson 8 GB unified memory.
- `Qwen3.5-2B-Q5_K_M.gguf`: Deleted. Requires llama.cpp build b8233+ (Gated DeltaNet architecture); unavailable in any llama-cpp-python release.
  - **2026-04-02 UPDATE**: llama-cpp-python 0.3.17 (b8475) 업그레이드 완료로 Qwen3.5 로딩 가능. → ADR 0034 참조.

## Consequences

- Response quality significantly improved (contextual understanding, empathy).
- Generation speed reduced from 17.4 to 11.4 tok/s (35% slower), acceptable
  for 2-3 sentence child dialogue (~3 s per turn).
- `_EXCLUDED_GGUF` set in `models/llm_runner.py` prevents auto-discovery of
  retired models.
- CLAUDE.md section 3 updated to reflect active model change.

## References

- ADR 0002 (Phase 0 baseline)
- ADR 0009 (sequential GPU loading)
- ADR 0016 (stage unload keeps unified-memory headroom available for 4B LLM offload)
- ADR 0013 (page cache drop — enables full GPU offload for 4B model)
