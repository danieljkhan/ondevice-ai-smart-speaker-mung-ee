# ADR 0028: Switch LLM from Qwen3-4B base to Qwen3-1.7B fine-tuned

- **Status**: Accepted
- **Date**: 2026-03-27

## Context

- Qwen3-4B-Q4_K_M (2.5GB) causes memory pressure on Jetson 8GB, especially with RAG active.
- QLoRA fine-tuning completed with 25,007 training samples on Qwen3-1.7B.
- GGUF Q4_K_M output: `qwen3-1.7b.Q4_K_M.gguf` (1.1GB).
- 1.7B model is 42% the size of 4B, freeing ~1.4GB GPU memory.
- Fine-tuning compensates for quality gap: persona, speech style, knowledge boundary all learned from data.

## Decision

- Replace `Qwen3-4B-Q4_K_M.gguf` with `qwen3-1.7b.Q4_K_M.gguf` as the active LLM.
- Restore sampling parameters closer to Qwen3 official recommendations:
  - `temperature`: 0.2 -> 0.7 (FT model has learned behavior, deterministic not needed)
  - `top_p`: 0.7 -> 0.8 (official recommendation)
  - `min_p`: 0.05 -> 0.0 (official recommendation)
  - `presence_penalty`: 0.8 -> 1.2 (compromise: official 1.5 causes language mixing)
  - `repeat_penalty`: 1.3 -> 1.0 (official: disabled; FT data handles repetition)
  - `max_tokens`: 128 -> 80 (1.7B optimal at shorter responses; avg 21.9 tok/turn)
- Keep `n_ctx=2048`, `flash_attn=True`, `top_k=20`.

## Alternatives Considered

- Keep 4B with ONNX: Achieves full offload but model quality unchanged, no fine-tuning benefit.
- Keep 4B with fine-tuning: QLoRA on 4B requires more VRAM than available on Jetson for deployment.
- Use 1.7B without fine-tuning: Quality insufficient for child conversation (ADR 0012 rejected this).

## Consequences

- Expected GPU memory savings: ~1.4GB (2.5GB -> 1.1GB).
- Expected full GPU offload with RAG active: high confidence.
- Expected LLM latency improvement: ~2x faster (fewer parameters).
- Risk: Fine-tuned quality may differ from base 4B; validated by E2E testing.
- `Qwen3-4B-Q4_K_M.gguf` retained as fallback in `/opt/mungi/ai_models/`.
