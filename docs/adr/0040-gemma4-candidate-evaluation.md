# ADR 0040: Gemma 4 E2B/E4B Edge Model Candidate Evaluation

- **Status**: Rejected
- **Date**: 2026-04-06
- **Decision makers**: Claude Code PM, maintainer
- **Related**: ADR 0034, ADR 0039

## Context

Current Mungi production model is Qwen3.5-2B-DPO.Q6_K (ADR 0039). Google released Gemma 4 with edge-
optimized E2B (effective 2B, total 5.1B) and E4B (effective 4B, total 8B) variants featuring
MatFormer selective activation, Per-Layer Embeddings (PLE), shared KV cache, 128K context, and native
multimodal (text+audio+vision) support. E2B-it selected as primary candidate — instruction-tuned with
safety alignment, suitable for QLoRA adaptation with frozen vision/audio encoders.

E4B ruled out: Q4_K_M (~4.9 GB) exceeds Jetson 8GB LLM memory budget (4.0-4.5 GB).

## Decision

### D1: Select Gemma 4 E2B-it as sole evaluation candidate

- **Model**: `google/gemma-4-E2B-it` (5.1B total, 2.3B effective)
- **Architecture**: Dense transformer with PLE + sliding/global hybrid attention + shared KV cache
- **Multimodal**: Vision + audio encoders frozen during fine-tuning, stripped at GGUF conversion
- **E2B-it over E2B (base)**: Already instruction-tuned + safety-aligned → LoRA for Korean + Mungi persona only

### D2: QLoRA fine-tuning pipeline (Colab)

- **SFT**: Unsloth FastVisionModel, LoRA r=16, alpha=16, `finetune_vision_layers=False`
- **DPO**: SFT LoRA base, beta=0.1, lr=5e-5, 2 epochs
- **Data**: batch_deduped.jsonl (7,733 samples, QA-supplemented) + dpo_pairs.jsonl (465 pairs)
- **GGUF**: Q4_K_M (3.11 GB), Q5_K_M (3.36 GB), Q6_K (4.50 GB), Q8_0 (5.05 GB)
- **Drive**: `mungi-finetune6/`

### D3: Config-based model switching on Jetson

- `MUNGI_LLM_MODEL_FAMILY` env var: `auto` (default, detects from filename) / `qwen` / `gemma`
- Stop sequences auto-switch: Qwen `<|im_end|>` ↔ Gemma `<end_of_turn>`
- No pipeline.py changes needed — dispatch in llm_runner.py

### D4: GGUF size constraints for Jetson

| Quant | Size | Jetson fit | Priority |
|-------|------|-----------|----------|
| Q4_K_M | 3.11 GB | OK | 1st |
| Q5_K_M | 3.36 GB | OK | 2nd |
| Q6_K | 4.50 GB | tight | 3rd |
| Q8_0 | 5.05 GB | OOM risk | ref only |

## Consequences

Positive: Potential upgrade path with 128K context, multimodal-ready architecture, MatFormer
efficiency. Config-based switching enables safe A/B testing without code changes.

Risk: Gemma 4 has no Korean-specific fine-tuning — Korean quality (반말, 자연스러움) may be inferior
to QLoRA-tuned Qwen3.5-2B. This is the primary evaluation hypothesis.

Risk: Vision/audio encoder stripping at GGUF conversion is Unsloth-dependent — verify output contains
text decoder only.

## Update Log

- 2026-04-06: ADR created (Proposed)
- 2026-04-06: Status → In Progress. E2B-it selected, E4B ruled out. QLoRA pipeline designed.
  Config-based model switching implemented (commit eebc365). Safety QA Phase 1 applied (8e00ade).
  Training data QA-supplemented (+23 SFT, +14 DPO). Colab notebooks ready (SFT + DPO).
- 2026-04-06: Qwen v6 retrain plan + notebooks created. Persona updated (encyclopedia-smart,
  bilingual mode, 5-10 age range). Simple Wiki RAG hybrid plan approved.
- 2026-04-06: Bilingual RAG index built (60,989 vectors, KO 53% + EN 47%). Language-aware
  retriever implemented (Codex, 6/6 tests). Gap-filling batch submitted (4,366 chunks, ~$2.95).
  Gemma 4 SFT completed on Colab. DPO notebook debugged (3 iterations: sloth typo,
  ClippableLinear, trl mergekit bug). Final DPO notebook verified (Codex + PM 3-round chain).
- 2026-04-06 (night): Gemma 4 DPO Colab 실행 시 3가지 추가 에러 발견 → `_fixed` 노트북에서 해결:
  (1) trl 0.24.0 `is_vision_model` read-only property — `model.config.model_type` 임시 교체 (`"gemma4"`→`"gemma"`, try/finally),
  (2) FastVisionModel processor vs text tokenizer 불일치 — `getattr(tokenizer, "tokenizer", tokenizer)` 분리,
  (3) trl 추가 미등록 모듈 — `llm_blender`, `weave` 더미 등록.
  Gap-filling 번역 배치 재제출: 모델 ID 오류(`claude-sonnet-4-6-20250514`→`claude-sonnet-4-6`) 수정 + Sea_of_Japan→the_East_Sea 정제. 새 배치 `msgbatch_017mGexTg9FQ3EoHtxwQYRRY` (3,989건).
- 2026-04-07: **Status → Rejected (Jetson 8GB 배포 불가)**. Phase 4 실행 결과:
  - llama-cpp-python 0.3.20 CUDA 소스 빌드 완료. Qwen3.5 회귀 테스트 Pass (18.39 tok/s).
  - Gemma 4 Q6_K (3.6 GB) 로드 성공, 하지만 **0.28 tok/s** (실사용 불가).
  - 근본 원인: PLE 텐서 `per_layer_token_embd.weight` (q6_K, 1,838 MB)가 CUDA_Host 미지원으로 CPU 강제 배치. 모델의 50%가 느린 CPU 경로.
  - 공식 ggml-org Q8_0 (4.63 GB): q8_1 타입이나 5 GB 전체가 Jetson 8GB에서 OOM.
  - NVIDIA 공식 컨테이너 (`ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin`): 동일 OOM.
  - **결론: Gemma 4 E2B는 Jetson Orin Nano 8GB에서 실행 불가. Qwen3.5-2B-DPO.Q6_K 유지.**

## Related ADRs

- ADR 0034: Qwen3.5-2B feasibility evaluation
- ADR 0039: Production model cleanup and selection override

## References

- [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- [Gemma 4 Fine-tuning Guide](https://unsloth.ai/docs/models/gemma-4/train)
- [Gemma 4 HuggingFace Blog](https://huggingface.co/blog/gemma4)
- `docs/archived/dev-plan/Gemma4-Candidate-Evaluation-Plan.md`
