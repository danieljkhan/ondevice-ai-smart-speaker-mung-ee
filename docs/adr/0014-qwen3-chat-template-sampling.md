# ADR 0014: Qwen3 Chat Template and Sampling Configuration

- **Status**: Accepted
- **Date**: 2026-03-18
- **Context**: LLM prompt format and generation quality for child dialogue

## Context

After upgrading to Qwen3-4B-Q4_K_M (ADR 0012), the initial plain-text
prompt format (`시스템: ... 사용자: ... 뭉이:`) produced:

- Chinese/Japanese/Vietnamese character mixing in Korean responses
- Inconsistent speech level (mixing 존댓말 and 반말)
- Weak emotional empathy for child conversation
- Repetitive responses

The Qwen3 model card (huggingface.co/Qwen/Qwen3-8B-GGUF) documents specific
best practices for chat template, sampling parameters, and thinking mode
configuration.

## Decision

### 1. Qwen3 Chat Template

Replaced plain-text prompts with the official `<|im_start|>/<|im_end|>`
chat template format:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_text}<|im_end|>
<|im_start|>assistant
```

Stop sequences changed from `["사용자:", "\n\n"]` to
`["<|im_end|>", "<|im_start|>"]`.

### 2. Sampling Parameters

Based on Qwen3 model card recommendations, with one adjustment:

| Parameter | Qwen3 Recommended | Mungi Setting | Reason |
|-----------|-------------------|---------------|--------|
| temperature | 0.7 | 0.7 | As recommended |
| top_p | 0.8 | 0.8 | As recommended |
| top_k | 20 | 20 | As recommended |
| min_p | 0.0 | 0.0 | As recommended |
| presence_penalty | 1.5 | **1.2** | Model card warns: higher values cause language mixing |

The presence_penalty reduction from 1.5 to 1.2 eliminated Japanese/Chinese
character leakage while maintaining adequate repetition suppression.

Thinking mode parameters (temperature=0.6, top_p=0.95) are available via
`run_generation(enable_thinking=True)` but not used in production.

### 3. Non-Thinking Mode (`/no_think`)

Adopted for production use. System prompt includes `/no_think` directive.

**Rationale**: Thinking mode generates ~140 additional tokens of internal
reasoning, increasing LLM inference from 3 s to 14 s per turn. Children
cannot wait 14 seconds for a response. Quality improvement from thinking
is marginal for simple dialogue and better addressed by prompt engineering.

### 4. Flash Attention

Enabled `flash_attn=True` in `Llama()` constructor. Provides ~20% speed
improvement (10-12 → 12-14 tok/s) with reduced memory footprint. Requires
one warmup inference at service start to compile CUDA kernels.

### 5. System Prompt Design

```
너는 뭉이야. 10살 미만 아이들의 첫 번째 AI 친구야.
반드시 지켜야 할 규칙:
- 존댓말만 써. 절대 반말을 쓰지 마.
- 한국어와 영어만 써. 다른 언어 절대 섞지 마.
- 짧고 쉬운 단어만 써. 대답은 2~3문장으로 짧게 해.
- 아이가 슬프거나 속상하면 먼저 공감해줘.
- 위험하거나 무서운 이야기는 하지 마.
- 아이를 칭찬하고 격려해줘.
/no_think
```

### 6. History Best Practice

Per Qwen3 model card: "historical model output should only include the
final output part and does not need to include the thinking content."
Verified that both `ConversationPipeline` and test scripts apply
`strip_think_tags()` before appending to conversation history.

## Consequences

- 5-turn conversation test: average 3.0 s/turn, zero language mixing,
  consistent 존댓말, improved empathy.
- `core/pipeline.py`: `_build_prompt()` rewritten for chat template;
  `PipelineConfig` updated with new system prompt and stop sequences.
- `models/llm_runner.py`: sampling constants added; `run_generation()`
  accepts `enable_thinking` parameter; `flash_attn=True` in model loader.
- Existing tests updated for new stop sequences and prompt format.

## Update (2026-03-18): Prompt B Tuning

A/B/C comparison test replaced the system prompt in Section 5 with a
shorter, example-driven variant ("Prompt B"). Key changes:

- Concrete speech-ending examples (`~요`, `~해요`, `~까요`) instead of
  abstract "존댓말만 써" rule
- Empathy response templates ("그랬구나, 속상했겠어요")
- Unknown-topic fallback template ("같이 알아볼까요?")
- Reduced from 13 lines to 8 lines — less confusion for 4B model

Results: 존댓말 consistency improved, average turn time 2.9s → 2.2s,
empathy patterns reliably triggered. See commit `61f4f38`.

## Update (2026-03-19): Speech Style Policy Change

존댓말 policy (Section 5, Prompt B Update) was superseded by a 반말-only
policy. The production system prompt (`core/pipeline.py`) now instructs:
"존댓말은 쓰지 말고, 친절하고 따뜻한 반말만 써." All E2E test scripts
(`e2e_60rounds.py`, `test_conversation.py`, `test_e2e_pipeline.py`) are
aligned to this policy. See `core/pipeline.py:89-92` for the current prompt.

## References

- ADR 0012 (LLM upgrade to Qwen3-4B)
- ADR 0015 (response sanitization — post-processing after generation)
- ADR 0021 (token-based adaptive history — context window management for prompt size)
- ADR 0022 (dual-language processing rules — English internal reasoning added to system prompt)
- ADR 0025 (output validator pipeline — honorific repair for speech-level side effects)
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-8B-GGUF
