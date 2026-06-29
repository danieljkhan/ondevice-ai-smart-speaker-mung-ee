# ADR 0020: Anti-Echo and Parrot Detection Layer

- **Status**: Accepted
- **Date**: 2026-03-25
- **Context**: Preventing LLM echo/parrot responses in child conversation

## Context

During E2E testing, Qwen3-4B frequently echoed the user's input back as a
question or minimal reformulation instead of generating a substantive response.
This "parrot" behavior is especially problematic in child conversation:

1. **STT garble echo**: When STT produces garbled text (e.g., "웅이 뭐해"),
   the LLM sometimes repeats it verbatim, confusing the child.
2. **Question reformulation**: The LLM rephrases the child's statement as a
   question ("공룡 좋아해?" → "공룡 좋아하는 거야?") without adding content.
3. **Minimal substitution**: The LLM changes 1–2 words but preserves the
   input structure, providing no new information.

Observed echo rate: ~6–12% of turns in 60-round E2E tests. Prompt-level
mitigation alone was insufficient — the 4B model's instruction following
is not reliable enough to prevent echo through prompting only.

## Decision

Implement a multi-layer anti-echo system combining prompt-level rules,
runtime detection, and fallback responses.

### Layer 1: System Prompt ANTI-ECHO RULE

Added to `core/pipeline.py` system prompt (lines 140–144):

```
ANTI-ECHO RULE:
- NEVER repeat the user's input back as a question.
- NEVER echo garbled STT text. Interpret the user's intent and respond substantively.
- Every response must contain NEW information, opinion, or emotion — not a restatement.
- If you cannot understand the input, say "뭐라고? 다시 말해줘!" instead of echoing.
```

### Layer 2: Runtime Echo Detection

`detect_echo(user_text, response_text)` in `models/llm_runner.py` (line 182):

Algorithm:
1. Strip punctuation and whitespace from both texts
2. Compute character overlap ratio: `overlap / len(clean_user)`
3. Return `True` (echo detected) if ALL conditions met:
   - Overlap ratio > 0.8 (80% character match)
   - Response length < user text length × 1.5
   - User text fully contained in response OR vice versa

### Layer 3: Fallback Response

When echo is detected in `core/pipeline.py._respond_to_text()` (line 544):

```python
if detect_echo(user_text, response_text):
    response_text = ECHO_FALLBACK  # "뭉이가 잘 못 알아들었어. 다시 말해줘!"
```

The fallback is a natural child-friendly response that acknowledges
misunderstanding rather than echoing garbled input.

## Consequences

- Echo rate reduced from baseline ~12% to ~6% in 2026-03-25 E2E test
- False positive risk: legitimate short responses that overlap with input
  may trigger fallback. Mitigated by the 1.5× length ratio and containment
  check requirements.
- Minimal performance impact (<1 ms string comparison per turn)
- QLoRA v4 data generation plan includes Category B (500 items) specifically
  for parrot/echo prevention training

## References

- ADR 0014: Qwen3 chat template (system prompt structure)
- ADR 0015: Response sanitization (echo detection runs before sanitization)
- ADR 0025: Output validator pipeline extension (honorific repair after echo check)
- `models/llm_runner.py:182`: `detect_echo()` implementation
- `core/pipeline.py:544`: Echo fallback integration point
- `docs/runbooks/weekly/archive/2026-03-25-e2e-text-tts-60round-summary.md`
