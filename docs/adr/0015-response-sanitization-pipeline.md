# ADR 0015: LLM Response Sanitization Pipeline

- **Status**: Accepted
- **Date**: 2026-03-18
- **Context**: Child-safety post-processing for LLM output on edge device

## Context

The Qwen3-4B-Q4_K_M model, despite system prompt instructions, occasionally
produces output containing:

- Foreign language fragments (Chinese, Japanese, Arabic)
- Emoji and special symbols
- Residual `</think>` tags from empty think-block prefill
- The word "think" echoed at response start

These artifacts are unacceptable in a product targeting children under 10.
Prompt-level mitigation alone is insufficient for a 4B parameter model.

## Decision

Implement a two-stage post-processing pipeline in `models/llm_runner.py`,
applied in `core/pipeline.py._run_llm()` after generation:

### Stage 1: `strip_think_tags(text)`

1. Remove closed `<think>...</think>` blocks (regex, DOTALL)
2. Remove unclosed `<think>...` blocks (truncated generation)
3. Remove standalone `</think>` closing tags (empty prefill echo)
4. Remove residual "think" word at response start (case-insensitive)

### Stage 2: `sanitize_response(text)`

1. Remove all characters outside the allowed set:
   - Korean Hangul syllables (U+AC00-U+D7A3)
   - Korean Jamo (U+3131-U+3163, U+1100-U+11FF)
   - English letters (a-zA-Z)
   - Digits (0-9)
   - Basic punctuation (.,!?~-:;'"())
   - Whitespace
2. Collapse multiple spaces to single space
3. Return safe fallback response if cleaned text is empty

### Regex pattern

```python
_ALLOWED_CHARS_RE = re.compile(
    r"[^\uAC00-\uD7A3\u3131-\u3163\u1100-\u11FF"
    r"a-zA-Z0-9"
    r"\s.,!?~\-\u2026:;'\"()]"
)
```

### Call chain

```
LLM output → strip_think_tags() → sanitize_response() → content_filter → TTS
```

## Consequences

- Foreign language fragments eliminated from child-facing output
- Emoji and special symbols stripped automatically
- Safe fallback prevents empty responses reaching TTS
- 11 new unit tests covering Korean passthrough, CJK removal, emoji removal,
  edge cases, and pipeline integration
- Minimal performance impact (regex on short strings, <1 ms)

## References

- ADR 0012 (LLM upgrade — 4B model produces more foreign text than 1.7B)
- ADR 0014 (chat template — think-block prefill creates residual tags)
- ADR 0020 (anti-echo detection — echo check runs before sanitization in pipeline)
- ADR 0022 (dual-language processing rules — source of English word leakage)
- ADR 0025 (output validator pipeline extension — extends this ADR with honorific repair, English word removal, and interjection stripping)
