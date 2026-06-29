# ADR 0025: Output Validator Pipeline Extension

- **Status**: Accepted
- **Date**: 2026-03-25
- **Extends**: ADR 0015 (Response Sanitization Pipeline)
- **Context**: Multi-pass output validation beyond character-level sanitization

## Context

ADR 0015 established a two-stage sanitization pipeline (`strip_think_tags` →
`sanitize_response`) focused on removing foreign characters, emoji, and
think-block artifacts. Since then, E2E testing revealed additional output
quality issues that require post-processing:

1. **Honorific leakage**: Despite 반말-only system prompt rules, the model
   produces 존댓말 endings (~27–37% of turns) that are inappropriate for
   the child-friend persona.
2. **English word sequences**: Occasional 2+ letter English words leak
   through, likely from the dual-language processing rules (ADR 0022).
3. **Leading interjections**: Repetitive "우~" interjections at response
   start add no value and sound unnatural via TTS.

These issues require rule-based repair beyond the character-level filtering
in the original `sanitize_response()`.

## Decision

Extend the `sanitize_response()` function in `models/llm_runner.py`
(line 213) with three additional processing stages, while preserving the
existing character-level sanitization.

### Extended Pipeline

```
LLM output
  → strip_think_tags()          [ADR 0015 Stage 1 — unchanged]
  → sanitize_response()         [ADR 0015 Stage 2 — extended below]
      1. Remove non-allowed characters  (original)
      2. Remove English word sequences  (NEW: regex 2+ letter words)
      3. Remove leading interjections   (NEW: "우~" pattern)
      4. Collapse whitespace            (original)
      5. repair_honorifics()            (NEW)
      6. Fallback if empty              (original)
  → content_filter
  → TTS
```

### Stage: English Word Removal

```python
re.sub(r"[a-zA-Z]{2,}", "", cleaned)
```

Removes sequences of 2+ English letters. Single letters (used in Korean
context, e.g., "A형") are preserved.

### Stage: Leading Interjection Removal

```python
_LEADING_INTERJECTION_RE = re.compile(r"^\s*우[!~,.\s]+\s*")
```

Strips repetitive "우!" / "우~" patterns at response start.

### Stage: Honorific Repair

`repair_honorifics(text)` in `models/llm_runner.py` (line 201):

**Approach**: Placeholder-based preservation + systematic repair.

1. **Preserve** specific honorifics that are intentional (greetings/praise):

| Honorific | Placeholder | Rationale |
|-----------|-------------|-----------|
| 안녕하세요 | `__MUNGI_HELLO__` | Standard greeting |
| 잘했어요 | `__MUNGI_PRAISE_GOOD_JOB__` | Praise pattern |
| 대단해요 | `__MUNGI_PRAISE_AMAZING__` | Praise pattern |
| 좋아요 | `__MUNGI_PRAISE_LIKE__` | Praise pattern |

2. **Repair** honorific endings to casual (반말) equivalents:

| Honorific Ending | Casual Replacement |
|------------------|--------------------|
| 해요 | 해 |
| 하세요 | 해 |
| 합니다 | 해 |
| 됩니다 | 돼 |
| 있어요 | 있어 |
| 없어요 | 없어 |
| 거예요 | 거야 |
| 줄게요 | 줄게 |
| 할게요 | 할게 |
| 볼까요 | 볼까 |
| 인가요 | 인 거야 |
| 하나요 | 하는 거야 |

3. **Restore** preserved placeholders to original honorifics.

### Fallback Constants

```python
SAFE_FALLBACK = "안녕하세요, 무슨 이야기를 할까요?"
ECHO_FALLBACK = "뭉이가 잘 못 알아들었어. 다시 말해줘!"
```

## Consequences

- Consistent 반말 speech level in output, maintaining the child-friend persona
- English leakage from dual-language rules (ADR 0022) caught and removed
- Preserved honorifics (greetings/praise) are intentional exceptions, not bugs
- Honorific repair coverage is pattern-based — novel endings not in the table
  will pass through. The table should be expanded as new patterns are observed.
- Total post-processing latency remains <1 ms (all regex operations on short
  strings)

## References

- ADR 0014: Qwen3 chat template (speech policy: 반말-only)
- ADR 0015: Response sanitization (original two-stage pipeline, now extended)
- ADR 0020: Anti-echo detection (echo check runs before sanitization)
- ADR 0022: Dual-language processing rules (source of English leakage)
- `models/llm_runner.py:201`: `repair_honorifics()` implementation
- `models/llm_runner.py:213`: `sanitize_response()` extended pipeline
