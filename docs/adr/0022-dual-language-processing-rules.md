# ADR 0022: Dual-Language Processing Rules for Qwen3-4B

- **Status**: Accepted
- **Date**: 2026-03-25
- **Context**: Optimizing instruction-following quality for a multilingual 4B model

## Context

Qwen3-4B-Q4_K_M is a multilingual model with stronger English
instruction-following capability than Korean. When given complex behavioral
rules entirely in Korean, the model exhibited:

1. **Rule drift**: Ignoring speech-level constraints (존댓말/반말 mixing)
2. **Response fixation**: Repeating the same phrase pattern (e.g., "와 진짜?
   대박!" appearing in 66.7% of turns in one test)
3. **Weak constraint adherence**: Difficulty following multiple simultaneous
   Korean-language rules

The hypothesis: if the model reasons internally in English (its stronger
language) and only produces Korean output, instruction-following improves
while output language remains correct.

## Decision

Add a LANGUAGE PROCESSING RULES section at the top of the system prompt
in `core/pipeline.py` (lines 121–124), marked as highest priority:

```
LANGUAGE PROCESSING RULES (highest priority):
- The user's input is Korean speech transcribed by STT.
- Internally, understand and reason about the input in English.
- Your final output MUST be ONLY in Korean. No English words in output.
```

### Placement

The rules are positioned at the very top of the system prompt, before
SPEECH RULES and CRITICAL RULES, to ensure the model processes them first.

### Safety Net

`sanitize_response()` (ADR 0015/0025) removes any English word sequences
(2+ letters) that leak into the output, providing a hard filter even if
the model fails to comply with the Korean-only output rule.

## Consequences

### Observed improvements
- Response fixation ("와 진짜? 대박!") dropped from 66.7% to 0%
- More varied and contextually appropriate responses

### Observed side effects
- 존댓말 rate increased from 27.9% to 37.1% in one test — the English
  internal reasoning may interfere with Korean speech-level grammar
  decisions. Mitigated by `repair_honorifics()` (ADR 0025).
- Effect isolation is incomplete: the rule was deployed alongside other
  prompt changes. A/B test with the rule removed is recommended to
  isolate its specific impact.

### Risks
- Model behavior with this rule is empirical, not guaranteed — future
  model updates may respond differently to the dual-language instruction
- The rule increases prompt length by ~3 lines, negligible impact on
  token budget

## References

- ADR 0014: Qwen3 chat template (system prompt structure and speech policy)
- ADR 0015: Response sanitization (English word removal as safety net)
- ADR 0025: Output validator pipeline (honorific repair for side effects)
- `core/pipeline.py:118-124`: System prompt LANGUAGE PROCESSING RULES
- `docs/runbooks/weekly/archive/2026-03-23-daily-worklog.md`: Initial deployment
