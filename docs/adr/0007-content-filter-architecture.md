# ADR 0007: Child-Safety Content Filter Architecture

- Status: Accepted
- Date: 2026-03-13
- Updated: 2026-03-16 (pipeline integration, naegi pattern fix)
- Decision makers: Daniel (user), project PM
- Related: AGENTS.md Section 1 (Product Vision), ADR 0005 (Jetson CUDA Policy)

## Context

Mungi must filter LLM output before TTS because the product is intended to be a child's first AI
friend and must prefer safety over conversational completeness.

The filter is intentionally lightweight, offline, and CPU-friendly so it can run on Jetson Orin
Nano 8GB without adding cloud dependencies or meaningful GPU memory pressure.

Sprint 3 Day 2 established the baseline two-phase filter. Sprint 3 Day 3 expanded that baseline to
close supervisor findings around missing categories, missing regexes, and missing empty-input
handling.

## Decision

### 1. The filter is a two-phase local pipeline with input/output guards

```text
User input (STT output)
  -> INPUT guard: blocklist + regex -> BLOCK → safe fallback, skip LLM/TTS
LLM output
  -> OUTPUT guard: blocklist + regex -> BLOCK → use filtered text for TTS
  -> TTS
```

The filter runs at two points in the pipeline:

- **Input guard**: after STT, before LLM. Blocks unsafe user input early.
- **Output guard**: after LLM, before TTS. Filters unsafe LLM responses.

`ConversationPipeline` accepts an optional `ContentFilter` via its constructor.
When provided, `run_turn()` applies both guards automatically.

### 2. Severity remains `BLOCK` or `REPLACE`

- `BLOCK`: return the safe fallback response instead of the original text.
- `REPLACE`: mask only the matched term or pattern with `***`.

If any `BLOCK` violation is detected, it takes precedence over all `REPLACE` results.

### 3. The blocklist now covers eight categories

The current categories are:

- `violence`
- `profanity`
- `sexual`
- `self_harm`
- `substance`
- `bullying`
- `gambling`
- `horror`

Sprint 3 Day 3 added:

- `bullying` with `BLOCK`
- `gambling` with `BLOCK`
- `horror` with `REPLACE`

This keeps high-risk content blocked while allowing low-risk scary or rude language to be softened
instead of fully dropping the reply.

### 4. The regex layer now covers 15 named patterns

The current regex rules are:

- `phone_number_kr`
- `phone_number_intl`
- `address_kr`
- `personal_info_request`
- `url_pattern`
- `ai_identity_kr`
- `ai_identity_en`
- `email_pattern`
- `profanity_en_short`
- `korean_ssn`
- `credit_card`
- `bank_account`
- `gambling_en_short`
- `pick_on_phrase`
- `gambling_kr_naegi`

Sprint 3 Day 3 added or updated:

- `korean_ssn`
- `credit_card`
- `bank_account`
- `gambling_en_short`
- `pick_on_phrase` — bullying phrase with `\bpick\s+on\b`
- `gambling_kr_naegi` — Korean betting term with lookbehind-only boundary
- a case-insensitive `url_pattern` via `(?i)`

### 5. Short English terms must use word boundaries

Short English terms remain too risky for naive substring matching. They must stay in the regex
layer with word boundaries to avoid false positives such as:

- `bet` inside unrelated words
- profanity fragments inside normal words

This rule applies to both profanity and gambling short forms.

### 6. Empty-input handling is part of the filter contract

`ContentFilter.filter()` must treat the following as safe no-op input:

- `None`
- empty string
- whitespace-only string

In these cases the filter returns an allowed result immediately and does not load configs only to
reject nothing.

## Consequences

### Positive

- The filter now covers the missing Day 3 safety categories and PII patterns.
- URL matching is case-insensitive.
- `filter(None)` and other empty-input cases no longer crash the pipeline.
- The design remains JSON-driven, so future vocabulary expansion does not require code changes.

### Negative

- Regex-based filtering still has false-positive risk on ambiguous numeric patterns such as
  `bank_account`.
- Korean morphology is still approximated with simple term and regex matching.
- Context is still ignored, so safe educational or fictional references may be softened.

## Follow-up Notes

Sprint 3 Day 3 supervisor warnings — status as of 2026-03-16:

- `bank_account` — **partially resolved**. Pattern now excludes
  phone-number-shaped sequences via negative lookahead.
  Date-like numeric strings remain a known edge case.
- `korean_ssn` — **resolved**. Pattern uses `(?<!\d)` lookbehind
  and `(?!\d)` lookahead for digit-boundary enforcement.
- `pick on` — **resolved**. Moved to regex layer as
  `pick_on_phrase` with `\bpick\s+on\b` word boundaries.
- Korean compound forms (e.g. 내기) — **resolved**. Pattern
  `gambling_kr_naegi` uses lookbehind-only boundary `(?<![가-힣])`.
  The lookahead was removed so that Korean particles (내기를, 내기도,
  내기에서) are correctly detected while compound verbs (해내기,
  참아내기, 이겨내기) remain excluded via the lookbehind.

Remaining edge cases:

- `bank_account` pattern correctly excludes standard date formats
  (2026-03-16) due to digit-count constraints. Edge cases with
  longer digit groups remain theoretically possible but have not
  been observed in practice.
- Korean morphology is still approximated with boundary matching;
  full morphological analysis remains out of scope for the
  lightweight offline filter.

## Related Documents

- `AGENTS.md`
- `safety/content_filter.py`
- `assets/filters/blocklist.json`
- `assets/filters/patterns.json`
- `tests/test_content_filter.py`
- `tests/test_sprint3_day3_filter.py`
- `docs/runbooks/weekly/archive/2026-03-13-sprint3-day2-worklog.md`
- `docs/runbooks/weekly/archive/2026-03-16-sprint3-day3-worklog.md`
- `docs/adr/0009-sequential-gpu-loading.md`
- `core/pipeline.py` (content filter pipeline integration)
