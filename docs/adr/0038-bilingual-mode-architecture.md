# ADR 0038: Bilingual Mode Architecture — Korean/English Language Routing

- **Status**: Accepted
- **Date**: 2026-04-05
- **Decision makers**: Claude Code PM, maintainer

## Context

Mungi targets children under 10 as an AI friend. The initial design was
Korean-only with aggressive English word stripping in `sanitize_response()`.
The `Mungi_Model_Selection_Report_v1.md` decision (2026-04-05) established
bilingual support as a requirement alongside the production model switch to
Qwen3.5-2B-DPO Q6_K. The selection override (quantitative runbook → qualitative
report) was subsequently formalized in ADR 0039.

The key challenge: existing sanitization rules (`re.sub(r"[a-zA-Z]{2,}", "")`)
destroyed English responses entirely (e.g., "Hello! How are you?" → "! ? ?").

## Decision

Implement bilingual mode via **code-level language routing** without
re-training the model. Qwen3.5-2B's base English capability is sufficient
for child-level English conversation.

### Architecture

1. **Language Detection** (`core/language.py`):
   - Unicode range check for Hangul Syllables (U+AC00-U+D7AF)
   - Any Korean character present → "ko", otherwise → "en"
   - Deterministic, zero-latency, no model required

2. **Prompt Routing** (`core/pipeline.py`):
   - `bilingual_mode=True` in `PipelineConfig`
   - Korean input → Korean system prompt (existing, with English prohibition)
   - English input → English system prompt (`child_safe_system_en.txt`)
   - Mixed input (any Korean) → Korean system prompt

3. **Sanitization** (`models/llm_runner.py`):
   - `sanitize_response(text, language="ko")` — new `language` parameter
   - Korean mode: unchanged (English word removal, allowed chars, honorific repair)
   - English mode: skip Korean-only rules, keep garbage detection active
   - Default "ko" preserves backward compatibility

4. **E2E Testing** (`scripts/e2e_bilingual_test.py`):
   - Configurable Korean/English ratio via `--ko-ratio`
   - Separate topic pools per language with independent cursors
   - Per-language metrics in summary output

### What we did NOT do

- **Did not re-train or fine-tune** the model for English
- **Did not modify** the Korean system prompt
- **Did not add** a separate English model
- **Did not use** an external language detection library

## Consequences

### Positive

- Zero additional memory cost (same model, different prompts)
- Zero additional latency (Unicode check is O(n) on input text)
- Full backward compatibility (`bilingual_mode=False` → existing behavior)
- English quality leverages Qwen3.5-2B's pre-trained English capability

### Negative

- English quality limited by base model (not fine-tuned for English child conversation)
- Language detection is binary (Korean/English only, no third languages)
- Mixed-language responses may occur if model doesn't fully follow prompt

### Risks

- English child safety relies on base model alignment + English system prompt
  (not validated by Korean safety filter training data)
- TTS quality for English depends on Supertonic TTS English support

## Related

- ADR 0034: Qwen3.5-2B Feasibility Evaluation
- ADR 0036: DPO Evaluation Q4/Q6
- ADR 0039: Production Model Cleanup and Selection Override (formalizes the model choice this ADR depends on)
- `Dev_Plan/Mungi_Model_Selection_Report_v1.md`: Production model decision
- `assets/prompts/child_safe_system_en.txt`: English system prompt
