# ADR 0024: STT Text Normalization via Alias Mapping

- **Status**: Accepted
- **Date**: 2026-03-25
- **Context**: Correcting STT misrecognition of the product name "뭉이"

## Context

The Sherpa-ONNX SenseVoice STT model frequently misrecognizes "뭉이"
(the product's character name) as phonetically similar Korean words.
Observed variants in live voice testing:

| STT Output | Frequency | Phonetic Distance |
|------------|-----------|-------------------|
| 웅이 | High | 1 initial consonant |
| 문이 | Medium | 1 vowel |
| 멍인 | Medium | 1 consonant + suffix |
| 멍이 | Medium | 1 vowel |
| 무이 | Low | 1 consonant |
| 멍의 | Low | 1 vowel |
| 붕이 | Low | 1 consonant |
| 몽이 | Low | 1 vowel |

This misrecognition causes two downstream problems:
1. The LLM does not recognize the user is addressing "뭉이", leading to
   confused or irrelevant responses
2. Echo detection (ADR 0020) may trigger on garbled name variants,
   producing fallback responses instead of natural conversation

Alternative approaches considered and rejected:
- **Hotword/keyword spotting model**: Requires additional model, memory,
  and tuning effort disproportionate to the single-word problem
- **STT language model hints**: SenseVoice ONNX runtime does not support
  custom vocabulary injection
- **Fine-tuning STT**: Prohibitive data collection and training cost for
  a single proper noun

## Decision

Implement a lightweight regex-based alias mapping as post-processing
between STT output and LLM prompt construction.

### Implementation

`_normalize_stt_text(text)` in `core/pipeline.py` (line 810):

```python
_STT_ALIAS_MAP: dict[str, str] = {
    "웅이": "뭉이",
    "문이": "뭉이",
    "멍인": "뭉이",
    "멍이": "뭉이",
    "무이": "뭉이",
    "멍의": "뭉이",
    "붕이": "뭉이",
    "몽이": "뭉이",
}
```

Simple string replacement applied to the full STT text. Non-destructive:
the original text is logged for debugging before replacement.

### Pipeline Position

```
Microphone → VAD → STT → _normalize_stt_text() → _build_prompt() → LLM
```

The normalization runs after STT transcription and before any prompt
construction or echo detection, ensuring downstream components see the
corrected text.

### Extensibility

The alias map is a plain dictionary. Additional product-specific terms
(e.g., feature names, character names) can be added without architectural
changes.

## Consequences

- Correct "뭉이" recognition for the 8 most common STT variants
- Zero additional memory or model loading cost
- Negligible latency (<0.1 ms string replacement)
- Does not fix novel misrecognitions not in the alias map — but the map
  covers all variants observed in live testing to date
- Risk of false positives is minimal: the alias words ("웅이", "문이" etc.)
  are not common Korean words in child conversation context

## References

- ADR 0006: Models layer architecture (STT runner contract)
- ADR 0020: Anti-echo detection (benefits from normalized input)
- `core/pipeline.py:810`: `_normalize_stt_text()` implementation
- `docs/runbooks/weekly/archive/2026-03-24-live-voice-test-report.md`: STT quality analysis

---

## Update — 2026-04-29

**Effective**: 2026-04-29
**Authority**: `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval)
**Disposition**: alias map retained pending Qwen3-ASR verification.

The original alias rationale was scoped to Sherpa-ONNX SenseVoice misrecognition of the product character name `뭉이`. After STT migration to Qwen3-ASR (per ADR 0055 + 2026-04-29 Update; SenseVoice fallback retired), the alias map remains in place as a pending-verification safeguard. Runtime team to verify whether Qwen3-ASR exhibits the same misrecognition pattern; if it does not, the alias map may be retired in a follow-up ADR.

This Update modifies applicability scope only. The original Decision body above remains immutable per the ADR immutability rule.
