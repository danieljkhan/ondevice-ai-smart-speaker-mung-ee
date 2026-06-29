# ADR 0070 — STT non-target-script drift guard (Hanzi / kana rejection at pipeline boundary)

- Status: Accepted
- Date: 2026-04-22
- Decision owner: Product orchestrator + Claude Code
- Related: `core/pipeline.py` (`_contains_non_target_script`, `_NON_TARGET_SCRIPT_RANGES`, `TurnResult.stt_script_drift_detected`), Session 7 continuation real-voice test (2026-04-22 ~02:50-03:10 KST, session `demo_2026-04-22_02-41-37` + `demo_2026-04-22_03-08-07`)

## Context

During the Session 7 continuation real-voice test on Jetson, a reproducible Qwen3-ASR multilingual-detect drift produced Chinese characters in `user_text` for English-prefix input:

```
Child (actual): "Hi, my name is Jongkyung. What's your name?"
STT output:     嗨，我的名字是钟景。What's your name?
```

Breakdown of the drift:

- `"Hi"` → `嗨` (Chinese transliteration)
- `"my name is"` → `我的名字是` (Chinese translation)
- `"Jongkyung"` → `钟景` (Chinese Hanzi rendering)
- `"What's your name?"` → preserved as English

Sherpa-ONNX `OfflineRecognizer.from_qwen3_asr` does NOT expose any `language` parameter or language-restriction beam API (verified via `help()` on Jetson in Session 6 + 7). The `language='ko'` hint that Mungi passes is explicitly ignored per log line: `language hint=ko ignored - model is multilingual auto-detect`. Qwen3-ASR has a Chinese-biased multilingual prior baked into its training, so acoustically ambiguous or short English-prefix utterances deterministically drift into Chinese.

This is a product-level violation:

- **CLAUDE.md §12** language policy: "internal English, user-facing Korean". Chinese is not a supported surface.
- **Product rule** (user-declared 2026-04-22): "뭉이는 한국어와 영어만 사용한다" — Mungi uses Korean and English only.
- **Log hygiene**: `conversation.jsonl` with Chinese `user_text` pollutes parental review.
- **History context risk**: Chinese tokens in multi-turn conversation history can destabilize LLM output on subsequent turns.

Session 6 already introduced two pipeline-boundary guards for the hotword prompt-echo pattern (`_is_hotword_hallucination` L1 + hotwords reduction L2). This ADR extends the same pattern to non-target scripts (Hanzi, Hiragana, Katakana, CJK symbols).

## Decision

Add a pipeline-layer guard that rejects any STT transcription containing characters outside the target script set (Hangul + Latin + basic punctuation). When triggered, skip the LLM call, do not append to conversation history, synthesize a language-appropriate re-prompt via the existing TTS path, and flag the turn in telemetry.

### Layer 1 — `_contains_non_target_script` helper

Module-level helper in `core/pipeline.py`:

```python
_NON_TARGET_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs (core Hanzi)
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2B73F),  # CJK Extension C
    (0x2B740, 0x2B81F),  # CJK Extension D
)

def _contains_non_target_script(text: str) -> bool:
    """Return True if text contains any Hanzi, kana, or CJK symbol."""
```

Any codepoint in these ranges triggers True. Hangul (U+AC00–U+D7AF) and Latin remain allowed.

### Layer 2 — Turn-loop guard integration

Immediately after the hotword hallucination guard (Session 6 implementation), invoke the script-drift guard:

1. Log a WARNING with the offending `user_text` (truncated).
2. Determine re-prompt language from `self._last_detected_language`:
   - `"en"` → `"Hmm? Say that again in Korean or English!"`
   - else → `"응? 한국어로 다시 말해줄래?"`
3. Synthesize + play the re-prompt via the regular TTS path. `llm_s = 0.0`, `llm_tokens = 0`.
4. Do NOT append to conversation history.
5. Set `TurnResult.stt_script_drift_detected = True`. Include this flag in the per-turn `demo_live.jsonl` record and `TurnMetrics.to_dict()` output for telemetry consistency with the existing `hotword_hallucination_detected` field.
6. Return early from the turn.

Guard order in the turn flow (first match wins):

```
1. empty-text check
2. _is_hotword_hallucination (Session 6, L1)
3. _contains_non_target_script (this ADR, L1)
4. regular LLM/safety-template path
```

### Layer 3 — Korean system-prompt reinforcement

In `core/pipeline.py` `llm_system_prompt` default, after the existing line `"Your final output MUST be ONLY in Korean. No English words in output."`, append:

```
"- NEVER use Chinese characters (Hanzi like 汉, 字, 猫) or Japanese kana (あ, ア). Output MUST be Korean Hangul only."
```

This is defense-in-depth: even if the LLM receives Chinese user_text (e.g. guard bypassed by future code change), the system prompt actively forbids Chinese output. The English prompt (`child_safe_system_en.txt`) already has this rule; the Korean side is brought to parity.

### What this ADR does NOT change

- Qwen3-ASR model or Sherpa-ONNX invocation: unchanged. The drift at the model level continues.
- Audio preprocessing (`_downmix_to_mono`, `_finite_float`, `_write_temp_wav`): ADR 0069 intact.
- Hotword hallucination guard (Session 6): unchanged.
- Bilingual dispatch logic: unchanged.
- Persona (`persona.md` ADR 0068): unchanged.
- Safety templates under `safety/`: unchanged.
- English system prompt (`child_safe_system_en.txt`): unchanged (Chinese ban rule was already present).
- Default backend (`qwen3_legacy`) invariant: unchanged.

## Rationale

1. **Application-layer fix is the only option without upstream changes**: Sherpa-ONNX Qwen3-ASR wrapper does not expose language constraints. Patching upstream is out of scope. Defensive rejection at pipeline boundary is predictable, low-risk, and test-covered.
2. **Same pattern as existing hotword guard** (Session 6): reuse of the established `TurnResult.*_detected` flag + re-prompt TTS convention keeps telemetry and UX consistent. Minimum new surface area.
3. **Defense-in-depth with L3**: the Korean system prompt now actively forbids Chinese. If the guard is bypassed by future code changes (e.g. a feature branch that skips the early-return), the LLM still has an explicit rule to ignore Chinese tokens.
4. **Empirical validation already in hand**: session `2026-04-22_03-08-07` 5-turn test showed:
   - T1 (Korean): `script_drift=False`, normal LLM response.
   - T2 (Chinese-heavy): `script_drift=True`, re-prompt in Korean.
   - T3 (English): `script_drift=False`, normal LLM response.
   - T4, T5 (mixed EN + Chinese characters): `script_drift=True`, re-prompt in English (language context from T3).
   - Zero false positives, zero missed drifts.
5. **Reject-over-translate is the safe choice**: a Chinese-to-Korean/English machine translation layer would cascade errors (Qwen3-ASR Chinese output is already partially broken; translating further would produce hallucinated content). Rejecting + re-prompting preserves accuracy.

## Alternatives considered

- **(A) L1 only (guard, no prompt rule)**: minimum change. Rejected — L3 prompt rule provides cheap defense-in-depth and aligns Korean prompt with English prompt that already has the rule.
- **(B) L3 only (prompt rule, no guard)**: relies solely on LLM to ignore Chinese in user_text. Rejected — LLM receives Chinese and may echo it, and conversation history is polluted regardless.
- **(C) Chinese-to-Korean machine translation**: translate STT output on drift. Rejected — error cascade, latency, and the broken Chinese output is often not recoverable.
- **(D) Force `language=ko` in STT API**: not possible (API doesn't support it).
- **(E) Switch to Whisper.cpp with `language` parameter**: upstream language forcing at STT model level. **Deferred to a dedicated A/B benchmark branch** (`Whisper` branch created in the same commit chain). If empirically superior on Korean child speech and English bilingual, Whisper.cpp adoption will supersede this ADR's guard (the guard would still be a safety net for any residual drift).

## Consequences

### Positive

- `conversation.jsonl` turns with Chinese `user_text` are explicitly flagged (`stt_script_drift_detected=True`), enabling parental review filtering and UX analytics.
- LLM is never asked to reason about Chinese tokens. Zero LLM cost + zero history pollution on drift turns.
- Korean system prompt is now at parity with English prompt for Chinese-ban rule.
- Re-prompt language follows conversational language context — natural UX.

### Neutral

- TTFS on drift turns is ~4 s (TTS re-prompt only), much faster than normal LLM turns (~18 s) — almost a UX bonus, at the cost of asking the user to repeat.

### Negative / risks

- Users saying legitimate Chinese words (e.g. a Korean child learning Mandarin) will be rejected. Not a current Mungi use case; re-evaluate only if product scope expands.
- Qwen3-ASR still produces Chinese in `user_text`; the guard cannot fix the STT output itself. Full root-cause fix requires STT model change (tracked as `Whisper` branch).
- A legitimate utterance that sounds acoustically like a Chinese prefix (e.g. user saying only `"Hi"`) may drift and trigger the guard even though the user spoke a supported language. UX cost: re-prompt. Mitigation: STT model replacement.

## Validation

- Codex task `stt-non-target-script-guard` (2026-04-22, 485.9 s, PASS).
- `ruff check .` + `ruff format --check .` + `mypy core/ models/ safety/ hardware/ scripts/ parental/` + `pytest tests/ --ignore=tests/integration_jetson -v --tb=short` → 965 passed / 3 skipped / 79.31 % coverage.
- Jetson real-voice 5-turn validation (session `2026-04-22_03-08-07`):
  - T2 `嗨，梦伊，我的名字是钟琼。What's your name?` → `script_drift=True`, `"응? 한국어로 다시 말해줄래?"` re-prompt.
  - T4 `嗨，萌萌。My name is Jong Kyung. What's your name?` → `script_drift=True`, English re-prompt.
  - T5 `嗨，梦伊。My name is 종경。What's your name?` → `script_drift=True`, English re-prompt.
  - T1 Korean, T3 English → `script_drift=False`, normal LLM.
- Polish loop: 2 cycles × 10 iterations, 0 fixes, terminated.

## References

- `core/pipeline.py::_contains_non_target_script`, `_NON_TARGET_SCRIPT_RANGES`
- `core/pipeline.py::TurnResult.stt_script_drift_detected`
- `tests/test_pipeline.py::TestNonTargetScriptGuard`
- Session 7 continuation worklog addendum: `docs/runbooks/weekly/archive/2026-04-22-daily-worklog.md`
- Related prior ADRs: 0068 (persona redefinition), 0069 (NaN/Inf audio sanitization)
- Jetson JSONL artifacts: `/var/lib/mungi/conversations/2026-04-22_02-41-37/` (drift reproduction pre-fix), `/var/lib/mungi/conversations/2026-04-22_03-08-07/` (post-fix 5-turn validation)
- Follow-up branch: `Whisper` (large-v3-turbo-q5_0 A/B bench to address root cause)
