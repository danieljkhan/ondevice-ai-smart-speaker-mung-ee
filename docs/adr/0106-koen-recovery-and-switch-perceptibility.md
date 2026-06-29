# ADR 0106: KO-EN Language Switch — Recovery Path and Switch Perceptibility

## Status

Proposed

(Will move to Accepted after implementation, verification chain, the G2 critic re-evaluation ≥8/10, and merge. Plan: `Dev_Plan/2026-06-08-koen-ux-improvement-plan.md` v4; review/discussion: `Dev_Plan/2026-06-08-koen-ux-plan-discussion-r1.md`; source evaluation: `docs/runbooks/weekly/2026-06-08-koen-ux-3critic-eval.md`.)

## Context

The KO-EN language switch (`한영전환`, shipped in PR #171) was evaluated by three independent critic agents against its child-UX acceptance bar (clause 7, ≥8/10). The mean score was ~4.2/10. Three convergent, PM-verified defects drove the failure:

1. **The switch was imperceptible.** The corner badge was static (no motion at the switch instant), the planned audio cue function `pick_language_switch()` was never wired and had no asset, and the badge flip preceded the EXCITED face + voice by a multi-second TTS-synthesis gap. The three "redundant confirmations" collapsed to roughly one ambiguous signal.
2. **Recovery could trap a pre-reading child in English.** The only way back to Korean was an instruction spoken **in English** telling the child to say a Korean sentence; the `en_to_ko` matcher was strict whole-turn-anchored (rejecting near-misses); there was no non-voice fallback. A confused 4-7-year-old could be stuck short of a reboot.
3. **The flag art carried no language meaning** to a non-reader and was crude placeholder quality (no anti-aliasing; a factually wrong taegeuk; broken stars).

Defects (2) and the recovery matcher live in the safety-routing layer (`safety/language_switch_router.py` + `assets/filters/language_switch_templates.json`), so this change is safety-policy-scoped and requires an ADR.

## Decision

### 1. Switch perceptibility (W1)
- Add a short, warm, **mono** non-verbal cue (`assets/sounds/feedback/language_switch/switch_01.wav`) played **at the switch instant** via the `language_sink` (`SessionManager.set_language_indicator`), before TTS synthesis, bridging the dead-air gap. Suppressed on cold-wake/boot (only fires on a real language change).
- Add a one-shot **badge transient animation** (scale-pulse) on a real change. The animation + cue are the **dedicated switch signals**, distinct from the generic EXCITED expression.

### 2. Meaningful, legible badges (W2)
- Replace national-flag placeholders with **anti-aliased glyph badges** (`한` / `A`) rendered from a bundled, RFN-de-branded SIL-OFL Pretendard subset (`assets/fonts/`). KO = solid disc, EN = ringed disc → distinguishable in grayscale (colorblind/low-vision). A glyph conveys *language* to a non-reader; a flag does not (and "English ≠ USA" is an i18n anti-pattern).

### 3. Recovery — comprehensible voice path (2B)
- **Confirmation spoken in the comprehensible (source) language.** TTS `language` selects voice only — it does not translate. We therefore (a) add a `confirmation_language` field per direction to the switch templates and `LanguageSwitchMatch`; (b) separate the confirmation *utterance* language from the *session/current* language in the pipeline by adding a `tts_language` parameter to `_return_fixed_tts_response` — `language=target_language` keeps `_current_language`/session correct, while only `_run_tts` uses `tts_language=confirmation_language`; (c) make the KO→EN `confirmation_text` actual Korean. Result: after KO→EN, a Korean-speaking child *hears, in Korean*, how to come back, while the session is genuinely English.
- **FP-safe relaxed recovery corpus.** The `en_to_ko` direction gains additional **whole-turn-anchored** shorter forms (e.g. `한국어`, `한국말 해줘`, `우리말로 해줘`) so a young/accented child is understood. The `ko_to_en` entry direction stays strict; **entry false-positive rate = 0 is preserved**, `fp_guards` intact, no substring matching introduced.

### 4. Recovery — non-voice path (2A')
- A **long-press** (`TouchEvent.press_duration_ms` ≥ 3000 ms) **reverts to Korean when the session language is `en`**, reusing the badge/cue/animation feedback path. When the session is `ko`, long-press keeps the existing parent-mode-request stub. This uses the existing touch contract (no coordinate hit-test) and cannot collide with tap/PTT (≤ 500 ms). It guarantees a child is never trapped even if speech recognition fails.

### 5. Robustness corrections
- **Re-entrant capture guard (F3):** the in-turn switch cue must not reopen the microphone during `RESPONDING`. `_play_audio_with_capture_guard` becomes save/restore — it only resumes/unmutes capture if capture was not already paused before the cue.
- **Missing-badge fallback (E3):** if a badge PNG fails to load, the renderer draws a generated colored-disc fallback (and logs) instead of silently showing nothing.

## Consequences

**Positive**
- A not-looking child now perceives the switch (motion + audio at the instant) and, after switching, *hears in Korean* how to return — plus a deterministic non-voice (long-press) escape. Recovery no longer depends on a pre-reader producing a precise English-prompted Korean phrase.
- Glyph badges convey language and are product-grade (AA, grayscale-distinguishable).
- Entry-direction FP=0 and `fp_guards` are preserved; the relaxation is confined to the recovery direction and stays whole-turn anchored.
- The in-turn cue no longer risks self-capture (re-entrant guard); the visual channel never silently vanishes.

**Negative / risks**
- Long-press in an EN session preempts the (currently stubbed) parent-mode request. A future real parent-mode gesture must be disambiguated from long-press-revert (tracked as follow-up).
- The switch-templates JSON gains a required `confirmation_language` field; existing template tests are migrated.

**Alternatives rejected**
- *Coordinate touch hit-test on the badge* — infeasible: `TouchEvent` exposes no x/y; long-press achieves a non-voice escape without extending the touch contract.
- *Two-segment / mixed-language TTS for a bilingual confirmation* — more complex than synthesizing the whole confirmation in the comprehensible source language.
- *Keeping national flags (real art)* — fixes quality but not the core "flag ≠ language for a non-reader" defect.

## References
- Plan: `Dev_Plan/2026-06-08-koen-ux-improvement-plan.md` (v4)
- Discussion record (PM↔Codex r1-r3): `Dev_Plan/2026-06-08-koen-ux-plan-discussion-r1.md`
- Source evaluation: `docs/runbooks/weekly/2026-06-08-koen-ux-3critic-eval.md`
- Related: ADR 0052 (ALSA mono→stereo plug upmix), ADR 0101 (crisis escalation router), ADR 0103 (parent-disclosure guardrail), PR #171 (original `한영전환`).
