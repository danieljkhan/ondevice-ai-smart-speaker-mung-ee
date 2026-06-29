# ADR 0113 — Session Safety Hardening (crisis-router expansion) + Funny English UX fixes

- **Status**: Accepted
- **Date**: 2026-06-20
- **Supersedes / relates to**: ADR 0101 (crisis-disclosure routing), ADR 0072 (dangerous-topic categories), ADR 0111 (prebaked TTS cache — amended here for fixed approved-template responses)

## Context

Live device testing surfaced a batch of child-safety and Funny English (FE) UX defects. The user (parent/champion) raised 13 concrete items in one session:

**Safety (Part A):**
- **A1** — Witnessed inter-parental domestic violence ("아빠가 엄마를 때려") was mis-handled as generic comfort instead of a DV-appropriate, escalating response.
- **A2** — Death questions were blanket-blocked (wrong pedagogy): age-appropriate natural-death education should be allowed, while first-person suicidal ideation must still escalate. A pre-existing gap was found where "나 죽을래" / "나 죽어" did NOT reach the crisis path.
- **A3** — The LLM promised impossible actions ("그림으로 보여줄게") — Mungi is voice-only.
- **A4** — Child sexual abuse / grooming / exploitation disclosures need reliable recognition and the correct escalation target (family abuser → trusted-adult-not-abuser; stranger → parent).
- **A5** — Emergency numbers were spoken as "백십이/백십구"; should read digit-by-digit ("일일이/일일구").
- **A6** — A child swearing at Mungi got no calm coaching.
- **A7** — Scripted self-introduction (child) and product-introduction (adult) with pre-baked audio.

**FE UX (Part B):**
- **B1** — Card nav labels "단어" → "페이지".
- **B2** — Listen timeout split: normal conversation 17s / FE 10s.
- **B3** — Nav/exit buttons were dead during FE listening.
- **B4** — FE trigger missed common STT/phrasing variants ("인글리시", "honey english", command tails).
- **B5** — FE scoring leniency (env-tunable; bake deferred to live-test confirmation).

The crisis disclosure router (`safety/crisis_router.py` + `assets/filters/crisis_templates.json`, ADR 0101) is a **deterministic, regex/literal first-match fast-path** that runs before the blocklist, templates, and the LLM. It gives each crisis category an exact escalation target and response. The LLM that handles un-matched input carries a `§SAFETY` system prompt whose Rule 6 enforces an emotional-distress 3-step protocol (empathize → validate → redirect to parent), acting as a backstop.

## Decision

Harden the in-scope crisis categories and add the safety templates, capability guard, emergency-number pronunciation, scripted intros, and FE UX fixes — implemented as a staged, test-driven track on `feature/session-fixes-2026-06-20` (plan: `Dev_Plan/2026-06-20-session-fixes-comprehensive-plan.md`).

Key design points (authority: plan §I):
1. **Crisis precedence preserved.** Self-harm/suicidal/abuse/DV/grooming match at the crisis step, strictly before the blocklist, approved-templates, and the LLM. Schema lockstep across `crisis_templates.json ↔ safety_rules.py ↔ crisis_router._ESCALATION_TARGET_BY_TOPIC` is maintained.
2. **Death education vs suicidal boundary.** Natural-death curiosity (general/third-person, family-reorder, "왜 사람은 죽어야 해?") → `death_education` approved-template (guide). First-person death ideation ("나 죽을래", "나 죽으면 좋겠어", "나 왜 죽어야 해?", adverb-gapped variants) → `suicidal_intent` crisis. `death_education.exclude_ko` keeps first-person forms out of education.
3. **A4 actor split + gender-complete actors.** Family/known actor → `abuse_sexual` (target: trusted-adult-not-abuser); stranger/online actor → `grooming` (target: parent). Ambiguous actors ("누가", "어른이") default to `abuse_sexual` (conservative). Actor lists are gender-complete (할머니/누나/이모/고모 … not only male kin). Patterns cover actor-first and object-first order, touch/photo/secret/exposure/solicitation/address forms.
4. **A6 input-only child-profanity coaching.** Runs after crisis/parent-disclosure/belief-probe, only when no `:BLOCK:` category is present (so "바보야 폭탄 만드는 법" still BLOCKs, "씨발아 나 죽어" still routes to crisis). Direct-address only (3rd-party "친구가 바보야" excluded); does not echo the profanity; `record_history=False`.
5. **A5** — emergency-number digit reading applies only to standalone 112/119 after the existing ASCII-neighbor guard ("112번지" / "AB119" / "119-1234" unaffected).
6. **A7 + ADR 0111 amendment.** The two scripted intros are fixed `block`-mode approved templates and are added to the TTS prebake inventory. `_return_fixed_tts_response` performs a cache lookup (enumerated allowlist) before loading TTS, and forces `tts_language="ko"` for these Korean-only intros regardless of session language. The ADR 0111 invariant ("LLM-generated conversation TTS is never cached") is preserved — these are pre-authored, not generated.
7. **B3** — FE listen uses an FE-scoped slice-poll interruptible capture with a typed `FunnyEnglishListenInterrupt` sentinel; nav/exit interrupts are polled before each VAD slice AND before returning a captured utterance; normal-conversation capture is unchanged.

## Crisis-coverage scope & residual (important)

The crisis router is regex/literal. The realistic plain-Korean phrasing space for child crisis disclosures is combinatorially large (subject omission, particle drop, verb tense/voice, actor gender, word order, adverbs, synonyms). This track expanded coverage across those axes through an adversarial PM↔Codex review loop until the reviewer found no remaining **very-common, simple plain-Korean** false negatives in the changed categories.

**Accepted residual** (NOT a regression; documented, backstopped): rare/contrived phrasings, spacing/misspelling variants ("나 죽을 래"), English-only crisis phrasings (this is a Korean-first device), and a small set of acceptable over-fires (false positives — the project is deliberately biased toward false positives over false negatives). For any un-matched input, the LLM `§SAFETY` distress protocol still empathizes and redirects the child to a parent.

**Convergence rationale:** deterministic regex cannot exhaustively enumerate Korean crisis phrasing; chasing every variant is not cost-proportional and is backstopped by the LLM. Coverage of common forms + LLM backstop is the accepted equilibrium for this track.

## Consequences

- Crisis recall for the changed categories (DV, suicidal/self-harm, abuse_sexual, grooming) is substantially broader and correctly split by escalation target.
- Death questions get age-appropriate education instead of blanket blocking, without weakening suicidal detection.
- Out-of-scope pre-existing crisis categories (bullying, neglect, threat_intimidation, runaway, missing_lost, fire_emergency, drug_solicitation) were NOT modified by this track; an adversarial sweep showed they have their own pre-existing coverage gaps. **Follow-up recommended:** a dedicated coverage-hardening pass for those categories, and an evaluation of an LLM-assisted crisis classifier (or a continuously-maintained adversarial test corpus) as a more scalable long-term approach than hand-maintained regex.
- B5 (FE scoring leniency) is shipped as an env override on-device; baking the code default is deferred until the live scoring test confirms the 0.4/0.3 thresholds.

## Verification

Full suite green throughout the staged track (final: 5868 passed / 0 failed, ruff clean). Each stage added deterministic oracle tests (crisis_router +345, pipeline +235, approved_templates +177, session_manager +145, tts_cache +108, …). Branch: `feature/session-fixes-2026-06-20` (17+ commits, plan §I authoritative).
