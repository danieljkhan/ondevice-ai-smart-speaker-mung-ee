# ADR 0101 — Crisis-distress signals route through deterministic mandatory adult/emergency escalation

- **Status**: Proposed (user-approved 2026-06-06; flips to Accepted after implementation validation)
- **Date**: 2026-06-06
- **Decision owner**: Claude Code (PM/orchestrator) + user final approval
- **Extends**: ADR 0072 (parent-disclosure invariance / deterministic-safety-routing precedent). Does NOT supersede it.
- **Related**: `Dev_Plan/2026-06-06-crisis-distress-escalation-plan.md` (v4, converged), discussion records `…-discussion-r1.md` / `…-discussion-r2.md`, ADR 0096 (parental monitoring — Phase 2 tie-in), `safety/approved_template_router.py`, `core/safety_rules.py`.

## Context

A 2026-06-06 on-device audit (full production persona prompt, temp=0.4, N=5/scenario) measured **adult-escalation rate** on child-welfare distress signals and found it **unreliable**: explicit suicide 80%, abuse-touch 60%, self-harm 40%, domestic violence 40%, **neglect 0%**, **runaway 0%**. Two controls established the cause is neither the model nor prompt plumbing: a marker system-instruction was obeyed (system prompt provably steers the model), and online-grooming hit **100%** because a deterministic `approved_template_router` block template fires for it.

Root cause: `m_safety_core` **Rule 6** treats parent escalation as **one of two optional redirects** ("같이 재밌는 얘기 할까?" OR "엄마한테 이야기해볼까?"), so a comfort-only reply is rule-compliant. Serious child-welfare signals fall under discretionary Rule 6, not under a deterministic escalation path. The gap is **model- and temperature-independent** and exists in current production.

## Decision

Crisis-distress signals route through a **deterministic, mandatory adult/emergency escalation path**, mirroring ADR 0072's deterministic-routing precedent.

1. **Crisis categories** (ADR-governed list, like `DANGEROUS_TOPIC_CATEGORIES`): `self_harm`, `suicidal_intent`, `neglect`, `abuse_physical`, `abuse_sexual`, `domestic_violence`, `bullying`, `threat_intimidation`, `runaway`, `grooming`, `drug_solicitation`, `missing_lost`, `fire_emergency`.
2. **Deterministic matcher** `match_crisis_disclosure(user_text, language)` runs **before** input content filtering (`core/pipeline.py` `_filter_text`); on a first-person crisis **disclosure** match it short-circuits to a fixed **`block`-mode** escalation response → 100% escalation, temperature/model-independent. This is required because `content_filter` returns `allowed=False` (→ `SAFE_FALLBACK_RESPONSE`) for both BLOCK and REPLACE violations, swallowing crisis disclosures otherwise.
3. **Disclosure vs request**: matcher fires only on first-person victim/intent disclosure (`disclosure_patterns_{ko,en}`), suppressed by `request_excludes_{ko,en}`. Third-person/how-to dangerous-content *requests* remain with the blocklist (→ `SAFE_FALLBACK`), unchanged.
4. **Escalation targets**: parent by default; **trusted adult (teacher/nurse/staff/police), not the implicated person**, for `abuse_*`/`domestic_violence`; **119** for `fire_emergency`; **112/stay-put** for `missing_lost`. All crisis categories are `block` (no guide mode).
5. **Bilingual**: matcher checks both `_ko` and `_en` pattern/exclude fields regardless of detected language; `priority: 100` (existing safety templates are 10).
6. **Rule 6 floor** (`m_safety_core.{ko,en}.txt`): safety-threatening distress makes trusted-adult escalation the **default, not optional**; self-harm disclosures must not route to "can't talk / fun redirect" (a current EN gap) — a second layer for paraphrases the matcher misses.

Layering note: `safety.crisis_router` may import `core.safety_rules` as a sanctioned exception because `core.safety_rules` is a shared constants leaf and does not import `safety`, so no cycle is introduced.

## Deterministic-floor scope & known limitations

The crisis matcher is a deterministic **floor** for common, clear first-person crisis
disclosures. It is not the exhaustive child-safety interpreter. The long tail of
paraphrases, arbitrary instruments, multi-clause phrasings, grooming/neglect
indirection such as `아무도 날 안 돌봐줘` or `사진 보내달래`, and similar indirect
signals is intentionally delegated to the strengthened Rule 6 persona prompt plus
the content-filter backstop.

Accepted deterministic-floor limitations include:
- abuse objects or instruments in arbitrary positions;
- grooming or neglect indirection that lacks the enumerated disclosure shape;
- non-enumerated self-harm or suicide paraphrases.

## Consequences

**Positive**
- Crisis-disclosure escalation becomes deterministic (target 100%), independent of model (E4B/E2B) and sampling temperature.
- Closes the audited 0%/40% gaps (neglect, runaway, self-harm, domestic violence) on the live device.

**Negative / trade-offs**
- Over-escalation (false positives) possible; mitigated by `request_excludes_*` and a benign near-miss corpus, with **ZERO** tolerance for 119/112 emergency-directive FPs and a small documented budget for "tell an adult" FPs.
- Block-mode crisis responses are fixed (not conversational) — intentional: the model must not free-form on a suicide/abuse disclosure.

**Out of scope (Phase 2)**: out-of-band parental-review event emission on crisis match (ties to ADR 0096).

## Validation criteria (before flipping Status to Accepted)
- Crisis-disclosure corpus → 100% escalation at temp=0.4 AND temp=1.0; live-pipeline reachability test proves crisis disclosure is not swallowed by `_filter_text`.
- Benign near-miss corpus → ZERO 119/112 FPs; documented small "tell-an-adult" FP budget.
- ADR 0072 parent-disclosure regression + full safety suite pass; `pytest` ≥70% cov; `ruff`/`mypy` clean (orchestrator runs tests).
