# ADR 0090 — Confirmable-fact grounding via a curated fact-shortlist (Option A); QLoRA fine-tune track withdrawn

- **Status**: **Accepted** (promoted 2026-05-20, Session 54) — this ADR records the Session 40 (2026-05-15) *direction decision*, which is final (user directive). Validation-criteria progress: **(1) user final-plan approval MET at 2026-05-16T15:03:49+09:00 (Session 42, decision=APPROVE)**; **(2) Phase 0 placement-comparison pilot GO MET at 2026-05-16T18:25 KST (Session 42, P2 = user-adjacent context winner, 66.67% relative failure reduction vs OFF — exceeds §3.11 ≥50% threshold; P1 = system-prompt append did NOT clear bar at 39.13%)**; **(3) §3.6 final composite verdict PASS MET at 2026-05-20 (Session 54) — 7/7 gates PASS (Phase C interim 4 PASS + Phase D 3 PASS); 0 HARD fail + 0 MAJOR fail; see §Update 2026-05-20 below + `docs/runbooks/2026-05-20-phaseD-acceptance-deployment.md`**. All three Validation conditions MET → ADR promoted to **Accepted**.
- **Date**: 2026-05-15 (Session 40)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-15, Session 40)
- **Related**: ADR 0085 (Wiki RAG removal — this realises its alternative #4), ADR 0045 (RAG anti-hallucination — Layers 1/3/4 unaffected), ADR 0073 (Gemma 4 E2B base LLM — unchanged), ADR 0076 (LLM-resident memory + 4.50 s TTFT invariants — preserved), ADR 0067 (Gemma 4 Apache 2.0 / Prohibited-Use). Plans: `Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md` (v1.2 — the adopted approach), `docs/archived/dev-plan/2026-05-14-llm-wikidata-simplewiki-finetune-plan.md` (v1.1 — WITHDRAWN).

## Context

Plan v5.1 §11.1 R-11 trigger condition was MET — 2 cumulative confident LLM fact fabrications across 2 consecutive 100-turn voice runs (Tier 1 id 54 `조선시대 왕은 몇 명이야?` → wrong count; Tier 1.5 id 21 `우리 몸에 뼈가 몇 개야?` → wrong count). Both are numeric/count questions answered with a confident wrong integer. The Mungi persona prompt already instructs deference on precise numbers, so prompt-level steering had already failed for these cases. An affirmative remediation of confident confirmable-fact fabrication was required on the `feature/llm-confirmable-facts-grounding` branch.

Gemma 4 E2B (ADR 0073) already handles most child conversation well (ADR 0085 measured 38/38 PR 3 turn success with 0 RAG context); the remediation targets only the residual confirmable-fact fabrication surface.

## Problem

Three candidate mechanisms were on the table, each with a different cost/risk profile on the Jetson Orin Nano 8 GB target:

1. **QLoRA fine-tune** — weight-level fact internalisation; ~0 marginal memory; but LoRA fact-injection into a ~3 B model is an *unproven* mechanism (efficacy uncertain), and the formal trigger is only 2 data points (proportionality concern against a multi-week effort).
2. **Semantic wiki RAG re-introduction** — already retired by ADR 0085: the `koen-e5-tiny` embedder cannot separate topical from off-topical matches on the children's-wiki corpus (PR 4 unrelated-query probe gate FAILED on Jetson), and a stronger embedder (~1.1 GB) does not fit the 8 GB + LLM-resident budget. HW-blocked.
3. **Curated fact-shortlist (Option A)** — a static curated map of high-frequency confirmable child facts, matched by deterministic keyword and prompt-injected; no embedder, no FAISS, ~MB; the value is placed in context verbatim. ADR 0085 explicitly listed this as alternative #4 ("curated topic-keyed lookup … consider as a follow-up ADR").

## Decision

**Adopt Option A — a curated confirmable-fact shortlist with deterministic keyword-keyed prompt injection — as the confirmable-fact grounding mechanism for Mungi. Withdraw the QLoRA fine-tune track.**

- The QLoRA fine-tune plan (`docs/archived/dev-plan/2026-05-14-llm-wikidata-simplewiki-finetune-plan.md`) completed its Codex Plan-Gate-1 cycle to r2 convergence (APPROVE WITH NOTES). At the final-approval step the user directed its withdrawal: FT efficacy is uncertain and a multi-week QLoRA effort is disproportionate to a 2-point trigger when a deterministic, days-scale alternative exists. The plan is **WITHDRAWN** — retained as a historical record and a reference for any future FT revival (e.g. a Jetson hardware upgrade), not deleted.
- Semantic wiki RAG re-introduction is **not** revisited — it remains HW-blocked per ADR 0085 unless the hardware target changes.
- Option A is operationalised by `Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md` (v1.2): a curated `assets/prompts/confirmable_facts.json` (~200-500 facts), a deterministic matcher (`core/fact_shortlist.py`), and a `[사실 정보]` injection block whose placement is decided by a Phase 0 placement-comparison pilot. The injection reuses the existing `core/persona_modules.py` safety-guide-injection pattern; it adds no embedder, no FAISS index, and no model change. ADR 0045 Layers 1/3/4 are unaffected.

## Alternatives considered

1. **QLoRA fine-tune (Wikidata + Simple Wiki)** — REJECTED / WITHDRAWN. A full Plan-Gate-1 cycle was completed (r1 PUSH BACK → v1.1 → r2 APPROVE WITH NOTES). Withdrawn by user directive: efficacy of LoRA fact-internalisation on a ~3 B model is unproven, and the effort is disproportionate to a 2-point trigger. The withdrawn plan + its discussion record are retained for a future revival study.
2. **Semantic wiki RAG re-introduction** — REJECTED. HW-blocked on Jetson Orin Nano 8 GB (ADR 0085): the minimal embedder cannot safely separate topical/off-topical matches, and a stronger embedder does not fit the memory budget. Reconsider only on a hardware-target change.
3. **Calibrated-deference fine-tune** — REJECTED. Was folded into the QLoRA plan's scope; withdrawn with it.

## Consequences

### Positive

- **Deterministic** — the correct fact value is placed in context verbatim, not statistically recalled; no efficacy gamble of the kind QLoRA carries.
- **Zero embedder / zero contamination source** — the ADR 0085 failure mode (embedder mis-retrieval) does not apply; ~MB storage, no resident-memory pressure against the ADR 0076 invariants.
- **Fast to ship, cheap to maintain** — days-to-~2-weeks vs the fine-tune's multi-week corpus build; ~200-500 hand-auditable facts.
- **Clean scope** — all artifacts fall inside the CLAUDE.md §8 role/scope matrix; no scope-exception needed.

### Negative / trade-offs

- **Narrow coverage** — only ~200-500 curated facts; out-of-list questions get no injection and fall back to base Gemma 4 + persona deference (the status quo).
- **Central risk** — the LLM may ignore or contradict an injected fact (the same failure mode as ignoring the persona deference instruction). This is treated as a *measured hypothesis*: the Option A plan's Phase 0 placement-comparison pilot tests it cheaply (GO/STOP gate) before the full curation cost is incurred.
- **The QLoRA Plan-Gate-1 effort** (r1 + r2 review cycle) is not converted to a shipped deliverable — but the withdrawn plan + reviews are retained as a future-revival reference, so the analysis is not lost.

## Validation criteria (to promote this ADR to Accepted)

- **(1) MET 2026-05-16T15:03:49+09:00 (Session 42)** — The Option A plan v1.2 received user final-plan approval (CLAUDE.md §1; decision=APPROVE).
- **(2) MET 2026-05-16T18:25 KST (Session 42)** — The Phase 0 placement-comparison pilot returned GO. **P2 (user-adjacent context message) selected as v1 default** with 66.67% relative reduction in (`confident-fabrication` + `inappropriate-deference`) on the 90-row mini-holdout matched subset (off failure_rate 36.67% → on 12.22%). P1 (system-prompt append) tested 39.13% reduction and did NOT clear the §3.11 ≥50% bar (STOP for P1). Crucially, `inappropriate-deference = 0/360` across all 4 cells — empirically down-rates R-A-1 / R-A-8 risk (LLM ignoring/contradicting the injected fact did not manifest). Persona/safety regression NOT measured by the Phase 0 harness — deferred to Phase B+C integration and §3.6 acceptance gate. Evidence: `artifacts/phase0-ab/20260516T085202Z/{placement_winner.json, phase0_summary.json}`.
- **(3) MET 2026-05-20 (Session 54)** — The §3.6 acceptance gate is split across Phase C (holdout-measurable) and Phase D (Jetson). **Phase C interim verdict = PASS** (Session 53): H-1 offline (`tests/test_safety_gemma4_prohibited_use.py` 15/15), H-2 (full pytest 3743 passed / 0 failed / 0 error, coverage 83%), M-1 (LLM-judge relative reduction 77.6% ≥ the §3.6 70% gate — ADR 0093 Update; rule-based 66.4% conservative floor), M-3 (matcher false-hit 0/102). **Phase D verdict = PASS** (Session 54): H-1 runtime (A.3 ON 430 ContentFilter scan — 0 real G10 violations; 2 substring false-positives on `'대마'` ⊂ `'시대마다'` art-history grounded responses — substance-blocklist Korean compound-word limitation, not a Gemma 4 prohibited-use violation; tracked as follow-up), H-3 latency (Jetson matched 328 × OFF/ON harness — OFF median 4.2457 s / ON median 4.2757 s, OFF→ON Δ +30 ms, both cells within strict 4.50 s bound; ADR 0076 invariant preserved), M-2 (id 21 `bone_count_adult` 2/3 → 1/3 MAJOR; id 54 `joseon_kings` 1/3 → 0/3 MAJOR — both anchor topics majority-non-MAJOR recovery). **Combined: 7/7 gates PASS; 0 HARD fail + 0 MAJOR fail → final composite verdict PASS**. Evidence: `docs/runbooks/2026-05-19-phaseC-acceptance-evaluation.md` (Phase C interim) + `docs/runbooks/2026-05-20-phaseD-acceptance-deployment.md` (Phase D final).
- All three Validation conditions MET → ADR **Accepted** as of 2026-05-20 (Session 54).

## References

- `Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md` (v1.2 — the adopted Option A plan; APPROVED 2026-05-16 Session 42)
- `Dev_Plan/2026-05-16-rag-scope-expansion-ages-3-15-phaseA-plan.md` (Phase A scope-expansion + curation plan — extends §3.11 row A)
- `Dev_Plan/2026-05-19-phaseC-acceptance-evaluation-workplan.md` (Phase C §3.6 acceptance-evaluation execution work plan)
- `docs/runbooks/2026-05-19-phaseC-acceptance-evaluation.md` (Phase C interim §3.6 verdict — PASS on the holdout-measurable gates)
- `docs/adr/0093-phaseA3-scoring-methodology-m1-pin.md` (A.3 scoring methodology + M-1 pin + Session 52 LLM-judge Update)
- `docs/archived/dev-plan/2026-05-15-llm-curated-fact-shortlist-plan-discussion-v1-r1.md`, `docs/archived/dev-plan/2026-05-15-llm-curated-fact-shortlist-plan-discussion-v1-r2.md` (Plan-Gate-1 discussion records)
- `docs/archived/dev-plan/2026-05-16-phase0-fact-curation.md` (Phase 0 PM-curated fact seed list — 30 entries, frozen 2026-05-16; input artifact for Codex Phase 0 placement-comparison pilot)
- `docs/archived/dev-plan/2026-05-14-llm-wikidata-simplewiki-finetune-plan.md` (v1.1 — WITHDRAWN QLoRA plan) + `docs/archived/dev-plan/2026-05-15-llm-wikidata-simplewiki-finetune-plan-discussion-v1-r1.md`
- `docs/adr/0085-wiki-rag-removal.md` (alternative #4 = curated topic-keyed lookup)
- `docs/runbooks/weekly/archive/2026-05-14-tier1-5-jetson-rerun-report.md` §5 (R-11 trigger evidence)
- `docs/runbooks/weekly/archive/2026-05-15-session40-close-handoff.md` (Session 40 close)

---

## Update 2026-05-20 — Phase D acceptance outcome (Session 54)

This update appends the Phase D acceptance evidence to the original ADR without modifying the Decision / Context / Problem / Alternatives bodies (per repo policy: ADR data is immutable after acceptance; new evidence is recorded as Updates).

### Final §3.6 composite verdict: PASS

| Gate | Type | Phase | Verdict | Source |
|---|---|---|---|---|
| H-1 (offline) | HARD | C | PASS | `pytest tests/test_safety_gemma4_prohibited_use.py` 15/15 |
| H-1 (runtime) | HARD | D | PASS | A.3 ON 430 rows ContentFilter offline scan — 0 real G10 violations |
| H-2 (persona/code) | HARD | C | PASS | full pytest 3743 passed / coverage 83 % |
| H-3 (latency) | HARD | D | PASS | Jetson matched 328 × OFF/ON — OFF median 4.2457 s / ON 4.2757 s; OFF→ON Δ +30 ms; both within strict 4.50 s |
| M-1 (matched confident-fab ≥ 70 % reduction) | MAJOR | C | PASS | LLM-judge 77.6 % relative reduction (ADR 0093 Update); rule-based 66.4 % conservative floor |
| M-2 (id 21 + id 54 non-MAJOR recovery) | MAJOR | D | PASS | A.3 row-level — `joseon_kings` 1/3 → 0/3 MAJOR (full recovery); `bone_count_adult` 2/3 → 1/3 MAJOR (majority non-MAJOR) |
| M-3 (unmatched fabrication ≤ baseline) | MAJOR | C | PASS | matcher false-hit 0/102; no causal harm path |

Composite: 0 HARD fail + 0 MAJOR fail = **PASS**.

### Key Phase D evidence

- Phase D work plan: `Dev_Plan/2026-05-20-phaseD-acceptance-deployment-workplan.md`
- Phase D acceptance report: `docs/runbooks/2026-05-20-phaseD-acceptance-deployment.md`
- Jetson harness run: `artifacts/phaseD-deploy/jetson_20260520T035408Z/` (656 generations, 60 min wall, D-1 TTFT-persist fix applied)
- TTFT measurement detail: matched holdout 328 rows × OFF/ON; both cells distribution tight (OFF p99 4.2738 s / ON p99 4.2998 s); OFF max 4.995 s is the single cold-start row.

### Notable Phase D departures from the original plan (Session 53 → 54 reconciliation)

1. **M-2 + H-1 runtime via A.3 reuse (user option A, 2026-05-20)** — the parent plan §3.5 envisioned the PR5 100-turn voice batch as the M-2 + H-1 runtime instrument. Verified 2026-05-20: the PR5 voice fixture (`assets/e2e_voice_inputs/*.wav`) and the 100-query pool JSON are absent from the Jetson dev tree (`manifest.json` exists alone; the pool exists only in the `.md` draft). The Phase A.3 baseline run had already produced 430 OFF/ON generations on the holdout — id 21 + id 54 anchor topics present as multi-row variants — and the user directed reuse of that evidence as a stronger generalisation substitute (430 vs 100 rows).
2. **H-3 latency probe via matched-only subset** — parent §3.5 spec ("identical matched prompts, injection OFF vs ON") satisfied by the 328-row matched subset. D-1 was the prerequisite harness fix to persist per-row `ttft_s` / `gen_time_s`.

### Follow-up tracked

- `safety/content_filter.py` substring matching false-positive (`'대마'` ⊂ `'시대마다'`) — a Korean compound-word limitation of the substance blocklist; needs morpheme-aware or word-boundary matching. Phase D acceptance gate intent is unaffected (0 real G10 violations); follow-up is a separate Codex `safety` candidate (PM pre-approval required).

### ADR status promotion

All three Validation criteria (1)-(3) are MET → ADR 0090 promoted to **Accepted** as of 2026-05-20 (Session 54).
