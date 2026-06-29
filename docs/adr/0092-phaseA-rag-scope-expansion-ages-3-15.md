# ADR 0092 — Phase A RAG knowledge scope expansion to ages 3-15 + Gemini 2.5 Flash batch primary judge

- **Status**: **Accepted** — records the Session 43 (2026-05-16) Phase A scope + judge decision; user-approved at the plan-level 2026-05-16T22:58:47+09:00 (Plan v1.4). **Amended 2026-05-16T23:20+09:00 (Plan v1.5)**: user directive — switch judge API from Vertex AI to Gemini API direct (Google AI Studio); same model + same batch-discounted pricing; simpler auth (single API key). Decision (i) + (v) updated; Vertex AI Gemini added as REJECTED-for-Phase-A alternative in §Alternatives considered. The ADR is promoted to **Accepted** once Phase A.0 spike validates (Gemini API key dry-run + agreement ≥ 80 % + Option-A/B selected + filtered article counts within plan ranges) AND Phase A.3 baseline measurement pins the absolute M-1 threshold — the two conditions enumerated in §Validation criteria below. **Accepted 2026-05-19**: both conditions now satisfied — Phase A.0 spike (condition 1, Session 45) + Phase A.3 baseline measurement pinned the absolute M-1 threshold (condition 2; `docs/runbooks/2026-05-19-phaseA3-baseline-measurement.md`).
- **Date**: 2026-05-16 (Session 43)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-16, Session 43)
- **Related**: ADR 0090 (Option A mechanism; this ADR is its Phase A scope decision), ADR 0091 (Session 41-42 Anthropic Haiku 4.5 batch wiki-filter; this ADR **reclassifies its outputs to alternative-judge cross-check material** without amending ADR 0091 itself), ADR 0085 (Wiki RAG removal — Phase A does NOT re-introduce semantic RAG; ADR 0085 stays operative), ADR 0089 (docs-only commit abbreviation — applies to plan commits). Plans: `Dev_Plan/2026-05-16-rag-scope-expansion-ages-3-15-phaseA-plan.md` v1.4 (APPROVED 2026-05-16T22:58:47+09:00; the agreed final plan).

## Context

Plan v1.2 (`Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md`, APPROVED Session 42) defined Option A — a curated confirmable-fact shortlist (~200-500 facts) with deterministic keyword matching and prompt injection — and successfully validated **Phase 0** (placement-comparison pilot, GO P2 at 18:25 KST Session 42, 66.67 % relative failure reduction, 0/360 inappropriate-deference; ADR 0090 conditions 1 + 2 of 3 MET). Plan v1.2 §3.11 row "A — Curate" defined Phase A as "full ~200-500 fact shortlist curation" but with the original CLAUDE.md §1 audience floor of children under 10.

In Session 43, the user directed two design extensions:

1. **Scope extension** to ages 3-15 (extending the outer audience without amending the CLAUDE.md §1 under-10 safety floor)
2. **Judge tool extension** from Anthropic Haiku 4.5 batch (ADR 0091) to **Vertex AI Gemini 2.5 Flash batch** (~2-3× cheaper at $0.15 / 1M input + $1.25 / 1M output, batch-discounted; the project's original pre-Session-41 wiki-filter judge)

Plan v1.4 (this ADR's operationalisation) integrates both extensions and went through a complete Plan-Gate-1 cycle (r1 PUSH BACK → v1.2 9 ACCEPT → v1.3 user-directive Gemini switch → r2 APPROVE WITH NOTES → v1.4 8 ACCEPT + 1 FOLDED + 4th-audit cleanup; 0 PM REJECT, 0 escalation across 25 substantive findings). r3 was explicitly not needed per Codex r2.

## Problem

Phase A needs concrete, recorded decisions on five inter-related design axes that Plan v1.4 specifies but ADR-level commitment requires for downstream audit and traceability:

1. **Primary judging tool** for the from-scratch raw-dataset refinement (Vertex Gemini vs Anthropic Haiku, considering cost / quality / project history)
2. **Status of the Session 41-42 Anthropic Haiku KEEP corpora** (re-seed as primary input vs reclassify as cross-check)
3. **Scope expansion** target audience (keep <10 vs extend to 3-15)
4. **Vertex AI auth + GCP project + region** (ADC vs service-account JSON; named project)
5. **Cross-judge audit metric** definition (full KEEP/DROP matrix vs KEEP-only intersection, given retained-asset constraints)

Without ADR-level pre-commitment on these axes, Phase A.0 implementation work (Codex `platform` dispatch + PM operational steps) would face decision uncertainty mid-execution.

## Decision

The following 5-point decision is committed by this ADR per Plan v1.4 §3.12 pre-commitment language. All five are independently auditable post-implementation.

### (i) Phase A primary judge = Gemini 2.5 Flash batch via Gemini API direct (Google AI Studio) — v1.5 amendment

The from-scratch raw-dataset refinement pipeline (Plan v1.5 §3.3 5-step pipeline) uses **Gemini 2.5 Flash batch** as the primary KEEP/DROP + target_age_band judge. The Gemini 2.5 Flash batch-discounted pricing is $0.15 / 1M input tokens + $1.25 / 1M output tokens, giving ~2-3× cheaper Phase A judging cost vs Anthropic Haiku 4.5 batch ($0.40 + $2.00).

**API surface (v1.5 amendment — supersedes v1 Vertex AI)**: Gemini API direct via the Google AI Studio endpoint (`generativelanguage.googleapis.com`) + the `google-genai` SDK. **NOT** Vertex AI. Rationale: same model + same batch-discounted pricing; simpler auth (single API key vs gcloud SDK + GCP project + region + ADC); no GCP project setup required.

Implementation: new Codex `platform` script `scripts/gemini_batch_childfilter.py` (renamed from `vertex_gemini_batch_childfilter.py` in v1.5). The tracked prototype helpers `Dev_Plan/submit_vertex_wiki_filter.py` + `monitor_vertex_batch.py` from the pre-Session-41 wiki-filter pipeline serve only as **conceptual reference** for the batch flow (submit → poll → download → extract); the SDK + auth differ entirely from Vertex.

### (ii) Session 41-42 Anthropic Haiku 4.5 KEEP corpora reclassified to alternative-judge cross-check material

`assets/rag/anthropic_keep_en.jsonl` (20,885 EN KEEP rows) + `assets/rag/anthropic_keep_ko_refined.jsonl` (2,190 KO KEEP rows) — both Session 41-42 outputs of the Anthropic Haiku 4.5 batch judge per ADR 0091 — are **NOT** seeded into Phase A as primary input. Their Phase A role is **alternative-judge cross-check**: the A.0 spike computes a KEEP-only intersection agreement metric (see decision (v)) using these KEEP corpora as the alternative-judge reference.

### (iii) ADR 0091 remains historically accurate and is NOT amended

ADR 0091 documented the Session 41 operational decision to use Anthropic Haiku 4.5 batch for wiki child-filter judging in the specific Session 41-42 context. That decision was correct for its context and produced the KEEP corpora that this ADR (0092) now reclassifies. ADR 0091 is **not amended** — it stays accurate as a historical record. This ADR (0092) contextualises ADR 0091 outputs for Phase A consumption.

### (iv) Age-graded scope = 12 baseline categories deepened + 5 new categories × 3 age bands

Phase A scope, per Plan v1.4 §3.1 + §3.2:
- **Categories**: 12 baseline (`animal`, `plant`, `science`, `culture`, `music`, `sports`, `vehicle`, `weather`, `body_health`, `math`, `nature`, `story` — historical 12 from the recovered `git show 9822cc5^:scripts/prepare_{wiki,simple_wiki}_rag.py`) deepened for ages 11-15 + 5 new categories for ages 8-15 (`world_geography`, `world_history_light` with date-event whitelist, `technology_intro`, `science_intro_deeper`, `arts_appreciation_intro`) = **17 total categories**.
- **Age bands**: 3 explicit (`preschool` 3-7 / `elementary` 8-12 / `middle_school` 13-15) with target distribution **40 / 40 / 20** (skewed toward primary audience).
- **Target shortlist size**: ~500-1,000 facts.
- **Safety floor**: CLAUDE.md §1 "children under age 10" is the **safety floor** (NOT amended by this ADR); the extension to 15 is outer-audience addition without loosening any of the 8 historical exclude families (violence / adult / crime / politics / finance / law / military / medical_sensitive).
- **Runtime band-cap**: matcher carries age-band trigger precedence (`preschool > elementary > middle_school`) + `MUNGI_FACT_SHORTLIST_MAX_BAND` env flag (default `elementary` — middle_school facts excluded from injection until per-turn age detection lands as a future-scope feature).

### (v) Gemini API direct auth = `GEMINI_API_KEY` env (Google AI Studio API key); KEEP-only intersection agreement metric — v1.5 amendment

- **Auth method (v1.5 amendment — supersedes v1 Vertex ADC)**: single API key from Google AI Studio (https://aistudio.google.com/apikey), supplied via `GEMINI_API_KEY` env var (fallback `GOOGLE_API_KEY`). PM provides the key separately (similar pattern to Anthropic `batch.md`); plan does NOT name a specific key location to avoid referencing user-side paths. No GCP project / region / gcloud SDK / ADC required.
- **SDK**: `google-genai` (preferred) or `google-generativeai` (legacy fallback). Both support the Gemini API Batch Mode (50% discount, async, up to 24h SLA).
- **Cross-judge audit metric** (Plan v1.4 §3.11 A.0 d, KEEP-only intersection per r2-B6; metric definition unchanged by v1.5): on the A.0 spike sample, compute `|Gemini-KEEP ∩ Anthropic-KEEP| / |Anthropic-KEEP|` on articles BOTH judges saw. Threshold ≥ 80 % triggers PASS; < 80 % triggers prompt redesign or escalation to Anthropic Haiku 4.5 for full Phase A pass at the higher cost. Caveat: Gemini DROP that Anthropic would have KEPT is undetectable from existing assets — Anthropic full KEEP/DROP was not retained beyond the KEEP extract (Session 42 data-retention choice). Audit artifact: `artifacts/phaseA-a0/<ts>/judge_agreement.jsonl` per-row + `judge_agreement_summary.json` aggregate.

## Alternatives considered

1. **Retain Anthropic Haiku 4.5 as Phase A primary judge** — REJECTED. Cost ~2-3× higher with no measurable Phase 0 / Phase A judging quality signal that would justify the price difference. The Session 41-42 outputs serve as cross-check reference; switching to Gemini for Phase A primary leverages the project's pre-Session-41 working Vertex pipeline + tracked helpers. Escalation path back to Anthropic is preserved (R-PA-11 mitigation).
2. **Keep audience floor at <10 children** (no 3-15 extension) — REJECTED per user directive. Phase A v1.4 retains the safety floor; the extension adds outer-audience scope without loosening policy.
3. **Amend ADR 0091** to reflect the Phase A judge switch — REJECTED. ADR 0091 documents a correct historical operational decision for Session 41 context; amending it would conflate two distinct decision points. ADR 0092 contextualises ADR 0091 outputs for Phase A consumption without rewriting the historical record.
4. **Hardcode service-account JSON auth** — REJECTED for Phase A. (v1 context: when Vertex was the planned API, ADC was simpler than service-account JSON for single-developer ops. v1.5 deprecates this entire axis: Gemini API direct uses a single API key, no service account needed.)
5. **Full KEEP/DROP matrix cross-judge audit** — REJECTED as infeasible. Anthropic full KEEP/DROP was not retained beyond the KEEP extract (Session 42 data-retention choice); only KEEP-only intersection agreement is computable from existing assets.
6. **Vertex AI Gemini 2.5 Flash batch (v1 decision, superseded by v1.5)** — REJECTED for Phase A per Session 43 user directive 2026-05-16T23:20+09:00. Same model + same batch-discounted pricing as Gemini API direct, but additional ops overhead: gcloud SDK install + ADC `gcloud auth application-default login` + active GCP project + region setup. Gemini API direct is operationally simpler (single API key) without sacrificing quality, cost, or capability. The original v1 decision rationale (project's pre-Session-41 working Vertex pipeline + tracked prototype helpers) is preserved as a fallback option: if Gemini API direct hits unexpected rate-limit or capability blockers, Vertex AI Gemini path is still viable with the existing `Dev_Plan/submit_vertex_wiki_filter.py` + `monitor_vertex_batch.py` references.

## Consequences

### Positive

- **Cost reduction**: Phase A judging cost ~$4.7-9.5 (Option-A) / $2.2-4.5 (Option-B) — single-digit USD for the full Phase A pass; ~2-3× cheaper than the Anthropic Haiku path that v1.2 originally assumed.
- **Audience widening**: ages 8-15 now have age-appropriate knowledge depth available (via middle_school band when band-cap flag permits) without compromising the under-10 safety floor.
- **Two-judge cross-check**: KEEP-only intersection agreement audit gives a quality signal for Gemini Flash judgment vs the established Anthropic Haiku baseline, with an explicit escalation path back to Anthropic if quality falls below threshold.
- **Reuse of historical infrastructure**: Vertex SDK + batch path was the pre-Session-41 working judge for `wiki_filter_batch_input.jsonl` + `wiki_filter_results.jsonl` (now-deleted artifacts) — the tracked Dev_Plan helpers provide a known-good reference.
- **ADR 0091 preserved**: historical record stays accurate without revision churn.

### Negative

- **Gemini Flash judgment quality risk** (R-PA-11) — Flash is a smaller/cheaper model than Haiku 4.5; child-suitability judgment is nuanced. Mitigation via A.0 ≥ 80 % agreement gate + escalation path. **UPDATE (2026-05-17 Session 46): R-PA-11 executed (800-row Haiku 4.5 re-judge, 30.2% agreement) and formally DISCARDED — root cause model calibration gap, not data quality. See ADR 0091 §R-PA-11.**
- **Cross-judge audit asymmetry**: Gemini DROP that Anthropic would have KEPT is undetectable from existing assets. Accepted trade-off.
- **Vertex auth + GCP setup overhead** (R-PA-12) — new auth method to set up + verify; mitigated by A.0 dry-run with a 5-row test batch BEFORE the full Phase A judge run.
- **Coverage extension cost**: 17 categories × 3 age bands = larger curation effort vs the original 12-category × 1-band v1.2 plan; Phase A.2 (consolidated curation per Plan v1.8 redefinition, Session 47; ~4-6 PM sessions + 1-2 Codex sessions, consolidated from original A.1 + A.2 split) absorbs the curation effort. Phase A.1 (full-corpus judge) already COMPLETE Session 45-46.

## Validation criteria (to promote this ADR to Accepted)

- **(1) ✅ Largely satisfied / partially superseded** (Session 45-47 updates) — Phase A.0 spike PASSED 8/8 batches (Session 45); Option-A 3-band prompt design selected; Phase A.1 full pass produced 23,925 KEEP rows within plan ranges. **Sub-conditions adjusted post-Session 43**: (a) Gemini API key dry-run was bypassed when judge switched from Gemini → OpenAI GPT-4.1-mini for A.1 (operational simplification, same Option-A 3-band design); (b) "Gemini-vs-Anthropic KEEP-only intersection agreement ≥ 80 %" → R-PA-11 cross-judge DISCARDED Session 46 (model calibration gap, not data quality; see ADR 0091 §R-PA-11); (c) ~~"age-band heuristic ≥ 75 % agreement vs PM-labeled sample"~~ → **r1-A2 SKIPPED Session 47** (Mungi 팀 구조상 독립 인간 레이블러 부재로 ground-truth 검증 성립 불가; `target_age_band` (GPT-4.1-mini judge) 신뢰). See `docs/runbooks/weekly/2026-05-17-session47-close-handoff.md`.
- **(2) ✅ Met (2026-05-19, Phase A.3 closed)** — Phase A.3 baseline measurement on the expanded 502-entry shortlist + 430-row holdout pinned the absolute M-1 threshold: matched-subset confident-fabrication ON 14.3 % / OFF 42.7 % / relative reduction 66.4 % (conservative floor; true value higher); M-3 false-match guard PASS (matcher false-hit 0/102); inappropriate-deference 0. See `docs/runbooks/2026-05-19-phaseA3-baseline-measurement.md`, `artifacts/phaseA-baseline/jetson_20260518T133850Z/baseline_summary.json`, and ADR 0093 (`docs/adr/0093-phaseA3-scoring-methodology-m1-pin.md`, the A.3 scoring-methodology + M-1-pin decision record). The HARD safety / regression / latency gates (H-1 / H-2 / H-3, parent plan §3.6) are evaluated in **Phase B**, not A.3 — consistent with the §Status promotion criterion above, which requires only the A.3 M-1 pin.
- Status note: Phase A.3 closed 2026-05-19; §Validation criteria conditions 1 + 2 both satisfied; the §Status field above is updated from **Proposed** to **Accepted**.

## References

- Phase A plan: `Dev_Plan/2026-05-16-rag-scope-expansion-ages-3-15-phaseA-plan.md` v1.4 (APPROVED 2026-05-16T22:58:47+09:00 Session 43)
- Discussion records: `Dev_Plan/2026-05-16-rag-scope-expansion-ages-3-15-phaseA-plan-discussion-v1-r{1,2}.md`
- Codex review spec archives: `.codex/specs/phaseA-rag-scope-expansion-ages-3-15-plan-review-v1-r{1,2}.md`
- Parent plan v1.2 (APPROVED Session 42; Phase 0 GO P2): `Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md`
- ADR 0090 (Option A mechanism; conditions 1+2 of 3 MET): `docs/adr/0090-confirmable-fact-grounding-curated-shortlist.md`
- ADR 0091 (Session 41 Anthropic batch decision; this ADR reclassifies its outputs without amending ADR 0091): `docs/adr/0091-wiki-corpus-child-filter-anthropic-batch.md`
- ADR 0085 (Wiki RAG removal — Phase A does NOT re-introduce semantic RAG; ADR 0085 stays operative): `docs/adr/0085-wiki-rag-removal.md`
- Historical RAG scope (recovered, 12 baseline categories source): `git show 9822cc5^:scripts/{prepare_wiki_rag, prepare_simple_wiki_rag}.py`
- Raw datasets at `assets/rag/raw material/`: `rahular/simple-wikipedia` (769,764 paragraph rows) + `lcw99/wikipedia-korean-20240501` (515,425 articles)
- Vertex SDK + batch path reference: `Dev_Plan/submit_vertex_wiki_filter.py` + `Dev_Plan/monitor_vertex_batch.py` (prototype-grade; structured reimplementation in Phase A.0)
- Anthropic KEEP corpora (cross-check references): `assets/rag/anthropic_keep_en.jsonl` + `assets/rag/anthropic_keep_ko_refined.jsonl`
- Vertex AI Gemini 2.5 Flash Flex/Batch pricing source: Google Cloud pricing page (current 2026-05; verified via Codex r2 authority log)
- A.2 work plan: `Dev_Plan/2026-05-18-phaseA2-curation-workplan.md` v1.4 (Champion-approved Session 49)
- Phase A.3 baseline measurement (M-1 pin): `docs/runbooks/2026-05-19-phaseA3-baseline-measurement.md` + `artifacts/phaseA-baseline/jetson_20260518T133850Z/baseline_summary.json`

## Update — 2026-05-18 (Session 49): age-band model consolidated 3 → 2

Per `feedback_adr_immutability`, the original decision data above is unchanged; this section records a subsequent decision.

The Champion (Session 49) consolidated the Phase A age-band model from **3 bands to 2**, reviewed + approved within the Plan-Gate-1 cycle of the A.2 work plan `Dev_Plan/2026-05-18-phaseA2-curation-workplan.md` v1.4 (Codex `reviewer` r1 PUSH BACK → r2 APPROVE WITH NOTES; Champion final-approved Session 49).

- **§Decision (iv)** — the "3 explicit age bands (`preschool` / `elementary` / `middle_school`), 40/40/20" statement is **superseded for Phase A.2+**: the 2-band model is `under_10` (ages ~3-9, primary audience) + `under_15` (ages ~10-14), distribution ≈ 80/20. The matcher `AgeBand` enum migrates to `under_10` / `under_15`; `MUNGI_FACT_SHORTLIST_MAX_BAND` default → `under_10`.
- **Rationale**: the A.1 KEEP pool is ~6 % preschool; a 40 % preschool curation target was infeasible from the pool. Merging preschool into the lower band removes the tension while preserving the under-10 safety/UX floor.
- **The A.1 KEEP corpus is NOT re-judged** — `assets/rag/cleaning_data/openai_keep_{en,ko}.jsonl` retains its 3-valued `target_age_band` (immutable A.1 artifact). A.2 maps `{preschool, elementary}` → `under_10` and `middle_school` → `under_15` at curation time.
- §Decision (i)/(ii)/(iii)/(v), the 17-category scope, the 8 exclude families / safety floor, and the ages-3-15 audience range are **unchanged**.
- This is a documentation reconciliation of an already-gated decision; the ADR Status is unaffected.
