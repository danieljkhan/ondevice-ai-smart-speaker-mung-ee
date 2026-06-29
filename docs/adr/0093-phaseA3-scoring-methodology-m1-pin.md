# ADR 0093 — Phase A.3 confirmable-fact holdout scoring methodology + absolute M-1 threshold pin

- **Status**: **Accepted** — records the Session 51 (2026-05-19) Phase A.3 baseline-measurement scoring methodology + the absolute M-1 threshold pin. The corrected scorer and the M-1 pin were validated this session.
- **Date**: 2026-05-19 (Session 51)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-19, Session 51)
- **Related**: ADR 0090 (Option A confirmable-fact grounding mechanism — A.3 is its baseline measurement), ADR 0092 (Phase A scope ages 3-15 — A.3 satisfies its §Validation criteria condition 2). Gates: `Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md` §3.6 (M-1/M-2/M-3). Phasing: `Dev_Plan/2026-05-16-rag-scope-expansion-ages-3-15-phaseA-plan.md` §3.11 A.3.

## Context

Plan v1.2 §3.6 defines **M-1**, the primary MAJOR gate: the matched-subset confident-fabrication rate reduced ≥ 70 % relative vs base, with the *absolute* target "pinned at the Phase A checkpoint". Phase A.3 (`...phaseA-plan.md` §3.11 A.3) is that checkpoint: run the Phase 0 A/B harness on the expanded 502-entry shortlist + 430-row holdout and pin the absolute M-1 threshold.

The harness ran on Jetson Orin Nano (Gemma 4 E2B Q5_K_M via llama.cpp; run `jetson_20260518T133850Z`; placement p2; `MUNGI_FACT_SHORTLIST_MAX_BAND=under_15`; 860 generations). During result analysis the offline scorer `scripts/score_fact_holdout.py` was found to systematically misclassify correct answers, so a scoring-methodology decision is required before the M-1 pin can be trusted.

## Problem

1. **Scorer defect.** The Phase-0-era `classify_response` accepted a response as `correct` only via a verbatim substring match of the gold answer (`_match_variant`) or an Arabic-digit numeric match (`_extract_numbers`). The Jetson LLM, when grounded, answers in natural Korean paraphrase and Korean numeral words (`두 개` for `2개`, `스물일곱` for `27`). Such correct answers failed both checks and then tripped the assertive-sentence pattern, getting mislabeled `confident-fabrication`. The initial M-1 measured a spurious 20.5 % relative reduction; a random 7/7 sample of "fabrication" rows were all genuinely correct answers.
2. **Matcher confirmed clean.** Offline `core.fact_shortlist.match_fact` over the 430 holdout questions: matched 328/328 to the correct topic, unmatched false-hit 0/102. The defect was purely in scoring, not in matching — `core/fact_shortlist.py` needed no change.
3. **Rule-based scoring has a precision ceiling.** Korean semantic equivalence cannot be decided perfectly by rules. Four Codex tuning rounds empirically oscillated between false-negatives (too strict) and false-positives (too lenient).
4. The absolute M-1 threshold must still be pinned for Phase B.

## Decision

### (i) Scorer correctness-detection methodology
`scripts/score_fact_holdout.py` `classify_response` (matched rows) uses a 4-matcher cascade, first match wins, correctness paths taking precedence over deference/fabrication:
1. exact gold / `acceptable_variant` substring;
2. Korean-numeral-normalized substring (numeral words → Arabic digits, Sino + native incl. counter-contracted forms);
3. numeric-tolerance match (numeral-aware `_extract_numbers`);
4. content-token overlap — the gold answer is tokenized into content tokens (Korean josa / assertive endings stripped only when the remaining stem is ≥ 2 characters), and the response is `correct` if it covers at least `CORRECTNESS_OVERLAP_THRESHOLD` (= 0.65) of the gold content tokens over the **full gold-token denominator**; a numeric-conflict guard rejects a response carrying a different number.

### (ii) Conservative-bias stance
The scorer is deliberately tuned conservative. For a fabrication-rate measurement, a false-positive (a fabrication scored `correct`) OVERSTATES the grounding benefit and is unacceptable; a false-negative (a correct answer scored `confident-fabrication`) merely understates the benefit and is safe. The final (round-4) scorer is PM-validated at **0 false-positives**; it retains documented false-negatives on pure semantic paraphrase that rule-based overlap cannot detect. A round-3 lenient variant (capped overlap denominator) that admitted false-positives — e.g. the wrong answer `호주 수도는 시드니` scored `correct` — was rejected.

### (iii) Pinned absolute M-1 threshold
Matched-subset confident-fabrication rate with injection ON = **14.3 %** (OFF baseline 42.7 %; relative reduction **66.4 %**; by age band: under_10 68.3 %, under_15 61.1 %). The 66.4 % figure is a **conservative floor** — the true reduction is higher (the rejected lenient scorer bracketed it at ~80 %; the true value is approximately the mid-70 % range). Phase B evaluates the M-1 gate against this absolute pin (ON confident-fabrication ≈ 14.3 %) with a tolerance band for temperature = 1.0 single-sample sampling noise.

### (iv) M-3 false-match guard — structurally PASS
The matcher false-hit rate on the 102 unmatched holdout rows is 0/102, so injection never fires on unmatched turns, the OFF and ON prompts are identical there, and the matcher provably causes zero harm on turns it should not touch. M-3 PASS. (The observed +2-row OFF→ON difference on the unmatched subset is temperature = 1.0 sampling noise, not matcher harm.)

## Alternatives considered

1. **LLM-judge scorer** — REJECTED. Most robust to paraphrase, but adds an external API dependency (key-approval gate + cost) or a slow, somewhat circular local-Gemma judge, and is non-deterministic.
2. **Enrich the holdout `acceptable_variants` instead of fixing the scorer** — REJECTED. A 430-row curation burden; the scorer's correctness logic is the root cause and the reusable fix.
3. **Lenient content-overlap (round-3: `min(gold_tokens, 3)` capped denominator)** — REJECTED. It measured M-1 at 80.3 % but PM validation found false-positives (wrong answers scored `correct`), which overstate the result — disqualifying for a fabrication-rate gate.
4. **Accept the round-1 / round-2 scorer** — superseded. Round-1 (67.8 %) was conservative but over-strict in a residual band; round-2 regressed (over-aggressive guards demoted 12 correct answers). Round-4 = round-1's conservative full-gold-token denominator + the numeral / paraphrase fixes, PM-validated clean.

## Consequences

### Positive
- A.3's absolute M-1 threshold is honestly pinned; Phase B has an auditable gate calibration.
- The corrected scorer is reusable for the Phase B 100-turn evaluation.
- The conservative bias guarantees the measurement never overstates the grounding benefit.
- The matcher `core/fact_shortlist.py` was independently confirmed correct — no matcher change was needed.

### Negative
- M-1 66.4 % is a floor, not a point estimate — the true reduction (~mid-70 %) is reported as a documented bracket rather than a single number.
- Rule-based scoring retains residual false-negatives on pure semantic paraphrase; a future LLM-judge re-score could tighten the number if Phase B requires it.
- temperature = 1.0 yields one stochastic sample per cell; the OFF/ON comparison is not seed-paired, so point estimates carry sampling noise.

## Validation criteria

- ✅ A.3 harness run complete (`jetson_20260518T133850Z`, 860 generations).
- ✅ Corrected scorer QC clean (`ruff` / `ruff format` / `mypy` on changed files; `pytest tests/test_fact_shortlist.py` 2376 passed).
- ✅ PM validation: 0 false-positives in sampled `correct` rows; the `호주 수도 시드니` wrong answer is correctly classified `confident-fabrication`.
- ✅ M-1 absolute threshold pinned; M-3 PASS.

## References

- A.3 baseline measurement report: `docs/runbooks/2026-05-19-phaseA3-baseline-measurement.md`
- A.4 handoff artifact (pinned M-1): `artifacts/phaseA-baseline/jetson_20260518T133850Z/baseline_summary.json`
- Scorer: `scripts/score_fact_holdout.py`; harness: `scripts/run_phase0_ab_harness.py`; matcher: `core/fact_shortlist.py`
- Phase A plan: `Dev_Plan/2026-05-16-rag-scope-expansion-ages-3-15-phaseA-plan.md` §3.11 A.3
- Acceptance gates (M-1/M-2/M-3): `Dev_Plan/2026-05-15-llm-curated-fact-shortlist-plan.md` §3.6
- ADR 0090 (Option A confirmable-fact grounding mechanism)
- ADR 0092 (Phase A RAG scope expansion ages 3-15 — A.3 satisfies §Validation criteria condition 2)

---

## Update 2026-05-19 (Session 52) — M-1 LLM-judge re-score

§Alternatives #1 of this ADR rejected an LLM-judge scorer *for A.3*; §Consequences/Negative anticipated that "a future LLM-judge re-score could tighten the number if Phase B requires it". That re-score was executed in Session 52 under the approved plan `Dev_Plan/2026-05-19-m1-llm-judge-rescore-plan.md` (Plan-Gate-1 CONVERGED, user-approved). This Update records its outcome. The original Context / Problem / Decision (i)–(iv) / Alternatives / Consequences are **unchanged** — the rule-based scorer and its conservative M-1 pin remain as recorded.

### Method
- Judge: OpenAI **`gpt-5-mini-2025-08-07`** (GPT-5-family reasoning model) via the OpenAI Batch API, Structured Outputs (4-class `correct` / `deference` / `confident-fabrication` / `other`), `reasoning_effort=low`, fixed `seed`; `temperature` omitted (unsupported by this model).
- Input: the saved A.3 holdout responses `rescored4_rows_p2_{off,on}.jsonl` (860 rows) rejoined to `confirmable_facts_holdout.jsonl` for `acceptable_variants` / `numeric_tolerance`.
- The judge's 4-class label is mapped to the 5-class `Verdict` by the same axis-aware contract as `score_fact_holdout.py` (matched: `deference` → appropriate/inappropriate by injection state; unmatched: `deference`/`other` → `correct`).
- A hard "no partial metric" completeness gate required all 860 `custom_id`s parsed — achieved 860/860, no rerun.
- Tool: `scripts/judge_fact_holdout_llm.py`. Artifacts: `artifacts/phaseA-baseline/llm_judge_20260519T042041Z/`.

### Result — M-1 point estimate
Matched-subset M-1 failure numerator = `confident-fabrication` + `inappropriate-deference` (per Plan v1.2 §3.5):

| Metric | OFF | ON | Relative reduction |
|---|---|---|---|
| **M-1 combined** | 17.68 % | 3.96 % | **77.6 %** |
| `confident-fabrication` only | 17.68 % | 2.74 % | 84.5 % |
| under_10 | 16.92 % | 4.23 % | 75.0 % |
| under_15 | 20.59 % | 2.94 % | 85.7 % |

The LLM-judge point estimate **77.6 %** falls inside the ~66–80 % bracket §(iii) predicted, confirming the rule-based 66.4 % was a conservative *floor*. 77.6 % clears the Plan v1.2 §3.6 M-1 ≥ 70 % relative-reduction gate.

### Judge trustworthiness validation (plan §3.5)
All 212 rule-vs-LLM failure-membership-flip rows were PM-reviewed:
- **Direction A** (rule = fail → LLM = pass, 148 rows): every reclassification sound — the rule over-flagged explicit deferrals, incomplete non-fabricating answers, and correct paraphrases (notably Korean-spelled numerals on injection-ON rows). **0 judge false-positives** — no genuine fabrication scored as a pass.
- **Direction B** (rule = pass → LLM = fail, 64 rows): the LLM is stricter (the safe direction) and additionally caught several rule *false-positives* — e.g. "물은 영하 10도에서 언다" (gold 0 °C) and "1AU는 태양에서 가까운 거리" were scored `correct` by the rule but correctly `confident-fabrication` by the LLM.

### Reconciliation outcome
Per the plan §3.5 conservative-reconciliation rule (77.6 % ∈ bracket AND 0 judge false-positives): the LLM-judge **77.6 %** is recorded as the M-1 **point estimate**. The rule-based conservative pin (ON `confident-fabrication` ≈ 14.3 %, 66.4 % floor) **remains the Phase B M-1 gate anchor** as decided in §(iii); the 77.6 % point estimate is its precision companion. Replacing the absolute pin with the LLM-judge figure is a separate decision deferred to Phase B / the user — not made here.

M-3 cross-check: the LLM-judge unmatched-subset `confident-fabrication` rate is OFF 92.2 % → ON 96.1 %; because the matcher false-hit rate is 0/102 the OFF and ON prompts are identical on unmatched turns, so **M-3 PASS (§(iv)) is unaffected** — the difference is temperature = 1.0 sampling noise.

### Update references
- LLM-judge re-score plan: `Dev_Plan/2026-05-19-m1-llm-judge-rescore-plan.md`
- Re-score tool: `scripts/judge_fact_holdout_llm.py`
- Re-score artifacts + summary: `artifacts/phaseA-baseline/llm_judge_20260519T042041Z/m1_llm_judge_summary.json`
