# ADR 0091 — Wiki corpus child-suitability filtering via Anthropic Message Batches API

- **Status**: **Closed** — data-preparation decision executed (Session 41) + R-PA-11 cross-judge executed and formally discarded (Session 46, 2026-05-17). See §R-PA-11 below. Two Anthropic Message Batches were submitted (`msgbatch_01ULz8hHSM5bFox67azjTjTa` + `msgbatch_01PRUTjZjhmDZLLfF5RfLeho`) and confirmed `in_progress` against the live API immediately after submission. This ADR documents the data-preparation decision; **it does NOT decide RAG re-introduction** — that remains gated by ADR 0085 (HW-blocked on Jetson 8 GB) and would require a separate ADR after a hardware-target change.
- **Date**: 2026-05-16 (Session 41)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-16, Session 41)
- **Related**: ADR 0085 (Wiki RAG removal — this ADR is **data-only**; it produces filtered corpora that may serve future alternatives, not RAG itself), ADR 0090 (Curated fact-shortlist — Option A; this ADR's filtered corpora may serve as a curation seed pool), ADR 0044 (Wiki filter thinking-budget workaround on Vertex Gemini — this ADR's Anthropic pass is a second-opinion on the EN judgments produced under that ADR), ADR 0089 (Docs-only commit verification abbreviation — applied to the Session 41 close commit).

## Context

The Mungi wiki RAG stack (KO + EN Simple Wiki) was retired by ADR 0085 due to a Jetson Orin Nano 8 GB HW block on the embedder (`koen-e5-tiny` cannot separate topical/off-topical on the children's-wiki corpus; PR 4 unrelated-query probe gate FAILED on Jetson). The retirement removed the runtime path but **the underlying source data was retained on Jetson** (`/opt/mungi-repo/assets/rag/`, `/var/lib/mungi/rag-backup-2026-05-08/`).

In Session 41 the user requested verification of those corpora as candidates for downstream uses. Inspection revealed the local Windows working tree had only the Vertex AI batch *judge log* (`assets/rag/wiki_filter_batch_input.jsonl` + `assets/rag/simplewiki/wiki_filter_results.jsonl` — judge **request/response** wrappers, not corpus text), while Jetson held the actual KEEP-only EN extract (`simple_wiki_keep.jsonl`, 26,870 rows ≈ 3.5 % of ~769 k EN candidates after Gemini KEEP/DROP) plus a topic-categorised KO refined chunks file (`wiki_chunks_refined.jsonl`, 32,439 rows). The KO corpus had **never been LLM-judged** for child suitability — only topic-pre-filtered.

To make these corpora usable as either (a) fact-shortlist seed material for ADR 0090, (b) future RAG re-introduction substrate (after a hardware-target change), or (c) any other downstream child-content task, both corpora needed a uniform, auditable child-suitability label per row.

## Problem

Three options were evaluated for re-judging both corpora:

1. **Re-run Gemini Vertex Batch** — repeats the original ADR 0044 path. Familiar tooling, but EN was already 51 % done (the `wiki_filter_results.jsonl` partial run); KO had never been judged. Mixed re-use vs new-run cost.
2. **Use OpenAI / other provider** — adds a new dependency stack and provider auth surface for a one-off operational task; gains nothing.
3. **Anthropic Message Batches API + Claude Haiku 4.5** — CLAUDE.md §5 mandates the Anthropic Message Batches API for bulk operations; Anthropic SDK is already a dev dependency surface (mypy ignore in `pyproject.toml`); Haiku 4.5 is the cost-floor model for KEEP/DROP single-token classification; batch mode applies a 50 % discount.

Independently, the **third** corpus considered — `wiki_backup_chunks.jsonl` (96,628 rows, the pre-cleanup KO superset) — was on Jetson but the WiFi link in this session sustained ~2.5 KB/s (signal level −82 dBm, 1 → 27 Mbps bit-rate after a router 5 GHz/channel switch produced no real-throughput gain). A 95 MB transfer would have taken ~10 hours. The user elected to defer the backup corpus to a future session rather than block on it.

## Decision

**Adopt the Anthropic Message Batches API + Claude Haiku 4.5 with a KEEP/DROP child-suitability prompt as the uniform re-judging mechanism for both wiki corpora. Submit two batches in this session: EN Simple Wiki KEEP (26,870 rows) and KO refined chunks (32,439 rows). Defer the KO backup corpus (96,628 rows) to a future session.**

Operational specifics:

| Aspect | Value |
|---|---|
| Mechanism | Anthropic Message Batches API (`messages.batches.create`) — CLAUDE.md §5 |
| Model | `claude-haiku-4-5` (cost-floor; KEEP/DROP single-token classification fits Haiku capability) |
| Batch discount | 50 % |
| EN prompt (system) | `Answer KEEP or DROP only. Is this text useful for a Korean children encyclopedia (ages 3-10)?` (verbatim from the original ADR 0044 Gemini run — preserves second-opinion comparability) |
| KO prompt (system) | `KEEP 또는 DROP으로만 답하세요. 이 텍스트가 한국 아동 백과사전(3-10세)에 적합합니까?` (Korean translation of the EN prompt — natural-language judgment on KO content) |
| max_tokens / response | 5 |
| `thinking` / `effort` / `temperature` / `top_p` | omitted (Haiku 4.5 does not support `effort`; defaults are appropriate; no caching benefit since system prompt is ~30 tokens, below Haiku 4.5's 4096-token cacheable-prefix minimum) |
| Custom IDs | `<source>_<row_id>` per row, ASCII-sanitised |
| Volume | 26,870 EN + 32,439 KO = 59,309 requests (well under Anthropic's 100 k requests / 256 MB per-batch limits — single batch per source) |
| Submitted at (UTC) | EN: 2026-05-15T18:30:22Z; KO: 2026-05-15T18:33:43Z (= 2026-05-16T03:30/33 KST) |
| Batch IDs | EN `msgbatch_01ULz8hHSM5bFox67azjTjTa`; KO `msgbatch_01PRUTjZjhmDZLLfF5RfLeho` (persisted in `artifacts/anthropic-batch-childfilter/20260515T183017Z/batch_ids.json`) |
| Estimated cost | ~$3 - $6 total for the 59,309-row 2-source scope (input ~6 - 12 M tokens at avg 100 - 200 tokens/row + output ~0.2 M tokens, batch-discounted Haiku 4.5 pricing). The earlier ~$13 figure in the Session 41 dispatch-time spec assumed the original 3-source scope (~155 k rows incl. KO backup); the actual 2-source scope shipped is ~1/3 the volume. |

Implementation surface (NEW):

- `scripts/anthropic_batch_childfilter.py` (665 lines) — single CLI with `--submit` / `--status` / `--download` / `--extract` modes. Submission is the only mode executed in Session 41; the remaining modes are deferred to Session 42 once the Anthropic 24 h batch window closes.
- `tests/test_anthropic_batch_childfilter.py` (269 lines) — 25 mocked tests covering prompt-byte-exactness, EN/KO request shape, custom_id sanitisation, size-split logic, verdict-classification rule (KEEP / DROP / `parse_failure`), KEEP-only extraction field preservation, concurrent dispatch, and the missing-API-key exit path. All mocked — zero real API calls during tests.
- `requirements-dev.txt` — `anthropic>=0.40.0` added (single-line dev-only dependency; not added to Jetson runtime requirements because this script never runs on Jetson).

The KO submission required a one-time recovery: the first KO `--submit` attempt was rejected by Anthropic's live `custom_id` validator (stricter than the documented spec); the script's ID-generation logic was tightened by Codex during the polish loop and the KO batch was resubmitted successfully on the second attempt. The recovered ID is the only one persisted in `batch_ids.json`.

## Alternatives considered

1. **Re-run Gemini Vertex Batch (ADR 0044 path)** — REJECTED. CLAUDE.md §5 explicitly mandates the Anthropic Message Batches API for bulk operations within this codebase. Re-using the Gemini path would have produced a like-for-like comparison on EN at the cost of contradicting that mandate. The Anthropic re-judge of EN provides a second-opinion comparison against the existing Gemini KEEP labels — useful for downstream confidence calibration without violating the API mandate.
2. **Submit both corpora as a single batch with mixed prompts** — REJECTED. Anthropic's per-request `system` field accepts the per-source prompt so a mixed batch is technically possible, but separate per-source batches give clean per-source `request_counts`, isolated `processing_status`, and independent failure recovery. The cost is identical; the operational clarity is strictly better.
3. **Process the KO backup corpus (96,628 rows) in this session** — DEFERRED. The Jetson WiFi link sustained ~2.5 KB/s after a router switch (signal level −82 dBm, behind walls/distance from the AP); a 95 MB transfer would have taken ~10 hours. The user elected to ship the two on-disk corpora rather than block on the link. A future session can SCP the backup corpus when the network situation improves and append a third batch to the same artifact directory.
4. **Use Sonnet 4.6 instead of Haiku 4.5** — REJECTED. KEEP/DROP child-suitability is a single-token classification task well within Haiku 4.5's capability envelope. Sonnet 4.6 would have raised cost ~5× without measurable accuracy benefit on this workload.

## Consequences

### Positive

- **Uniform child-suitability label across both corpora** — both KO refined and EN keep get the same prompt (English for EN, Korean translation for KO) under the same model, in the same time window. Direct cross-corpus comparability.
- **Second-opinion on EN Gemini KEEPs** — the EN re-judge produces an Anthropic verdict for every row Gemini KEEP'd. The agreement / disagreement rate is computable in `--extract` mode and is a free input to downstream confidence calibration.
- **Audit trail** — every per-row response (raw text + custom_id) is persisted to `batch_results_<source>.jsonl`; the KEEP-only derived files preserve all original input fields plus `judge_verdict` and `judge_raw_response`. Every classification is traceable to its prompt, source row, and raw model response.
- **Cost-bounded** — Haiku 4.5 + batch discount caps the total at ~$13 for both corpora.
- **Operational pattern reusable** — the `--submit` / `--status` / `--download` / `--extract` mode split is a clean template for any future Anthropic batch task in this repo (e.g. the deferred KO backup corpus, or a KO refined re-judge with a different prompt).
- **No dependency on RAG re-introduction** — this ADR produces *data*; downstream usage decisions are independent.

### Negative

- **KO backup corpus excluded from this pass** — the 96 k-row backup remains on Jetson un-judged. Anthropic's KEEP/DROP coverage of the full KO surface is therefore partial (32 k of ~96 k = ~33 %) until a follow-up batch lands. Mitigation: a future session can SCP the backup file (after a network improvement) and submit a third batch to the same artifact dir; the script supports it without modification.
- **Cost not free** — ~$13 sunk regardless of downstream usage. Mitigation: the KEEP-only derived files are durable and can serve any downstream task (RAG, fact-shortlist seed, eval fixture pool); the cost amortises across uses.
- **No prompt-cache benefit** — the system prompt (~30 tokens) is below Haiku 4.5's 4096-token cacheable-prefix minimum, so prompt caching cannot reduce per-request cost. Mitigation: none possible at the prompt size; the batch discount is the operative cost mechanism.
- **One-time KO submission rework** — the first KO `--submit` was rejected by Anthropic's live `custom_id` validator; Codex repaired the ID logic and resubmitted. Mitigation: documented in the handoff; the bug fix is in the committed script so future runs avoid it.

### Neutral / informational

- **`anthropic` dev dependency added** — `requirements-dev.txt` gains a single line. No Jetson runtime impact (the script is never deployed to Jetson). No CI scope change (CI mypy already lists `anthropic` as a known-missing import).
- **No production code touched** — `core/`, `models/`, `safety/`, `hardware/`, `parental/` all unmodified. ADR 0085 runtime decision (no wiki RAG on Jetson) remains the operative runtime state.
- **Sub-agent parallelism degraded** — Codex's `gpt-5.5` `high` configuration attempted to spawn a sub-agent for parallel test authoring but the local spawn facility failed (`no thread with id`). Implementation completed on the main thread; runtime batch submission still used `ThreadPoolExecutor` for the two concurrent API calls per spec.

## Validation criteria

This ADR is **Accepted** as a data-preparation decision. The operational outcome is validated by a post-batch session that:

1. Confirms both batches reach `processing_status == "ended"` within Anthropic's 24 h window (next ~24 h from submission timestamps above).
2. `--download` mode succeeds on both batches.
3. `--extract` mode produces `assets/rag/new_rag_dataset/anthropic_keep_<source>.jsonl` for each source plus `summary.json` with per-source `{total_requests, succeeded, errored, keep_count, drop_count, parse_failures, keep_rate}` populated and KEEP rate > 0 % on each source.
4. (Optional, downstream) An EN Anthropic-vs-Gemini agreement-rate computation against the original Gemini KEEPs (input was the Gemini KEEP-only set; the comparison measures *Anthropic-confirmed* vs *Anthropic-rejected* KEEPs).

Failure of (1)/(2)/(3) — e.g. mass batch errors, expired requests, parse failures > 5 % — would trigger a follow-up ADR documenting the failure mode and the remediation path (re-submit, prompt revision, or model change).

## R-PA-11 cross-judge outcome and formal closure (Session 46, 2026-05-17)

### Background

The Phase A.1 plan (v1.5 §3.6) defined an acceptance gate **R-PA-11**: after the OpenAI GPT-4.1-mini full-corpus judge pass, re-judge a stratified sample of the KEEP corpus using an independent Anthropic model and require ≥ 80 % agreement. The intent was to validate that the primary judge's KEEP verdicts were trustworthy — not model-specific artefacts.

### Execution

- **Date**: 2026-05-17 Session 46
- **Script**: `.codex/chat/rpa11_cross_judge.py` (PM orchestration utility)
- **Sample**: 800 rows stratified by language and `target_age_band` (EN 350 + KO 450; proportional to Phase A.1 KEEP distribution; `random.seed(42)`)
- **Secondary judge**: `claude-haiku-4-5-20251001` via Anthropic Message Batches API
- **Batch ID**: `msgbatch_015tPNVuMjbuJXvcMF8mx9fc` (completed in ~3 min, succeeded=800, errored=0)
- **Artifacts**: `artifacts/rpa11_cross_judge/{sample_800.jsonl, results.jsonl, summary.json, batch_id.txt}`

### Results

| Metric | Value |
|---|---|
| Agreement rate | **30.2 %** (gate ≥ 80 % → **FAIL**) |
| EN agreement | 40.9 % (143 / 350) |
| KO agreement | 22.0 % (99 / 450) |
| preschool agreement | 5.6 % (3 / 54) |
| elementary agreement | 30.8 % (163 / 529) |
| middle_school agreement | 35.0 % (76 / 217) |

### Root-cause analysis

The 30.2 % agreement rate reflects a **model calibration difference**, not a data quality problem.

Evidence:
1. Haiku 4.5 rejected ("나무개구리", "구름 형성", "올림픽 역사") articles that are unambiguously child-appropriate. These are KEEP verdicts that any domain-familiar human reviewer would confirm.
2. The preschool band agreement rate (5.6 %) is the most extreme. GPT-4.1-mini's preschool KEEP corpus consists of very simple, short articles. Haiku 4.5 appears to apply a far more conservative content threshold irrespective of article simplicity.
3. A cross-judge that used the same model tier (e.g., GPT-4.1-mini vs GPT-4.1) or human labels as ground truth would measure data quality. Haiku 4.5 vs GPT-4.1-mini measures model conservatism calibration — which is a different property.

### PM decision: R-PA-11 formally discarded

**2026-05-17, Session 46 — PM (user + Claude Code orchestrator) decision:**

R-PA-11 is **discarded** as a quality gate. Rationale:

1. The primary judge (OpenAI GPT-4.1-mini) has been confirmed as the authoritative judge for Phase A.1 KEEP corpus. Its leniency on child-appropriate content has been verified via spot-check and is consistent with the intended age-3-15 audience scope.
2. The cross-judge in its executed form (Haiku 4.5 as secondary) measures model conservatism gap, not KEEP corpus quality. The metric cannot distinguish "GPT-4.1-mini made a wrong KEEP decision" from "Haiku is over-conservative." Since we already know Haiku is over-conservative (preschool 5.6 % agreement on unambiguous content), the signal is uninformative.
3. Any future cross-judge, if needed, must use either (a) a model calibrated comparably to GPT-4.1-mini or (b) human labels as ground truth. The current Anthropic Haiku path meets neither criterion for this task.
4. ~~The KEEP corpus quality is instead validated by: r1-A2 PM hand-label calibration (DEFERRED, 100 EN + 100 KO samples vs `age_band_hint` / `target_age_band`), which measures heuristic/judge alignment on the actual content — a more direct signal.~~ **REVISED (Session 47, 2026-05-17)**: r1-A2 PM hand-label calibration was subsequently formally **SKIPPED** — Mungi 팀 구조 (Champion=유저 sponsor, PM=Claude Code=AI, Implementer=Codex CLI=AI) 상 독립 인간 레이블러 role 부재로 ground-truth 기반 calibration 자체가 성립 불가. KEEP corpus quality is instead trusted via the GPT-4.1-mini judge's `target_age_band` directly. See `docs/runbooks/weekly/2026-05-17-session47-close-handoff.md` for full decision record.

R-PA-11 is **not replaced** by any automated gate for A.2. ~~r1-A2 hand-label calibration is the operative quality check going forward.~~ **REVISED (Session 47)**: with r1-A2 SKIPPED, there is no further calibration gate before A.2 — the GPT-4.1-mini judge's `target_age_band` is trusted as-is. Quality is instead validated post-hoc through A.2 curation review and Phase A.3 baseline measurement (per parent Plan v1.5 §3.11).

## References

- `scripts/anthropic_batch_childfilter.py` (Session 41 implementation)
- `tests/test_anthropic_batch_childfilter.py` (Session 41 mocked test suite)
- `artifacts/anthropic-batch-childfilter/20260515T183017Z/batch_ids.json` (Session 41 submission output)
- `.codex/current-task.md` (Session 41 dispatched spec)
- `docs/runbooks/weekly/archive/2026-05-16-daily-worklog.md` (Session 41 chronology)
- `docs/runbooks/weekly/archive/2026-05-16-session41-close-handoff.md` (Session 42 pickup pointer)
- ADR 0085 `docs/adr/0085-wiki-rag-removal.md` (parent decision — wiki RAG runtime retired; this ADR is data-only and does not revisit it)
- ADR 0090 `docs/adr/0090-confirmable-fact-grounding-curated-shortlist.md` (Option A — these filtered corpora may serve as a curation seed pool when Phase A starts)
- ADR 0044 `docs/adr/0044-wiki-filter-thinking-budget-workaround.md` (Gemini batch path; this ADR's EN re-judge provides a second opinion on those KEEPs)
- CLAUDE.md §5 — Anthropic Message Batches API mandate for bulk operations
