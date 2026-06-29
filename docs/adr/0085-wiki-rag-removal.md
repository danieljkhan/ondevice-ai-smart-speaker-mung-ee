# ADR 0085 — Wiki RAG removal

- **Status**: **Accepted** (promoted 2026-05-13 after PR-5 Option C PASS on 100-turn voice run `pr5_100_voice_option_c_20260513` and PR #97 merge to dev as `22ff20f`; see ADR 0087). Implementation originally merged via PR 4-B in Session 30.
- **Date**: 2026-05-09 (drafted at Session 29 close)
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-05-08 / -09 Session 29)
- **Superseding target**: ADR 0019 (Wiki RAG factual grounding) and ADR 0045 Layer 2 (RAG anti-hallucination threshold). Layers 1, 3, 4 of ADR 0045 (sampling / RAG-query separation / system prompt rules) are PRESERVED. Conv-memory RAG (ADR 0082) is OUT OF SCOPE — it is a separate index with a separate purpose.
- **Related**: `Dev_Plan/2026-05-08-pr4-per-language-threshold-plan.md` (v3 — archived; superseded), `docs/archived/dev-plan/2026-05-07-session28-3findings-synthesis-plan.md` v3 §3 (F27-9), `docs/archived/dev-plan/2026-05-10-pr5-100-query-pool-draft.md` (ACCEPTED 2026-05-10), `docs/runbooks/stage2-runs/2026-05-08-pr3-rerun/2026-05-08-pr3-rerun-report.md`, `docs/runbooks/stage2-runs/2026-05-08-pr4-unrelated-probe/unrelated_probe.json`, `assets/prompts/persona.md`, `core/rag_retriever.py`, ADR 0073 (Gemma 4 promotion).

## Context

The Mungi runtime carries a wiki-content RAG path (`RAGRetriever` in `core/rag_retriever.py`, originally specified in ADR 0019, hardened by ADR 0045 Layer 2, and rebuilt as bilingual KO+EN under PR 0 in Session 28). Throughout Sessions 22-28, Stage-2 measurements consistently reported **0 RAG hits at the production-default 0.70 threshold** (38 of 38 turns in Session 28's PR 3 rerun: `rag_miss_reason="threshold_miss"`). PR 4 (Session 29) attempted to recalibrate the threshold to KO=0.45 + EN=0.50 with a Codex Plan Gate 1 cycle (2 rounds, APPROVE WITH NOTES) and a Codex implementation that produced 1149 passing tests + 81.20% coverage. The mandatory unrelated-query probe gate then **failed unambiguously** on Jetson (KO p95=0.5228 vs limit 0.40, EN p95=0.5767 vs limit 0.45, 9 of 10 ADR 0045 v3 contamination-trigger proxies reproducing at the new thresholds).

Per-query analysis showed the embedding model has broad semantic recall over the children's wiki corpus. Many "unrelated" queries had legitimate semantic neighbors: `"이 노래 제목 뭐야"` matched a music article at 0.5228; `"tell me about cinderella"` matched the closely-related `"Ella Enchanted"` article at 0.6029. The matches are not "RAG contamination" in the ADR 0045 v3 sense — they are the embedding model functioning normally over a children's-wiki corpus. The implication: **Layer 2 threshold cannot reliably separate "topical" from "off-topical" matches under `koen-e5-tiny` + child-wiki**, so Layer 2 single-knob calibration is the wrong intervention.

Concurrently, the LLM backend has been promoted to Gemma 4 E2B Q5_K_M (ADR 0073) — a 2026-era 2-3B-parameter model with substantially stronger factual baseline than the Qwen3.5-2B-DPO model that was active when ADR 0019 (Wiki RAG factual grounding) and ADR 0045 v3 (RAG anti-hallucination threshold) were authored. PR 3's measured behavior — 38/38 turn success WITH 0 RAG context, including age-appropriate handling of cosmology / drowning safety / hypothetical reasoning prompts that fall in 8–10세 territory — demonstrates that **Gemma 4 + persona deference rules + ADR 0045 layers 1+3+4 are jointly sufficient for safe child conversation without wiki RAG context**.

This ADR proposes formally retiring the wiki RAG path. Conv-memory RAG (ADR 0082) is preserved.

## Problem

The wiki RAG path consumes ~300 MB of resident memory on a Jetson Orin Nano 8 GB device that is already at the LLM-resident memory budget boundary (G1 5,500 MB / G2a 6,000 MB invariants — see ADR 0076). It currently provides **0 hits** at production-default threshold and cannot be safely lowered (probe evidence). Maintaining it imposes:

1. **Memory cost**: KO FAISS ~50 MB + EN FAISS ~53 MB + sentence-transformers + torch ~150 MB + chunk metadata ~50 MB ≈ ~300 MB resident.
2. **Latency cost**: per-turn RAG retrieval (embedding + FAISS search) adds to TTFT.
3. **Hallucination attack surface (paradoxically)**: ADR 0045 v3 was authored after observing that lowered thresholds (0.5) caused cross-domain RAG contamination (zip-line, Cinderella). Wiki RAG is a contamination *source* whose risk has to be actively mitigated by the threshold; eliminating the path eliminates the source.
4. **Operational maintenance**: bilingual rebuilds, threshold calibration cycles, probe gates, embedder model upgrades — all are sunk costs against a path that already does nothing in production.

## Decision (implementation merged; pending PR 5 validation)

Retire the wiki RAG path:

1. **`RAGRetriever` removal**: clean-delete `core/rag_retriever.py`. No conv-memory RAG implementation exists in `core/` as of 2026-05-10, so there is no conv-memory code path to preserve inside the deleted module. Future conv-memory RAG work (ADR 0082) must land as a separate implementation.
2. **Pipeline injection path**: `core/pipeline.py` no longer constructs / injects a `RAGRetriever` for wiki retrieval; the `rag_context` system message slot is removed.
3. **Asset retirement**: `assets/rag/wiki_chunks.jsonl`, `wiki_metadata.jsonl`, `wiki_faiss.index`, `wiki_embeddings.npy`, `simple_wiki_*.jsonl`, `simple_wiki_*.npy`, `simple_wiki_*.index`, tracked wiki build artifacts, and the `assets/rag/koen-e5-tiny/` embedder bundle are removed from the source tree. Matching Jetson assets are deleted during the PR 4-B deploy gate after backup.
4. **ADR retirement**: ADR 0019 (Wiki RAG factual grounding) → Status: Superseded by ADR 0085. ADR 0045 (RAG anti-hallucination) → Status: Accepted (updated; Layer 2 retired in favor of ADR 0085; Layers 1, 3, 4 remain).
5. **Persona update**: minor — `assets/prompts/persona.md` already directs Gemma 4 to defer on precise factual questions ("정확한 수치가 필요한 정보는 정말 확실할 때만 말한다") and to use graceful fallbacks ("잘 모르겠는데", "엄마아빠한테 같이 물어보자"). No change required.
6. **Tests**: remove `tests/test_rag_bilingual.py` wiki-specific tests; preserve conv-memory RAG tests (reorganize fixtures if shared).
7. **Validation gate**: PR 5 Stage-2 rerun on Jetson with the implementing PR merged. Pool = 38 (existing PR 3 pool) + 100 (per `docs/archived/dev-plan/2026-05-10-pr5-100-query-pool-draft.md` ACCEPTED 2026-05-10). Configuration A2 production-default (LLM-resident; no wiki RAG). Pass condition:
   - core_success ≥ 95% (≥ 131 of 138 turns).
   - Hallucination 0 (orchestrator spot-check 5 KO + 5 EN random hit responses).
   - Persona violation 0 (존댓말, 60자 초과, 이모지, mixed-language artifacts).
   - Safety violation 0.
   - G2a ram_used_mb peak < 6,000 MB (with F28-1 standalone fix already merged).

### Decision scope — what is in / out

**IN scope**:
- Wiki RAG read path (`RAGRetriever` wiki branch in `core/rag_retriever.py`).
- Bilingual KO + EN wiki indices (PR 0 artifacts on Jetson and source tree).
- Wiki-RAG-specific tests in `tests/test_rag_bilingual.py`.
- ADR 0019 supersedence + ADR 0045 Layer 2 retirement.
- PR 5 validation against the expanded 138-turn pool.
- The `koen-e5-tiny` embedder bundle on Jetson (`/opt/mungi-repo/assets/rag/koen-e5-tiny/` + F27-6 symlink target) and source tree (`assets/rag/koen-e5-tiny/` 11 tracked files). Corrected from v1 ADR 0085 draft assumption: conv-memory RAG (ADR 0082) has no implementation in `core/` as of 2026-05-10 (verified via `grep -l "conversation_memory|koen-e5-tiny" core/` returning only `rag_retriever.py` which itself is being deleted). No current consumer exists. Future conv-memory RAG implementation per ADR 0082 will install its own embedder bundle as part of that work.

**OUT of scope** (preserved):
- Conv-memory RAG (ADR 0082) — separate index, separate embedder lifecycle, separate purpose (conversational continuity).
- ADR 0045 Layers 1, 3, 4 (sampling / RAG-query separation / system prompt rules).

## Alternatives considered

1. **Keep wiki RAG; recalibrate thresholds (PR 4 path)** — REJECTED. Codex round-1 + 2 plan-gate review approved the calibration path WITH NOTES, but the mandatory probe gate failed on Jetson (KO p95=0.52, EN p95=0.58 vs limits 0.40, 0.45). Per-query analysis indicated the embedding model has broad semantic recall over the wiki corpus, so Layer 2 single-knob calibration cannot reliably separate "topical" from "off-topical" without rejecting all useful retrieval. The PR 4 archive (`feature/pr4-per-language-threshold`) preserves the implementation for future reuse if the embedding model or wiki corpus changes.
2. **Switch to a stronger embedding model** (e.g., `multilingual-e5-base`, `gte-multilingual-base`) — REJECTED for the immediate next step. Memory cost (768-dim, ~1.1 GB) is prohibitive on Jetson Orin Nano 8 GB with LLM-resident; the `koen-e5-tiny` baseline is already near-minimum for memory. A future PR may revisit this if Mungi's hardware target changes (e.g., Orin Nano 16 GB or NX).
3. **Flag-gate the wiki RAG path** (default disabled, opt-in via `MUNGI_WIKI_RAG=enabled`) — DEFERRED. Smaller blast radius for the PR but leaves dead-code surface. Decision: prefer clean removal (option above) for clarity; the PR 4 archive branch already preserves the code if future reuse is needed.
4. **Replace wiki RAG with curated topic-keyed lookup** (small JSON map of high-frequency child topics → fact snippets) — DEFERRED. Could augment Gemma 4's answers for specific factual queries (planet distances, dinosaur sizes) without an embedding-based retrieval. If PR 5 validation shows specific factual gaps, consider as a follow-up ADR.

## Consequences (implementation merged — awaiting PR 5 validation)

### Positive

- **~300 MB resident memory freed** on Jetson Orin Nano 8 GB (KO FAISS ~50 + EN FAISS ~53 + embedder ~150 + chunks ~50). Combined with F28-1 (G2a measurement-window fix), PR 5 should report cleanly under G2a 6,000 MB and likely under G1 5,500 MB invariant.
- **Lower per-turn TTFT** (no embedding + FAISS retrieval step).
- **Hallucination attack surface reduced**: removing the wiki RAG path eliminates the contamination source that ADR 0045 v3 had to actively mitigate.
- **Operational simplification**: no more wiki rebuild cycles, threshold calibration cycles, probe gates, embedder model upgrades for the wiki path.
- **Persona-aligned behavior**: persona.md already directs deference on precise factual queries; Gemma 4's "Moong-ee doesn't know that" fallback is already validated in PR 3.
- **Code surface reduction**: ~500–800 lines deletable across `core/rag_retriever.py`, `tests/test_rag_bilingual.py`, and related test fixtures. Codebase becomes simpler to maintain.

### Negative / trade-offs

- **Loss of factual grounding for niche queries** that Gemma 4 may not know directly (e.g., very specific Korean cultural / historical / school-curriculum facts). Mitigation: persona deference rule + parent-recommendation fallback ("엄마아빠한테 같이 물어보자"). PR 5 138-turn pool with 100 new 5–15세 reinforcement queries (10 categories × 10 each, 53 KO + 47 EN) explicitly tests this.
- **ADR 0019 retirement** is a meaningful documentation pivot — readers of the historical ADR record need clear cross-reference (this ADR + handoff trail).
- **PR 0 work (bilingual rebuild) becomes archival**: The KO+EN bilingual indices built in Session 28 are no longer used. The work is not wasted — the PR 0 lineage (refined article inputs, deterministic chunker, ADR 0085 evidence) feeds future RAG re-introduction studies if Mungi's hardware or LLM target evolves.
- **Conv-memory RAG embedder dependency**: ADR 0082 conv-memory RAG is plan-level only and has no implementation in `core/` as of 2026-05-10 (verified via grep). The wiki RAG path is the only consumer of the `koen-e5-tiny` embedder bundle in the current codebase, so PR 4-B can remove the bundle without affecting any active code. When conv-memory RAG is later implemented per ADR 0082, the embedder bundle will be re-installed as part of that work.
- **Reverting RAG removal** would require resurrecting the PR 4 archive branch, redeploying assets, and reverting ADR 0019 + ADR 0085. Not impossible, but a meaningful fall-back cost.

### Unknown / to be measured in PR 5 validation

- Whether removing the wiki RAG path improves or unchanges TTFT measurably (current Session 27 baseline TTFT under RAG-with-zero-hits is 22,721 ms / 22.7 s; the RAG retrieval cost is small relative to total).
- Whether PR 5's 100 new 5–15세 reinforcement queries reveal a class of factual queries where Gemma 4's knowledge gaps + persona deference is insufficient (degraded response quality).
- Whether the PR 4 archive branch's F28-1 fix needs adjustments before standalone cherry-pick (no expected dependencies, but needs verification).

## PR 4-B implementation requirements

The approved PR 4-B implementation plan had to:

1. Specify the exact `core/rag_retriever.py` deletion / restructuring strategy (clean removal vs split into wiki + conv-memory modules).
2. Specify the `core/pipeline.py` injection-path changes (remove `rag_retriever` parameter? keep for conv-memory only? rename for clarity?).
3. Audit and remove or refactor wiki-RAG-specific tests; preserve conv-memory RAG tests.
4. Produce the exact ADR diff for ADR 0019 (Superseded) and ADR 0045 (Layer 2 retired).
5. Specify the Jetson asset cleanup procedure (`/opt/mungi-repo/assets/rag/{wiki_*,simple_wiki_*}` deletion; backup-first).
6. Decide whether `scripts/probe_unrelated_rag.py` is retained as a future tool, retired, or repurposed.
7. Use the accepted 100-query PR 5 reinforcement pool (`docs/archived/dev-plan/2026-05-10-pr5-100-query-pool-draft.md`): 10 categories × 10 each, 53 KO + 47 EN.
8. Define the rollback story (revert PR 4-B; restore PR 4 archive branch artifacts; revert ADR 0085).
9. Submit to a Codex reviewer cycle before user final-plan approval (CLAUDE.md §1 Gate 1).

## Validation criteria (to be satisfied before flipping this ADR's Status to Accepted)

- Plan document exists, completed the Codex reviewer cycle, and carries user final approval.
- PR 4-B implementing PR is merged into `dev`.
- PR 5 Stage-2 rerun passes:
  - core_success ≥ 95% (≥ 131 of 138).
  - Hallucination 0 in spot-check (5 KO + 5 EN).
  - Persona / safety violations 0.
  - G2a ram_used_mb peak < 6,000 MB (preferably < 5,500 MB).
- ADR 0019 Status updated to "Superseded by ADR 0085" with cross-link.
- ADR 0045 Status updated to "Accepted (updated 2026-MM-DD; Layer 2 retired)" with cross-link.

Until PR 5 validation is satisfied, this ADR remains in the PR-4-B-merged intermediate state and is not yet Accepted.

## Implementation owner + timeline

- **Plan Gate 1**: completed before PR 4-B implementation.
- **Codex impl**: PR 4-B clean-removal implementation.
- **Jetson redeploy + PR 5 measurement**: post-PR-4-B merge under the deployment approval gate.
- **ADR 0085 promotion to Accepted**: post-PR 5 validation, ~next-session close handoff.
