# ADR 0110: Conversation Memory — Nightly On-Device Batch + Daytime Encoder-Free Keyword Recall

## Status

Accepted (2026-06-18)

(Moved to Accepted on 2026-06-18 after passing **G2** — unit tests green, nightly job
dry-run on synthetic + real-sample fixtures with 0 errors, and on-device validation: one real
nightly run over that day's conversations + next-day smoke (gated first-turn injection + keyword
recall mid-session) with 0 runtime errors and the memory guard never crossed. Plan:
`Dev_Plan/2026-06-12-conversation-memory-nightly-plan.md` v4, converged through Codex 3-round
review — r1 PUSH BACK → r2 PUSH BACK → r3 APPROVE WITH NOTES; user final approval granted
2026-06-12; G0 pre-implementation verification ALL PASS 2026-06-12.)

**Implementation status (see Update 2026-06-18 at the end):** the day/night paths, the
explicit-recall intercept, and the v0 scope refinements are merged and deployed to the device. The
live-session smoke PASSED on 2026-06-18 — a real spoken query ("내 이름 뭐라고 했어?") returned the
child's stored name verbatim ("종경") with 0 recall-path runtime errors — so this ADR is **Accepted**
for the v0 scope (first-turn day-summary injection remains a v1 feature).

- **Date**: 2026-06-12
- **Decision owner**: Claude Code (primary orchestrator) + user direction (2026-06-12: Option A
  confirmed; tiered retention 5 years / 90 days; time-aware recall)
- **Supersedes (partially)**: ADR 0082 (daytime-vector/FAISS requirements — see Supersession
  section), with a status note owed to ADR 0027 (see Supersession section)
- **Related**: ADR 0085 (wiki RAG removal — retired the index ADR 0082 was paired against),
  ADR 0090 (fact-shortlist — the deterministic-injection precedent the day path follows),
  ADR 0045 (RAG anti-hallucination gating precedent), ADR 0102 (E4B primary LLM)

## Context

Cross-session conversation memory ("the child said X yesterday") is the last declared-but-unbuilt
pipeline layer. Two prior ADRs specified it as a daytime vector-RAG stack:

- **ADR 0027** (2026-03-27): per-turn FAISS retrieval (`IndexFlatL2` + `IndexIDMap`), embeddings
  shared with the wiki RAG, post-session incremental indexing, 30-day retention.
- **ADR 0082** (2026-04-29, plan-level): `koen-e5-tiny` ONNX embedding + a separate
  `conversation_memory.faiss` index queried at conversation time, ≤100-token prompt budget.

Neither was ever implemented: grep-verified (G0/plan §3.0), **no** `faiss` / `koen-e5` /
`conversation_memory` runtime code exists; ADR 0082 is plan-level only. Meanwhile the runtime
landscape changed materially:

1. **Daytime RAM is the binding constraint.** The live conversation runtime (demo_live kiosk path)
   has been observed at up to **5.1 GB RSS over 2.2 h** against a 6000 MB guard (Session 95).
   A resident daytime encoder + FAISS index has no safe headroom; the same embedder also failed
   the topical-separation gate that retired wiki RAG (ADR 0085) — daytime vector recall is both
   memory-hostile and quality-unproven on this device.
2. **The wiki FAISS index ADR 0082 was "separate from" is gone** (ADR 0085), and the deterministic
   fact-shortlist (ADR 0090) established a working precedent: deterministic keyword matching +
   verbatim context injection, no embedder, measured before shipped.
3. **The device-resolved LLM config differs from old assumptions** (G0.1): the device
   `/var/lib/mungi/config/config.json` resolves to Gemma 4 **E4B Q4_K_XL** with **`n_ctx=6144`**
   (not the E2B/4096 code defaults), `max_tokens=256`, live `llm_max_tokens=80`,
   `max_history_turns=1`, fact-shortlist p2 ACTIVE.
4. **The device sleeps when the child does.** The charging-dock usage pattern (powered overnight,
   RTC attached — user-confirmed 2026-06-12) makes a nightly batch window free compute: latency
   and GPU contention are irrelevant at 03:00 KST.

The user approved **Option A** (2026-06-12): move ALL heavy work (embedding, clustering, LLM
summarization) to a nightly on-device batch, and keep the daytime conversation path
**encoder-free** — keyword recall + snippet injection only.

## Decision

Adopt **Option A — nightly on-device batch + daytime encoder-free keyword recall** as the
conversation-memory architecture, superseding the daytime-vector/FAISS design of ADR 0082/0027.
Authoritative spec: `Dev_Plan/2026-06-12-conversation-memory-nightly-plan.md` (v4).

### 1. Night path — `scripts/nightly_memory_build.py` + `systemd/mungi-memory-nightly.timer`

- Systemd timer (default 03:00 KST), `Persistent=true` with a bounded catch-up window,
  `RuntimeMaxSec` + `MemoryMax` + explicit env-file sourcing in the unit; refuses to run while a
  conversation session is active (quiescence check; defer + retry).
- **Ingest quarantine (denylist FIXED at G0.3)** — dropped at turn level: top-level
  `hotword_hallucination_detected` / `stt_script_drift_detected`, plus any TRUE among
  `metrics.{crisis_matched, parent_disclosure_matched, template_matched, belief_matched,
  content_filter_blocked, history_mode_matched, funny_english_matched, language_switch_matched,
  hotword_hallucination_detected, stt_script_drift_detected}`. **Session level: any
  crisis-flagged turn quarantines the whole session**, and the following morning's first-turn
  summary is suppressed for that day (child-safety: no distressing-memory resurfacing).
- **v0 output**: Hangul-aware normalized keyword inverted index over filtered raw turns — raw
  layer `turns.jsonl` + `index.json` with `{layer, id}` references.
- **v1 addition**: `koen-e5-tiny` ONNX **CPU** embedding of filtered turns → clustering/dedup →
  **Gemma 4 (llama.cpp) CPU-only summarization** (`n_gpu_layers=0`; stage order: encoder load →
  embed/cluster → encoder unload → LLM load → summarize → LLM unload) producing ≤3 validated
  day-summary snippets/day into the summary layer `summaries.jsonl`. Template-constrained prompt;
  output validated (script check, length cap); every snippet carries provenance
  `{session_dir, turn_refs, timestamp_range, source_hashes}` — unmappable snippets are rejected.
  Vectors persist to `vectors.npy` on disk only and are **never loaded by the day path**.
- **Idempotency/publish contract**: `session_end.json` completion sentinel (mtime-quiescence
  fallback ≥10 min); processed-session manifest keyed by SHA-256 of `conversation.jsonl` bytes
  (re-ingest on change); each run publishes a fresh `generations/<gen-id>/` tree behind an
  atomically-renamed `current` pointer file; GC keeps the last 2 generations.
- **Tiered retention (user decision)**: day-summaries **5 years** (≈2.5 MB total); raw-turn
  keyword index **90 days**. Rotation runs inside the nightly job — this closes ADR 0082's
  open unbounded-growth item.

### 2. Day path — `core/conversation_memory.py` (encoder-free)

- **Load** at session start: resolve the `current` generation pointer, load `index.json` +
  referenced layers (a few MB). No encoder, no vectors, no FAISS — a regression test asserts the
  day path never opens a vector file. Load failure = feature silently off.
- **First-turn injection (gated)**: the most recent validated benign day-summary, one snippet
  max; suppressed entirely after a quarantined (crisis) day or when no snippet validates.
- **Mid-session recall**: normalized keyword match of `user_text` against the index with recency
  weighting; threshold gate (≥2 content-bearing keyword hits, OR 1 exact child/entity hit +
  recency ≤7 days); below threshold inject nothing (ADR 0045 gating precedent).
  **Matcher spec (G0.2-validated)**: particle-tolerant matching — strip ONE trailing single-char
  particle (가/이/은/는/을/를/도/만/에/야/의/랑, stem ≥2 chars) + prefix match with
  **PREFIX_TOL=3** trailing chars; hits count DISTINCT query tokens; recall-intent expressions
  (기억나/말했잖아/얘기했/했었지 family) and temporal words are excluded from content keywords,
  with particle-tolerant exclusion.
- **Time-aware recall**: per-turn KST timestamps (second precision) and per-summary time ranges
  are preserved; a deterministic Korean time-expression parser extracts an optional window used
  **only as an AND filter** on keyword matches. Time-only recall is a QUERY-level mode: only when
  the query carries no content keywords, requires an explicit recall-intent expression, bounded
  to 3 days. Injection phrasing is child-friendly ("어제 저녁에").
- **Injection seam**: the recall block is added in `_build_messages()` immediately before the
  current user message, gated before prompt assembly on the resolved router/metric context —
  skipped whenever the turn was claimed by any fixed path (crisis, parent-notice, template,
  guide-mode, language-switch, history, funny-english).
- **Budget**: recall block hard cap **≤100 tokens** (CLAUDE.md §6 budget unchanged from
  ADR 0082/0027). Runtime trim condition (G0.1-fixed):
  `ceil(estimated_prompt_tokens × 1.15) + 256 ≤ n_ctx`; on violation trim in order —
  (a) drop history turn, (b) trim recall block, (c) omit recall.

### 3. Single rollout flag

`MUNGI_CONV_MEMORY=1` enables BOTH the daytime recall path AND the nightly builder. The systemd
unit ships installed but inert: the nightly job reads the flag from the sourced env file as its
first action and exits 0 (logged no-op) when absent/`0`. Default OFF until G2 passes — one
switch, no half-enabled states.

### 4. Staged rollout

**v0**: keyword index over filtered raw turns, no LLM summarization. **v1**: nightly clustering +
LLM day-summaries. Daytime vector recall (options B/C) is a future escalation gated on measured
keyword-recall quality — it is NOT part of this decision.

### 5. Privacy and deletion semantics

Everything on-device; no data leaves the device. Artifacts under
`/var/lib/mungi/conversation_memory/` are owned by the service user with restrictive permissions.
Deletion semantics are documented and tested to cover ALL artifacts (snippets, index, turns,
vectors, manifest) so a parent-initiated wipe of a child's data is complete.

## Key measured facts (G0 pre-verification, 2026-06-12)

Evidence: `artifacts/g0-conv-memory-20260612/G0-results.md` (scripts + reports in the same dir).

1. **Token budget (G0.1)** — exact backend-formatted prompts measured on the real E4B GGUF at the
   device-resolved `n_ctx=6144`: worst case (long turn + history1 + recall100 + fact context)
   = **3356 tokens ⇒ headroom 2532**; the ≤100-token recall block adds only **~96 formatted
   tokens** (+3.0%). Even under the old 4096 assumption the worst case leaves 484 tokens. The
   `_estimate_tokens` heuristic underestimates by up to 10.5% (max +317 tokens) — hence the
   `×1.15 + 256 ≤ n_ctx` trim condition. The trim ladder is a safety net, not an expected path.
2. **Recall quality (G0.2)** — v0 matcher probed offline against all 142 real device sessions /
   2013 turns (1800 indexed after quarantine) with 20 PM-labeled queries incl. 5 FP traps:
   final spec scores **13 hit / 0 miss / 0 false-positive / 7 true-negative (20/20)**. The four
   matcher-spec adjustments (particle strip + PREFIX_TOL=3; distinct-query-token hit counting;
   query-level time-only gating; intent-token exclusion) are REQUIRED — they are part of this
   decision, not implementation freedom.
3. **Schema audit (G0.3)** — the quarantine denylist keys above are fixed from a 5-site code
   audit + real-data audit. **4 historical `conversation.jsonl` record variants exist** (oldest
   lacks the top-level hotword/drift keys entirely) ⇒ the shared schema module implements a
   **versioned reader that treats absent flag keys as False**. Real crisis fixtures exist
   (2 sessions, 2026-06-07). Guide-mode turns are LLM turns but carry `template_matched=true`
   and stay quarantined (intentional — they carry safety-guidance content).

## Alternatives considered

1. **Option B/C — daytime vector RAG (resident or lazy-loaded encoder + FAISS), per ADR
   0082/0027** — REJECTED for v0/v1. Daytime RAM is the binding constraint: the conversation
   runtime already peaks at 5.1 GB against a 6000 MB guard, and a resident encoder + index has no
   safe headroom. The same `koen-e5-tiny` embedder also failed topical separation on-device
   (ADR 0085), so daytime semantic recall quality is unproven exactly where it would be trusted.
   Retained as a measured escalation path if keyword recall quality proves insufficient.
2. **Keyword-only daytime recall WITHOUT a nightly batch (index at session end, in-process)** —
   REJECTED. Post-session work inside the conversation process competes with the runtime memory
   guard and provides no place for LLM summarization or rotation; a separate nightly process with
   `MemoryMax`/`RuntimeMaxSec` isolates leaks from the conversation runtime entirely.
3. **No summarization (v0 forever)** — REJECTED as an end state. Raw-turn snippets cannot serve
   the 5-year retention tier (size and recall quality both degrade); day-summaries compress a day
   to ≤3 validated snippets with provenance. v0 is retained as the shipping stage and as the
   per-snippet fallback when a summary fails source mapping.
4. **Cloud-assisted summarization** — REJECTED outright: product privacy invariant (no
   conversation data leaves the device).

Note: ADR 0027 rejected keyword search ("SQLite FTS5 only") on Korean-morphology grounds. That
rejection assumed unmeasured morphology risk; G0.2 measured it on 1800 real child turns and the
particle-tolerant prefix matcher resolved it (20/20) without a morpheme dependency. The evidence,
not the preference, changed.

## Consequences

### Positive

- **Daytime RAM stays flat** (+5–20 MB estimated; no encoder, no FAISS, no vectors in the
  conversation process) — compatible with the observed 5.1 GB runtime peak.
- **Deterministic and measured** — recall is keyword-gated with a G0-validated 0-FP matcher spec;
  below threshold, silence (no wrong-memory injection UX).
- **Token budget proven before implementation** — 2532-token worst-case headroom measured on the
  real device-resolved config; the ≤100-token budget of ADR 0082 is preserved.
- **Child safety by construction** — crisis sessions quarantined whole, next-morning first-turn
  summary suppressed, fixed-path turns (crisis/parent/template/guide/etc.) never indexed and
  never recall-injected.
- **Closes ADR 0082's open items** — rotation/pruning (tiered retention in-job), index population
  pipeline (nightly builder), and the unbounded-growth risk.
- **Cheap nights** — CPU-only night LLM avoids GPU contention and the `n_gpu_layers` MemFree
  trap; latency is irrelevant at 03:00.

### Negative / trade-offs

- **Memory freshness lags one night**: today's turns are recallable only after the next nightly
  run (in-session continuity is still covered by live history).
- **Keyword recall is semantically shallower than vector recall** — paraphrase recall is weaker
  by design; mitigated by the v1 summary layer and bounded by the measured G0.2 quality bar;
  vector escalation (B/C) stays available.
- **Depends on overnight power + RTC** — a device powered off all night defers the batch
  (`Persistent=true` bounded catch-up + manifest idempotency mitigate; user-confirmed dock
  pattern).
- **Summary hallucination risk ("기억 조작")** — mitigated by template-constrained prompts,
  output validation, and per-snippet provenance with reject-on-unmappable; residual risk is
  measured at G2.
- **New runtime surface**: a systemd timer/service, a generation store, and a manifest add
  operational artifacts that must stay aligned with transcript storage (mirrors ADR 0027's
  negative, now bounded by the publish/GC contract).

## Supersession and document ownership

### What this ADR supersedes — and what survives

- **ADR 0082 (Conversation Memory RAG — koen-e5-tiny ONNX, plan-level)**: this ADR **supersedes
  its daytime-vector/FAISS requirements** — the `conversation_memory.faiss` index, daytime
  embedding inference per turn, and daytime vector search are NOT built. **What survives of
  ADR 0082**: (a) the `koen-e5-tiny` ONNX model adoption itself — repurposed to **nightly
  CPU-only clustering/dedup ONLY**, never daytime search; (b) the **≤100-token conversation-
  memory prompt budget** (unchanged, CLAUDE.md §6); (c) the source-contamination principle —
  conversation memory never mixes with factual grounding (the fact-shortlist path stays
  separate). ADR 0082's open items (rotation, population pipeline) are closed by this ADR.
  A status note pointing here is added to ADR 0082 at acceptance (originals are immutable;
  Update sections only).
- **ADR 0027 (Conversation Memory RAG via separate FAISS index)**: its runtime design (per-turn
  FAISS `IndexFlatL2` retrieval, embeddings shared with the wiki RAG, post-session in-process
  indexing, 30-day retention) is **fully superseded** — its premise of a co-resident wiki
  embedding stack was already invalidated by ADR 0085, and none of it was ever implemented.
  Its product intent (on-device, privacy-preserving recall of "what did we talk about
  yesterday?") is what this ADR delivers. A status note pointing here is added to ADR 0027 at
  acceptance.

### Document ownership — acceptance-time sweep checklist (NON-NEGOTIABLE scope)

ADR 0110 **owns all active documents matching "conversation-memory FAISS / koen-e5 / daytime
vector recall"** (plan §4 N1, PM-verified list). The sweep below executes **at ADR acceptance
(post-G2), not now**; until then the listed docs intentionally still describe the superseded
architecture:

- [ ] `CLAUDE.md` §3 (pipeline item 6: "koen-e5-tiny ONNX + separate FAISS index" → nightly
      batch + daytime keyword recall) and §6 (FAISS contamination + 100-token budget wording)
- [ ] `Dev_Plan/mungi-pipeline-master.md`
- [ ] `Dev_Plan/mungi-architecture-master.md`
- [ ] `Dev_Plan/mungi-runtime-ops-master.md`
- [ ] `Dev_Plan/mungi-safety-policy-master.md`
- [ ] `docs/runbooks/baseline-stack-and-models.md`
- [ ] `docs/PROJECT_STATUS.md` (AI pipeline diagram + model table conversation-memory row)
- [ ] `docs/runbooks/jetson-setup-guide.md` (§7-6 koen-e5-tiny bundle path — now night-only)
- [ ] `docs/adr/0082-conversation-memory-rag-koen-e5-tiny.md` — status note / Update section
      pointing to ADR 0110 (original body untouched)
- [ ] `docs/adr/0027-conversation-memory-rag.md` — status note / Update section pointing to
      ADR 0110 (original body untouched)

## Implementation (planned; Codex delegation per plan §4)

- `core/conversation_memory_schema.py` — NEW, shared night/day schema incl. the versioned
  `conversation.jsonl` reader (lands FIRST; both paths depend on it)
- `core/conversation_memory.py` — NEW, day-path store/matcher/gates/injection formatting
- `core/pipeline.py` — recall wiring in `_build_messages()`, first-turn hook,
  router/metric-context gating, session-end completion marker
- `scripts/nightly_memory_build.py` — NEW, nightly batch (quarantine, index, v1
  embed/cluster/summarize, manifest, generation publish, rotation)
- `systemd/mungi-memory-nightly.service` / `.timer` — NEW (`Persistent=true`, `RuntimeMaxSec`,
  `MemoryMax`, env sourcing, dedicated service user)
- `tests/test_conversation_memory.py`, `tests/test_nightly_memory_build.py` — NEW (incl. the
  no-daytime-vector-load regression assert, flag-off no-op for BOTH paths, static systemd
  directive tests, generation-pointer atomicity, deletion-semantics completeness)

## References

- `Dev_Plan/2026-06-12-conversation-memory-nightly-plan.md` (v4 — authoritative plan; user
  final approval 2026-06-12; G0 results integrated)
- `artifacts/g0-conv-memory-20260612/G0-results.md` (G0.1/G0.2/G0.3 evidence — ALL PASS)
- `Dev_Plan/2026-06-12-conversation-memory-discussion-r1.md`,
  `Dev_Plan/2026-06-12-conversation-memory-discussion-r2.md` (Codex mutual-discussion records;
  r3 APPROVE WITH NOTES)
- `docs/adr/0082-conversation-memory-rag-koen-e5-tiny.md` (partially superseded — see
  Supersession section)
- `docs/adr/0027-conversation-memory-rag.md` (runtime design fully superseded — see
  Supersession section)
- `docs/adr/0085-wiki-rag-removal.md` (retired the wiki index 0082 was paired against; embedder
  topical-separation failure evidence)
- `docs/adr/0090-confirmable-fact-grounding-curated-shortlist.md` (deterministic-injection
  precedent; fact-shortlist p2 coexists with recall within measured headroom)
- `docs/adr/0045-anti-hallucination-sampling-rag-gating.md` (below-threshold-silence gating
  precedent)
- `docs/adr/0102-e4b-primary-e2b-autofallback.md` (device-resolved E4B primary LLM)

## Update 2026-06-18 — Explicit-recall intercept + v0 scope refinement; deployed and on-device validated; live smoke pending

This update records what landed since the Decision, the on-device validation, and the one
remaining gate for Acceptance. The Decision body above is unchanged (ADR bodies are immutable).

### Landed (merged to `dev`)

- **Explicit-recall intercept (PR #238)** — a deterministic day-path *answer* mechanism added
  alongside the §2 injection seam. When the child directly asks the device to recall
  ("내 이름 뭐라고 했어?", "내가 뭐라고 했지?"),
  `safety/recall_query_router.py::match_recall_query` matches a whole-turn anchored intent and
  `core/pipeline.py::_conversation_memory_recall_answer` answers verbatim from the index via a
  fixed TTS response, bypassing the LLM (sibling to the crisis/parent/template/history/
  funny-english/datetime intercepts). False-positive guards reject third-party recall
  ("엄마가…"), idioms ("기억력"), and store-requests ("기억해 둬"). This is additive to — and
  distinct from — the mid-session keyword *injection* of §2, which is unchanged.
- **Runtime flag toggle (PR #240)** — `scripts/mungidev.sh` now sources
  `/var/lib/mungi/config/mungi.env` (`set -a`), the same file `mungi-memory-nightly.service`
  reads via `EnvironmentFile`. One mutable file toggles `MUNGI_CONV_MEMORY` for both the kiosk and
  the nightly job without editing tracked files.
- **systemd directive fix (PR #241)** — `mungi-memory-nightly.service` used `RuntimeMaxSec=`,
  which `Type=oneshot` ignores (flagged by `systemd-analyze verify`); corrected to
  `TimeoutStartSec=1800`.
- **v0 scope refinement (PR #242)** — two on-device-discovered quality fixes:
  1. **Interrogative filter.** Explicit recall could select a turn the child phrased as a
     *question* and echo it back ("네가 '내 이름이 뭐야?'라고 했었지!"). `recall()` now takes an
     optional `snippet_predicate`; `recall_for_intent` passes one that excludes interrogative
     snippets (`_is_interrogative`: trailing `?`/`？`, plus the ASR-dropped wh-endings
     `뭐야`/`뭐니`/`뭘까`/`뭐냐`), so a declarative statement always wins over a higher-scored
     question.
  2. **`preference` sub_kind deferred to v1.** The v0 keyword seed "좋아" cannot distinguish a
     child's *stated* preference from questions, second-person statements about the device, or
     general usage (41 "좋아" turns in the real device index, predominantly non-preference; a real
     query echoed the child's own question). Per the §4 staged rollout, reliable preference recall
     requires the v1 summary layer, so `preference` was removed from the explicit-recall router,
     seeds, the pipeline answer branch, and tests. **v0 explicit recall now ships `name` +
     `general_recall` only.** The §2 mid-session injection still surfaces preference-relevant
     context for the LLM when appropriate.

### On-device validation (real device index, `dev@df4c66e`, `MUNGI_CONV_MEMORY=1`)

- Index built over the device's real conversations: **1822 indexed turns**, generation
  `20260618T094242KST`; the nightly builder was run manually (exactly what the timer will do):
  exit 0, sub-second, `turns_indexed=1832`, `quarantined=2`, 0 errors.
- Recall validated against the real index: name recall returns the declarative `'나의 이름은 종경.'`;
  the index also holds **9 interrogative name-turns** ('…너의 이름은 뭐냐?', '…너 이름은 뭐니?', and an
  ASR-dropped-`?` '…너의 이름은 뭐야') — all correctly excluded by the filter; `general_recall`
  declines vague queries (no fabrication); FP guards hold.
- A full deployed-path conversation simulation produced the exact child-facing sentences (e.g.
  "내 이름 뭐라고 했어?" → "네가 '나의 이름은 종경.'라고 했었지!"). Tests green: the `dev` full suite
  passed 5296; the recall scope passes 365; mypy + ruff clean.

### Acceptance (G2) — PASSED 2026-06-18

The operator installed + enabled `mungi-memory-nightly.timer` (next run 03:08 KST confirmed) and
ran `systemctl restart mungi-kiosk`; the live kiosk (PID 21720) now carries `MUNGI_CONV_MEMORY=1`
and loaded the store (1822 snippets). **Live-session smoke PASSED:** a real spoken query
"내 이름 뭐라고 했어?" returned the child's stored name verbatim ("종경") in conversation, with **0
recall-path runtime errors** in `demo_live.log`. A second spoken query "어제 한 말 기억나?" correctly
declined (no indexed turns existed for the prior calendar day) rather than fabricating — the
time-window was parsed correctly (`day_offset=1`); there was simply nothing to recall. This
satisfies the G2 keyword-recall-mid-session criterion for the v0 scope, so this ADR is **Accepted**.

Notes for v1 / follow-up (do not block this acceptance):
- **First-turn day-summary injection** is a v1 feature (summary layer), not part of v0; its G2
  sub-criterion is deferred with the v1 summarization work.
- **TTS unsupported-character hardening (separate track):** Supertonic rejects certain characters
  (CJK ideographs, the middle dot `·`); the **main** TTS path normalizes via `normalize_tts_text`
  but the **history-narration** path (`core/history_mode.py`) passes raw text and can crash on such
  content. This is a pre-existing history/TTS bug surfaced during the smoke, unrelated to recall
  correctness; it is tracked separately. The recall answer path is protected (it does not echo the
  ~10 CJK-drift index turns because the interrogative/declarative gating + main-path normalization
  keep them out of spoken output).
- The "document ownership — acceptance-time sweep" checklist above may now execute as follow-up
  documentation work.
