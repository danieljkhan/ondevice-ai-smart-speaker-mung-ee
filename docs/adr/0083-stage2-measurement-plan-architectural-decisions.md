# ADR 0083 — Stage-2 Full-Audio + RAG + Resident Mode Measurement: architectural decisions

- **Status**: Accepted (plan-level decisions; execution pending prerequisite measurement-infrastructure patch PR + Phase 0 audit + Phase 1/2 Jetson runs)
- **Date**: 2026-05-05
- **Author**: Claude Code (orchestrator)
- **Related plan**: `docs/archived/dev-plan/2026-05-03-stage2-jetson-full-audio-rag-resident-measurement-plan.md` (v4 FINAL, 3-round Codex review + user-resolved escalation)
- **Related ADRs**: 0073 (Gemma 4 swap), 0076 (L1 LLM resident default), 0058 (TTS-resident deferred), 0055 (Qwen3-ASR primary), 0082 (conversation-memory RAG)

## Context

ADR 0076 §"Operator notes" requires Stage-2 measurement of full-audio + RAG + optional STT residency before the L1 env-var rollback path (`MUNGI_LLM_RESIDENT=0`) can be retired. The L1 rollout plan §2 enumerates 4 deferrals + R44 routing real-audio validation (Session 19 watchpoint #5). Stage-2 plan v1 (2026-05-03) was authored to close all five items in a single measurement campaign; CLAUDE.md §1 Gate 1 mutual-discussion ran the full 3-round cap (Codex `reviewer` round 1: 16 findings PUSH BACK; round 2: 4 findings PUSH BACK; round 3: 3 findings PUSH BACK escalate). User escalation directives on 2026-05-05 closed the round-3 BLOCK + 2 MAJORs and produced Plan v4 FINAL.

This ADR records the architectural-level decisions that emerged from the v1→v4 evolution and that bind the prerequisite patch PR + the eventual Phase 0/1/2/3 execution.

## Decisions

### D1 — Measurement-infrastructure patches separated as prerequisite PR (Q1=b)

The audio runner (`scripts/e2e_qwen3_asr_mix.py`), report generator (`scripts/generate_e2e_report.py`), and pipeline observability (`core/pipeline.py` `TurnMetrics`) modifications required for Stage-2 evidence are scoped into a **separate prerequisite PR** under standard CLAUDE.md §7 branch policy (branch → PR → CI → review → merge). Stage-2 plan v4 assumes that PR is merged into `dev` and synced to Jetson `/opt/mungi-repo` before Phase 1.

**Rationale**: keeps the measurement-plan execution surface free of code-author scope, enables independent CI/coverage gates on the patch, and avoids a single PR that mixes 300-470 LoC of measurement infrastructure with measurement evidence. Considered alternative: single combined PR with an expanded plan-scope budget (rejected for review-economy and gate-isolation reasons).

### D2 — R44 operational identity = `template_topic_id == "swimming"` AND `template_mode == "guide"`

R44 routing real-audio validation (G7) is operationally bound to a single approved-template identity: `topic_id == "swimming"` AND `mode == "guide"` AND `template_matched == true`. Any other approved-template match is NOT R44 and does not count toward the Phase 0.5 ≥ 5-stimuli threshold or the G7 success condition.

**Rationale**: codified by Codex round-3 audit (CONFIRMED-COMPLETE) of `safety/approved_template_router.py:117,133-143` + `assets/filters/approved_templates.json:1005-1024,1043-1044` + `docs/archived/dev-plan/2026-04-28-swimming-face-submersion-keywords-plan.md:17,27,44`. Bath/ocean variants in the swimming plan (`:119-121, 222-228`) are guard cases that route AWAY from `swimming`; they are not additional accepted R44 identities.

### D3 — `core/pipeline.py` `TurnMetrics` extended for live router observability (R3-F2)

`TurnMetrics` is extended to expose `template_topic_id: str | None`, `template_mode: str | None`, and `template_matched=True` for **both `guide`-mode and `block`-mode** approved-template matches.

**Pre-decision state**: current `TurnMetrics` exposes only `template_matched`; guide-mode matches return without setting `template_matched=True` (only block-mode does, per `core/pipeline.py:512-568, 1189-1201`).

**Post-decision state**: both match modes set `template_matched=True`; topic/mode are sourced from `safety.approved_template_router.check_approved_template` return at `safety/approved_template_router.py:140-143`.

**Rationale**: G7 live route capture is the faithful R44 audit field. Without this extension, G7 evidence would be either fictitious (no topic/mode) or missing for guide-mode matches (which is the R44 mode per D2). Considered alternative: post-hoc replay against STT predictions only (rejected — input-equivalent but does not exercise the live router path through `pipeline.run_turn`).

**Compatibility**: `template_matched` becoming True for guide-mode is a **semantics change** for any downstream consumer that reads this field. Audit before patch PR merge: only the audio runner consumes `template_matched` as part of Stage-2 instrumentation; existing telemetry fields (`response_source`, `template_id`) are unchanged. No backwards-compatibility shim is required.

### D4 — Configuration B = single-process 5-pass (no cold restart between passes)

Configuration B (sustained thermal, ≥ 30-min continuous load) runs a **single Python process** with single `ModelManager` and single per-language `ConversationPipeline` pair, invoked once with `--repeat-passes 5`. The 24-WAV pool is iterated 5 times in-process, yielding 120 turns. Resident state (LLM + STT + TTS + RAG retriever) is preserved across passes by construction — no `unload_all()` until process exit at run end.

**Rationale**: ADR 0076 §"Operator notes" requires sustained-residency evidence; 3 cold-restart processes (Plan v1 design, Codex F3 BLOCK) preserve zero resident state across pass boundaries. Same-process repeat-passes is the only design that meaningfully tests "does residency hold under repeated load over 30+ minutes". Patch PR adds `--repeat-passes N` semantics with strict input validation (`N >= 1`), per-pass `--max-rounds` scope, `pass_id`/`global_turn_id` schema, and history-reset rules conditional on `--conversation-per-lang`.

### D5 — Stage-2 acceptance gate set = G1-G10 (with G3 thermal-only)

The acceptance gate set is finalized at G1-G10:

| Gate | Surface | Threshold |
|---|---|---|
| G1 | tegrastats system RAM peak | < 5500 MB (ADR 0076 invariant) |
| G2a | tegrastats system RAM peak | < 6000 MB (operational ceiling) |
| G2b | `MemoryHealth.CRITICAL` log events | == 0 (RSS+CUDA > 6500 MB guard) |
| G3 | `thermal_max_c = max(cpu_temp_c.max, gpu_temp_c.max)` over ≥ 30 min | < 80.0 °C |
| G4 | Turn success rate over 120 turns (Config B aggregate) | ≥ 95 % |
| G5 | RAG enabled + per-turn `rag_hit=true` AND `rag_context_chars>0` for ≥ 50 % of KO rounds | per per-turn fields |
| G6 | First-turn positive `*_load_ms`; subsequent-turn `*_load_ms == 0` or resident-skip log | residency evidence |
| G7 | R44 routing per D2 (`swimming`/`guide`); ≥ 5 stimuli; failed-STT/mismatched = failure | operational definition |
| G8 | Korean hot-turn TTFT mean (excludes EN + LLM-bypass safety-template turns) | ≤ 4.50 s (ADR 0076) |
| G9 | TTS load+synth errors == 0; non-empty WAV (`tts_wav_bytes > 0` AND `tts_wav_frames > 0`) | hard gate |
| G10 | Sherpa-ONNX provider resolved + `sherpa_onnx_version` recorded + `stt_load_count == 1` | hard gate |

**Rationale**: G2 was originally a single gate conflating system RAM with RSS+CUDA (Codex F5 MAJOR). G3 originally included a "no throttle event" clause unmeasurable from the existing tegrastats parser (Codex R3-F1 BLOCK; user directive (A) dropped the clause given Jetson Orin Nano Super HW throttle thresholds GPU 88 °C / CPU 95 °C make the 80 °C ceiling conservative). G9/G10 hardened from "tolerate STT errors and defer" → "fail Stage-2 on TTS/STT runtime regression" (Codex F12 + R2-F2). Failed-STT R44 count as failures (Codex F4 + R2-F1, R5 strengthened) — no STT-confidence-threshold loophole.

### D6 — Conversation-memory FAISS = path/size verification only (Q3=c)

Stage-2 verifies that wiki FAISS path ≠ conversation-memory FAISS path (CLAUDE.md §6 separate-FAISS invariant) and records `ntotal` of each at Phase 0.3, but does NOT measure active growth. ADR 0082 deferred status is preserved; Stage-2 does not depend on or activate the conversation-memory update path.

**Rationale**: per CLAUDE.md §6 the invariant is index separation, not active update. Activating ADR 0082's deferred update path would expand Stage-2 scope by ≥ 2 sessions and is out of scope. Considered alternatives: (a) full removal — rejected, separation invariant still requires Phase 0 verification; (b) activate update path — rejected, scope creep.

### D7 — Reporting columns reference `docs/templates/e2e-report-format.md:6-20` directly (R3-F3)

Plan + patch PR + final reports use the canonical column set defined in `docs/templates/e2e-report-format.md:6-20` by **direct reference** rather than inline restatement.

**Rationale**: Plan v3 inline column lists drifted from the mandatory template (omitted `playback` and `first_sound` columns, both already computed by the mix runner — Codex R3-F3 MAJOR / F8 regression). Direct reference makes future template revisions propagate automatically without plan-doc drift.

## Consequences

### Positive

- Stage-2 measurement scope is now precisely defined and Codex-audited across 3 rounds. R44 expected identity is operational (D2). Live router observability is pipeline-wired (D3). Configuration B genuinely tests sustained residency (D4). Acceptance gates G1-G10 cover the L1 plan §2 deferrals + R44 audio-path + STT/TTS hard gates. Patch PR is bounded and CI-gated (D1).
- ADR 0076 status update path is well-defined: on G1-G10 pass, ADR 0076 → "Accepted (validated for full-audio + RAG + STT-resident)"; TTS-resident is documented as ADR 0058 opt-in evidence (not ADR 0076 closure). On any gate miss, Stage-3 plan is authored.
- Env-var rollback retirement (`MUNGI_LLM_RESIDENT=0`) is a separate user-approved decision after Stage-2 closes, recorded as a future ADR amendment.

### Negative

- D3's `template_matched=True` semantics change for guide-mode is a **behavior change** for any future downstream consumer. Mitigation: pre-merge audit confirmed only the Stage-2 audio runner consumes `template_matched`; existing telemetry fields are unchanged. Documented in this ADR for future reference.
- Patch PR prerequisite (D1) adds 1 cycle to the Stage-2 timeline (~1 session) and requires its own Plan Gate 1 cycle (separate Codex `reviewer` round). Mitigation: bounded scope (300-470 LoC across 3 source files + tests) and clean CI/coverage isolation.
- Configuration B (D4) requires ~30-50 minutes of unattended Jetson runtime. Mitigation: tmux preserves session across SSH disconnects; post-hoc evidence is intact even if mid-run monitoring is lost (R7 in Plan v4 §9).

### Neutral / monitored

- D5 G3 thermal-only gate: if 80 °C is reached without HW throttle (which Jetson Orin Nano Super does not initiate until GPU 88 °C / CPU 95 °C), the gate fails as designed. The "no throttle event" check would have provided defense-in-depth but is rejected as over-engineering for the child-product safety intent.
- D6 conversation-memory deferred update path: if a future plan activates ADR 0082, Stage-2-equivalent measurement of conversation-memory growth becomes a separate scope item.

## Status notes

- Plan Gate 1 cycle for Plan v4: COMPLETE (3-round cap reached; user escalation closure on 2026-05-05; final approval recorded in `.claude/plan-review-status.json`).
- Prerequisite patch PR plan: NOT YET DRAFTED (next-session work; will run its own Plan Gate 1 cycle).
- Phase 0 audit / Phase 1/2 Jetson runs / Phase 3 report / Phase 4 env-var rollback decision: gated on patch PR merge.

## References

- `docs/archived/dev-plan/2026-05-03-stage2-jetson-full-audio-rag-resident-measurement-plan.md` (v4 FINAL)
- `Dev_Plan/2026-05-03-stage2-jetson-full-audio-rag-resident-measurement-plan-codex-review-round{1,2,3}.md`
- `Dev_Plan/2026-05-03-stage2-jetson-full-audio-rag-resident-measurement-plan-discussion-round{1,2,3-final}.md`
- ADR 0076 §"Operator notes"
- `Dev_Plan/2026-04-26-l1-llm-resident-rollout-plan.md` §2
- `safety/approved_template_router.py:117,133-143`
- `assets/filters/approved_templates.json:1005-1024,1043-1044`
- `core/pipeline.py:512-568,1189-1201`
- CLAUDE.md §1 Gate 1, §6, §7
