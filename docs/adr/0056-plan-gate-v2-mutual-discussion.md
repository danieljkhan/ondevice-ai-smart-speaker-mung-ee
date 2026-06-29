# ADR 0056: Plan Gate v2 — Codex review + mutual-discussion cycle for plan approval

- **Status**: Accepted
- **Date**: 2026-04-14

## Context

Prior to this decision, Plan Gate v1 required only that Claude draft a plan and obtain explicit user
approval before implementation. In the 2026-04-14 E2E Bottleneck Improvement Plan session, Claude
delegated the initial plan to Codex for a read-only deep review and received a MAJOR REVISIONS
NEEDED verdict listing 4 BLOCK findings and 9 WARN findings. Two of the BLOCKs were execution-
critical: (1) the plan targeted assets/prompts/persona.md for a runtime change, but CLAUDE.md:644
explicitly classifies persona.md as a developer-reference document while the actual LLM-facing
Korean prompt lives inline at core/pipeline.py:130-227, and (2) the plan did not address the
unconditional unload_stt() call at core/pipeline.py:472-482 that would have defeated the STT
resident feature entirely. These errors would have reached the user and potentially production if
the review had not happened. Simultaneously, it became clear that a single-pass review-then-
integrate cycle lacks the convergence mechanics to resolve contested findings. The PM can accept,
but the reviewer cannot verify whether the response is sound before the user sees it. A mutual-
discussion cycle with bounded rounds and an explicit verdict protocol (APPROVE / APPROVE WITH NOTES
/ PUSH BACK) gives both agents the chance to reach convergent agreement before involving the user.
The user is the strategic decision-maker, not the technical fact-checker.

## Decision

Every plan document under Dev_Plan/ or docs/ MUST pass through a mandatory Codex deep review (role:
reviewer) before user approval. After the initial review, Claude Code and Codex conduct a mutual-
discussion cycle capped at 3 rounds: Claude writes a per-finding response (ACCEPT/MODIFY/REJECT) in
a discussion record, Codex re-reviews and returns APPROVE / APPROVE WITH NOTES / PUSH BACK, and the
loop continues until agreement or round 3. Items still contested after round 3 are ESCALATED to the
user. Only when the discussion yields APPROVE or APPROVE WITH NOTES does Claude Code request user
final approval, presenting the revised plan AND the discussion summary together. Fast-path
exception: if Codex's initial verdict is APPROVE AS-IS, the mutual-discussion phase is skipped.
Enforcement is implemented via a PreToolUse hook (.claude/hooks/enforce_plan_gate.py wrapper +
mungi-codex-plugin/scripts/hooks/enforce_plan_gate.py) that blocks Codex implementation delegations
(role: feature/platform/safety/test/qc) while plan_review_pending is true; role: reviewer is always
allowed. State is tracked in .claude/plan-review-status.json (schema v2).

## Consequences

Positive: (a) execution-critical plan errors are caught before user approval, improving first-pass
quality of implementations; (b) users see better artifacts with less cognitive load, since reviewers
have already resolved technical disputes; (c) the implementer (Codex) gains a formal channel to push
back on spec decisions it finds infeasible, reducing downstream rework; (d) enforcement via
PreToolUse hook prevents process drift. Trade-offs: (a) each plan now requires at least 2 Codex
calls (initial review + round-1 mutual discussion) instead of 1, adding roughly 15-30 minutes of
orchestration time per plan; (b) the mutual-discussion record adds a new artifact per plan that PM
must maintain; (c) round cap of 3 requires the PM to accept escalation when agreement is not
reached, shifting some burden back to the user for genuinely contested items; (d) PUSH BACK
reasoning quality depends on Codex having sufficient codebase context, which assumes ongoing read-
only scope access. Fast-path exception (APPROVE AS-IS bypasses discussion) limits overhead for
trivial or well-aligned plans. Operational impact: Plan Gate hook enforcement reuses the existing
thin-wrapper pattern (enforce_verification.py equivalent); the state file plan-review-status.json
uses schema_version so future revisions can migrate cleanly.

## Related ADRs

- 0003 — QA/QC agent split
- 0050 — Hook wrapper relative-path fail-loud

## References

- CLAUDE.md §1 (4대 승인 게이트) and §8 (Sub-Agent System + Plan Review Workflow)
- .claude/hooks/enforce_plan_gate.py (Claude-side thin wrapper)
- mungi-codex-plugin/scripts/hooks/enforce_plan_gate.py (plugin-side enforcement logic)
- .claude/plan-review-status.json (state schema v2)
- docs/archived/dev-plan/2026-04-14-E2E-Bottleneck-Improvement-Plan.md (first plan gated under Plan Gate v2)
- docs/archived/dev-plan/2026-04-14-E2E-Bottleneck-Improvement-Plan-codex-review.md (first Codex review artifact)
- docs/archived/dev-plan/2026-04-14-E2E-Bottleneck-Improvement-Plan-discussion.md (first mutual-discussion record)
