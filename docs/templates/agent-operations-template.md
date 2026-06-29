# Mungi Sub-Agent Operations Template

Updated 2026-03-24 — implementation delegation to Codex CLI (GPT-5.4). Claude Code retains only qc and review internal sub-agents.

## 1. Operational Objectives

- Keep orchestrator-centered decision authority.
- Run a consistent loop: readiness review, implementation, verification, acceptance.
- Keep code, docs, and verification evidence traceable in the same repository.

## 2. Work Session Cadence

### Start of Session

- Orchestrator reviews the latest worklog under `docs/runbooks/weekly/`.
- Orchestrator identifies priorities and reports readiness to user.
- User approves the plan before implementation begins.

### Implementation Phase

- Orchestrator writes task spec to `.codex/current-task.md` with role, scope, goal, and constraints.
- User runs Codex CLI. Codex implements within assigned scope.
- Codex performs self-verification (3-round + polish loop) and produces English handoff note.
- Orchestrator collects Codex handoff results.

### Verification Phase

- Orchestrator spawns qc sub-agent to run verification suite.
- Orchestrator spawns review sub-agent for independent inspection.
- Verification sub-agents can run in parallel after Codex handoff is received.

### Acceptance Phase

- Orchestrator reviews qc and review results.
- Accepts only if no BLOCK items exist.
- Reports final status to user in Korean.

## 3. Quality Gate Checklist

- Required tests written and passed.
- Safety regression passed.
- Performance or memory targets met when relevant.
- Ops checks passed when relevant.
- ADR or runbook updates completed when required.
- QC verification: no BLOCK items.
- Review inspection: no BLOCK items.

## 4. Ticket Template

```md
## Background
- Why this is needed:

## Scope
- In:
- Out:

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Codex Delegation
- Role: [feature | platform | safety | test]
- Scope: [exact files]

## Validation
- QA: tests written
- QC: verification passed
- Review: inspection passed
- Evidence: [summary or reference]

## Risk
- Main risks:
- Rollback plan:
```

## 5. PR Template

```md
## Summary
- Goal:
- Main changes:

## Impact
- User impact:
- System impact:

## Validation
- [ ] QA (tests written)
- [ ] QC (lint + type + tests pass)
- [ ] Review (inspection passed)

## Evidence
- QC results:
- Review findings:

## Rollback
- Rollback steps:
```

## 6. Sub-Agent Prompt Composition

When composing a sub-agent prompt, include:

1. **Role**: Which Codex role (feature, platform, safety, test, qc, reviewer).
2. **Goal**: What needs to be accomplished.
3. **Scope**: Exact files or directories Codex may edit.
4. **Context**: Relevant background (worklog, blockers, dependencies).
5. **Constraints**: CLAUDE.md rules, scope limits, task-specific rules.
6. **Deliverable**: Expected output format.

## 7. Repository Locations

- R&R docs: `docs/agents/`
- Sub-agent prompt templates: `docs/agents/instructions/`
- Operations templates: `docs/templates/`
- Weekly reports: `docs/runbooks/weekly/`
- Release records: `docs/runbooks/releases/`
- ADRs: `docs/adr/`

## 8. Governance Addendum

- No direct commit to `main`.
- Branch -> PR -> review -> merge remains mandatory.
- Required checks:
  - `ruff check .`
  - `ruff format --check .`
  - `mypy core/ models/ safety/ hardware/ scripts/ parental/`
  - `pytest tests/ -v`
- Architecture, runtime-path, or safety-policy changes require ADR updates.
- Orchestrator cannot close a code task without qc and review evidence.

## 9. Model Tier Policy

All sub-agents run on **Claude Opus 4.6 exclusively**. No lower-tier models (Sonnet, Haiku) are permitted. Do not set `model` overrides in Agent tool calls.

## 10. Language Clause

- Policy and process documents remain in English.
- User-facing summaries and reports are written in Korean.
