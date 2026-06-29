# ADR 0018: Implementation Delegation to Codex — Claude Code as Non-Implementing Orchestrator

- **Status:** Accepted
- **Date:** 2026-03-24
- **Extends:** ADR 0017 (Two-Tool Workflow)
- **Partially supersedes:** ADR 0008 (Sub-agent Migration) — write-mode sub-agents retired

## Context

ADR 0017 established a 2-tool workflow (Claude Code + Codex) by removing Gemini CLI. However, under that model Claude Code could still implement code directly via internal write-mode sub-agents (feature, platform, safety, qa) OR delegate to Codex. This dual-path created ambiguity:

1. **Inconsistent quality gates**: Claude Code's internal implementations bypassed Codex's self-verification (1st filter), going straight to the formal verification chain. Codex-delegated work went through dual verification.
2. **Role confusion**: Claude Code acted simultaneously as implementer and reviewer of its own code — no separation of concerns.
3. **Underutilized Codex**: With Claude Code handling some implementations directly, Codex's role as implementer was partial, leading to inconsistent task routing decisions.
4. **Polish loop enforcement gap**: Codex's AGENTS.md had a conditional polish loop ("If Round 2 found issues"), allowing skips when self-verification found 0 issues — contradicting CLAUDE.md's mandatory polish loop requirement.

## Decision

**All implementation is delegated to Codex CLI (GPT-5.4). Claude Code does NOT write implementation code. This is NON-NEGOTIABLE.**

### Claude Code role: PM + Orchestrator + Architect + Reviewer

| Responsibility | Description |
|---------------|-------------|
| Project Management | Priority, scope, schedule, release decisions |
| Planning | Worklog analysis, priority derivation, work planning |
| Design & Architecture | All design decisions, ADR authoring, system architecture — core competency |
| Task Spec Writing | Compose `.codex/current-task.md` with role, scope, goal, constraints |
| Verification | Formal verification chain (QC + Review × 3 rounds + polish loop) |
| Documentation | Worklogs, ADRs, runbooks, test reports |
| Commit/Push | Only after verification passes + user approval |

### Codex role: Sole Implementer

| Responsibility | Description |
|---------------|-------------|
| All code writing | Features, platform, safety, tests — 6 role-based invocation modes |
| QC execution | ruff, mypy, pytest — run and auto-fix |
| Self-verification | Mandatory 3-round (consistency + scope, self-review, final QC) |
| Polish loop | Mandatory — 2 consecutive 0-fix cycles (10 iterations each). Skipping is a process violation. |
| Handoff | English note with mandatory fields. Missing polish loop results → automatic rejection by Claude Code. |

### Architectural changes

1. **Claude Code internal sub-agents reduced from 6 to 2**: Only `qc` (execute) and `review` (read-only) remain. Write-mode sub-agents (`feature`, `platform`, `safety`, `qa`) are ARCHIVED.
2. **Codex sub-agent roles**: 6 role-based invocation modes defined in `.codex/config.json` and `AGENTS.md`, mirroring the retired Claude Code sub-agent scopes.
3. **Handoff rejection policy**: Claude Code MUST reject Codex handoffs missing self-verification or polish loop results.
4. **Polish loop mandatory**: AGENTS.md updated from conditional to mandatory, with explicit termination criteria.

### Verification flow

```
Claude Code: write task spec (.codex/current-task.md)
  └─ User: run Codex CLI
       └─ [1st filter] Codex: implement → QC → self-verify 3-round → polish loop (mandatory)
            └─ English handoff note (mandatory fields)
                 └─ [2nd filter] Claude Code: formal verification
                      ├─ QC sub-agent (ruff, mypy, pytest)
                      └─ Review sub-agent (read-only inspection)
                           └─ 3 rounds → polish loop → commit (user approval)
```

## Consequences

### Positive

- **Clean separation of concerns**: Claude Code thinks (design, architecture, review); Codex builds (implementation). No self-review conflict.
- **Consistent dual verification**: Every code change passes through both Codex self-verification and Claude Code formal verification. No bypass path.
- **Stronger quality enforcement**: Mandatory polish loop + handoff rejection policy prevents process shortcuts.
- **Focused expertise**: Claude Code concentrates on architectural decisions and review quality rather than splitting attention between implementation and orchestration.

### Negative

- **User involvement required**: Codex CLI must be invoked manually by the user (Claude Code cannot spawn Codex directly). Each implementation cycle requires a user action.
- **Latency increase**: Dual verification adds overhead compared to Claude Code implementing directly. Mitigated by Codex's self-verification reducing issues before formal verification.
- **No hook enforcement for Codex**: Codex CLI lacks Claude Code's hook system. Polish loop and self-verification are enforced by documentation rules and handoff rejection, not by automated hooks.

### Neutral

- Code quality remains the same or improves — dual verification catches more issues.
- Repository structure unchanged — only the agent workflow is different.

## Related

- ADR 0017: Two-Tool Workflow (establishes Claude Code + Codex; this ADR constrains the division of labor)
- ADR 0008: Sub-agent Migration (partially superseded — write-mode sub-agents retired)
- CLAUDE.md §8: Active workflow definition
- AGENTS.md: Codex self-verification and polish loop specification

## Clarification (2026-03-25): Mandatory Polish Loop

The polish loop specified in Section "Codex role" is **unconditionally mandatory**,
not conditional on verification findings. Codex must execute the polish loop after
every self-verification 3-round, regardless of whether issues were found. Termination
requires 2 consecutive 0-fix cycles (10 iterations each, max 20 iterations total).
Skipping the polish loop is a process violation and grounds for handoff rejection.
