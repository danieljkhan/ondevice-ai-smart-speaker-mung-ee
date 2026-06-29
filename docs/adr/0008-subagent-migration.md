# ADR 0008: Migrate from Multi-Agent Team to Claude Code Sub-Agent Architecture

- **Status:** Accepted (partially superseded by ADR 0017 + 2026-03-24 Codex delegation rule)
- **Date:** 2026-03-16
- **Supersedes:** ADR 0003 (QA/QC Agent Split), ADR 0004 (Verification Enforcement Hooks)
- **Partially superseded by:** ADR 0017 (Two-Tool Workflow) + 2026-03-24 rule change: Claude Code no longer uses write-mode internal sub-agents (feature, platform, safety, qa). All implementation is delegated to Codex CLI (GPT-5.4). Only qc and review sub-agents remain active. See CLAUDE.md §8.

## Context

The project has been running an 11-agent team model (PM + 3 lane writer/reviewer pairs + QA + QC + 2 supervisors) across separate terminal sessions. This model required:

- 11 separate `.cmd` launcher scripts
- File-based coordination board (`docs/agents/board/assignments/`, `handoffs/`)
- Operating declarations per session
- Complex VS Code task configurations to launch groups of agents
- Manual synchronization between agents via status files

While this provided clear separation of concerns, the operational overhead was significant for a solo developer working part-time. Key pain points:

1. **High ceremony**: Each work session required launching multiple terminals, waiting for agents to read board files, and manually routing handoffs.
2. **Coordination latency**: File-based board system meant agents polled for status updates rather than receiving direct communication.
3. **Resource waste**: Running 11 Codex sessions simultaneously consumed significant API quota for coordination overhead.
4. **Single-developer mismatch**: The team model was designed for multi-person collaboration but the project has one developer.

## Decision

**Replace the 11-agent team model with Claude Code's built-in sub-agent architecture.**

The main Claude Code session (Opus 4.6) acts as the orchestrator (PM role), spawning specialized sub-agents via the Agent tool for parallel execution. Sub-agents communicate results directly back to the orchestrator — no file-based coordination needed.

### New Architecture

```text
Main Claude Session (Orchestrator/PM)
├── Sub-agent: feature    — core/, models/, parental/, assets/prompts/, assets/sounds/
├── Sub-agent: platform   — hardware/, systemd/, scripts/, .github/, requirements-*.txt
├── Sub-agent: safety     — safety/, assets/filters/
├── Sub-agent: qa         — tests/ (write tests)
├── Sub-agent: qc         — run ruff, mypy, pytest (verification)
└── Sub-agent: review     — read-only inspection (CLAUDE.md compliance, security, architecture)
```

### Key Design Choices

1. **Direct communication**: Sub-agents return results to the orchestrator directly. No board files needed.
2. **Parallel execution**: Independent sub-agents are spawned in parallel via multiple Agent tool calls in a single message.
3. **Worktree isolation**: Sub-agents that write to overlapping paths use `isolation: "worktree"` to prevent conflicts.
4. **Scope enforcement**: Each sub-agent prompt includes explicit scope restrictions.
5. **Verification chain preserved**: QA → QC → Review pipeline remains mandatory, but orchestrated programmatically instead of via file polling.
6. **Single model**: All sub-agents use Claude Opus 4.6. No Codex dependency.

## Consequences

### Positive

- **Dramatically lower ceremony**: One session, one orchestrator, instant sub-agent dispatch.
- **Faster feedback loops**: Sub-agent results return in-context, no file polling.
- **Lower resource usage**: Sub-agents are spawned on demand, not running continuously.
- **Simpler infrastructure**: No launcher scripts, no VS Code task configurations, no board files.
- **Better context**: Orchestrator sees all sub-agent results in one conversation.

### Negative

- **No persistent agent state**: Sub-agents start fresh each invocation (mitigated by detailed prompts).
- **Context window pressure**: Large sub-agent results consume orchestrator context (mitigated by sub-agents summarizing their findings).

### Neutral

- Verification quality remains the same — the chain is enforced by orchestrator logic, not file existence.
- Repository structure unchanged — only coordination infrastructure is removed.

## Migration Actions

1. Rewrite `CLAUDE.md` section 8 (Agent Team System) and section 14 (Model Tier Policy).
2. Rewrite `AGENTS.md` for sub-agent model.
3. Rewrite `AGENT_TEAM_SETUP.md` for sub-agent model.
4. Rewrite `docs/agents/agent-team-rr.md` for sub-agent roles.
5. Convert `docs/agents/instructions/*.md` to sub-agent prompt templates.
6. Update `docs/templates/agent-operations-template.md`.
7. Delete all `.vscode/scripts/mungi_*.cmd` launcher scripts.
8. Simplify `.vscode/tasks.json` (remove agent launcher tasks).
9. Archive board structure (`docs/agents/board/`).
