# ADR 0017: Two-Tool Workflow — Claude Code + Codex Only

- **Status:** Accepted
- **Date:** 2026-03-24
- **Supersedes:** Three-tool workflow (Claude Code + Codex + Gemini) established in commit 28b12fe

## Context

The project adopted a 3-tool workflow (Claude Code as orchestrator, Codex as implementer, Gemini as reviewer) on 2026-03-22. After the first real usage cycle (system prompt improvement task, 2026-03-23/24), Gemini's review capability proved insufficient:

1. **Rubber-stamp reviews**: Gemini gave all-PASS verdicts with no code-line-level analysis, no critical insight, and no detection of issues that Codex and Claude Code subsequently found.
2. **Outdated references**: Gemini cited figures (e.g., "42% latency reduction") that had already been corrected in the conversation, indicating it did not read the latest analysis.
3. **Context-reset limitation**: Gemini CLI resets context each session. Review tasks require deep understanding of design decisions, prior discussions, and architectural tradeoffs — information that cannot be transmitted via a file-based review request.
4. **Codex had the same limitation but different impact**: Codex's context reset is acceptable for implementation tasks (specs can be written precisely), but review tasks require contextual judgment that only the orchestrator (with full conversation history) can provide.

Adding a second Codex instance for review was also considered and rejected — same model, same blind spots, same context limitation.

## Decision

**Adopt a 2-tool workflow: Claude Code (PM + orchestrator + architect + reviewer) + Codex (sole implementer).**

### Active roles

| Agent | Role | Responsibilities |
|-------|------|-----------------|
| Claude Code (Opus 4.6) | Orchestrator + Reviewer | Planning, code review, verification chain (3-round), polish loop, documentation, commit/push management |
| Codex CLI (GPT-5.4) | Implementer | Code writing, bug fixes, QC tool execution, self-verification before handoff |

### Gemini status

Gemini CLI is **dormant**. All configuration files (`.gemini/`, `GEMINI.md`, `GEMINI_CONSTITUTION.md`) are retained with DORMANT labels. Gemini may be reactivated if its review capabilities demonstrably improve.

### Dual verification structure

```
Codex self-verification (1st filter):
  1. Implement according to task spec
  2. Run QC (ruff, mypy, pytest)
  3. Self-verification 3-round (consistency, self-review, final QC)
  4. Polish loop (0-fix x2)
  5. English handoff note to Claude Code

Claude Code formal verification (2nd filter):
  6. Review Codex handoff
  7. Formal verification chain (3-round QA/QC/Review)
  8. Polish loop
  9. Commit (user approval required)
```

## Consequences

### Positive

- **Higher review quality**: Claude Code has full conversation context, design history, and architectural understanding.
- **Faster iteration**: No file-based handoff to external reviewer, no waiting for Gemini execution.
- **Simpler workflow**: Two tools instead of three, fewer coordination files, fewer failure modes.
- **Codex self-verification**: Reduces issues reaching Claude Code's formal chain.

### Negative

- **Single reviewer**: Claude Code is both orchestrator and reviewer, losing independent review perspective.
- **Codex review instability**: Codex's self-verification quality varies across sessions (observed FAIL->FAIL->FAIL->PASS pattern with shifting blocker interpretations).

### Mitigations

- Codex self-verification acts as a complementary filter with different focus (mechanical consistency) vs Claude Code (architectural/contextual review).
- If independent review is needed for critical changes (e.g., safety module), a human review or future capable AI reviewer can be introduced.

## Related

- ADR 0008: Sub-agent migration (partially superseded — only qc and review internal sub-agents remain active)
- ADR 0018: Implementation delegation to Codex (extends this ADR — constrains Claude Code to non-implementing role, mandates polish loop)
- ADR 0003: QA/QC agent split (historical)
- CLAUDE.md §8: Active workflow definition
- AGENTS.md: Codex self-verification procedure
