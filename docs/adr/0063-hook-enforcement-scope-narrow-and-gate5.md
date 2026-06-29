# ADR 0063: Narrow hook enforcement scope; introduce CLAUDE.md §1 Gate 5 for PR merge

- **Status**: Accepted
- **Date**: 2026-04-17

## Context

On 2026-04-17, during the Wave 3 Step 8 close-out session (Phase A —
merging PRs #18 / #19 / #20 into `dev`), the `mungi-codex-plugin`
`enforce_verification.py` hook blocked `gh pr merge 18 --squash`
because its `BLOCKED_PATTERNS` regex list treats the PR-merge command
identically to the commit and push commands:

```python
BLOCKED_PATTERNS = [
    re.compile(r"\bgit\b.*\bcommit\b", re.IGNORECASE),
    re.compile(r"\bgit\b.*\bpush\b",   re.IGNORECASE),
    re.compile(r"\bgh\b.*\bpr\b.*\bmerge\b", re.IGNORECASE),
]
```

The block fired because an immediately prior auto-memory file edit
(`C:\Users\danie\.claude\projects\E--python-vscode-mungi\memory\MEMORY.md`
— outside the mungi workspace) had triggered `reset_verification.py`
to full-reset the local verification state. A workaround attempt
(manually restoring `verified=true` via the plugin state API) was
correctly denied by the permission system as audit fabrication.

Two independent root causes combined to produce the incident:

1. **`gh pr merge` is the wrong layer for a hook-level verification
   block.** PR merges integrate already-reviewed, CI-green,
   human-reviewed code. Per-PR verification (branch-level
   `verification-status.json` during the Codex-implementation phase,
   PR-level CI checks, human review on the PR) is a lifecycle
   distinct from the orchestrator / Codex-delegation verification
   lifecycle captured in the live session's
   `verification-status.json`. The hook confused them.

2. **`reset_verification.py` full-resets on any file edit whose
   absolute path cannot be made relative to workspace root.** The
   current fallback in `_normalize_file_path` returns the absolute
   path unchanged when `Path.resolve().relative_to(workspace)` raises,
   then prefix matching against `.claude/` and
   `docs/runbooks/weekly/` fails, and `reset_for_file` falls through
   to the default `_full_reset`. Auto-memory paths (outside the
   workspace) deterministically hit this path and reset verification
   state on every memory write.

The incident blocked the Phase A PR-merge sequence. A plan was
drafted at `docs/archived/dev-plan/2026-04-17-hook-enforcement-scope-fix.md`,
reviewed by Codex in two mutual-discussion rounds
(`MAJOR REVISIONS NEEDED` → `APPROVE WITH NOTES`; record at
`docs/archived/dev-plan/2026-04-17-hook-enforcement-scope-fix-discussion.md`),
and user-approved.

## Decision

### Plugin-repo changes

1. **Narrow `BLOCKED_PATTERNS`** in
   `mungi-codex-plugin/scripts/hooks/enforce_verification.py` to two
   entries only — the commit pattern and the push pattern. The PR
   merge pattern is removed. Module and `is_blocked_command`
   docstrings updated accordingly.

2. **Add workspace-boundary awareness** to
   `mungi-codex-plugin/scripts/hooks/reset_verification.py`. A new
   helper `_is_outside_workspace(file_path, workspace_root)`
   resolves relative paths against `workspace_root` (not process
   `cwd`) before checking `relative_to`. Paths outside the workspace
   skip reset entirely.

3. **Fail-open on unresolvable paths** — the helper returns `True`
   (= skip reset) on any `OSError` / `RuntimeError` / `ValueError`
   from `Path.resolve()`. This is deliberate: for external paths
   the choice eliminates false-positive resets; for in-workspace
   paths whose resolve raises (permission error, broken symlink,
   symlink loop), the cost is a stale-verification false-negative
   until the next successful resolve. The implementer must include
   a regression test and a code comment citing this ADR.

4. **Test coverage** — two new test files
   (`tests/test_enforce_verification.py` and
   `tests/test_reset_verification.py`) cover: retained `git commit`
   / `git push` gating, non-block of `gh pr merge` in various
   forms, workspace-relative resolution from a non-workspace cwd
   (regression guard), platform-parametrized out-of-workspace
   classifications (Windows user-profile / UNC; POSIX `/home` /
   `/tmp`), and the fail-open branch.

### Mungi-repo changes

5. **Add Gate 5 (`Merge (PR)`) to CLAUDE.md §1.** `gh pr merge`
   approval moves from hook regex to the orchestrator protocol.
   Gate 5 shape mirrors Gate 3 (Push): state target PR number + base
   branch + commit list, confirm PR CI is `SUCCESS`, wait for user
   approval. The section heading changes from "Four Approval Gates"
   to "Five Approval Gates"; the `.claude/settings.local.json`
   explainer is updated to describe the two-layer model
   (hook-gated vs orchestrator-gated).

6. **Update CLAUDE.md §8 hook description.** The
   `enforce_verification.py` row no longer claims to block
   `gh pr merge`; it explicitly references Gate 5 and this ADR for
   the new approval surface.

### Non-decisions (explicitly deferred)

- **Do not** widen `BLOCKED_PATTERNS` to include other `gh`
  subcommands (`gh release create`, `gh pr close`, etc.). No live
  incidents; future work if needed.
- **Do not** edit `.claude/settings.local.json`. The existing
  `Bash(gh pr:*)` allowlist already provides execution capability;
  Gate 5 is the approval layer.
- **Do not** modify `lib/state.py`, `DEFAULT_IGNORED_PATHS`, or
  `DEFAULT_SAFETY_PATHS`. Only the ingress filter is fixed.
- **Task #53 Hook strengthening** (verification-status.json hash
  chain + exclusive CLI write path) remains an orthogonal follow-up.

## Consequences

### Positive

- **Eliminates the false-positive hook block on PR merges**;
  orchestrator sessions can merge pre-approved PRs without the
  spurious reset-storm chain from auto-memory edits.
- **Auto-memory file edits no longer reset project verification
  state** — session continuity preserved across the default
  Claude Code memory workflow.
- **Architectural separation is explicit**: per-PR lifecycle (CI +
  human review on the PR) vs orchestrator-Codex lifecycle (local
  `verification-status.json`) are no longer conflated at the regex
  layer. The ADR freezes this distinction so future edits cannot
  re-merge the two.
- **CLAUDE.md §1 now captures PR-merge approval explicitly**,
  making the merge decision auditable alongside commit / push /
  deploy.
- **Test coverage grows** from zero dedicated hook unit tests to
  two test modules exercising both regression and bug-fix
  behaviors, including platform-parametrized cases.

### Negative / Risks

- **Orchestrator now carries sole local responsibility for
  PR-merge approval** (previously partially shared with the hook).
  Mitigated by: (a) Gate 5 protocol enforcement in the orchestrator
  session, (b) PR-level CI as the independent automated gate,
  (c) human PR review, (d) branch protection on `dev` / `main` can
  be tightened if audit evidence ever shows orchestrator drift.
- **In-workspace unresolvable paths (permission error / broken
  symlink / symlink loop) get a stale-verification false-negative**
  until the next successful resolve — deliberate fail-open
  trade-off. Implementer must add a test exercising this branch
  and a code comment citing this ADR.
- **CLAUDE.md §1 renumbered 4 → 5 gates.** External references to
  "four gates" must update; discoverability: the heading + intro
  sentence both changed, so a single grep finds references.
- **Two-repo coupled landing** — plugin + mungi edits must land in
  the same session. Mitigation captured in plan §7 delivery
  sequence (plugin-first ordering; mungi in the same session).

## References

- `docs/archived/dev-plan/2026-04-17-hook-enforcement-scope-fix.md` — approved
  plan (Plan Gate v2, Round 2 `APPROVE WITH NOTES`).
- `docs/archived/dev-plan/2026-04-17-hook-enforcement-scope-fix-discussion.md` —
  Round 1 + Round 2 mutual-discussion record.
- `docs/runbooks/weekly/archive/2026-04-16-next-session-handoff.md` —
  Phase A context (the PR-merge sequence blocked by the incident).
- ADR 0056 — Plan Gate v2 mutual discussion (the process this
  change went through).
- ADR 0062 — Wave 2 final verdict (the Wave 3 baseline; Phase A
  PRs being merged relate to Wave 3 Step 8).
- Plugin repo feature branch: `fix/hook-enforcement-scope-narrow`.
- Mungi repo feature branch: `chore/claude-md-gate5-hook-docs-sync`.
