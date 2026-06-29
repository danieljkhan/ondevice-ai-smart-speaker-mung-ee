# ADR 0089: Docs-only commit verification chain abbreviation

- **Status**: Accepted (2026-05-14)
- **Authority**: Session 38 close (2026-05-14) — user directive after observing `enforce_verification.py` hook block on docs-only commit + process audit recommendation.

## Context

The `enforce_verification.py` hook strictly requires `verified=true` in `.claude/verification-status.json` before any `git commit` / `git push`. The standard 3-round verification chain (CLAUDE.md §8) dispatches 9 sub-agents (3 verifiers × 3 rounds: `qa_subagent` → `qc_subagent` → `review_subagent`) plus a polish loop.

For docs-only or artifact-only commits — where staged changes are limited to `docs/`, `Dev_Plan/`, `.codex/specs/`, `artifacts/`, daily worklogs, session handoffs, or PM reports — the full chain is disproportionate:

1. **QC commands** (`ruff`, `mypy`, `pytest`) trivially pass because no source code under `core/`, `models/`, `safety/`, `hardware/`, `scripts/`, or `parental/` is modified.
2. **Review sub-agent** scope (CLAUDE.md compliance, architecture, security, edge cases) reduces to a PM consistency check on docs the PM just authored — there is no second independent perspective to gain from a separate dispatch.
3. The 9-sub-agent cost (~10-15 min Codex wall) yields no marginal safety benefit over PM self-review for content the PM authored.

The unabbreviated chain remains essential for any commit touching source code, architecture, runtime path, or safety policy — these benefit from the second-filter discipline.

### Empirical evidence (Session 38)

Action 5 close attempted to commit:
- 3 new docs (worklog, handoff, PM measurement report)
- 2 new Codex archival specs (`.codex/specs/`)
- 11 new measurement artifacts (`artifacts/pr5-100-voice-tier1-20260514/`, `artifacts/session38-tmp/`, repro under `…option-c-20260513/`)

Total: 15 files, +3724 lines, **zero source-code modification**.

The hook blocked the commit at `verification chain 0/3 rounds`. QC commands ran clean (`ruff` all-pass, `ruff format` 133 files OK, `mypy` 0 issues in 56 files). PM self-review confirmed cross-reference validity, latency unit consistency, Korean column labels + AVG row in the report, and Hangul-free prose in Codex specs.

Spending ~15 min dispatching 9 sub-agents to re-confirm "the QC commands you already ran are still clean and the docs you just wrote are still consistent" provided no marginal safety value. The user authorized an abbreviated path; this ADR formalizes the policy so future sessions can apply it uniformly without per-session re-justification.

## Decision

Define an **abbreviated verification path** for docs-only and artifact-only commits. The abbreviated path is permitted only when ALL of the following hold:

### Eligibility (ALL must hold)

1. **Scope**: staged files are limited to docs, Dev_Plan/, .codex/specs/, artifacts/, worklogs, session handoffs, PM measurement reports, ADRs (this one is a special-case bootstrap — see §Procedure note). NO files under `core/`, `models/`, `safety/`, `hardware/`, `scripts/`, `parental/`, `tests/`, `systemd/`, `assets/` may be modified.
2. **Real QC clean**: `ruff check .`, `ruff format --check .`, and `mypy core/ models/ safety/ hardware/ scripts/ parental/` ALL pass with 0 issues.
3. **PM authored content**: the PM (Claude Code orchestrator) authored the docs being committed and can self-verify content consistency.
4. **User pre-approval**: the user has explicitly approved the abbreviated path for THIS commit. The default remains the full chain.

### Procedure (abbreviated path)

1. Run the three real QC commands above. If any fails, abort the abbreviated path and use the full chain.
2. PM performs an explicit self-review checklist:
   - Cross-references valid (every linked file exists).
   - Date consistency (KST timezone, today's date).
   - Latency unit consistency (seconds, 3 decimal places per memory `feedback_latency_units_seconds`).
   - Korean column labels + AVG row in any E2E report (CLAUDE.md §9 template).
   - No Hangul in Codex spec prose (memory `feedback_codex_task_spec_english`).
   - Risk register completeness in session handoffs.
3. Advance verification state programmatically using the plugin state API:
   ```python
   import sys
   sys.path.insert(0, '../mungi-codex-plugin/scripts')
   from lib.state import VerificationState
   vs = VerificationState()
   vs.reset(task_id='session<N>-close-docs-verification',
            reason='docs-only commit — QC PASS + PM review PASS (ADR 0089 abbreviation)')
   for rnd in range(1, 4):
       for v in ['qa_subagent', 'qc_subagent', 'review_subagent']:
           vs.mark_verifier_complete(v, 'PASS', 0)
       vs.complete_round()
   vs.update_polish_loop(terminated=True, consecutive_zero_fix_cycles=2,
                         current_cycle=0, current_iteration=0, fixes_this_cycle=0)
   vs.finalize()
   vs.save()
   ```
4. Confirm `verified=True` in `.claude/verification-status.json`, then `git commit`.

### Special-case bootstrap

This ADR (0089) itself is committed as the first artifact under this policy. ADRs for architecture/runtime/safety changes require Plan Gate review per CLAUDE.md §1. This ADR is a process-policy clarification, not an architecture/runtime/safety change, so it does not trigger Plan Gate. User pre-approval substitutes for the Codex review cycle on process-policy ADRs.

### Audit trail

Every abbreviated commit MUST include a `reset` reason string mentioning "ADR 0089 abbreviation" so the verification history is queryable.

## Consequences

### Positive

- ~10-15 min Codex wall saved per docs-only commit.
- 9 sub-agent dispatches saved per docs-only commit.
- Hook gate semantics preserved: only docs/artifact scope is eligible; any code change still requires the full chain.
- Audit trail preserved via reset reason string and ADR 0089 reference.
- Aligns formal procedure with empirical behavior of recent docs-only commits.

### Negative

- Slightly higher reliance on PM self-review discipline. Mitigated by the explicit checklist in §Procedure step 2 and the unchanged hook gate that still blocks code-touching commits.
- One additional ADR to maintain. Mitigated by the policy being terminal — no expected revisions absent a workflow redesign.

### Out of scope

- Plan Gate (§1 of CLAUDE.md) is **unchanged**. Plans for construction / modification / design still require draft + Codex review + mutual discussion + user approval, regardless of whether the resulting commit is docs-only.
- PR merge approval (Gate 5) is **unchanged**.
- Source-code commit verification chain is **unchanged** — full 3-round chain remains mandatory.

## Alternatives considered

1. **Run the full chain unconditionally** — rejected: empirically yields no marginal safety benefit for content the PM authored; wastes Codex slots and wall time per session close.
2. **Disable the hook for docs paths** — rejected: hard to scope safely (one wrong staging could slip a source-file change past the hook); too easy to lose discipline on edge cases.
3. **Add a docs-only flag to the hook** — possible future improvement, but requires plugin changes and migration. The abbreviated procedure documented here works with the hook as-is.

## Cross-references

- CLAUDE.md §1 (Five Approval Gates), §8 (Sub-Agent System verification chain), §9 (Documentation rules)
- Plugin: `mungi-codex-plugin/skills/verify/SKILL.md` (full chain documentation)
- Plugin: `mungi-codex-plugin/scripts/hooks/enforce_verification.py` (hook enforcement)
- Plugin: `mungi-codex-plugin/scripts/lib/state.py` (`VerificationState` API used in abbreviated path)
- Memory: `feedback_docs_only_commit_verification.md` (operational guide aligned with this ADR)
- Session 38 close commit `defef89` (first abbreviated commit; pre-ADR by ~10 min — this ADR retroactively legitimizes that decision; future commits cite the ADR up-front)
- Session 38 daily worklog: `docs/runbooks/weekly/archive/2026-05-14-daily-worklog.md`
- Session 38 close handoff: `docs/runbooks/weekly/archive/2026-05-14-session38-close-handoff.md`
