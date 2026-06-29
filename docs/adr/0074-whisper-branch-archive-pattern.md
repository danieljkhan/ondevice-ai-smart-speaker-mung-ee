# ADR 0074 — Branch archival pattern for discarded experimental tracks

- **Status**: Accepted
- **Date**: 2026-04-25
- **Decision owner**: Claude Code (orchestrator) + user approval
- **Related**: `docs/runbooks/weekly/archive/2026-04-25-whisper-branch-archive-note.md` (precedent application), `docs/runbooks/weekly/archive/2026-04-25-daily-worklog.md` §Whisper branch archived (this session's execution log)

## Context

When an experimental feature branch (e.g., `Whisper`) is discarded by user decision but has accumulated meaningful work product (Plan documents, ADR drafts, Codex review reports, daily worklogs, session handoffs) over a multi-day cycle, deleting the remote branch outright loses the history and forces future contributors to chase commit SHAs by memory. Conversely, leaving the discarded branch named after its original feature creates persistent confusion: future sessions reading the branch list see "Whisper" and assume Whisper-related work is in progress, even though the actual content has drifted into Safety hardening / Reboot observability / Pipeline analysis docs.

This ADR was written immediately after the first concrete application of the pattern, on 2026-04-25, when the `Whisper` branch (Whisper large-v3-turbo A/B bench plan) was archived after its plan was discarded 2026-04-24. The branch had drifted to hold 17 commits worth of unrelated docs (Safety hardening drafts, Silent-reboot observability Plan Gate artifacts, pipeline bottleneck analysis, Session 11/12 close handoffs) authored in parallel with `dev`'s own work on the same topics (PRs #45/#46/#48). Three branch-cleanup options were considered:

1. **Squash-merge `Whisper` → `dev`** — would have created ADR-number conflicts (Whisper's ADR 0073 = `silent-reboot-preventive-observability`; dev's ADR 0073 = `llm-primary-gemma4-swap`) and forced cherry-pick decisions on every duplicated Session 11/12 close doc. Estimated 2–4 h of conflict resolution.
2. **Cherry-pick selective files to `dev`** — would have required ADR renumbering (`gemma3` → `gemma4` mass replace etc.) and a per-file value judgment for ~17 files. High cognitive load, easy to miss something.
3. **Archive rename + selective cherry-pick later** — preserve full Whisper history under a clearly archive-named branch, delete the original branch, leave cherry-pick to be done at the moment a specific archive artifact becomes load-bearing for ongoing work. Low immediate cost, high optionality.

Option 3 was chosen and executed. This ADR codifies the pattern so it can be applied uniformly to future discarded experimental tracks without re-debating the choice each time.

## Decision

When an experimental feature branch is discarded by user decision and the branch contains work product worth preserving:

1. **Push the branch to a new archive-namespace ref**: `origin/archive/<original-branch-name>-<YYYY-MM-DD>` where the date is the archive date (the day the decision is recorded), not the branch's last-commit date. This is a non-destructive new-ref push, not a force-push.
2. **Delete the original ref**: `git push origin --delete <original-branch-name>` after confirming the archive ref exists at the same SHA. Delete the local branch only after the remote is confirmed gone.
3. **Author a one-page archive note** under `docs/runbooks/weekly/<YYYY-MM-DD>-<branch-name>-archive-note.md` documenting: why archived, where the work went, what was NOT cherry-picked into the active branches and why, how to recover anything from the archive (literal `git show` / `git checkout` commands), and the preserved unique commit SHAs with their merge base.
4. **Commit the archive note to the active integration branch** (typically `dev`), passing the orchestrator's normal Gate 2/3 approval flow.
5. **Do NOT cherry-pick speculatively** at archive time. Cherry-picking is deferred until a specific archive artifact becomes load-bearing for ongoing work (e.g., when implementation of a deferred plan begins on a fresh feature branch). At that point, cherry-pick the specific commit/file with full discussion of any renaming or renumbering required at the cherry-pick PR review time, NOT at archive time.

## Operational rules

- **Archive ref naming**: always `archive/<original-name>-<YYYY-MM-DD>`. The date suffix prevents collision if a future branch reuses the same name.
- **Force-push prohibited** on archive refs: archives are immutable history; if an archived branch needs further commits, create a new archive with a new date.
- **Discoverability**: the archive note is the canonical pointer. Future sessions reading the active branch list should see only live work; "where did the old work go?" must be answerable by `grep -ri archive docs/runbooks/weekly/`.
- **No archive-branch cleanup policy**: archives accumulate over project lifetime. They are cheap (refs only; the actual commits are deduplicated in the object store). Do not introduce a deletion policy without a separate ADR.
- **Cherry-pick at consumption time, not at archive time**: forces the cherry-pick reasoning to happen with the consuming-feature context fresh. Speculative cherry-picks tend to introduce churn and conflicts that the consumer doesn't even need.

## Alternatives considered

1. **Git tags instead of archive branches** — tags are lighter but harder to discover via `git branch -a`. Branches under `archive/` namespace appear in branch listings but are visually segregated by prefix. The branch form also keeps the work cleanly checkout-able for inspection (`git checkout origin/archive/whisper-2026-04-25`), which a tag-only flow makes one step longer.
2. **Permanent branch retention without archive prefix** — leaving `Whisper` alive forever was rejected because the name no longer described content (drifted into Safety + Reboot observability docs). Stale-named branches are cognitive load on every future session.
3. **Cherry-pick everything immediately + delete branch outright** — was rejected because of the ADR-number conflict and the Session 11/12 close doc duplication. Each cherry-pick would have required value judgment in real time, and several files genuinely have no ongoing-work consumer (e.g., the discarded Whisper A/B bench plan).

## Consequences

### Positive

- Future sessions see only live work in the branch list; the "what is `Whisper`?" cognitive question disappears.
- Discarded plan history is preserved indefinitely at near-zero storage cost (refs only).
- Archive notes give future contributors a deterministic recovery procedure with literal commands.
- ADR-number / file-name conflicts that would have blocked a forced merge are sidestepped; renaming/renumbering happens at the cherry-pick PR moment with full consuming-feature context.

### Negative / trade-offs

- The archive-ref naming convention requires discipline; one-off `tmp-old-whisper`-style names defeat the discoverability guarantee.
- Archive notes need to be authored at archive time, not deferred. A discarded branch with no archive note is harder to recover from than just the ref alone (the recovery procedure isn't obvious to a contributor who didn't run the original archive).
- Cherry-pick-on-demand means future PRs that want archived material pay a small extraction cost; this is by design, but is a non-zero tax compared to "everything is already in `dev`".

### Out of scope

- Deciding when to delete archive refs (never, by default; revisit only with an explicit ADR).
- Automation for archive-note generation (manual for now; if pattern is applied >5 times, consider a `scripts/archive_branch.py` helper).

## First application

Applied to the `Whisper` branch on 2026-04-25:
- Archive ref: `origin/archive/whisper-2026-04-25` (`4381ec2`).
- Archive note: `docs/runbooks/weekly/archive/2026-04-25-whisper-branch-archive-note.md`.
- Original branch deletion: planned 2026-04-25; not executed on that date (verified by `git ls-remote origin Whisper` on 2026-04-26 returning the live ref). Re-execution scheduled for §3.3 of the 2026-04-26 Whisper-divergence resolution plan (`docs/archived/dev-plan/2026-04-26-whisper-divergence-resolution-plan.md` v4 FINAL), immediately after the docs PR carrying ADR 0075 + this correction merges. See `docs/runbooks/weekly/archive/2026-04-25-whisper-branch-archive-note.md` §"2026-04-26 correction note" for context.
- Archive note committed to `dev` as `b959ff3 [docs] archive Whisper branch — record rationale + recovery procedure`.

## Validation criteria

- [x] Archive ref `origin/archive/whisper-2026-04-25` created and verified at `4381ec2`.
- [x] (executed 2026-04-26) Original `origin/Whisper` ref confirmed deleted via `git push origin --delete Whisper` per §3.3 of the Whisper-divergence resolution plan. Post-delete validation: `git ls-remote origin Whisper` returned empty; `git branch --list Whisper` returned empty; archive ref `origin/archive/whisper-2026-04-25` @ `4381ec2` confirmed intact.
- [x] Archive note `docs/runbooks/weekly/archive/2026-04-25-whisper-branch-archive-note.md` exists, lists what was NOT cherry-picked and why, and provides literal recovery commands.
- [x] Archive note committed to `dev` and pushed to `origin/dev`.
- [ ] No cherry-picks from `archive/whisper-2026-04-25` performed yet (intentional — defer until consuming feature begins).

## Future review

If this pattern is applied 3+ more times, review whether to:
- Promote `scripts/archive_branch.py` automation (currently manual).
- Add a CI check that prevents `archive/*` refs from being force-pushed.
- Add documentation to `CLAUDE.md` §7 Git policy referencing this ADR.

Until then, this ADR alone is sufficient.
