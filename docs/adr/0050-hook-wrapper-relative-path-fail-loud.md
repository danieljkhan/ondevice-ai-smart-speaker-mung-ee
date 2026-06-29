# ADR 0050: Hook Wrapper Architecture — Relative PLUGIN_ROOT Resolution + Fail-Loud Degradation

- **Status**: Accepted
- **Date**: 2026-04-11

## Context

On 2026-04-10 the three .claude/hooks/*.py thin wrappers (enforce_verification, reset_verification,
post_commit_worklog) were discovered to have been silently no-op since the mungi-codex-plugin
migration commit (167c244). Root cause was three concurrent design flaws: (1) the PLUGIN_ROOT
default was a hardcoded absolute path 'E:/Python_vscode/mungi-codex-plugin' that did not exist on
the current developer machine where the plugin actually sat at 'D:/python_vscode/mungi-codex-
plugin'; (2) the 'plugin script not found' branch returned exit code 0 with no stderr output,
turning an exceptional condition into silent success; (3) the CLAUDE_PLUGIN_ROOT environment
variable override was undocumented, so no operator knew it existed. The combination meant
enforce_verification, reset_verification, and post_commit_worklog all ran as no-ops for weeks.
Actual product-code harm was zero because every commit during the affected window was docs-only, but
the exposure was real: any attempted code commit would have bypassed the 3-round verification chain
and polish loop with no operator signal. Full post-mortem at
docs/runbooks/weekly/archive/2026-04-10-verification-hook-no-op-postmortem.md.

## Decision

The three thin wrappers in .claude/hooks/ must compute PLUGIN_ROOT via runtime-relative path
resolution from Path(__file__).resolve(), rooted in the mungi repository layout: MUNGI_ROOT =
Path(__file__).resolve().parents[2]; DEFAULT_PLUGIN_ROOT = MUNGI_ROOT.parent / 'mungi-codex-plugin'.
The CLAUDE_PLUGIN_ROOT environment variable remains an explicit override with higher precedence.
When the resolved plugin script does not exist, each wrapper MUST emit a single-line stderr warning
in the format '[hook-wrapper:<name>] Plugin script not found at <resolved_path>. Set
CLAUDE_PLUGIN_ROOT or place mungi-codex-plugin as sibling of mungi. Hook enforcement DISABLED.' and
then return 0 (graceful degradation preserved, but the bypass is now visible). Drive-letter absolute
paths are banned from .claude/hooks/*.py entirely. CLAUDE.md §8 'Plugin Workflow' is updated to
document the expected sibling on-disk layout, the relative resolution strategy, and
CLAUDE_PLUGIN_ROOT as an optional override.

## Consequences

Positive: (1) Hook wrappers now work on any drive, any OS, and any clone layout because no absolute
paths are hardcoded. (2) Silent failures are structurally impossible — operators see the stderr
warning on the first affected invocation. (3) CLAUDE_PLUGIN_ROOT is discoverable via CLAUDE.md §8 so
new developers can override the default without reading wrapper source. (4) The remediation
establishes a reusable pattern for any future cross-repo thin wrapper in the mungi ecosystem.
Negative / accepted trade-offs: (1) The wrappers now depend on their own on-disk location being
'inside a mungi repo rooted at parents[2]' — if someone moves the wrappers elsewhere the resolution
breaks, but this is caught by the fail-loud warning. (2) There is still no automated smoke test that
invokes each wrapper on every PR — this is tracked as a follow-up in the post-mortem §7 'Follow-ups'
list. (3) The rule 'no hardcoded drive letters anywhere in .claude/hooks/' is currently enforced by
manual grep during self-verification, not by CI — a ruff custom rule is also a follow-up item.

## Related ADRs

- ADR 0008 — Subagent Migration (established the original plugin architecture that these wrappers delegate to)
- ADR 0042 — Safety Approved Template Router (safety-critical work that the restored verification chain now properly protects)

## References

- docs/runbooks/weekly/archive/2026-04-10-verification-hook-no-op-postmortem.md — Full three-flaw post-mortem with timeline and remediation plan
- CLAUDE.md §5 — 'Paths: always use relative paths in code, configs, and docs' coding rule that was violated by the original hardcoded default
- CLAUDE.md §8 Plugin Workflow — Updated in the same change-set to document sibling layout and CLAUDE_PLUGIN_ROOT override
- Auto-memory feedback_relative_paths_only.md — User feedback rule that existed at migration time and would have caught this at author time if consulted
- .claude/hooks/enforce_verification.py, reset_verification.py, post_commit_worklog.py — The three wrappers refactored in this change-set
