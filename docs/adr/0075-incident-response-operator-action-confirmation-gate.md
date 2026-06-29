# ADR 0075: Incident response — operator-action hypothesis confirmation gate before preventive-infrastructure design

- **Status**: Accepted (process decision)
- **Date**: 2026-04-26
- **Origin**: drafted on local Whisper branch (now archived) at 2026-04-25 Session 13 close; promoted to dev as a fresh ADR on 2026-04-26 (Session 14, Whisper-divergence resolution plan §3.2). The local branch ADR was numbered 0074 in its source context, but on dev that slot is held by `whisper-branch-archive-pattern`; this ADR is renumbered to 0075 and metadata-normalized for dev.
- **Companion**: ADR 0072 (parent-disclosure invariance) for the "absolute rule" pattern; ADR 0074 (whisper-branch-archive-pattern) as the sibling Session-13-lesson ADR codifying the archive workflow; the now-archived reboot-observability program (preserved at `origin/archive/whisper-2026-04-25:docs/adr/0073-preventive-observability-for-silent-reboot.md`) as the motivating incident.

## Context

On 2026-04-23 at 21:59 KST the Jetson development unit appeared to reboot silently. Persistent journald was not enabled at the time, so post-incident forensic was structurally impossible. The Mungi orchestrator initiated a 9-hypothesis Codex audit (H1 user-initiated power-cycle through H9 NVMe transient) followed by a 3-round Plan Gate 1 over a "preventive observability" program (a draft ADR on the now-archived Whisper branch numbered ADR 0073 there, plan r1 → r2 → r3, 4 Codex review reports, 2 mutual-discussion records — all preserved at `origin/archive/whisper-2026-04-25:docs/adr/0073-preventive-observability-for-silent-reboot.md` and the surrounding `Dev_Plan/` paths). The Plan Gate closed with `APPROVE AS-IS` on r3 and the user final-approved the plan.

Two days later, at the start of the local Session 13 (2026-04-25), the user disclosed that the 21:59 KST event was a manual hard power-cycle they performed (power-cord disconnect). Hypothesis H1 in the original audit had marked this as `High likelihood`. The disclosure invalidated every downstream artifact in the preventive-observability program. The local-branch ADR 0073 was flipped to `Superseded — root cause confirmed` and the supporting documents were archived. (NOTE: dev's ADR 0073 is `llm-primary-gemma4-swap`, an unrelated track; the Superseded ADR 0073 lives only on the archive ref `origin/archive/whisper-2026-04-25` and is NOT the same artifact.)

Approximate effort cost of the wasted cycle:
- 1 Codex 9-hypothesis audit (~30 min)
- 3 rounds of Plan Gate 1 (~25 min Codex per round + ~30 min orchestrator per round = ~165 min)
- 1 ADR draft (~15 min)
- 2 discussion records (~30 min)
- Total ~4 hours of orchestrator + Codex compute on a problem that did not exist.

The Codex H1 verdict did not block the program because the orchestrator interpreted "High likelihood" as one hypothesis among many, deferring confirmation to the broader telemetry-collection design. That deferral is the systemic defect this ADR addresses.

## Decision

**Before designing preventive infrastructure for an incident with one or more plausible operator-action hypotheses on the candidate list, the orchestrator MUST escalate user-confirmation of those hypotheses to a hard gate.** The gate fires before any Codex Plan Gate review of preventive infrastructure is dispatched.

Specifically:

1. **Trigger condition** — The Codex audit (or equivalent root-cause enumeration step) returns one or more hypotheses tagged as operator-action with likelihood `Medium` or higher, AND post-incident forensic data is missing or insufficient to discriminate between operator-action and equipment-fault.

2. **Required step** — The orchestrator presents the operator-action hypotheses to the user with the question: "Did any of these match an action you took during the incident window?" The orchestrator does NOT proceed to Plan Gate 1 dispatch on preventive infrastructure until the user has answered.

3. **Three answer paths**:
   - **Yes (operator action confirmed)** — Close the incident as user-attributable. Document in worklog. Do NOT design preventive infrastructure for this specific incident class unless the user explicitly requests it (e.g., "I do not want to be able to do this again, add a guard").
   - **No (operator action ruled out)** — Proceed to preventive-infrastructure Plan Gate 1 with the operator-action hypotheses formally eliminated.
   - **Unsure / cannot recall** — Treat as `No` for design purposes but flag the residual uncertainty in the resulting plan's risk table. Add detection telemetry that would reliably distinguish operator-action from equipment-fault on next occurrence.

4. **Effort budget heuristic** — If the projected preventive-infrastructure investment exceeds 2 hours of orchestrator + sub-agent effort, the operator-action confirmation gate is mandatory. Below that, judgment-call with default-toward-confirmation.

5. **Audit trail** — The user's confirmation answer is recorded in the daily worklog with timestamp, and the gate's outcome is referenced in any downstream ADR or plan document that proceeds from the incident.

## Consequences

### Positive

- Eliminates the failure mode that consumed ~4 hours during the 2026-04-23 incident response.
- Preserves the value of preventive-infrastructure design for genuine equipment / software faults.
- Makes the operator-action hypothesis a first-class consideration in the incident-response decision tree, instead of one item buried in a hypothesis list.
- Compatible with CLAUDE.md §1 Gate 1 (Plan Gate) — the gate fires before Plan Gate 1 dispatch, so it does not modify Gate 1 mechanics.

### Negative

- Adds one user-touchpoint per significant incident. For minor incidents this is fine; for major user-affecting incidents the user is already involved.
- If the user is unavailable when the gate fires, incident response stalls. Mitigation: the orchestrator MAY proceed with parallel non-design work (forensic data collection, telemetry hygiene like persistent journald) while waiting for the user's confirmation, but MUST NOT begin Plan Gate 1 on preventive infrastructure.
- The gate cannot help if the operator-action hypothesis is missing from the audit. This is a Codex-quality concern, not a process concern; orthogonal to this ADR.

### Risks not eliminated

- A user may answer the gate inaccurately (e.g., misremember whether they pulled the cord). Mitigation: telemetry hygiene (persistent journald, process audit logs) reduces dependence on user recall over time.
- A genuinely-mixed incident (operator action AND equipment fault co-occurred) is not captured by yes/no. Mitigation: if the user answers "Yes but I also think there might be a fault," fall through to the `Unsure` path and proceed to design a narrowly-scoped detection (not full preventive infrastructure) for the residual fault hypothesis.

## Alternatives considered

1. **Status quo (rely on Codex audit ranking)** — Rejected. The Codex audit correctly ranked H1 `High`, but the orchestrator did not treat that ranking as a gating signal. Without explicit gate semantics, ranking alone is too easily deferred.

2. **Always require user confirmation for any preventive-infrastructure design** — Rejected as too heavyweight. For incidents with no plausible operator-action hypothesis (e.g., a kernel panic with full panic trace), the gate adds friction without value.

3. **Add a pre-Plan-Gate "incident classification" sub-step that runs Codex with a stricter operator-action prior** — Rejected as redundant with this ADR. The Codex audit already enumerates hypotheses; the missing piece is the human escalation gate, not more Codex work.

4. **Capture the lesson in CLAUDE.md prose only** — Rejected because process gates that are not formal ADRs tend to be diluted over time. A numbered ADR with explicit trigger conditions is the strongest binding form.

## Implementation

This ADR is self-implementing: it documents a process gate the orchestrator must follow. No code changes are required. CLAUDE.md does not yet need to be updated; if the gate is invoked or skipped in a future incident, that experience will inform whether CLAUDE.md should reference this ADR explicitly.

## References

- `origin/archive/whisper-2026-04-25:docs/adr/0073-preventive-observability-for-silent-reboot.md` — the Superseded ADR (preserved on the archive ref; NOT the same artifact as dev's ADR 0073 `llm-primary-gemma4-swap`).
- `origin/archive/whisper-2026-04-25:docs/runbooks/2026-04-24-jetson-reboot-codex-audit-round1.md` — the 9-hypothesis audit; H1 was `High likelihood`.
- `docs/runbooks/weekly/archive/2026-04-25-daily-worklog.md` §Key events (root-cause disclosure entry).
- `docs/runbooks/weekly/archive/2026-04-25-whisper-branch-archive-note.md` — recovery procedure for the archived reboot-observability content.
- `docs/adr/0074-whisper-branch-archive-pattern.md` — sibling Session-13-lesson ADR codifying the archive workflow used for the superseded preventive-observability program.
- CLAUDE.md §1 Gate 1 (Plan Gate process) — the gate this ADR fires before.
