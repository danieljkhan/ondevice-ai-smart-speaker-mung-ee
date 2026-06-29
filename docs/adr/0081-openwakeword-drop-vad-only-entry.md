# ADR 0081: openWakeWord Drop — VAD-Driven Single Entry Path

- **Status**: Accepted
- **Date**: 2026-04-29
- **Authority**: `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval, 2026-04-29)

## Context

Plan v2.1 listed `openWakeWord` (~50 MB CPU) as a wake-word detector preceding Silero VAD. In current operational baseline (`CLAUDE.md §3` and `docs/runbooks/baseline-stack-and-models.md`), the entry path is already **Silero VAD only** — openWakeWord was never deployed as the active conversation entry mechanism. The Plan v2.1+v2.7 update cycle (Session 20) makes that ground truth official by removing residual openWakeWord references from active documents.

## Decision

Drop `openWakeWord` from the project entirely:

1. **Pipeline**: Conversation entry is gated by Silero VAD only. No separate wake-word detector.
2. **Model artifacts**: No openWakeWord model is downloaded or stored. The previous `docs/runbooks/jetson-setup-guide.md §7-5 (openWakeWord ~50 MB)` install step is removed.
3. **Dependencies**: `openwakeword` package reference in `requirements-core.txt` (or equivalent) is marked retired. Any test fixtures referencing openWakeWord are dropped (handled in Plan v4 Phase D — Codex `test` role).
4. **Memory budget**: −50 MB CPU steady-state vs original Plan v2.1.

## Consequences

### Positive
- Simpler entry path; one less model to load and version.
- −50 MB CPU memory.
- Aligns documentation with operational truth (which has been VAD-only since deployment).

### Negative
- No "always-listening" wake-word UX. The current product UX requires PTT activation OR VAD-detected speech onset within an active session. This is acceptable for the child-conversation use case (sessions are explicitly opened, not always-on).
- If a future product variant requires hands-free always-on wake-word, a new ADR would re-introduce a wake-word path; this ADR does NOT preclude that future option.

## Verification

This ADR ratifies a phased rollout. Verification clauses are split by PR landing:

**PR-1 (this ADR + truth-layer docs)**:
- `git grep -E "openWakeWord|openwakeword|wakeword\.py"` over modified active documentation in PR-1 returns only retirement-marker contexts (e.g., struck-through table rows, "폐기" annotations).
- `docs/runbooks/jetson-setup-guide.md` does not contain an active openWakeWord install step (§7-5 marked retired, §6 dependency table struck through).
- No `wakeword.py` module exists under `core/`, `models/`, or `hardware/` (verified absent).

**Deferred to PR-2 (Codex `platform` role)**:
- Removal of `openwakeword>=0.6.0` from `Dev_Plan/requirements-core.txt:28`. PR-1 does not edit `requirements-core.txt` because that file's edits fall outside Plan v4 Phase A scope (`Dev_Plan/requirements-jetson.txt` is the only deps file in Phase A; root `requirements-*.txt` and `requirements-core.txt` are platform-tier per AGENTS.md).
- Removal of any `openwakeword` test fixtures or imports under `tests/` (Phase D — Codex `test` role).

## Related

- `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` §3 model stack mapping (WakeWord row → DROPPED)
- `CLAUDE.md §3` baseline stack (Silero VAD as entry; no wake-word listed)
