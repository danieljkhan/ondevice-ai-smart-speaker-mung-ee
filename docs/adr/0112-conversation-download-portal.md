# ADR 0112: Conversation Data Download Portal (Tailscale-only, PIN)

## Status

Accepted (2026-06-18)

Converged through Codex review r1 (PUSH BACK, 5 BLOCK + 7 MAJOR) → r2 (PUSH BACK, residuals) → r3
(APPROVE WITH NOTES) with all findings accepted; user pre-approved the design and the implementation
gate (2026-06-18). Authoritative plan: `Dev_Plan/2026-06-18-conversation-download-portal-plan.md`
(v3).

- **Date**: 2026-06-18
- **Decision owner**: Claude Code (PM) + user direction (Tailscale-only access from any network;
  PIN authentication; text + audio download)
- **Related**: ADR 0110 (conversation memory — produces the data alongside the raw transcript
  store), CLAUDE.md §6 (all conversation data is permanently stored for parental review)

## Context

Conversation data (transcripts + input/output audio) is stored under
`/var/lib/mungi/conversations/` precisely so parents can review it (CLAUDE.md §6). Until now there
was no way to retrieve it off-device without SSH. The parent asked for a simple web page to select
and download sessions to a local PC. The Jetson is **offline** (no inbound internet) but is on a
**Tailscale** mesh (`jetson.local`). Exposing a child's private conversations over a network is
safety-critical and must default to closed.

## Decision

Serve `conversations/` **read-only** from a Python **stdlib `http.server`** daemon
(`parental/download_portal/`) bound to the **`tailscale0` interface only**, gated by a hashed PIN +
per-session CSRF, **disabled by default** behind `MUNGI_DOWNLOAD_PORTAL`, installed but inert until
the operator sets a PIN, writes a Tailscale ACL, and enables the unit.

- **Tailscale-only binding**: the LAN IP changes with the WiFi the Jetson joins; the Tailscale IP is
  stable, encrypted (WireGuard), and device-authenticated. Binding there gives "access from any
  network" with **zero LAN and zero public exposure**, and is resolved dynamically (never
  hardcoded); fail-closed (no LAN/`0.0.0.0` fallback) when Tailscale is down.
- **stdlib http.server**: no offline web framework is available and the device is offline; zero new
  runtime deps and a minimal surface, with security implemented per the plan and a security-first
  test suite.

## Threat model

- **T1** malicious/curious same-LAN device → mitigated: not bound to LAN.
- **T2** unauthorized tailnet device → mitigated: a **Tailscale ACL** limits port 8765 to the
  parent's device/group + a `100.64.0.0/10` peer check (operator step, runbook-verified).
- **T3** lost/compromised parent device → residual: the holder has tailnet + PIN-gated access until
  the device is removed from the tailnet and/or the PIN is rotated (documented).
- **T4** path traversal / TOCTOU → mitigated: fd-anchored per-component resolution with
  `O_NOFOLLOW`, symlink rejection at every level.
- **T5** resource exhaustion → mitigated: download semaphore, `MemoryMax`/`TasksMax`, timeouts,
  `429`/`503`, streamed STORE-zip.
- **T6** secret leakage → mitigated: PBKDF2-hashed PIN, never logged, CLI-set only; the throttle is
  checked before any PBKDF2 work.

## Alternatives considered

- **LAN / `0.0.0.0` binding** — REJECTED: exposes child data to the home network and is fragile
  across WiFi changes.
- **Public / port-forward** — REJECTED: exposes child data to the internet.
- **Tailscale Funnel/Serve** — REJECTED: publishes the port to the public internet; explicitly
  banned for this service.
- **A web framework (Flask/FastAPI)** — REJECTED: no offline wheel on the device + larger surface;
  stdlib suffices with rigorous hand-implemented guards.
- **HTTPS in v1** — DEFERRED to v2: Tailscale (WireGuard) already encrypts the only transport
  end-to-end.

## Consequences

### Positive
- Child data never touches the LAN or the public internet; reachable only by the parent's
  Tailscale-authorized devices, PIN-gated, audit-logged, read-only.
- Off by default; no new runtime dependency; bounded resource use within the 8 GB budget.

### Negative / trade-offs
- The strongest parent-only guarantee lives in the **Tailscale ACL** (operator-configured), not the
  app; the runbook makes it a required, verified step, with the `100.64/10` peer check as
  defense-in-depth.
- Hand-rolled stdlib server carries more security-implementation responsibility — mitigated by the
  security-first test suite and the 3-round review.
- v1 is download-only (no in-portal deletion); plaintext HTTP on the (already-encrypted) Tailscale
  hop until HTTPS lands in v2.

## Implementation

Per plan v3: `parental/download_portal/` (`auth`, `data`, `server`, `cli`, `__main__`, `assets`),
`systemd/mungi-download-portal.service` (flag-gated via `ExecCondition`, sandboxed), a security-first
test suite, and an operator runbook (PIN init → flag → Tailscale ACL → enable). Default OFF.

## References

- `Dev_Plan/2026-06-18-conversation-download-portal-plan.md` (v3; r1/r2/r3 dispositions inline)
- `.codex/chat/handoff.md` review rounds (r1 PUSH BACK → r2 PUSH BACK → r3 APPROVE WITH NOTES)
