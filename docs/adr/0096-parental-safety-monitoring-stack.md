# ADR 0096: Parental Safety Monitoring Stack (WiFi CSI sensing — Workstream C)

- **Status**: Accepted (2026-05-29 — user mini-approval 완료, Plan v3 §8 P-ADR gate 통과; P0 manual phase 진입 가능)
- **Date**: 2026-05-29
- **Authority**: `Dev_Plan/2026-05-26-parental-safety-monitoring-plan.md` v3 (Codex-APPROVED 2026-05-29 r3 APPROVE WITH NOTES; 28/28 cumulative r1+r2 closures)
- **Related**: ADR 0001 (repo-runtime separation — this ADR extends `/var/lib/mungi/`), CLAUDE.md §1 product vision ("the safest AI friend — a child's very first"), CLAUDE.md §7 rule 4 (architecture/runtime-path change requires ADR)
- **Memory refs**: `project_parental_monitoring_scope.md` (non-commercial academic/PoC; legal compliance descoped per user directive)

## Context

Mungi's vision is the safest AI friend for children under 10. Active conversation safety is delivered by the existing voice pipeline (Silero VAD → Qwen3-ASR → Gemma 4 → Supertonic → safety filters). What is missing is **passive ambient safety** — presence, motion, breathing, heart rate — without a camera (camera explicitly rejected as privacy-incompatible with the product persona).

Session 77 (2026-05-26) verified end-to-end feasibility of `ruvnet/RuView` ecosystem on Jetson Orin Nano:
- `ruvnet/wifi-densepose:latest` Docker image pulled on `mungi@jetson.local`
- Idle footprint: 1.75 MiB RAM / 1.23% CPU (vs. 7.4 GiB Jetson budget — negligible)
- Hardware: ESP32-S3 × 4 + 18650 5V battery packs (user-owned)

This ADR records the architecture / runtime-path / safety-policy decisions for adopting WiFi CSI sensing as Mungi's third orthogonal workstream (Workstream C), parallel to Workstream A (voice pipeline) and Workstream B (touchscreen).

Plan v3 went through three Codex review rounds (r1 BLOCK with 27 findings → v2 → r2 PUSH BACK with 5 → v3 → r3 APPROVE WITH NOTES) and integrated 28/28 actionable findings before this ADR was drafted.

## Decision

### D1 — Adopt WiFi CSI sensing as Mungi extension (Workstream C)

Add `parental/` as a new top-level Python module. Use the `ruvnet/RuView` upstream ecosystem (Docker container `ruvnet/wifi-densepose:<pinned-digest>` + ESP32-S3 firmware from `firmware/esp32-csi-node/`).

- **Why CSI, not camera**: privacy-compatible with the product persona (no imaging).
- **Why RuView, not from-scratch**: full upstream stack (firmware + inference + dashboard) already exists; building from scratch is out of v1 scope.
- **Why Docker, not PyPI wheel**: container isolation + reproducibility + bundled inference dependencies (the PyPI wheel ecosystem is unstable for this ARM64 use case). Per Plan v3 §11 D1.

### D2 — Runtime path: Docker on Jetson, NOT bare Python

`ruvnet/wifi-densepose:<sha256-pinned>` runs on Jetson via systemd unit `systemd/wifi-densepose.service` with Docker compose `systemd/wifi-densepose.compose.yml`. Container is restart-on-failure with explicit `--cpus 1.5 --memory 1500m` limits + log rotation `--log-opt max-size=10m --log-opt max-file=3`.

Network mode = `host` (CSI UDP/TCP ingress on Jetson LAN IP, not Docker NAT).

### D3 — Mungi consumer: `parental/sensing_daemon.py` (single service, multi-threaded)

- Main thread: Flask + waitress web server (dashboard at port 8080).
- Background thread: paho-mqtt `client.loop_start()` subscribed to `mungi/sensing/<event_type>` topics on localhost broker (127.0.0.1:1883).
- Shared state: lock-protected ring buffer (`deque(maxlen=1000)`) + daily summary + notification state.
- Audio playback: independent `SupertonicEngine` instance owned by `parental/notifier.py` (NOT coupled to `core/pipeline.ConversationPipeline`; pipeline-aware suppression deferred to a future Plan that adds a public `is_idle()` accessor).

### D4 — Mutable runtime path extension: `/var/lib/mungi/sensing/`

Matches the existing `/var/lib/mungi/conversations/` pattern from ADR 0001.

- `events-YYYY-MM-DD.jsonl` — daily-rotated event log
- Retention: `MUNGI_PARENTAL_RETENTION_DAYS` (default 30 days)
- Read tolerance: missing file → empty result; truncated/malformed last line → skip with warn; oversize day file (> 50 MB) → warn but read continues

### D5 — Dashboard auth: HTTP Basic + bcrypt PIN + LAN UFW (defense-in-depth)

- Stack: Flask + waitress + Jinja2 + bcrypt + flask-httpauth (all already in `Dev_Plan/requirements-core.txt:31-37`; no new web deps).
- PIN: 6-digit numeric, bcrypt-hashed, stored in `/var/lib/mungi/config/parental.env` (root:root 0600) as `MUNGI_PARENTAL_DASHBOARD_PIN=<hash>`.
- Manual rotation procedure documented in `docs/runbooks/parental-monitoring-operations.md` (Plan v3 §15).
- LAN firewall (`ufw allow from 192.168.0.0/16 to any port 8080 proto tcp`) as secondary control.
- Non-diagnostic safety banner mandatory on every dashboard page (Korean default): "실험용 환경 모니터링이며 의료기기가 아닙니다. 진단/응급 대응에 의존하지 마세요."

### D6 — Notification 5-gate stack (per Plan v3 §6.5.3)

All five gates must pass in order before audio dispatch:
1. **Master enable**: `MUNGI_PARENTAL_NOTIFY_ENABLED=true` (default `false`)
2. **Presence gate**: child currently in room
3. **Quiet hours gate**: outside `MUNGI_PARENTAL_QUIET_HOURS` (default `00:00-06:00`)
4. **Cooldown gate**: no same-kind dispatch within 10 min
5. **Sustained gate**: condition held ≥ 60 s

Non-diagnostic safety prefix prepended to first daily notification: "실험용 모니터링입니다. 진단이 아닙니다."

### D7 — MQTT event schema (v1, normative)

Per Plan v3 §6.6:
- Topic: `mungi/sensing/<event_type>` ∈ {presence, motion, breathing, heart_rate}
- Payload fields: `schema_version=1`, `event`, `node_id` (1–4), `timestamp` (ISO 8601), `value` (typed per event), `unit`, optional `raw`
- `max_payload_bytes = 16384` (16 KiB) at subscriber; oversize → drop + counter
- `raw` field truncation at 8192 bytes JSON-serialized → drop + `raw_truncated: true` flag

### D8 — Cognitum upstream registry suppression: pre-start UFW deny ordering

Per Plan v3 §6.1: install UFW outbound deny rule BEFORE first networked container start (or use `--network none` diagnostic mode). Evidence: chronological log + tcpdump from deploy start through 60 s of container life shows zero outbound packets to `storage.googleapis.com`.

### D9 — Codex implementation role split (per CLAUDE.md §8)

Plan v3 §8 rollout:
- P-ADR: PM direct (this ADR)
- P0: PM + user manual (Jetson Phase 0 verification)
- P0-closeout: PM direct (plan/spec refresh)
- P0-deps: Codex `platform` (paho-mqtt requirement)
- P1a: Codex `feature` (parental/ production code)
- P1b: Codex `test` (parental/ tests)
- P2a: Codex `feature` (parental/dashboard/)
- P2b: Codex `test` (parental/dashboard/ tests)
- P2c: Codex `platform` (systemd/ + scripts/)
- P3: PM + user manual (4-node deployment + 24h soak)

## Consequences

### Positive

- Passive ambient safety becomes the third Mungi safety layer (camera-free, privacy-compatible)
- `parental/` module is namespace-isolated (no collision with existing layers)
- Reuses existing Flask/waitress/bcrypt stack (no new web deps)
- Container isolation contains the upstream RuView blast radius
- 5-gate notification stack provides defense-in-depth against false alarms waking the child

### Negative / trade-offs

- Adds Docker as a runtime dependency for the parental track (not previously required for voice pipeline)
- ESP32-S3 firmware trust contract relies on upstream release tag + SHA256 checksums (Phase 0 O8 must record these)
- WebKit-style continuous WebView for the dashboard adds ~80-150 MB system memory (per Plan v3 §7.2 D scenario)
- Manual operator PIN rotation (no web reset endpoint in v1)
- Child vital-sign data (breathing, heart rate) requires non-diagnostic UX wording to avoid false confidence/panic

### Out of scope (deferred)

- Fall detection (Plan v3 §10; P3 deferred per user directive)
- Mobile app / parent webhook push (web dashboard is sufficient for v1)
- Cloud sync (intentional privacy-first; local-only)
- Pipeline-aware notification suppression (deferred; needs new `ConversationPipeline.is_idle()` accessor — future feature plan)
- Commercial certification (academic/PoC scope; legal compliance descoped per user directive)
- Multi-room mesh (single bedroom only in v1)

## Validation criteria

- [ ] User mini-approval of this ADR (P-ADR gate)
- [ ] Plan v3 §7.2 Phase 0 verification completed (O1–O8 + memory scenarios resolved on Jetson)
- [ ] Phase 0 runbook authored at `docs/runbooks/parental-monitoring-phase0.md`
- [ ] paho-mqtt ARM64 wheel availability verified (P0-deps gate)
- [ ] P1a-P1b-P2a-P2b-P2c Codex dispatches each pass §7.1.b phase gates + full repo gates
- [ ] §15 operator runbook `docs/runbooks/parental-monitoring-operations.md` authored before P3
- [ ] P3 user 24-h soak signs off acceptable false-positive rate
- [ ] Combined active-state Jetson RAM ≤ 5500 MB (1-ESP32 active conversation) and ≤ 6000 MB critical guard (4-ESP32 + concurrent conversation + dispatched notification)

## Reference map

| Artifact | Role |
|---|---|
| Plan v3 | Implementation plan + verification gates (28/28 Codex r1+r2 closures) |
| Plan v3 §6.5 | 5-gate notification stack normative contract |
| Plan v3 §6.6 | MQTT event schema v1 |
| Plan v3 §7.2 | Phase 0 verification (O1–O8 + active-state memory) |
| Plan v3 §8 | Codex role split rollout |
| Plan v3 §14 | This ADR gate |
| Plan v3 §15 | Operator runbook references |
| Codex r1/r2/r3 reviews | `.codex/specs/parental-safety-monitoring-plan-review-v{1-r1,2-r2,3-r3}.md` |
| r1/r2 discussion records | `Dev_Plan/2026-05-29-parental-safety-monitoring-plan-codex-review-discussion-r{1,2}.md` |
| ADR 0001 | `/var/lib/mungi/` mutable-root pattern (extended here) |
| CLAUDE.md §1 | Product vision |
| CLAUDE.md §6 | Jetson 8 GB memory budget |
| CLAUDE.md §7 rule 4 | ADR requirement for architecture/runtime-path/safety-policy changes |
| CLAUDE.md §8 | Sub-agent scope ownership (P1/P2 role split per H1 BLOCK fix) |

## Update (2026-05-29 — Acceptance event)

- ADR status: Proposed → **Accepted** via user mini-approval (Session 86 start-of-work).
- P-ADR gate (Plan v3 §8) 통과 — Plan v3 §8 P0 manual phase 진입 가능.
- Validation criterion `[x] User mini-approval of this ADR` 충족.
- 잔여 validation criteria는 Plan v3 §7 verification gates와 §8 rollout phases 진행 시 순차 충족 예정.
- 원본 Context / Decision / Consequences 섹션 미변경 (`feedback_adr_immutability` 준수). 본 Update 노트는 acceptance event 감사 추적 목적.
