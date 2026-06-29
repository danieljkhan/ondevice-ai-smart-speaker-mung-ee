# ADR 0080: 4inch HDMI Capacitive Touch IPS LCD Adoption

- **Status**: Accepted
- **Date**: 2026-04-29
- **Authority**: `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval, 2026-04-29)
- **Related**: ADR 0079 (LED retirement, supersedes ADR 0010)

## Context

Plan v2.1 originally relied on a WS2812B LED ring for visual feedback. The Plan v2.1+v2.7 update cycle (Sessions 18–20) replaces that LED path with a richer visual surface: a 4inch HDMI capacitive touch IPS LCD. The display covers face/mood expressions, alert overlays (`BATTERY_LOW`, `WATCHDOG_TIMEOUT`), and a Phase 2+ touch-input modality that the LED path could not.

## Decision

Adopt the **Waveshare 4inch HDMI Capacitive Touch IPS LCD (C)** (720×720, IPS panel, capacitive multi-touch) as the sole visual feedback + interaction surface, with the following integration shape:

1. **Connection**: DisplayPort → HDMI Active adapter (~₩15,000) + USB-C cable (combined power + multi-touch HID over USB-C).
   - The Jetson Orin Nano Super exposes DisplayPort, NOT HDMI directly. An Active DP→HDMI adapter is required (passive adapters do not work).
2. **Mode line**: 720×720 @ ~60 Hz via xrandr/xorg.conf custom mode line (specifics in `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part1_HW_Setup.md`).
3. **HDMI audio passthrough**: disabled (system audio remains on Waveshare USB Audio, see ADR 0052).
4. **Software stack** (design only; runtime build is a separate Plan track):
   - `hardware/display.py` (target ≥350 LOC) implementing `DisplayController` + `FaceExpression` + `DisplayAlert` + touch-event handler. **(superseded by ADR 0095, adopted 2026-05-27 — runtime renderer landed in `core/character_renderer.py` instead of `hardware/display.py` per ADR 0095 §Context item 1.)**
   - Rendering: `pygame>=2.5.0` (deferred dep — added when `hardware/display.py` runtime Plan begins, not in Plan v4 to avoid orphan deps). **(superseded by ADR 0095, adopted 2026-05-27 — dependency added as `pygame>=2.6.0,<2.7` in `Dev_Plan/requirements-core.txt` per ADR 0095 D2.)**
   - Multi-touch input: `python-evdev>=1.6.0` (deferred dep, same rationale).
5. **PTT button retained**: child-UX safety net for cases where the display is unresponsive or in `BATTERY_LOW` state. Touch is supplementary to PTT, not replacement.
6. **BOM rows added** (in `docs/archived/dev-plan/Mungi_Development_Plan_v2_1_clean.md §6`):
   - 디스플레이: ₩110,000 (단가 검증 필요 — `🔴`)
   - DP→HDMI Active 어댑터: ₩15,000 (DP↔HDMI 호환성 검증 필요 — `🔴`)
   - USB-C 케이블 (전원+HID): 잡비 처리

## Consequences

### Positive
- Richer visual feedback than LED ring (face expressions, text overlays).
- Touch input becomes available for Phase 2+ child interactions.
- Single screen handles all visual responsibility (mood, alerts, optional UI).
- BOM net delta vs LED: +₩108,000 (per Plan v4 §9 risk #2).

### Negative
- Single point of visual failure (mitigated by retained PTT + audio cues — see ADR 0079).
- DP→HDMI active adapter is an extra HW dependency requiring compatibility verification.
- Display power draw and mounting impact battery sizing and chassis volume — flagged as `🔴` items pending mock-up measurement (Phase 0 protocol in Part1).
- The 720×720 unconventional resolution requires custom xorg.conf mode line — risk of platform regressions across Jetson kernel/Mesa updates.
- `pygame` + `python-evdev` are deferred until the `hardware/display.py` runtime Plan (no orphan deps in Plan v4).

## Verification

Phased verification by PR landing:

**PR-1 (this ADR + initial truth-layer docs)**:
- `CLAUDE.md §3` Hardware interfaces lists the Display bullet (Waveshare 4inch HDMI Capacitive Touch IPS LCD).
- `docs/runbooks/baseline-stack-and-models.md` mirrors the Display bullet.
- `docs/PROJECT_STATUS.md` P2-5 work item rewritten as the display work item.
- `docs/PRODUCT_VISION.md` interface bullet lists the 4inch HDMI 정전식 터치 디스플레이.

**Deferred to a follow-up Phase A commit on the same PR-1 branch (4 Dev_Plan target files A4–A7)**:
- `docs/archived/dev-plan/Mungi_Development_Plan_v2_1_clean.md §7` BOM budget table sums correctly with the +₩108,000 net (3 new lines: 디스플레이 ₩110,000 + DP→HDMI Active ₩15,000 + USB-C 잡비; LED + Pico rows removed).
- `docs/archived/dev-plan/Mungi_Development_Plan_v2_1_clean.md` 부록 A includes the `hardware/display.py` skeleton (DisplayController + FaceExpression + DisplayAlert).
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part1_HW_Setup.md` includes Pre-Phase 0 mock-up measurement protocol, DP→HDMI adapter warning, HDMI audio passthrough disable, and xrandr/xorg 720×720 mode-line setup.
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part3_Product_Build.md` includes the `hardware/display.py` module section + state machine display calls.

**Deferred to follow-up runtime build Plan**:
- Actual `hardware/display.py` source code + `pygame`/`python-evdev` dependency adds.
- Procurement / mock-up / mounting work.

## Related

- ADR 0079 (LED retirement)
- ADR 0052 (Jetson ALSA default routing override — audio remains on USB card, NOT HDMI)
- `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` §4 Touchscreen swap-in
- Waveshare product reference (resolved at draft time)
