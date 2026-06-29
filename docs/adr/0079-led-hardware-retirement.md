# ADR 0079: LED Hardware Retirement (WS2812B + RP2040)

- **Status**: Accepted
- **Date**: 2026-04-29
- **Authority**: User direction 2026-04-28 + `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval, 2026-04-29)
- **Supersedes**: ADR 0010 (LED Indicator via RP2040 USB Serial — Option B)

## Context

Plan v2.1 originally specified visual feedback via a WS2812B LED ring driven by a Raspberry Pi Pico (RP2040) over USB serial (`pyserial`). ADR 0010 ratified this Option B in 2026-03-17. Since then:

1. Hardware procurement was deferred and never executed; `hardware/led.py` was removed from the repository before runtime cutover.
2. The Plan v2.1+v2.7 update cycle (2026-04-28) introduced a 4inch HDMI capacitive touch IPS LCD (Waveshare 720×720) as the visual feedback + interaction surface.
3. Maintaining a parallel LED indicator path alongside the new display would duplicate the visual feedback responsibility and consume a USB port + dependency footprint without product value.

## Decision

Retire the LED Option B hardware path entirely:

1. **Hardware**: WS2812B LED ring + RP2040 Raspberry Pi Pico are not part of the product BOM.
2. **Software**: `hardware/led.py` remains absent; no LED control module is reintroduced. Visual feedback states (`BATTERY_LOW`, `WATCHDOG_TIMEOUT`, persona/mood expressions) are routed to `hardware/display.py` (runtime build is a separate Plan track) via `DisplayController` / `DisplayAlert` / `FaceExpression` abstractions.
3. **Dependencies**: `pyserial` is removed from `Dev_Plan/requirements-jetson.txt` (verified LED-only — no Python `import serial` usage in any active code path).
4. **Documentation**: `CLAUDE.md §3`, `docs/runbooks/baseline-stack-and-models.md`, `docs/PROJECT_STATUS.md`, `docs/PRODUCT_VISION.md`, and the 4 Dev_Plan target files (v2.1 + Part1/2/3) drop all WS2812B / RP2040 / Pico / USB-serial references in their active descriptions.
5. **ADR 0010**: Marked SUPERSEDED by this ADR via Update-section append (body immutable).

## Consequences

### Positive
- Eliminates an entire hardware subsystem with zero product loss (display covers the visual feedback role).
- Removes one USB port consumer and one Python dependency (`pyserial`).
- Closes a long-pending procurement and assembly path that was never going to materialize.
- BOM net delta: −₩7,000 (LED ring) − ₩10,000 (Pico) = −₩17,000 toward the +₩125,000 display addition (net BOM impact in Plan v4 §9 risk #2: +₩108,000).

### Negative
- The 4inch display becomes a single point of failure for visual feedback. Mitigation: `BATTERY_LOW` and `WATCHDOG_TIMEOUT` overlays in the display layer; if the display itself fails, audio cues (existing TTS) and the PTT button (kept for child UX safety net) remain operational.
- Documents authored under Option B (e.g., the LED-Option-B section of `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part2_Dev_Environment.md`) require rewrites to remove orphan references (handled in Plan v4 Phase A).

## Verification

Phased verification by PR landing:

**PR-1 (this ADR + initial truth-layer docs + Phase A13 deps cleanup)**:
- `CLAUDE.md §3` Hardware interfaces no longer lists the LED bullet.
- `docs/runbooks/baseline-stack-and-models.md` Hardware interfaces LED bullet removed.
- `docs/PROJECT_STATUS.md` P2-5 work item rewritten away from LED.
- `docs/PRODUCT_VISION.md` interface bullet no longer lists WS2812B LED.
- `docs/runbooks/jetson-setup-guide.md` `pyserial` install line + `openWakeWord` and other LED-only references retired.
- `Dev_Plan/requirements-jetson.txt` `pyserial>=3.5` line removed (Plan v4 Phase A13).
- ADR 0010 has appended Update section marking it Superseded by this ADR.

**Deferred to a follow-up Phase A commit on the same PR-1 branch (4 Dev_Plan target files A4–A7)**:
- `docs/archived/dev-plan/Mungi_Development_Plan_v2_1_clean.md` ASCII diagram + directory tree + 부록 A LED references replaced by display references.
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part1_HW_Setup.md` LED block removal verification.
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part2_Dev_Environment.md` `spi` group / `adafruit-circuitpython-neopixel-spi` / WS2812 hardware table row removed (per FINDING-4 lines 214–217, 377–386, 1269–1273).
- `docs/archived/dev-plan/Mungi_SW_Build_Plan_v2_7_Part3_Product_Build.md` LED control calls replaced with display calls.

**Deferred to PR-2 (Codex `platform` and `test` roles)**:
- `hardware/led.py` confirmed absent in current checkout (Codex `platform` Phase B verification).
- `systemd/` confirmed empty of LED-related units (Codex `platform` Phase B verification).
- LED-related test fixtures removed (Phase D — Codex `test` role).
- Cross-reference scan over the entire repository (Plan v4 §10 regex) returns only retirement-marker contexts — runs as part of PR-2 polish loop.

## Related

- ADR 0010 (LED Option B RP2040) — superseded
- ADR-NEW-2 / proposed slot 0080 (4inch HDMI Touchscreen Adoption)
- `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` §5 LED retirement table
- `CLAUDE.md §3` (Hardware interfaces — Display bullet)
