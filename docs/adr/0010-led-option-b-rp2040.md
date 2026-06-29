# ADR 0010: LED Indicator via RP2040 USB Serial (Option B)

- **Status**: Accepted
- **Date**: 2026-03-17
- **Context**: LED feedback hardware for child conversation device

## Context

The Mungi device requires visual LED feedback so children can
understand the device state (listening, thinking, speaking, etc.).
Two options were evaluated:

- **Option A**: Drive LEDs directly from Jetson Orin Nano GPIO.
- **Option B**: Offload LED control to a Raspberry Pi Pico (RP2040)
  connected via USB serial.

Option A was rejected because:

1. Jetson Orin Nano GPIO pinout is limited and shared with other
   peripherals (audio codec, future sensors).
2. WS2812B LEDs require precise timing that competes with the main
   CPU running AI inference workloads.
3. Tight coupling between LED firmware and the main system
   complicates updates and debugging.

## Decision

**Use Raspberry Pi Pico (RP2040) + WS2812B LED ring + USB serial
(pyserial) for LED status indication.**

### Architecture

```text
Python host (Jetson)          USB serial           RP2040 Pico
  LedController  ──pyserial──►  /dev/ttyACM0  ──►  Firmware
                                                     │
                                                 WS2812B LED ring
```

The Python host sends simple ASCII state commands over serial.
The RP2040 firmware maps each command to an LED animation.

### LED States

| Command      | Animation              | Meaning           |
|--------------|------------------------|--------------------|
| `IDLE`       | Breathing blue         | Standby            |
| `LISTENING`  | Solid green            | Capturing speech   |
| `THINKING`   | Pulsing yellow         | LLM generating     |
| `SPEAKING`   | Flowing cyan           | TTS playback       |
| `ERROR`      | Red flash              | Error condition    |

### Key Design Choices

1. **CPU offload** — RP2040 handles timing-critical NeoPixel
   operations independently from Jetson AI workloads.
2. **Clean protocol boundary** — Simple serial ASCII commands
   decouple host software from LED firmware.
3. **Independent firmware updates** — RP2040 firmware can be
   flashed via UF2 without touching the main system.
4. **Minimal BOM impact** — RP2040 costs approximately $4.
5. **pyserial dependency** — Added to
   `Dev_Plan/requirements-jetson.txt`.

### Parts Status

Hardware purchase and assembly proceeds after Phase 0 AI
pipeline validation is complete.

## Consequences

- **Positive**: Frees Jetson GPIO for other peripherals; isolates
  timing-critical LED control from AI inference; enables
  independent LED firmware iteration; simple serial protocol is
  easy to test and mock.
- **Negative**: Adds a secondary microcontroller to the BOM;
  requires RP2040 firmware development and maintenance; USB
  serial adds a potential failure point (cable, enumeration).
- **Risks**: USB device enumeration order may vary across reboots
  (mitigated by udev rules for stable `/dev/ttyACM0` path).
  RP2040 firmware bugs could cause LED hangs (mitigated by
  watchdog timer in firmware).

## Related

- CLAUDE.md section 3 (Hardware Interfaces)
- Development specification (hardware integration plan)

---

## Update — 2026-04-29

**Effective**: 2026-04-29
**Authority**: User direction 2026-04-28 + `docs/archived/dev-plan/2026-04-28-Plan-v21-v27-Update-Plan-v4.md` (Gate 1 final-approval)
**Status change**: **SUPERSEDED** by ADR-NEW-1 (proposed slot 0079 — LED hardware retirement).

The Option B LED hardware (Raspberry Pi Pico RP2040 + WS2812B LED ring + USB serial via `pyserial`) is retired across all active project documentation, code, and dependencies. Visual feedback responsibilities are transferred to the new 4inch HDMI capacitive touch IPS LCD (Waveshare 720×720) per ADR-NEW-2 (proposed slot 0080).

**Downstream impact** (per Plan v4 §5 + §7):
- `hardware/led.py` confirmed absent in current checkout (Codex `platform` verification — Phase B no-op).
- `Dev_Plan/requirements-jetson.txt` `pyserial>=3.5` line removed (Phase A — Claude direct).
- `docs/runbooks/jetson-setup-guide.md` `pyserial`/LED install steps removed.
- `CLAUDE.md §3` Hardware interfaces LED bullet replaced with Display bullet.
- `docs/PROJECT_STATUS.md` P2-5 work item rewritten as display work item.
- `docs/PRODUCT_VISION.md` interface bullet rewritten.
- 4 Dev_Plan target files (v2.1 + Part1/2/3) LED→display swap-in (separate Phase A subtasks).

This Update modifies disposition only. The original Decision body above remains immutable per the ADR immutability rule.
