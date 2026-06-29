# ADR 0052 — Jetson ALSA Default Routing Override for USB PnP Audio

- **Status**: Accepted (user-level `~/.asoundrc` applied on Jetson)
- **Date**: 2026-04-12
- **Author**: Claude Code (PM)
- **Related ADRs**: 0049 (E2E Voice Pilot Methodology), 0051 (Pilot Runner Debugging)
- **Related worklog**: `docs/runbooks/weekly/archive/2026-04-11-daily-worklog.md` §14.10

---

## 1. Context

During the 2026-04-11 evening session, PM discovered that the Jetson's
default ALSA routing is misconfigured for the installed audio hardware.
This caused approximately 60-90 minutes of debugging confusion before
the root cause was identified.

### Hardware reality vs. documented specification

**CLAUDE.md §3 states**:
> Audio: Waveshare Audio Card for Jetson Nano (model 19491, SSS1629A5
> codec + dual MEMS mics + PAM8403 amp)

This describes an **I2S HAT** variant: codec chip wired directly to the
Tegra Orin Nano via GPIO/I2S pins, internal MEMS microphones, internal
Class-D amplifier.

**The installed product is different**:

```bash
$ lsusb | grep Audio
Bus 001 Device 004: ID 0c76:1229 JMTek, LLC. USB PnP Audio Device

$ aplay -l | head
card 0: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]
card 1: HDA [NVIDIA Jetson Orin Nano HDA], device 3: HDMI 0
card 2: APE [NVIDIA Jetson Orin Nano APE], device 0: tegra-dlink-0
```

The actual device is:
- **USB 2.0** (not I2S)
- **JMTek chipset** 0c76:1229 (not SSS1629A5)
- **USB card 0** with both input (capture) and output (playback) channels
- **External 3.5 mm mic/headphone jacks** (not internal MEMS/amp)

User confirmed Waveshare sells this USB variant under the same product
branding ("Waveshare Audio Card for Jetson Nano"), hence the naming
confusion. The CLAUDE.md §3 description predates this purchase and
represents an out-of-date plan for an I2S HAT that was never acquired.

### NVIDIA default ALSA configuration

NVIDIA ships Jetson with a factory `/etc/asound.conf` that assumes an
I2S HAT connected to the Tegra APE (Audio Processing Engine):

```
# /etc/asound.conf (NVIDIA default)
pcm.!default {
    type plug
    slave {
        pcm "hw:APE,0"     # ← Tegra APE sound card (I2S HAT interface)
        channels 2
        rate 48000
    }
    hint.description "Tegra APE Soundcard"
}

ctl.!default {
    type hw
    card APE
}
```

This sets the **ALSA "default" pseudo-device** to route to the Tegra APE
hardware, which is the correct choice **if** a HAT is attached to the
I2S pins. For the USB variant, this is the wrong routing — the USB card
is on `hw:0,0`, not `hw:APE,0`.

### Symptoms caused by the mismatch

```bash
# Explicit USB card (works)
$ arecord -D plughw:0,0 -f S16_LE -r 16000 -c 1 -d 1 /tmp/test.wav
$ sox /tmp/test.wav -n stats | grep RMS
RMS lev dB     -55.24     ← normal ambient noise

# ALSA default (silent — routes to APE which has no physical input)
$ arecord -f S16_LE -r 16000 -c 1 -d 1 /tmp/test.wav
$ sox /tmp/test.wav -n stats | grep RMS
RMS lev dB     -inf       ← nothing captured
```

```python
# sounddevice via default (goes through ALSA default → APE → silence)
>>> import sounddevice as sd
>>> sd.default.device
[36, 36]                   # device 36 = "default" pseudo-device
>>> sd.query_devices(36)
{'name': 'default', 'max_input_channels': 128, ...}  # pseudo, routes via asound.conf
>>> a = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32')
>>> sd.wait()
>>> 20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-9)
-180.0                     # effectively zero — silent capture
```

Any Python library using sounddevice/PortAudio (including the E2E voice
runner) that relies on default device selection captures silence, because
PortAudio's default device resolution on Linux goes through ALSA's
`pcm.!default`, which points to the non-functional APE route.

### Impact on debugging

For roughly 60-90 minutes of the 2026-04-11 evening session, PM attempted
to diagnose the E2E voice runner's "no speech detected" symptoms under
the false assumption that the runner had a code bug. The actual problem
was routing: even a correct runner would have captured silence under
the NVIDIA default configuration.

Resolution moment: PM ran a standalone `sd.rec` test (not the runner)
and compared `arecord -D plughw:0,0` (works) vs `arecord` (fails). The
asymmetry pointed to the ALSA default device. Reading `/etc/asound.conf`
confirmed the APE routing.

## 2. Decision

Apply a **user-level ALSA override** via `~/.asoundrc` on the Jetson to
redirect the default device to the USB PnP card (hw:0,0). Do NOT modify
`/etc/asound.conf` (system-level, ships with NVIDIA L4T package, may be
replaced on OS updates).

### Configuration

```
# ~/.asoundrc (created 2026-04-11 ~22:20 KST)
pcm.!default {
    type plug
    slave {
        pcm "hw:0,0"           # USB PnP Audio Device (JMTek 0c76:1229)
        channels 2
        rate 48000
    }
}

ctl.!default {
    type hw
    card 0                     # USB card 0
}
```

The `type plug` slave allows ALSA to automatically convert between
requested formats (e.g., 16 kHz mono float32 for sounddevice) and the
USB hardware's native 48 kHz stereo 16-bit. This is the same plug
conversion ALSA performs for `plughw:0,0` under `arecord -D`.

### Verification after applying

```bash
# Default capture now works
$ arecord -f S16_LE -r 16000 -c 1 -d 1 /tmp/test.wav
$ sox /tmp/test.wav -n stats | grep RMS
RMS lev dB     -55.51        ✅ (ambient noise captured)

# Default playback now routes to USB
$ aplay /usr/share/sounds/alsa/Front_Center.wav
Playing WAVE '...' : Signed 16 bit Little Endian, Rate 48000 Hz, Mono
                                                  ← heard via USB earphone jack
```

```python
>>> import sounddevice as sd
>>> import numpy as np
>>> a = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32')
>>> sd.wait()
>>> 20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-9)
-30.20                       ← ambient room noise, correctly captured via USB
```

**Both capture and playback paths now route to the USB PnP card**,
matching the installed hardware reality.

## 3. Alternatives Considered

### A. Modify `/etc/asound.conf` (system-level)

**Rejected**. Reasons:
- System-level changes require sudo and affect all users.
- NVIDIA L4T updates may overwrite `/etc/asound.conf`, losing the fix.
- Harder to revert or version-track.
- User-level `~/.asoundrc` achieves the same result with smaller
  blast radius.

### B. Hardcode device index in every application

**Rejected**. Reasons:
- Every Python script using sounddevice would need `device=0` explicitly.
- `sd.default.device` would still be wrong for scripts that omit the
  parameter.
- Fragile — one missed script and silent capture returns.
- Propagates hardware-specific assumptions into application code.

### C. Keep NVIDIA default, instruct users to always use `-D plughw:0,0`

**Rejected**. Reasons:
- Only works for arecord/aplay, not for sounddevice/portaudio Python code.
- Requires remembering the flag in every invocation.
- Documentation burden in every runbook and script.

### D. Install the I2S HAT variant that matches CLAUDE.md §3

**Rejected (for now)**. Reasons:
- User has already purchased and installed the USB variant.
- No budget/time to replace hardware.
- USB variant is functional for the pilot test (proven by arecord
  capture + STT smoke test).
- CLAUDE.md §3 can be updated to match reality instead of forcing
  reality to match CLAUDE.md.

## 4. Consequences

### Positive

1. **USB audio Just Works**: After `~/.asoundrc`, all ALSA and PortAudio
   consumers (including sounddevice-based Python code) route through
   the correct USB device by default.
2. **No code changes**: Applications don't need to specify
   `--input-device` or `device=0` anywhere. Default behavior is correct.
3. **System config untouched**: `/etc/asound.conf` stays pristine, no
   sudo required, safe across OS updates.
4. **Easy rollback**: `rm ~/.asoundrc` restores NVIDIA default (useful
   if a future I2S HAT replacement happens).

### Negative

1. **Non-portable**: The `~/.asoundrc` is specific to the current
   Jetson's user account (`mungi`). Fresh Jetson deployments need to
   recreate it. Should be automated via deployment script or documented
   in a runbook.
2. **Depends on USB card ordering**: Uses `hw:0,0`. If another USB audio
   device is plugged in and ordered before the JMTek, `hw:0,0` would
   resolve to the wrong card. Acceptable for the current test setup but
   fragile for production multi-device environments.
3. **Fixed sample rate in config**: The `rate 48000` slave spec is the
   native USB device rate. If the USB device is replaced with one that
   has a different native rate, the config needs updating.

### Documentation updates needed

1. **CLAUDE.md §3** — update audio hardware description from the I2S
   HAT (SSS1629A5 + MEMS + PAM8403) to the actual USB variant (JMTek
   0c76:1229 USB PnP). Note that Waveshare sells both variants under
   the same product branding.
2. **`docs/archived/dev-plan/2026-04-10-E2E-Voice-Pilot-Plan.md` §6 Hardware Setup**
   — add USB variant note and `~/.asoundrc` requirement.
3. **New runbook**: `docs/runbooks/jetson-audio-setup.md` documenting
   the `~/.asoundrc` creation step for fresh Jetson deployments.
4. **Jetson deployment script** (future): add `~/.asoundrc` creation to
   any automated Jetson setup script.

## 5. Operational Notes

### Detection script

A simple check to verify the override is in place:

```bash
$ [ -f ~/.asoundrc ] && \
  arecord -d 1 -f S16_LE -r 16000 -c 1 /tmp/_audio_check.wav 2>&1 | \
  grep -q "Recording WAVE" && \
  sox /tmp/_audio_check.wav -n stats 2>&1 | \
  awk '/RMS lev/ { if ($NF > -80) print "OK"; else print "SILENT"; }'
```

Expected output: `OK`.

### Permanent fix for production

If Mungi ships on Jetson devices with USB audio, consider:
1. Shipping a pre-configured `~/.asoundrc` as part of the installation.
2. Adding a systemd service that verifies ALSA default routing at boot.
3. Or, switching to a system-level `/etc/asound.conf` snippet installed
   by the Mungi deployment package (`/etc/asound.conf.d/10-mungi.conf`
   if the distro supports fragment-based ALSA config).

### Relationship to ADR 0051 (runner debugging)

The runner debugging failures documented in ADR 0051 occurred **after**
the ALSA routing issue was already fixed. The `~/.asoundrc` override is
therefore a **precondition** for any runner testing, not a runner bug
itself. The runner bugs (streaming VAD misuse, off-by-one segmentation,
audio corruption in callback+queue path) are independent of the ALSA
routing fix.

## 6. References

- Jetson `/etc/asound.conf` (NVIDIA L4T default)
- `~/.asoundrc` created on Jetson 2026-04-11 22:20 KST
- `lsusb` output: `0c76:1229 JMTek, LLC. USB PnP Audio Device`
- `aplay -l` output showing `card 0: USB PnP Audio Device`
- Worklog: `docs/runbooks/weekly/archive/2026-04-11-daily-worklog.md` §14.10-14.11
- ADR 0051: Pilot Runner Debugging Rollback
- Related to ADR 0049: E2E Voice Pilot Methodology (mentions Sony recorder + mic + hardware chain)
- CLAUDE.md §3 (current — needs correction)
- `docs/archived/dev-plan/2026-04-10-E2E-Voice-Pilot-Plan.md` §6 (current — needs correction)
