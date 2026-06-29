# ADR 0099: Boot-persistent touchscreen kiosk runtime (systemd + Weston + always-on display)

- **Status**: Accepted (2026-06-03 — documents shipped decisions: PR #153 `3ce3f56` boot-persistent kiosk;
  PR #154 `f805f5f` runtime fixes; PR #156 `c67f5bf` always-on display. Post-hoc ADR per CLAUDE.md §7 rule 4 —
  the systemd kiosk service is a runtime-path change that lacked an ADR.)
- **Date**: 2026-06-03
- **Authority**: `Dev_Plan/2026-06-03-wake-ordering-screen-lead-plan-v1.md` (v2 always-on pivot); Session 93–95
  worklogs/handoffs (`docs/runbooks/weekly/`); project memory `project_touchscreen_display_solution`.
- **Related**: ADR 0080 (4inch HDMI touchscreen H/W adoption), ADR 0095 (Pygame character renderer runtime),
  ADR 0097 (HTML+CSS renderer — deferred, EGL-blocked), ADR 0079 (LED retirement — touchscreen is the sole
  visual channel).

## Context

ADR 0080 adopted the Waveshare 4inch 720×720 HDMI touchscreen (via a DP→HDMI active adapter) and ADR 0095
adopted the Pygame character renderer, but neither defined **how the UI starts and stays up** on the Jetson.
PR #153 (`3ce3f56`) introduced a boot-persistent kiosk — a new systemd service plus launcher/ensure-display
scripts — which is a **runtime-path change** (new service + runtime bring-up) requiring an ADR per CLAUDE.md
§7 rule 4. Sessions 94–95 then fixed several live runtime defects (audio routing, screen-lock prompt,
cold-boot USB race) and pivoted screen-idle behavior. None of this was captured in an ADR. This ADR documents
the resulting kiosk runtime model as the accepted architecture.

Key hardware constraints discovered live: (a) Tegra needs the **NVIDIA** Weston (stock Weston DRM backend
fails "no drm device found"); (b) `nvidia-drm` KMS only finishes loading at boot **+~100 s** (kernel/GPU gate;
not fixable from userspace — kernel cmdline / modprobe.d both failed, only a runtime `modprobe` works); (c) the
DP→HDMI panel has a **~2.5–3 s physical DPMS-on latency**, so idle screen-off makes wake feel laggy and lets
audio lead video; (d) the USB PnP audio card enumerates with 0 ALSA output channels (reachable for output only
via the PulseAudio sink), and the touchscreen's HDMI sink would otherwise steal default audio output.

## Decision

### D1 — systemd service `mungi-kiosk.service`
A boot-persistent unit (`WantedBy=multi-user.target`) runs the kiosk as user `mungi`:
- `Type=simple`, `User=mungi`, `PAMName=login`, `TTYPath=/dev/tty1` (a real login seat so libseat/Weston get a
  session); `After/Wants=seatd.service`.
- `ExecStartPre=+/opt/mungi-repo/scripts/mungi-ensure-display.sh` (the `+` = run as root) ensures
  `nvidia-drm modeset=1 fbdev=1` is loaded before Weston (the only reliable KMS-enable path).
- `ExecStart=/opt/mungi-repo/scripts/mungi-kiosk-start.sh` brings up Weston + the `demo_live` app.
- `Restart=on-failure`, `RestartSec=5`, `TimeoutStartSec=120` (covers the ~100 s cold-boot nvidia-drm gate;
  Weston fails a few times then converges at ~+116 s).

### D2 — NVIDIA Weston compositor, DP-1 auto-detection
`mungi-kiosk-start.sh` launches the NVIDIA Weston (`--idle-time`, `--config=systemd/weston-kiosk.ini`) on DRM
`card1` connector `DP-1` (720×720). `weston-kiosk.ini` is intentionally minimal (`[shell] locking=false`, **no
`[output]` block**) to preserve DP-1 auto-detection. The renderer runs as `MUNGI_RENDERER=pygame`
`MUNGI_SDL_DRIVER=wayland` `MUNGI_RENDERER_WINDOWED=1` against `wayland-0`.

### D3 — Always-on display (idle screen-off disabled)
`DEFAULT_MUNGI_SCREEN_IDLE_S=0` → `weston --idle-time=0` → the panel never DPMS-blanks. Rationale: the panel's
~2.5–3 s DPMS-on latency makes idle-off → tap-wake feel laggy and causes the wake voice to lead the screen
(empirically, a screen-lead audio delay up to 3 s could not reliably beat it — see the wake-ordering plan v2).
For the practicum-demo device, always-on gives an instant, correctly-ordered wake and fits the "always-present
friend" vision. `MUNGI_SCREEN_IDLE_S` remains env-overridable to re-enable blanking (e.g. `=180`). The
companion `MUNGI_WAKE_SCREEN_LEAD_S` screen-lead (ADR-less, code default 0.4 s in `core/session_manager.py`)
is pinned to `0` for the kiosk since the screen is always on.

### D4 — `[shell] locking=false` (no unlock prompt)
Weston's desktop-shell screen lock ("unlock your device") is disabled — inappropriate for a child-facing
device. With always-on (D3) the lock would not trigger anyway, but `locking=false` is kept as defense in depth
if idle-off is ever re-enabled.

### D5 — Audio output pinned to PulseAudio; resilience hardening
- `MUNGI_AUDIO_OUTPUT_DEVICE=pulse` pins playback to the USB speaker via the pulse sink (the HDMI sink the
  DP→HDMI adapter introduced would otherwise be the blind fallback → silent speaker). See ADR-less audio fix,
  `hardware/audio_player.py` resolution chain.
- Cold-boot USB-audio race: `scripts/demo_live.py::_build_audio_capture` retries/waits for the USB input
  (`MUNGI_USB_AUDIO_WAIT_S`, default 15 s) instead of crashing on boot attempt 1.
- `demo_live` stdout/stderr is redirected to `/var/lib/mungi/logs/demo_live.log` (`MUNGI_DEMO_LOG`) because the
  unit's `TTYPath=/dev/tty1` keeps it out of the journal.
- Initial IDLE frame: `SessionManager.run()` emits the IDLE expression once at startup so the first Wayland
  buffer commits and the character is visible without a tap.

## Consequences

- The kiosk auto-starts on boot and self-heals (`Restart=on-failure`); operators restart it with
  `kill -9 $(pgrep -x weston) $(pgrep -f 'm scripts.demo_live')` as `mungi` (no sudo) or `systemctl restart`
  (needs sudo, not passwordless). ⚠️ Do **not** rapid-kill Weston during diagnosis.
- Always-on means the panel is continuously lit (power/burn-in tradeoff accepted for the demo device); idle
  blanking can be restored via `MUNGI_SCREEN_IDLE_S` if needed.
- The ~100 s cold-boot nvidia-drm gate is accepted (kernel-level, not fixable from userspace); leaving the
  device powered (always-on, no reboot) avoids it.
- Split-mode operation: the Jetson runtime tree `/opt/mungi-repo` is kept current by `scp` (git intentionally
  behind); changes to these files must be deployed, not just merged.
- This decision will be treated as the default kiosk runtime unless superseded by a later ADR (e.g. if the
  HTML+CSS renderer of ADR 0097 ever clears the NVIDIA EGL export blocker).

## Related ADRs

- ADR 0080 — Waveshare 4inch HDMI touchscreen hardware adoption (this ADR is its runtime counterpart).
- ADR 0095 — Pygame character renderer runtime (the renderer this kiosk launches).
- ADR 0097 — HTML+CSS character renderer (deferred, NVIDIA EGL-blocked; would supersede the Pygame path).
- ADR 0079 — LED hardware retirement (touchscreen is the sole visual feedback channel).
- ADR 0052 — Jetson ALSA default routing override (USB audio card context).

## References

- PR #153 `3ce3f56` (boot-persistent kiosk), PR #154 `f805f5f` (audio routing + character display + Weston
  lock + cold-boot USB), PR #155 `1f77994` (wake-ordering screen-lead), PR #156 `c67f5bf` (always-on display).
- `systemd/mungi-kiosk.service`, `systemd/weston-kiosk.ini`, `scripts/mungi-kiosk-start.sh`,
  `scripts/mungi-ensure-display.sh`, `core/character_renderer.py`, `core/session_manager.py`.
- `docs/runbooks/mungi-touchscreen-kiosk.md`; project memory `project_touchscreen_display_solution`.
