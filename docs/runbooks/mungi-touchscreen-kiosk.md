# Mungi Touchscreen Kiosk

## Purpose

Make the verified NVIDIA Weston + Pygame touchscreen demo repeatable after boot.
This runbook does not deploy to the protected runtime tree and does not enable the
service automatically during local development.

## Prerequisites

These are already present on the Jetson and should not be recreated by this
runbook:

- `nvidia-drm` available for runtime KMS enablement by the service pre-start
- `seatd.service` enabled and active
- User `mungi` in the `video`, `render`, and `weston-launch` groups
- `weston-launch` group present
- Git-managed runtime tree available at `/opt/mungi-repo`

The service uses `PAMName=login` plus `TTYPath=/dev/tty1` so systemd gives `mungi`
a logind seat session and `XDG_RUNTIME_DIR=/run/user/1000`.

## Install the Service

Run on the Jetson after the branch is present in `/opt/mungi-repo`:

```bash
cd /opt/mungi-repo
chmod +x scripts/mungi-kiosk-start.sh scripts/mungi-ensure-display.sh
systemd-analyze verify systemd/mungi-kiosk.service
sudo cp systemd/mungi-kiosk.service /etc/systemd/system/mungi-kiosk.service
sudo systemctl daemon-reload
sudo systemctl enable --now mungi-kiosk.service
```

Check startup logs:

```bash
sudo systemctl status mungi-kiosk.service --no-pager
sudo journalctl -u mungi-kiosk.service -n 100 --no-pager
tail -n 100 /tmp/mungi-weston.log
```

## Display Pre-start

At boot on this Jetson, `nvidia-drm` KMS is not enabled reliably before Weston
starts. The kiosk unit runs `scripts/mungi-ensure-display.sh` as a root
`ExecStartPre` step with the systemd `+` prefix, then the main launcher still
runs as `User=mungi`.

The helper exits immediately when `/sys/class/drm/card1-DP-1` already exists.
Otherwise, it reloads `nvidia_drm` and runs:

```bash
modprobe nvidia-drm modeset=1 fbdev=1
```

It then waits for the known 720x720 `card1-DP-1` output, or another connected
DP/HDMI DRM output, before allowing Weston to start. A kernel command-line
`nvidia-drm.modeset=1` attempt was tried and did not reliably fix boot startup;
the root `ExecStartPre` runtime `modprobe` is the verified working approach.

## Deploy Derived Frame Assets

`assets/character/frames/` (18 expressions × 120 PNG = 2160, ~44 MB) is gitignored because the frame
sequences are **derived** from the tracked CSS/HTML source in `assets/emoji/HTML/` (340 KB, committed for
reproducibility). The large external illustrator shipment (`assets/emoji/*.mp4` / `*.png`) stays gitignored.

Use the formal deploy script (regenerates from source on demand, verifies the frame count, backs up the
remote set, and optionally restarts the kiosk):

```bash
# defaults: host mungi@jetson.local, REMOTE_REPO=/opt/mungi-repo
scripts/deploy_character_frames.sh [mungi@<jetson-host>] [--render] [--restart]
```

- `--render` regenerates the frames first via `scripts/render_emoji_frames.py` (needs Playwright + Chromium).
- `--restart` runs `sudo systemctl restart mungi-kiosk.service` on the Jetson (passwordless sudo not
  guaranteed; otherwise restart manually — see "Manual Fallback Launch" / the kill-restart note).

Manual equivalent (if the script is unavailable):

```bash
python scripts/render_emoji_frames.py
ssh mungi@<jetson-host> "mkdir -p /opt/mungi-repo/assets/character/frames"
scp -r assets/character/frames/* mungi@<jetson-host>:/opt/mungi-repo/assets/character/frames/
ssh mungi@<jetson-host> "sudo systemctl restart mungi-kiosk.service"
```

## Manual Fallback Launch

Use this when systemd seat details need on-device tuning:

```bash
export XDG_RUNTIME_DIR=/run/user/1000
unset DISPLAY
weston &
cd /opt/mungi-repo
source scripts/mungidev.sh
WAYLAND_DISPLAY=wayland-0 \
MUNGI_RENDERER=pygame \
MUNGI_SDL_DRIVER=wayland \
MUNGI_RENDERER_WINDOWED=1 \
.venv/bin/python -m scripts.demo_live
```

## Notes

- The launcher writes Weston output to `/tmp/mungi-weston.log`.
- `MUNGI_SCREEN_IDLE_S` controls standby display blanking through Weston
  `--idle-time` and defaults to `180` seconds. After that idle period with no
  touch input, the screen turns off; a tap wakes it. Set `MUNGI_SCREEN_IDLE_S=0`
  to disable blanking. The systemd unit can set this with an `Environment=`
  line.
- The launcher polls `${XDG_RUNTIME_DIR}/wayland-0` for up to 15 seconds before
  failing with a clear error.
- The runtime WPE/cog path is blocked on NVIDIA EGL dmabuf export. The blocker
  and the reason for the Pygame + Weston path are documented in the
  `project_touchscreen_display_solution` memory and the CSS/HTML renderer plan.
