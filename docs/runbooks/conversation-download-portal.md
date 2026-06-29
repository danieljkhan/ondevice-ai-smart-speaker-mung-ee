# Runbook — Conversation Data Download Portal

Operator setup for the PIN-protected, Tailscale-only download portal
(`parental/download_portal/`, ADR 0112). **Default: disabled.** Follow the steps in order; the
portal must not be enabled before the PIN and Tailscale ACL are in place.

## 1. Set the PIN (required first)

```bash
ssh mungi@<jetson>
cd /opt/mungi-repo
.venv/bin/python -m parental.download_portal set-pin     # prompts on stdin; ≥8 digits
```

This writes `/var/lib/mungi/config/portal.json` (mode `0600`, PBKDF2-hashed — the PIN is never
echoed or logged). The daemon **refuses to start** until this exists.

## 2. Enable the feature flag

Add to `/var/lib/mungi/config/mungi.env` (the same env file the kiosk + nightly job read):

```
MUNGI_DOWNLOAD_PORTAL=1
```

## 3. Write the Tailscale ACL (required — this is the parent-only guarantee)

In the Tailscale admin console:

1. **Tag the Jetson** (e.g. `tag:mungi`) — assign it to this device.
2. **Restrict port 8765** to the parent's device/group only, e.g.:

```jsonc
"acls": [
  { "action": "accept", "src": ["<parent-user-or-group>"], "dst": ["tag:mungi:8765"] }
]
```

3. **Ensure Tailscale Funnel/Serve is OFF** for this port (the portal must never be published to the
   internet): `tailscale serve status` / `tailscale funnel status` should show nothing for 8765.

## 4. Install the service — on-demand, not auto-started (sudo)

```bash
sudo cp /opt/mungi-repo/systemd/mungi-download-portal.service /etc/systemd/system/
sudo systemctl daemon-reload
# On-demand: do NOT `enable`. Start the portal only when you need to download:
sudo systemctl start mungi-download-portal.service
systemctl status mungi-download-portal.service          # active; bound to the tailscale0 IP
# When done, stop it to free memory — use the web "포털 종료" button, or:
sudo systemctl stop mungi-download-portal.service
```

The portal is **on-demand**: with no `[Install]` section it never auto-starts on boot, so it
costs zero resident memory when idle. If it was previously enabled, remove the boot hook once
with `sudo systemctl disable mungi-download-portal.service` (the unit can no longer be `enable`d).

(If the flag is unset or the PIN is uninitialized, the unit stays inactive / exits cleanly — by
design.)

## 5. Verify (do not skip)

- **Allowed device**: from a Tailscale-connected PC in the parent group, open
  `http://jetson.local:8765` → login page → PIN → session list → download. ✅
- **Denied device**: from a Tailscale device NOT in the parent group → connection refused (ACL). ✅
- **LAN**: from a same-WiFi device using the Jetson's LAN IP → connection refused (not bound to
  LAN). ✅
- **Audit**: `tail /var/lib/mungi/logs/portal-audit.log` shows the download + any failed login.

## 6. Access (day-to-day)

From any of the parent's Tailscale devices (home or away): `http://jetson.local:8765`, enter the
PIN, select sessions, download (text `conversation.jsonl` + `input/output` WAVs, with a
`manifest.json` of SHA-256 checksums).

## Disable / rotate

- Disable: `sudo systemctl disable --now mungi-download-portal.service` (or unset the flag).
- Rotate PIN: re-run `set-pin` (invalidates existing sessions).
- Revoke a lost device: remove it from the tailnet in the Tailscale console.


## Touchscreen on-demand activation (no CLI at use time)

The portal is on-demand (it does not auto-start). To activate it without a shell:

- **Start**: double-tap the **hidden top-left corner** of the kiosk screen. The kiosk
  runs `systemctl start mungi-download-portal.service` through the narrow passwordless
  rule in `systemd/mungi-portal-control.sudoers`; a short ack tone confirms it. (Normal
  single taps stay push-to-talk — only a double-tap in that corner activates. The
  top-right corner remains the KO/EN toggle.)
- **Stop**: the portal web page's **"포털 종료"** button, or `sudo systemctl stop …`.

One-time install of the narrow sudo rule (sudo), so the kiosk user may start/stop ONLY
this service:

```bash
sudo cp /opt/mungi-repo/systemd/mungi-portal-control.sudoers /etc/sudoers.d/mungi-portal-control
sudo chmod 0440 /etc/sudoers.d/mungi-portal-control
sudo visudo -c        # validate before it takes effect
```
