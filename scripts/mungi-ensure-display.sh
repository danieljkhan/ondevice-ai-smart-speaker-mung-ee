#!/usr/bin/env bash
set -uo pipefail

KNOWN_CONNECTOR="/sys/class/drm/card1-DP-1"
POLL_ATTEMPTS=100
POLL_INTERVAL_S=0.1

log() {
    echo "[mungi-ensure-display] $*" >&2
}

connector_ready() {
    if [[ -e "${KNOWN_CONNECTOR}" ]]; then
        log "Display connector already available: ${KNOWN_CONNECTOR}"
        return 0
    fi

    local connector=""
    local status_file

    shopt -s nullglob
    for status_file in /sys/class/drm/card*-DP-1/status /sys/class/drm/card*-HDMI*/status; do
        if [[ -r "${status_file}" ]] && [[ "$(cat "${status_file}")" == "connected" ]]; then
            connector="${status_file%/status}"
            break
        fi
    done
    shopt -u nullglob

    if [[ -n "${connector}" ]]; then
        log "Connected DRM output available: ${connector}"
        return 0
    fi

    return 1
}

if connector_ready; then
    exit 0
fi

log "Ensuring nvidia_drm KMS modeset before Weston startup"
rmmod nvidia_drm 2>/dev/null || true
modprobe nvidia-drm modeset=1 fbdev=1 || true

for ((_attempt = 1; _attempt <= POLL_ATTEMPTS; _attempt++)); do
    if connector_ready; then
        exit 0
    fi
    sleep "${POLL_INTERVAL_S}"
done

log "ERROR: display connector did not appear after 10 seconds; expected ${KNOWN_CONNECTOR}"
log "ERROR: no connected DP/HDMI DRM fallback output was found"
exit 1
