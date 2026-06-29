#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[mungi-kiosk] $*" >&2
}

fail() {
    log "ERROR: $*"
    exit 1
}

MUNGI_REPO="${MUNGI_REPO:-/opt/mungi-repo}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

WAYLAND_DISPLAY_NAME="wayland-0"
WAYLAND_SOCKET="${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY_NAME}"
WESTON_LOG="/tmp/mungi-weston.log"
DEFAULT_MUNGI_SCREEN_IDLE_S=0 # 0=never blank (always-on display); ~3s DPMS-on hurts wake; see plan

if [[ -z "${MUNGI_SCREEN_IDLE_S+x}" ]]; then
    MUNGI_SCREEN_IDLE_S="${DEFAULT_MUNGI_SCREEN_IDLE_S}"
elif [[ ! "${MUNGI_SCREEN_IDLE_S}" =~ ^[0-9]+$ ]]; then
    log "WARNING: invalid MUNGI_SCREEN_IDLE_S='${MUNGI_SCREEN_IDLE_S}'; falling back to ${DEFAULT_MUNGI_SCREEN_IDLE_S}"
    MUNGI_SCREEN_IDLE_S="${DEFAULT_MUNGI_SCREEN_IDLE_S}"
fi

if [[ ! -d "${XDG_RUNTIME_DIR}" ]]; then
    fail "XDG_RUNTIME_DIR does not exist: ${XDG_RUNTIME_DIR}; start through a login-backed session"
fi

if [[ ! -d "${MUNGI_REPO}" ]]; then
    fail "MUNGI_REPO does not exist: ${MUNGI_REPO}"
fi

if ! command -v weston >/dev/null 2>&1; then
    fail "weston executable not found in PATH"
fi

if pgrep -u "$(id -u)" -x weston >/dev/null 2>&1; then
    log "Weston already running; waiting for ${WAYLAND_SOCKET}"
else
    log "Starting Weston; idle-time=${MUNGI_SCREEN_IDLE_S}s; log=${WESTON_LOG}"
    unset DISPLAY
    weston --idle-time="${MUNGI_SCREEN_IDLE_S}" \
        --config="${MUNGI_REPO}/systemd/weston-kiosk.ini" \
        >"${WESTON_LOG}" 2>&1 &
fi

for _attempt in {1..150}; do
    if [[ -S "${WAYLAND_SOCKET}" ]]; then
        log "Wayland socket ready: ${WAYLAND_SOCKET}"
        break
    fi
    sleep 0.1
done

if [[ ! -S "${WAYLAND_SOCKET}" ]]; then
    fail "Wayland socket did not appear after 15 seconds: ${WAYLAND_SOCKET}; see ${WESTON_LOG}"
fi

export WAYLAND_DISPLAY="${WAYLAND_DISPLAY_NAME}"
cd "${MUNGI_REPO}"

log "Loading Mungi runtime environment from scripts/mungidev.sh"
# shellcheck disable=SC1091
source scripts/mungidev.sh

log "Starting touchscreen demo with Pygame over Wayland"
DEMO_LOG="${MUNGI_DEMO_LOG:-/var/lib/mungi/logs/demo_live.log}"
mkdir -p "$(dirname "${DEMO_LOG}")"
exec env \
    MUNGI_RENDERER=pygame \
    MUNGI_SDL_DRIVER=wayland \
    MUNGI_RENDERER_WINDOWED=1 \
    MUNGI_AUDIO_OUTPUT_DEVICE=pulse \
    MUNGI_WAKE_SCREEN_LEAD_S=0 \
    "${MUNGI_REPO}/.venv/bin/python" -m scripts.demo_live >>"${DEMO_LOG}" 2>&1
