#!/usr/bin/env bash
set -euo pipefail

SERVICE="mungi-kiosk.service"
RUNNER_LOCK="/run/lock/mungi-tts-bake-run.lock"
INTERLOCK_LOCK="/run/mungi-tts-bake.lock"
DROPIN_DIR="/etc/systemd/system/${SERVICE}.d"
DROPIN_PATH="${DROPIN_DIR}/zz-tts-bake-lock.conf"
MIN_FREE_KB=$((12 * 1024 * 1024))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="/var/lib/mungi/tts_cache"
STEPS="30"
DEVICE="cuda"
ONLY=""
LIMIT=""
we_stopped_kiosk=0
interlock_started=0

log() {
    printf '[tts-cache-bake] %s\n' "$*"
}

fail() {
    printf '[tts-cache-bake] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Usage: scripts/bake_tts_cache_run.sh [options]

Options:
  --out-dir PATH       Cache output directory (default: /var/lib/mungi/tts_cache)
  --steps N            Supertonic total_steps; must be 30 for runtime identity
  --device cuda|cpu    Bake device (default: cuda; CPU is explicit opt-in)
  --only ko|en         Stage only one language
  --limit N            Limit items for smoke/resume runs
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            [[ $# -ge 2 ]] || fail "--out-dir requires a value"
            OUT_DIR="$2"
            shift 2
            ;;
        --steps)
            [[ $# -ge 2 ]] || fail "--steps requires a value"
            STEPS="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || fail "--device requires a value"
            DEVICE="$2"
            shift 2
            ;;
        --only)
            [[ $# -ge 2 ]] || fail "--only requires a value"
            ONLY="$2"
            shift 2
            ;;
        --limit)
            [[ $# -ge 2 ]] || fail "--limit requires a value"
            LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ "${STEPS}" == "30" ]] || fail "--steps must be 30"
[[ "${DEVICE}" == "cuda" || "${DEVICE}" == "cpu" ]] || fail "--device must be cuda or cpu"
[[ -z "${ONLY}" || "${ONLY}" == "ko" || "${ONLY}" == "en" ]] || fail "--only must be ko or en"
if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[0-9]+$ ]]; then
    fail "--limit must be a non-negative integer"
fi
OUT_DIR="$(realpath -m "${OUT_DIR}")"
case "${OUT_DIR}" in
    /var/lib/mungi/tts_cache|/var/lib/mungi/tts_cache/*)
        ;;
    *)
        fail "--out-dir must stay under /var/lib/mungi/tts_cache"
        ;;
esac

cleanup() {
    local status=$?
    trap - EXIT ERR INT TERM
    set +e
    if [[ "${interlock_started}" == "1" ]]; then
        log "restoring kiosk interlock"
        sudo -n rm -f "${INTERLOCK_LOCK}"
        sudo -n rm -f "${DROPIN_PATH}"
        sudo -n systemctl daemon-reload
    else
        log "no kiosk interlock was applied"
    fi
    if [[ "${we_stopped_kiosk}" == "1" ]]; then
        log "starting ${SERVICE} because this runner stopped it"
        sudo -n systemctl start "${SERVICE}"
    else
        log "leaving ${SERVICE} stopped/inactive because this runner did not stop it"
    fi
    exit "${status}"
}

trap cleanup EXIT INT TERM
trap 'exit $?' ERR

exec 9>"${RUNNER_LOCK}"
flock -n 9 || fail "another TTS cache bake runner is already active"

SYSTEMCTL="$(command -v systemctl)"
log "preflighting non-interactive sudo/systemctl"
sudo -n -v >/dev/null || fail "sudo -n cannot validate credentials"
sudo -n -l "${SYSTEMCTL}" stop "${SERVICE}" >/dev/null || fail "sudo cannot stop ${SERVICE}"
sudo -n -l "${SYSTEMCTL}" start "${SERVICE}" >/dev/null || fail "sudo cannot start ${SERVICE}"
sudo -n -l "${SYSTEMCTL}" restart "${SERVICE}" >/dev/null || fail "sudo cannot restart ${SERVICE}"
sudo -n -l "${SYSTEMCTL}" daemon-reload >/dev/null || fail "sudo cannot daemon-reload"

export MUNGI_REPO="${REPO_ROOT}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/mungidev.sh"

mkdir -p "${OUT_DIR}"
available_kb="$(df -Pk "${OUT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ -z "${available_kb}" || "${available_kb}" -lt "${MIN_FREE_KB}" ]]; then
    fail "need at least 12 GB free under ${OUT_DIR}"
fi

MODEL_DIR="${MUNGI_TTS_MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
    if [[ -n "${MUNGI_MODEL_ROOT:-}" ]]; then
        MODEL_DIR="${MUNGI_MODEL_ROOT}/supertonic-2"
    elif [[ -n "${MUNGI_MODEL_DIR:-}" ]]; then
        MODEL_DIR="${MUNGI_MODEL_DIR}/supertonic-2"
    else
        MODEL_DIR="/opt/mungi/ai_models/supertonic-2"
    fi
fi
[[ -d "${MODEL_DIR}" ]] || fail "Supertonic model directory not found: ${MODEL_DIR}"
[[ -n "${MUNGI_TTS_VOICE_STYLE_KO:-}" ]] || fail "MUNGI_TTS_VOICE_STYLE_KO is empty"
[[ -n "${MUNGI_TTS_VOICE_STYLE_EN:-}" ]] || fail "MUNGI_TTS_VOICE_STYLE_EN is empty"
[[ -f "${MUNGI_TTS_VOICE_STYLE_KO}" ]] || fail "KO voice JSON not found: ${MUNGI_TTS_VOICE_STYLE_KO}"
[[ -f "${MUNGI_TTS_VOICE_STYLE_EN}" ]] || fail "EN voice JSON not found: ${MUNGI_TTS_VOICE_STYLE_EN}"

was_active=0
if sudo -n systemctl is-active --quiet "${SERVICE}"; then
    was_active=1
fi
was_enabled="$(sudo -n systemctl is-enabled "${SERVICE}" 2>/dev/null || true)"
was_masked=0
if [[ "${was_enabled}" == "masked" ]]; then
    was_masked=1
fi
log "prior state: was_active=${was_active} was_enabled=${was_enabled:-unknown} was_masked=${was_masked}"

log "installing ConditionPathExists interlock"
interlock_started=1
sudo -n mkdir -p "${DROPIN_DIR}"
printf '%s\n' '[Unit]' 'ConditionPathExists=!/run/mungi-tts-bake.lock' \
    | sudo -n tee "${DROPIN_PATH}" >/dev/null
sudo -n systemctl daemon-reload
sudo -n touch "${INTERLOCK_LOCK}"

if [[ "${was_active}" == "1" ]]; then
    log "stopping ${SERVICE}"
    sudo -n systemctl stop "${SERVICE}"
    we_stopped_kiosk=1
fi

if sudo -n systemctl is-active --quiet "${SERVICE}"; then
    fail "${SERVICE} is still active after interlock/stop"
fi

holders="$(pgrep -af '([s]cripts/demo_live|[d]emo_live\.py|[m]ungi-kiosk-start|[p]ygame|[r]enderer)' || true)"
if [[ -n "${holders}" ]]; then
    printf '%s\n' "${holders}" >&2
    fail "demo_live or renderer process is still running"
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

cmd=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/scripts/bake_tts_cache.py"
    --out-dir "${OUT_DIR}"
    --steps "${STEPS}"
    --device "${DEVICE}"
)
if [[ -n "${ONLY}" ]]; then
    cmd+=(--only "${ONLY}")
fi
if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

log "running bake: ${cmd[*]}"
"${cmd[@]}"
log "bake completed"
