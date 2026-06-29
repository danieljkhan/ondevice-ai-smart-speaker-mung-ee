#!/usr/bin/env bash
# mungidev.sh — Mungi development environment setup for Jetson
#
# Usage:
#   source scripts/mungidev.sh          # Set up env in current shell
#   scripts/mungidev.sh <command>        # Run command with env (e.g. via SSH)
#   ssh mungi@jetson "/opt/mungi-repo/scripts/mungidev.sh python ..."
#
# When sourced: activates venv + sets LD_LIBRARY_PATH (like the .bashrc function)
# When executed with args: sets env then runs the given command
# When executed without args: sets env and prints status

set -euo pipefail

MUNGI_REPO="${MUNGI_REPO:-/opt/mungi-repo}"
MUNGI_VENV="${MUNGI_REPO}/.venv"

# Optional runtime config (feature flags such as MUNGI_CONV_MEMORY) sourced from
# the mutable config dir, so features can be toggled without editing this file.
# The mungi-memory-nightly.service reads the same file via EnvironmentFile.
_MUNGI_ENV_FILE="${MUNGI_ENV_FILE:-/var/lib/mungi/config/mungi.env}"
if [[ -f "${_MUNGI_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${_MUNGI_ENV_FILE}"
    set +a
fi

# Core environment
export LD_LIBRARY_PATH="${MUNGI_REPO}/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${MUNGI_VENV}/lib/python3.10/site-packages/nvidia/cu12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda/bin:${PATH}"
# Hotwords default to empty: the Qwen3-ASR FT bundle echoes the hotword list
# verbatim on low-confidence audio (hotword hallucination), and the wake word is
# unused, so no decoder biasing is needed. Set the env var to override.
export MUNGI_QWEN3_ASR_HOTWORDS="${MUNGI_QWEN3_ASR_HOTWORDS:-}"
export MUNGI_TTS_VOICE_STYLE_KO="${MUNGI_TTS_VOICE_STYLE_KO:-/var/lib/mungi/voices/mung-ee.json}"
export MUNGI_TTS_VOICE_STYLE_EN="${MUNGI_TTS_VOICE_STYLE_EN:-/var/lib/mungi/voices/mung-ee.json}"
export MUNGI_TOUCH_DEVICE="${MUNGI_TOUCH_DEVICE:-}"
# 63 frames x 512 samples / 16kHz = ~2.0s end-of-utterance silence window.
# 25 (=0.8s) cut children off mid-utterance -> empty STT -> no answer;
# 156 (=5.0s) felt sluggish. ~2.0s tolerates a child's thinking pause without lag.
export MUNGI_VAD_SILENCE_FRAMES="${MUNGI_VAD_SILENCE_FRAMES:-63}"
export MUNGI_VAD_THRESHOLD="${MUNGI_VAD_THRESHOLD:-0.5}"
# Legacy MUNGI_TTS_VOICE_STYLE is retained as a safety-net fallback.
export MUNGI_TTS_VOICE_STYLE="${MUNGI_TTS_VOICE_STYLE:-/var/lib/mungi/voices/mung-ee.json}"

cd "${MUNGI_REPO}" || exit 1
# LLAMA_SET_ROWS removed: llama-cpp-python 0.3.17 uses ggml_set_rows() exclusively.
# Legacy ggml_cpy() path no longer exists. No env var needed.

# Activate venv
if [[ -f "${MUNGI_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${MUNGI_VENV}/bin/activate"
fi

# If executed (not sourced) with arguments, run the command
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -gt 0 ]]; then
        exec "$@"
    else
        echo "mungidev environment ready"
        echo "  MUNGI_REPO: ${MUNGI_REPO}"
        echo "  VENV: ${VIRTUAL_ENV:-not activated}"
        echo "  LD_LIBRARY_PATH set: $(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | wc -l) entries"
        echo "  CUDA: $(nvcc --version 2>/dev/null | grep 'release' || echo 'not found')"
        echo "  Python: $(python --version 2>/dev/null || echo 'not found')"
    fi
fi
