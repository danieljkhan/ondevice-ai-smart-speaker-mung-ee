#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 TARGET" >&2
}

log_step() {
    echo "[step $1] $2"
}

log_ok() {
    echo "[ok] $1"
}

log_skip() {
    echo "[skip] $1"
}

log_error() {
    echo "[error] $1" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

TARGET=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASHRC_SNIPPET="$REPO_ROOT/scripts/_jetson_ssh_tmux.bashrc.snippet"
TMUX_SNIPPET="$REPO_ROOT/scripts/_jetson_ssh_tmux.tmux.conf.snippet"
BASHRC_BEGIN="# --- Mungi: auto-tmux on SSH login (begin) ---"
BASHRC_END="# --- Mungi: auto-tmux on SSH login (end) ---"
TMUX_BEGIN="# --- Mungi: tmux configuration (begin) ---"
TMUX_END="# --- Mungi: tmux configuration (end) ---"

log_step 1 "Verifying local snippet files"
if [[ ! -f "$BASHRC_SNIPPET" ]]; then
    log_error "missing local snippet: $BASHRC_SNIPPET"
    exit 1
fi
if [[ ! -f "$TMUX_SNIPPET" ]]; then
    log_error "missing local snippet: $TMUX_SNIPPET"
    exit 1
fi
log_ok "local snippet files found"

log_step 2 "Testing SSH reachability to $TARGET"
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$TARGET" true; then
    log_error "unable to reach $TARGET via SSH"
    exit 1
fi
log_ok "SSH reachability confirmed"

log_step 3 "Checking tmux availability on the remote host"
if ! ssh "$TARGET" 'command -v tmux >/dev/null 2>&1'; then
    log_error "tmux is not installed on the remote host"
    echo "Install it manually on the Jetson:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y tmux" >&2
    exit 3
fi
log_ok "remote tmux prerequisite satisfied"

log_step 4 "Installing the ~/.bashrc auto-attach snippet"
if ssh "$TARGET" "grep -Fqx '$BASHRC_BEGIN' ~/.bashrc 2>/dev/null"; then
    log_skip "bashrc snippet already installed"
else
    if ! {
        printf '\n%s\n' "$BASHRC_BEGIN"
        cat "$BASHRC_SNIPPET"
        printf '%s\n' "$BASHRC_END"
    } | ssh "$TARGET" "cat >> ~/.bashrc"; then
        log_error "failed to append the ~/.bashrc snippet"
        exit 1
    fi
    log_ok "installed the ~/.bashrc auto-attach snippet"
fi

log_step 5 "Installing the ~/.tmux.conf snippet and reloading tmux"
if ssh "$TARGET" "grep -Fqx '$TMUX_BEGIN' ~/.tmux.conf 2>/dev/null"; then
    log_skip "tmux.conf snippet already installed"
else
    if ! {
        printf '\n%s\n' "$TMUX_BEGIN"
        cat "$TMUX_SNIPPET"
        printf '%s\n' "$TMUX_END"
    } | ssh "$TARGET" "cat >> ~/.tmux.conf"; then
        log_error "failed to append the ~/.tmux.conf snippet"
        exit 1
    fi
    log_ok "installed the ~/.tmux.conf snippet"
fi
if ! ssh "$TARGET" 'tmux source-file ~/.tmux.conf 2>/dev/null || true'; then
    log_error "failed to reload ~/.tmux.conf on the remote host"
    exit 1
fi
log_ok "remote tmux configuration reloaded"

log_step 6 "Verifying the installed remote markers"
if ! ssh "$TARGET" "grep -Fqx '$BASHRC_BEGIN' ~/.bashrc 2>/dev/null \
    && grep -Fqx '$TMUX_BEGIN' ~/.tmux.conf 2>/dev/null"; then
    log_error "remote marker verification failed"
    exit 1
fi
log_ok "remote markers verified"

echo "[ok] Installation complete for $TARGET"
echo "Next steps:"
echo "  ssh $TARGET"
echo "  Detach with Ctrl+A D"
echo "  Add a new window with Ctrl+A C"
