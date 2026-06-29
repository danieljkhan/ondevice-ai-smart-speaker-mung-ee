#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-mungi@jetson.local}"
REMOTE_REPO="${REMOTE_REPO:-/opt/mungi-repo}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_TMP="/tmp/mungi_option_c_pipeline_${STAMP}.py"
REMOTE_BACKUP="/tmp/mungi_option_c_pipeline_backup_${STAMP}.py"

if [[ ! -f "core/pipeline.py" ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

scp -O "core/pipeline.py" "${REMOTE_HOST}:${REMOTE_TMP}"
ssh "${REMOTE_HOST}" \
    "set -e; test -d '${REMOTE_REPO}/core'; cp '${REMOTE_REPO}/core/pipeline.py' '${REMOTE_BACKUP}'; cp '${REMOTE_TMP}' '${REMOTE_REPO}/core/pipeline.py'; cd '${REMOTE_REPO}' && python3 -m py_compile core/pipeline.py; echo 'Synced core/pipeline.py to ${REMOTE_REPO}; backup: ${REMOTE_BACKUP}'"
