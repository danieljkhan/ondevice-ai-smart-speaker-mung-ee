#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REMOTE_HOST="mungi@jetson.local"
REMOTE_HOST="${DEFAULT_REMOTE_HOST}"
REMOTE_HOST_SET=0
REMOTE_REPO="${REMOTE_REPO:-/opt/mungi-repo}"
DO_RESTART=0

LOCAL_IMAGES_DIR="assets/history/images"
LOCAL_MANIFEST="assets/history/manifest.json"
LOCAL_TOTAL=0
LOCAL_TARBALL=""

usage() {
    cat <<'USAGE'
Usage: scripts/deploy_history_assets.sh [REMOTE_HOST] [--restart]

Deploy derived Korean-history image assets to the Jetson runtime repository.

Arguments:
  REMOTE_HOST   SSH target. Defaults to mungi@jetson.local.

Options:
  --restart     Restart mungi-kiosk.service after deploy verification.
  -h, --help    Show this help text.

Environment:
  REMOTE_REPO   Remote runtime repo. Defaults to /opt/mungi-repo.
USAGE
}

usage_error() {
    echo "error: $1" >&2
    usage >&2
    exit 2
}

cleanup() {
    if [[ -n "${LOCAL_TARBALL}" && -f "${LOCAL_TARBALL}" ]]; then
        rm -f "${LOCAL_TARBALL}"
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restart)
            DO_RESTART=1
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --*)
            usage_error "unknown flag: $1"
            ;;
        -*)
            usage_error "unknown option: $1"
            ;;
        *)
            if (( REMOTE_HOST_SET == 1 )); then
                usage_error "multiple REMOTE_HOST values are not supported"
            fi
            REMOTE_HOST="$1"
            REMOTE_HOST_SET=1
            ;;
    esac
    shift
done

if [[ ! -f "${LOCAL_MANIFEST}" ]]; then
    echo "error: run this script from the repository root." >&2
    exit 1
fi

verify_local_images() {
    if [[ ! -d "${LOCAL_IMAGES_DIR}" ]]; then
        echo "local_verify error=missing_dir path=${LOCAL_IMAGES_DIR}" >&2
        exit 1
    fi

    LOCAL_TOTAL="$(
        find "${LOCAL_IMAGES_DIR}" -type f -name '*.jpg' -print |
            wc -l |
            awk '{print $1}'
    )"

    if (( LOCAL_TOTAL == 0 )); then
        echo "local_verify error=no_jpg path=${LOCAL_IMAGES_DIR}" >&2
        exit 1
    fi

    echo "local_verify jpg_total=${LOCAL_TOTAL}"
}

create_archive() {
    local stamp="$1"
    LOCAL_TARBALL="${TMPDIR:-/tmp}/mungi_history_images_${stamp}.tar.gz"

    echo "archive_create path=${LOCAL_TARBALL}"
    tar -C "${LOCAL_IMAGES_DIR}" -czf "${LOCAL_TARBALL}" .
}

transfer_archive() {
    local remote_tarball="$1"

    echo "transfer_start host=${REMOTE_HOST} remote_tarball=${remote_tarball}"
    scp -O "${LOCAL_TARBALL}" "${REMOTE_HOST}:${remote_tarball}"
}

deploy_remote_archive() {
    local remote_tarball="$1"
    local stamp="$2"

    ssh "${REMOTE_HOST}" \
        "set -e;
remote_repo='${REMOTE_REPO}';
remote_tarball='${remote_tarball}';
stamp='${stamp}';
parent_dir=\"\${remote_repo}/assets/history\";
images_dir=\"\${parent_dir}/images\";
backup_dir=\"\${images_dir}.bak-\${stamp}\";
stage_dir=\"\${images_dir}.stage-\${stamp}\";
trap 'rm -f \"\${remote_tarball}\"' EXIT;
mkdir -p \"\${parent_dir}\";
if [ -e \"\${stage_dir}\" ]; then
    echo \"remote_deploy error=stage_exists path=\${stage_dir}\" >&2;
    exit 1;
fi;
if [ -e \"\${backup_dir}\" ]; then
    echo \"remote_deploy error=backup_exists path=\${backup_dir}\" >&2;
    exit 1;
fi;
mkdir \"\${stage_dir}\";
tar -xzf \"\${remote_tarball}\" -C \"\${stage_dir}\";
if [ -e \"\${images_dir}\" ]; then
    mv \"\${images_dir}\" \"\${backup_dir}\";
fi;
mv \"\${stage_dir}\" \"\${images_dir}\";
echo \"remote_deploy images_dir=\${images_dir} backup=\${backup_dir}\""
}

verify_remote_images() {
    ssh "${REMOTE_HOST}" \
        "set -e;
images_dir='${REMOTE_REPO}/assets/history/images';
expected_total='${LOCAL_TOTAL}';
if [ ! -d \"\${images_dir}\" ]; then
    echo \"remote_verify error=missing_dir path=\${images_dir}\" >&2;
    exit 1;
fi;
actual_total=\$(find \"\${images_dir}\" -type f -name '*.jpg' -print |
    wc -l |
    awk '{print \$1}');
echo \"remote_verify jpg_total=\${actual_total}\";
if [ \"\${actual_total}\" -ne \"\${expected_total}\" ]; then
    echo \"remote_verify error=local_remote_mismatch\" >&2;
    echo \"remote_verify expected_total=\${expected_total} actual_total=\${actual_total}\" >&2;
    exit 1;
fi"
}

restart_kiosk() {
    if ssh "${REMOTE_HOST}" "sudo systemctl restart mungi-kiosk.service"; then
        echo "restart status=ok service=mungi-kiosk.service"
    else
        echo "restart status=warning service=mungi-kiosk.service" >&2
        echo "restart hint=passwordless_sudo_may_be_unavailable" >&2
        echo "restart hint=restart_manually_on_remote_host" >&2
    fi
}

verify_local_images

STAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_TARBALL="/tmp/mungi_history_images_${STAMP}.tar.gz"

create_archive "${STAMP}"
transfer_archive "${REMOTE_TARBALL}"
deploy_remote_archive "${REMOTE_TARBALL}" "${STAMP}"
verify_remote_images

if (( DO_RESTART == 1 )); then
    restart_kiosk
fi

echo "deploy_complete host=${REMOTE_HOST} remote_repo=${REMOTE_REPO}"
