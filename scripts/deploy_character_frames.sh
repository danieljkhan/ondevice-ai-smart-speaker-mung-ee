#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REMOTE_HOST="mungi@jetson.local"
REMOTE_HOST="${DEFAULT_REMOTE_HOST}"
REMOTE_HOST_SET=0
REMOTE_REPO="${REMOTE_REPO:-/opt/mungi-repo}"
DO_RENDER=0
DO_RESTART=0

LOCAL_FRAMES_DIR="assets/character/frames"
LOCAL_EXPRESSIONS=0
LOCAL_FRAMES_PER=0
LOCAL_TOTAL=0
LOCAL_TARBALL=""

usage() {
    cat <<'USAGE'
Usage: scripts/deploy_character_frames.sh [REMOTE_HOST] [--render] [--restart]

Deploy derived character frame assets to the Jetson runtime repository.

Arguments:
  REMOTE_HOST   SSH target. Defaults to mungi@jetson.local.

Options:
  --render      Regenerate frames before deploy.
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
        --render)
            DO_RENDER=1
            ;;
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

if [[ ! -f "scripts/render_emoji_frames.py" ]]; then
    echo "error: run this script from the repository root." >&2
    exit 1
fi

verify_local_frames() {
    local dir
    local count
    local expression
    local expressions=0
    local frames_per=0
    local total=0

    if [[ ! -d "${LOCAL_FRAMES_DIR}" ]]; then
        echo "local_verify error=missing_dir path=${LOCAL_FRAMES_DIR}" >&2
        exit 1
    fi

    while IFS= read -r -d '' dir; do
        expression="$(basename "${dir}")"
        count="$(
            find "${dir}" -maxdepth 1 -type f -name '*.png' -print |
                wc -l |
                awk '{print $1}'
        )"

        if (( count == 0 )); then
            echo "local_verify error=empty_expression expression=${expression}" >&2
            exit 1
        fi

        if (( frames_per == 0 )); then
            frames_per="${count}"
        elif (( count != frames_per )); then
            echo "local_verify error=frame_count_mismatch expression=${expression}" >&2
            echo "local_verify expected=${frames_per} actual=${count}" >&2
            exit 1
        fi

        expressions=$((expressions + 1))
        total=$((total + count))
    done < <(find "${LOCAL_FRAMES_DIR}" -mindepth 1 -maxdepth 1 -type d -print0)

    if (( expressions == 0 )); then
        echo "local_verify error=no_expression_dirs path=${LOCAL_FRAMES_DIR}" >&2
        exit 1
    fi

    LOCAL_EXPRESSIONS="${expressions}"
    LOCAL_FRAMES_PER="${frames_per}"
    LOCAL_TOTAL="${total}"

    echo "local_verify expressions=${LOCAL_EXPRESSIONS} frames_per=${LOCAL_FRAMES_PER}" \
        "total=${LOCAL_TOTAL}"
}

create_archive() {
    local stamp="$1"
    LOCAL_TARBALL="${TMPDIR:-/tmp}/mungi_character_frames_${stamp}.tar.gz"

    echo "archive_create path=${LOCAL_TARBALL}"
    tar -C "${LOCAL_FRAMES_DIR}" -czf "${LOCAL_TARBALL}" .
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
parent_dir=\"\${remote_repo}/assets/character\";
frames_dir=\"\${parent_dir}/frames\";
backup_dir=\"\${frames_dir}.bak-\${stamp}\";
stage_dir=\"\${frames_dir}.stage-\${stamp}\";
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
if [ -e \"\${frames_dir}\" ]; then
    mv \"\${frames_dir}\" \"\${backup_dir}\";
fi;
mv \"\${stage_dir}\" \"\${frames_dir}\";
echo \"remote_deploy frames_dir=\${frames_dir} backup=\${backup_dir}\""
}

verify_remote_frames() {
    ssh "${REMOTE_HOST}" \
        "set -e;
frames_dir='${REMOTE_REPO}/assets/character/frames';
expected_expressions='${LOCAL_EXPRESSIONS}';
expected_frames_per='${LOCAL_FRAMES_PER}';
expected_total='${LOCAL_TOTAL}';
expressions=0;
frames_per=0;
total=0;
if [ ! -d \"\${frames_dir}\" ]; then
    echo \"remote_verify error=missing_dir path=\${frames_dir}\" >&2;
    exit 1;
fi;
for dir in \"\${frames_dir}\"/*; do
    [ -d \"\${dir}\" ] || continue;
    expression=\$(basename \"\${dir}\");
    count=\$(find \"\${dir}\" -maxdepth 1 -type f -name '*.png' -print |
        wc -l |
        awk '{print \$1}');
    if [ \"\${count}\" -eq 0 ]; then
        echo \"remote_verify error=empty_expression expression=\${expression}\" >&2;
        exit 1;
    fi;
    if [ \"\${frames_per}\" -eq 0 ]; then
        frames_per=\"\${count}\";
    elif [ \"\${count}\" -ne \"\${frames_per}\" ]; then
        echo \"remote_verify error=frame_count_mismatch expression=\${expression}\" >&2;
        echo \"remote_verify expected=\${frames_per} actual=\${count}\" >&2;
        exit 1;
    fi;
    expressions=\$((expressions + 1));
    total=\$((total + count));
done;
echo \"remote_verify expressions=\${expressions} frames_per=\${frames_per} total=\${total}\";
if [ \"\${expressions}\" -ne \"\${expected_expressions}\" ] ||
    [ \"\${frames_per}\" -ne \"\${expected_frames_per}\" ] ||
    [ \"\${total}\" -ne \"\${expected_total}\" ]; then
    echo \"remote_verify error=local_remote_mismatch\" >&2;
    echo \"remote_verify expected_expressions=\${expected_expressions}\" >&2;
    echo \"remote_verify expected_frames_per=\${expected_frames_per}\" >&2;
    echo \"remote_verify expected_total=\${expected_total}\" >&2;
    echo \"remote_verify actual=\${expressions}/\${frames_per}/\${total}\" >&2;
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

if (( DO_RENDER == 1 )); then
    echo "render_start command=python scripts/render_emoji_frames.py"
    python scripts/render_emoji_frames.py
fi

verify_local_frames

STAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_TARBALL="/tmp/mungi_character_frames_${STAMP}.tar.gz"

create_archive "${STAMP}"
transfer_archive "${REMOTE_TARBALL}"
deploy_remote_archive "${REMOTE_TARBALL}" "${STAMP}"
verify_remote_frames

if (( DO_RESTART == 1 )); then
    restart_kiosk
fi

echo "deploy_complete host=${REMOTE_HOST} remote_repo=${REMOTE_REPO}"
