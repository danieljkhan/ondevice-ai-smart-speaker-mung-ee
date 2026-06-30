#!/usr/bin/env bash
set -euo pipefail

TARGET_VERSION="1.12.38"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHERPA_ONNX_RELEASE_URL="${SHERPA_ONNX_RELEASE_URL:-https://github.com/danieljkhan/ondevice-ai-smart-speaker-mung-ee/releases/download/v1.12.38-sherpa-gpu/sherpa_onnx-1.12.38-cp310-cp310-linux_aarch64.whl}"
SHERPA_ONNX_SHA256="${SHERPA_ONNX_SHA256:-858f730cc605c6289a17fbde22d399e38c239e9e5636d1b8df8825b606857bde}"
SHERPA_ONNX_WHEEL_PATH="${SHERPA_ONNX_WHEEL_PATH:-wheelhouse/sherpa_onnx-1.12.38-cp310-cp310-linux_aarch64.whl}"

usage() {
  cat <<'EOF'
Usage:
  scripts/install_sherpa_onnx.sh --from-release
  scripts/install_sherpa_onnx.sh --from-source
  scripts/install_sherpa_onnx.sh --help

Options:
  --from-release  Download the 1.12.38 wheel from GitHub Release and install it.
                  This is the default mode.
  --from-source   Delegate to scripts/build_sherpa_onnx.sh for a local source build.
  --help          Show this help message.
EOF
}

print_recovery_hint() {
  printf "Recovery hint: rerun 'scripts/install_sherpa_onnx.sh --from-source' to build locally.\n" >&2
}

fail_with_code() {
  local exit_code="$1"
  shift
  printf '%s\n' "$@" >&2
  exit "${exit_code}"
}

require_commands() {
  local cmd

  for cmd in "$@"; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      fail_with_code 1 "Missing required command: ${cmd}"
    fi
  done
}

resolve_wheel_path() {
  local candidate="$1"

  case "${candidate}" in
    /*)
      printf '%s\n' "${candidate}"
      ;;
    *)
      printf '%s\n' "${REPO_ROOT}/${candidate}"
      ;;
  esac
}

verify_wheel_sha256() {
  local wheel_path="$1"
  local wheel_dir
  local wheel_name
  local checksum_file
  local observed_sha256

  wheel_dir="$(dirname -- "${wheel_path}")"
  wheel_name="$(basename -- "${wheel_path}")"
  checksum_file="$(mktemp)"

  printf '%s  %s\n' "${SHERPA_ONNX_SHA256}" "${wheel_name}" > "${checksum_file}"
  if (cd -- "${wheel_dir}" && sha256sum -c "${checksum_file}" >/dev/null 2>&1); then
    rm -f "${checksum_file}"
    return 0
  fi

  observed_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
  rm -f "${checksum_file}" "${wheel_path}"
  printf 'SHA256 mismatch for %s\n' "${wheel_path}" >&2
  printf 'Expected: %s\n' "${SHERPA_ONNX_SHA256}" >&2
  printf 'Observed: %s\n' "${observed_sha256}" >&2
  print_recovery_hint
  exit 3
}

ensure_platform() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    fail_with_code 1 "This script must run on Linux."
  fi

  if [[ "$(uname -m)" != "aarch64" ]]; then
    fail_with_code 1 "This script is intended for Linux aarch64 systems."
  fi
}

warn_if_no_venv() {
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    printf 'Warning: no virtual environment detected. Activate the mungi venv before continuing.\n' >&2
  fi
}

install_from_release() {
  local wheel_path
  local wheel_dir
  local observed_sha256

  require_commands python pip curl sha256sum mktemp

  wheel_path="$(resolve_wheel_path "${SHERPA_ONNX_WHEEL_PATH}")"
  wheel_dir="$(dirname -- "${wheel_path}")"
  mkdir -p "${wheel_dir}"

  if [[ -f "${wheel_path}" ]]; then
    observed_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
    if [[ "${observed_sha256}" == "${SHERPA_ONNX_SHA256}" ]]; then
      printf 'Using cached wheel: %s\n' "${wheel_path}"
    else
      printf 'Cached wheel hash mismatch, re-downloading: %s\n' "${wheel_path}"
      rm -f "${wheel_path}"
    fi
  fi

  if [[ ! -f "${wheel_path}" ]]; then
    if ! curl -fL --retry 3 --retry-delay 5 -o "${wheel_path}" "${SHERPA_ONNX_RELEASE_URL}"; then
      rm -f "${wheel_path}"
      printf 'Failed to download sherpa-onnx 1.12.38 from %s\n' "${SHERPA_ONNX_RELEASE_URL}" >&2
      print_recovery_hint
      exit 2
    fi
  fi

  verify_wheel_sha256 "${wheel_path}"

  if ! pip install --force-reinstall "${wheel_path}"; then
    printf 'Failed to install wheel: %s\n' "${wheel_path}" >&2
    print_recovery_hint
    exit 4
  fi
}

install_from_source() {
  require_commands bash python pip

  if ! (
    export SHERPA_ONNX_VERSION="${TARGET_VERSION}"
    cd "${REPO_ROOT}"
    exec bash "${SCRIPT_DIR}/build_sherpa_onnx.sh"
  ); then
    printf 'Local source build failed in scripts/build_sherpa_onnx.sh\n' >&2
    print_recovery_hint
    exit 4
  fi
}

post_install_verify() {
  if ! python -c "from importlib.metadata import version; import sherpa_onnx; v = version(\"sherpa-onnx\"); print(f'sherpa-onnx: {v}'); print(f'sherpa_onnx module: {sherpa_onnx.__file__}'); target = \"${TARGET_VERSION}\"; raise SystemExit(0 if v == target or v.startswith(target + \"+\") else 1)"; then
    printf 'Post-install verification failed: expected sherpa-onnx %s or %s+<local>\n' "${TARGET_VERSION}" "${TARGET_VERSION}" >&2
    exit 5
  fi
}

main() {
  local mode="${1:---from-release}"

  if [[ $# -gt 1 ]]; then
    fail_with_code 1 "Expected at most one option. Use --help for usage."
  fi

  case "${mode}" in
    --help)
      usage
      exit 0
      ;;
    --from-release|--from-source)
      ;;
    *)
      fail_with_code 1 "Unknown option: ${mode}"
      ;;
  esac

  ensure_platform
  warn_if_no_venv

  if [[ "${mode}" == "--from-release" ]]; then
    install_from_release
  else
    install_from_source
  fi

  post_install_verify
}

main "$@"
