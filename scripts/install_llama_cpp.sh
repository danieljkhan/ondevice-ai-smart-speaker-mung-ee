#!/usr/bin/env bash
set -euo pipefail

TARGET_VERSION="0.3.17"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LEGACY_WHEEL_FILENAME="llama_cpp_python-0.3.17-cp310-cp310-linux_aarch64.whl"
LEGACY_RELEASE_TAG="v0.3.17-llama"
GITHUB_RELEASE_DOWNLOAD_BASE="https://github.com/danieljkhan/ondevice-ai-smart-speaker-mung-ee/releases/download"
LEGACY_RELEASE_BASE_URL="${GITHUB_RELEASE_DOWNLOAD_BASE}/${LEGACY_RELEASE_TAG}"
LLAMA_CPP_RELEASE_URL="${LLAMA_CPP_RELEASE_URL:-\
${LEGACY_RELEASE_BASE_URL}/${LEGACY_WHEEL_FILENAME}}"
LLAMA_CPP_SHA256="${LLAMA_CPP_SHA256:-\
12bffe3f7c5a3c445debb34b235f22276090a3d0aac1d806b29a32cd17c4f503}"
LLAMA_CPP_WHEEL_PATH="${LLAMA_CPP_WHEEL_PATH:-wheelhouse/${LEGACY_WHEEL_FILENAME}}"

WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-wheelhouse}"
B8772_WHEEL_FILENAME="llama_cpp_python-0.3.20-py3-none-linux_aarch64.whl"
B8772_RELEASE_TAG="v0.3.20-llama-b8772"
B8772_RELEASE_BASE_URL="${GITHUB_RELEASE_DOWNLOAD_BASE}/${B8772_RELEASE_TAG}"
LLAMA_CPP_B8772_RELEASE_URL="${LLAMA_CPP_B8772_RELEASE_URL:-\
${B8772_RELEASE_BASE_URL}/${B8772_WHEEL_FILENAME}}"
LLAMA_CPP_B8772_SHA256="${LLAMA_CPP_B8772_SHA256:-}"
LLAMA_CPP_B8772_WHEEL_PATH="${LLAMA_CPP_B8772_WHEEL_PATH:-\
${WHEELHOUSE_DIR}/${B8772_WHEEL_FILENAME}}"
PROVENANCE_FILENAME="release-provenance.json"

POST_INSTALL_EXPECTED_VERSION="${TARGET_VERSION}"
POST_INSTALL_PROVENANCE_PATH=""
POST_INSTALL_WHEEL_PATH=""

usage() {
  cat <<'EOF'
Usage:
  scripts/install_llama_cpp.sh --from-release
  scripts/install_llama_cpp.sh --from-source
  scripts/install_llama_cpp.sh --from-release-b8772
  scripts/install_llama_cpp.sh --from-source-b8772
  scripts/install_llama_cpp.sh --help

Options:
  --from-release        Download the 0.3.17 wheel from GitHub Release and install it.
                        This is the default mode.
  --from-source         Delegate to scripts/build_llama_cpp.sh for a legacy 0.3.17 build.
  --from-release-b8772  Download the opt-in 0.3.20+b8772 wheel and provenance JSON.
  --from-source-b8772   Build the opt-in 0.3.20+b8772 wheel from a pinned wrapper SHA.
  --help                Show this help message.
EOF
}

print_recovery_hint() {
  printf '%s\n' \
    "Recovery hint: rerun scripts/install_llama_cpp.sh --from-source to build locally." >&2
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

# Verify the pinned Jetson Python ABI before installing wheels.
validate_python_version() {
  local current_version
  local error_message

  if ! current_version="$("${PYTHON_BIN}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
  )"; then
    error_message="Mungi Phase A pins Python 3.10 (current: ${current_version:-unknown})."
    error_message+=" Set PYTHON_BIN to a 3.10 interpreter and retry."
    fail_with_code 1 "${error_message}"
  fi
}

require_python_pip() {
  if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    fail_with_code 1 "Missing required Python module for ${PYTHON_BIN}: pip"
  fi
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
  local expected_sha256="${2:-${LLAMA_CPP_SHA256}}"
  local wheel_dir
  local wheel_name
  local checksum_file
  local observed_sha256

  wheel_dir="$(dirname -- "${wheel_path}")"
  wheel_name="$(basename -- "${wheel_path}")"
  checksum_file="$(mktemp)"

  printf '%s  %s\n' "${expected_sha256}" "${wheel_name}" > "${checksum_file}"
  if (cd -- "${wheel_dir}" && sha256sum -c "${checksum_file}" >/dev/null 2>&1); then
    rm -f "${checksum_file}"
    return 0
  fi

  observed_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
  rm -f "${checksum_file}" "${wheel_path}"
  printf 'SHA256 mismatch for %s\n' "${wheel_path}" >&2
  printf 'Expected: %s\n' "${expected_sha256}" >&2
  printf 'Observed: %s\n' "${observed_sha256}" >&2
  print_recovery_hint
  exit 3
}

ensure_platform() {
  local kernel_name
  local machine_name

  kernel_name="$(uname -s)"
  if [[ "${kernel_name}" != "Linux" ]]; then
    fail_with_code 1 "This script must run on Linux."
  fi

  machine_name="$(uname -m)"
  if [[ "${machine_name}" != "aarch64" ]]; then
    fail_with_code 1 "This script is intended for Linux aarch64 systems."
  fi
}

warn_if_no_venv() {
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' \
      "Warning: no virtual environment detected. Activate the mungi venv before continuing." >&2
  fi
}

# Return one string value from a provenance JSON file.
json_get() {
  local provenance_path="$1"
  local key="$2"

  "${PYTHON_BIN}" - "${provenance_path}" "${key}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
value = data[sys.argv[2]]
if not isinstance(value, str):
    raise SystemExit(f"{sys.argv[2]} is not a string")
print(value)
PY
}

# Validate release-provenance.json has every required key and full-length digests.
validate_provenance_json() {
  local provenance_path="$1"

  "${PYTHON_BIN}" - "${provenance_path}" <<'PY'
import json
import re
import sys

required = [
    "build_mode",
    "llama_cpp_python_version",
    "llama_cpp_python_remote",
    "llama_cpp_python_branch_requested",
    "llama_cpp_python_commit_sha40",
    "llama_cpp_python_ref_resolved",
    "llama_cpp_remote",
    "llama_cpp_tag",
    "llama_cpp_commit_sha40",
    "wheel_filename",
    "wheel_sha256",
    "libllama_so_relative_path",
    "libllama_so_sha256",
    "standalone_tarball_filename",
    "standalone_tarball_sha256",
    "cmake_flags_wheel",
    "cmake_flags_standalone",
    "jetpack_version",
    "l4t_release",
    "os_release",
    "uname_a",
    "cuda_version_full",
    "cmake_version",
    "python_version_full",
    "python_abi",
    "cuda_architectures",
    "built_at",
    "built_on_hostname",
]

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

errors = []
for key in required:
    if key not in data:
        errors.append(f"missing key: {key}")
    elif not isinstance(data[key], str) or not data[key]:
        errors.append(f"empty or non-string key: {key}")

for key in ("llama_cpp_python_commit_sha40", "llama_cpp_commit_sha40"):
    value = data.get(key, "")
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        errors.append(f"{key} must be exactly 40 lowercase hex chars")

for key in ("wheel_sha256", "libllama_so_sha256", "standalone_tarball_sha256"):
    value = data.get(key, "")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        errors.append(f"{key} must be exactly 64 lowercase hex chars")

if data.get("build_mode") != "release":
    errors.append("build_mode must be release")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)
PY
}

# Download or validate release-provenance.json and return its cached path.
_fetch_provenance_json() {
  local wheel_url="$1"
  local local_candidate="${2:-}"
  local provenance_path
  local provenance_dir
  local provenance_url

  if [[ -n "${local_candidate}" && -f "${local_candidate}" ]]; then
    # validate_provenance_json maps schema failures to exit code 6 here.
    # shellcheck disable=SC2310
    if ! validate_provenance_json "${local_candidate}"; then
      fail_with_code 6 "Invalid local provenance JSON: ${local_candidate}"
    fi
    printf '%s\n' "${local_candidate}"
    return
  fi

  provenance_path="$(resolve_wheel_path "${WHEELHOUSE_DIR}/${PROVENANCE_FILENAME}")"
  provenance_dir="$(dirname -- "${provenance_path}")"
  provenance_url="${wheel_url%/*}/${PROVENANCE_FILENAME}"
  mkdir -p "${provenance_dir}"

  if ! curl -fL --retry 3 --retry-delay 5 -o "${provenance_path}" "${provenance_url}"; then
    rm -f "${provenance_path}"
    fail_with_code 6 \
      "Failed to download release-provenance.json from ${provenance_url}" \
      "The b8772 release is not yet published." \
      "See Phase B WI-5 in Dev_Plan/2026-04-19-llama-cpp-b8772-parity-plan.md."
  fi

  # validate_provenance_json maps schema failures to exit code 6 here.
  # shellcheck disable=SC2310
  if ! validate_provenance_json "${provenance_path}"; then
    fail_with_code 6 "Invalid provenance JSON downloaded from ${provenance_url}"
  fi

  printf '%s\n' "${provenance_path}"
}

# Return the expected b8772 wheel SHA, preferring the explicit env override.
b8772_expected_wheel_sha256() {
  local provenance_path="$1"

  if [[ -n "${LLAMA_CPP_B8772_SHA256}" ]]; then
    printf '%s\n' "${LLAMA_CPP_B8772_SHA256}"
    return
  fi

  json_get "${provenance_path}" "wheel_sha256"
}

# Validate LLAMA_CPP_PY_SHA before delegating to build_llama_cpp.sh.
require_b8772_source_sha() {
  if [[ -z "${LLAMA_CPP_PY_SHA:-}" ]]; then
    fail_with_code 1 \
      "LLAMA_CPP_PY_SHA is required for --from-source-b8772." \
      "Use the canonical SHA from the GH Release release-provenance.json."
  fi

  if [[ ! "${LLAMA_CPP_PY_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    fail_with_code 1 \
      "LLAMA_CPP_PY_SHA must be a 40-char lowercase hex commit SHA of abetlen/llama-cpp-python."
  fi
}

install_from_release() {
  local wheel_path
  local wheel_dir
  local observed_sha256

  require_commands "${PYTHON_BIN}" curl sha256sum mktemp
  require_python_pip

  wheel_path="$(resolve_wheel_path "${LLAMA_CPP_WHEEL_PATH}")"
  wheel_dir="$(dirname -- "${wheel_path}")"
  mkdir -p "${wheel_dir}"

  if [[ -f "${wheel_path}" ]]; then
    observed_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
    if [[ "${observed_sha256}" == "${LLAMA_CPP_SHA256}" ]]; then
      printf 'Using cached wheel: %s\n' "${wheel_path}"
    else
      printf 'Cached wheel hash mismatch, re-downloading: %s\n' "${wheel_path}"
      rm -f "${wheel_path}"
    fi
  fi

  if [[ ! -f "${wheel_path}" ]]; then
    if ! curl -fL --retry 3 --retry-delay 5 -o "${wheel_path}" "${LLAMA_CPP_RELEASE_URL}"; then
      rm -f "${wheel_path}"
      printf 'Failed to download llama-cpp-python 0.3.17 from %s\n' "${LLAMA_CPP_RELEASE_URL}" >&2
      print_recovery_hint
      exit 2
    fi
  fi

  verify_wheel_sha256 "${wheel_path}"

  if ! "${PYTHON_BIN}" -m pip install --force-reinstall "${wheel_path}"; then
    printf 'Failed to install wheel: %s\n' "${wheel_path}" >&2
    print_recovery_hint
    exit 4
  fi

  POST_INSTALL_EXPECTED_VERSION="${TARGET_VERSION}"
  POST_INSTALL_PROVENANCE_PATH=""
  POST_INSTALL_WHEEL_PATH="${wheel_path}"
}

install_from_source() {
  require_commands bash "${PYTHON_BIN}"
  require_python_pip

  if ! (
    export LLAMA_CPP_VERSION="${TARGET_VERSION}"
    export PYTHON_BIN
    unset LLAMA_CPP_MODE
    cd "${REPO_ROOT}"
    exec bash "${SCRIPT_DIR}/build_llama_cpp.sh"
  ); then
    printf 'Local source build failed in scripts/build_llama_cpp.sh\n' >&2
    print_recovery_hint
    exit 4
  fi

  POST_INSTALL_EXPECTED_VERSION="${TARGET_VERSION}"
  POST_INSTALL_PROVENANCE_PATH=""
  POST_INSTALL_WHEEL_PATH=""
}

# Install the opt-in b8772 wheel from the future GH Release asset.
install_from_release_b8772() {
  local wheel_path
  local wheel_dir
  local observed_sha256
  local provenance_path
  local expected_sha256
  local expected_version

  require_commands "${PYTHON_BIN}" curl sha256sum mktemp
  require_python_pip

  wheel_path="$(resolve_wheel_path "${LLAMA_CPP_B8772_WHEEL_PATH}")"
  wheel_dir="$(dirname -- "${wheel_path}")"
  mkdir -p "${wheel_dir}"

  provenance_path="$(_fetch_provenance_json "${LLAMA_CPP_B8772_RELEASE_URL}")"
  expected_sha256="$(b8772_expected_wheel_sha256 "${provenance_path}")"

  if [[ -f "${wheel_path}" ]]; then
    observed_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
    if [[ "${observed_sha256}" == "${expected_sha256}" ]]; then
      printf 'Using cached wheel: %s\n' "${wheel_path}"
    else
      printf 'Cached wheel hash mismatch, re-downloading: %s\n' "${wheel_path}"
      rm -f "${wheel_path}"
    fi
  fi

  if [[ ! -f "${wheel_path}" ]]; then
    if ! curl -fL --retry 3 --retry-delay 5 \
      -o "${wheel_path}" "${LLAMA_CPP_B8772_RELEASE_URL}"; then
      rm -f "${wheel_path}"
      printf 'Failed to download llama-cpp-python b8772 from %s\n' \
        "${LLAMA_CPP_B8772_RELEASE_URL}" >&2
      printf 'The b8772 release is not yet published; see Phase B WI-5 in %s.\n' \
        "Dev_Plan/2026-04-19-llama-cpp-b8772-parity-plan.md" >&2
      exit 2
    fi
  fi

  verify_wheel_sha256 "${wheel_path}" "${expected_sha256}"

  if ! "${PYTHON_BIN}" -m pip install --force-reinstall "${wheel_path}"; then
    printf 'Failed to install wheel: %s\n' "${wheel_path}" >&2
    exit 4
  fi

  expected_version="$(json_get "${provenance_path}" "llama_cpp_python_version")"
  POST_INSTALL_EXPECTED_VERSION="${expected_version}"
  POST_INSTALL_PROVENANCE_PATH="${provenance_path}"
  POST_INSTALL_WHEEL_PATH="${wheel_path}"
}

# Build and install the opt-in b8772 wheel from a pinned wrapper SHA.
install_from_source_b8772() {
  local provenance_path
  local local_provenance_path
  local wheel_filename
  local wheel_path
  local expected_version

  require_commands bash "${PYTHON_BIN}" curl sha256sum
  require_python_pip
  require_b8772_source_sha

  if ! (
    export LLAMA_CPP_MODE="release"
    export LLAMA_CPP_PY_SHA
    export LLAMA_CPP_BRANCH="b8772"
    export PYTHON_BIN
    cd "${REPO_ROOT}"
    exec bash "${SCRIPT_DIR}/build_llama_cpp.sh"
  ); then
    printf 'Local b8772 source build failed in scripts/build_llama_cpp.sh\n' >&2
    exit 4
  fi

  local_provenance_path="$(resolve_wheel_path "${WHEELHOUSE_DIR}/${PROVENANCE_FILENAME}")"
  provenance_path="$(
    _fetch_provenance_json "${LLAMA_CPP_B8772_RELEASE_URL}" "${local_provenance_path}"
  )"
  wheel_filename="$(json_get "${provenance_path}" "wheel_filename")"
  wheel_path="$(resolve_wheel_path "${WHEELHOUSE_DIR}/${wheel_filename}")"

  if [[ ! -f "${wheel_path}" ]]; then
    fail_with_code 4 "Built b8772 wheel not found at ${wheel_path}"
  fi

  expected_version="$(json_get "${provenance_path}" "llama_cpp_python_version")"
  POST_INSTALL_EXPECTED_VERSION="${expected_version}"
  POST_INSTALL_PROVENANCE_PATH="${provenance_path}"
  POST_INSTALL_WHEEL_PATH="${wheel_path}"
}

post_install_verify() {
  local expected_version="$1"
  local provenance_path="${2:-}"
  local wheel_path="${3:-}"
  local installed_version
  local module_path
  local libllama_path
  local expected_lib_sha256
  local observed_lib_sha256
  local expected_wheel_sha256
  local observed_wheel_sha256

  if ! installed_version="$("${PYTHON_BIN}" - <<'PY'
import llama_cpp

print(llama_cpp.__version__)
PY
  )"; then
    printf 'Post-install verification failed: unable to import llama_cpp.\n' >&2
    exit 5
  fi

  if ! module_path="$("${PYTHON_BIN}" - <<'PY'
import llama_cpp

print(llama_cpp.__file__)
PY
  )"; then
    printf 'Post-install verification failed: unable to resolve llama_cpp module path.\n' >&2
    exit 5
  fi

  printf 'llama_cpp.__version__: %s\n' "${installed_version}"
  printf 'llama_cpp module: %s\n' "${module_path}"

  if [[ "${installed_version}" != "${expected_version}" ]]; then
    printf 'Post-install verification failed: expected llama-cpp-python %s\n' \
      "${expected_version}" >&2
    exit 5
  fi

  if [[ -z "${provenance_path}" ]]; then
    return
  fi

  # validate_provenance_json maps schema failures to exit code 6 here.
  # shellcheck disable=SC2310
  if ! validate_provenance_json "${provenance_path}"; then
    printf 'Post-install verification failed: invalid provenance JSON %s\n' \
      "${provenance_path}" >&2
    exit 6
  fi

  expected_lib_sha256="$(json_get "${provenance_path}" "libllama_so_sha256")"
  if ! libllama_path="$("${PYTHON_BIN}" - <<'PY'
import os
import llama_cpp

print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib/libllama.so"))
PY
  )"; then
    printf 'Post-install verification failed: unable to resolve libllama.so.\n' >&2
    exit 5
  fi

  if [[ ! -f "${libllama_path}" ]]; then
    printf 'Post-install verification failed: libllama.so not found at %s\n' \
      "${libllama_path}" >&2
    exit 5
  fi

  observed_lib_sha256="$(sha256sum "${libllama_path}" | awk '{print $1}')"
  if [[ "${observed_lib_sha256}" != "${expected_lib_sha256}" ]]; then
    printf 'Post-install verification failed: libllama.so SHA256 mismatch.\n' >&2
    printf 'Expected: %s\n' "${expected_lib_sha256}" >&2
    printf 'Observed: %s\n' "${observed_lib_sha256}" >&2
    exit 5
  fi
  printf 'Verified libllama.so SHA256: %s\n' "${observed_lib_sha256}"

  expected_wheel_sha256="$(json_get "${provenance_path}" "wheel_sha256")"
  if [[ -z "${wheel_path}" || ! -f "${wheel_path}" ]]; then
    printf 'Wheel cache not found; skipping advisory wheel SHA256 verification.\n'
    return
  fi

  observed_wheel_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
  if [[ "${observed_wheel_sha256}" != "${expected_wheel_sha256}" ]]; then
    printf 'Warning: cached wheel SHA256 mismatch; installed lib verification passed.\n' >&2
    printf 'Expected wheel: %s\n' "${expected_wheel_sha256}" >&2
    printf 'Observed wheel: %s\n' "${observed_wheel_sha256}" >&2
    return
  fi

  printf 'Verified cached wheel SHA256: %s\n' "${observed_wheel_sha256}"
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
    --from-release|--from-source|--from-release-b8772|--from-source-b8772)
      ;;
    *)
      fail_with_code 1 "Unknown option: ${mode}"
      ;;
  esac

  ensure_platform
  require_commands "${PYTHON_BIN}"
  validate_python_version
  warn_if_no_venv

  case "${mode}" in
    --from-release)
      install_from_release
      ;;
    --from-source)
      install_from_source
      ;;
    --from-release-b8772)
      install_from_release_b8772
      ;;
    --from-source-b8772)
      install_from_source_b8772
      ;;
    *)
      fail_with_code 1 "Unhandled mode: ${mode}"
      ;;
  esac

  post_install_verify \
    "${POST_INSTALL_EXPECTED_VERSION}" \
    "${POST_INSTALL_PROVENANCE_PATH}" \
    "${POST_INSTALL_WHEEL_PATH}"
}

main "$@"
