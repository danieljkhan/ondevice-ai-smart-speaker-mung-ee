#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheels for Jetson Orin Nano Super.
# Target environment: JetPack 6.2, CUDA 12.6, Python 3.10, sm_87.
# Preferred legacy install path: `scripts/install_llama_cpp.sh --from-release`.
# Use this script as the fallback when a local source build is required.
#
# Version history:
#   0.3.14 | KV cache broken on full offload in E2E
#   0.3.16 | Intermediate build; ggml_cpy() fallback still present
#   0.3.17 | ggml_cpy() removed (previous default; legacy-0.3.17 mode preserves
#          | byte-identical install)
#   0.3.20 | llama.cpp b8772 parity (NVIDIA dusty-nv reference; discovery + release modes)
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   LLAMA_CPP_MODE=legacy-0.3.17|discovery|release
#   LLAMA_CPP_VERSION=0.3.17
#   LLAMA_CPP_PY_SHA=<40-char lowercase hex SHA>  # release mode only
#   LLAMA_CPP_PY_BRANCH=main                      # discovery mode only
#   LLAMA_CPP_BRANCH=b8772                        # discovery + release modes
#   CUDA_HOME=/usr/local/cuda
#   CUDA_ARCHITECTURES=87                         # legacy mode only
#   WHEELHOUSE_DIR=wheelhouse
#   CMAKE_BUILD_PARALLEL_LEVEL=6
#   PIP_EXTRA_ARGS="--index-url ..."
#   MUNGI_KEEP_BUILD_ROOT=1  # retain the discovery/release build tempdir on failure
#                            # (any non-empty value is treated as truthy per POSIX convention)

LLAMA_CPP_PY_REMOTE="https://github.com/abetlen/llama-cpp-python"
LLAMA_CPP_REMOTE="https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_B8772_VERSION="0.3.20"
STANDALONE_TARBALL_FILENAME="llama-cpp-bin-b8772.tar.gz"

PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMA_CPP_VERSION_WAS_SET=0
if [[ -n "${LLAMA_CPP_VERSION+x}" ]]; then
  LLAMA_CPP_VERSION_WAS_SET=1
fi
LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-0.3.17}"
LLAMA_CPP_PY_BRANCH="${LLAMA_CPP_PY_BRANCH:-main}"
LLAMA_CPP_BRANCH="${LLAMA_CPP_BRANCH:-b8772}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-wheelhouse}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-6}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

declare -a extra_pip_args=()
if [[ -n "${PIP_EXTRA_ARGS}" ]]; then
  # Preserve existing space-delimited PIP_EXTRA_ARGS behavior.
  # shellcheck disable=SC2206
  extra_pip_args=( ${PIP_EXTRA_ARGS} )
fi

# Print an error message and exit non-zero.
fail() {
  printf '%s\n' "$@" >&2
  exit 1
}

# Verify every required command is available in PATH.
require_commands() {
  local cmd

  for cmd in "$@"; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      fail "Missing required command: ${cmd}"
    fi
  done
}

# Verify the pinned Jetson Python ABI before building wheels.
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
    fail "${error_message}"
  fi
}

# Verify this build is running on the supported Jetson Linux platform.
ensure_platform() {
  local kernel_name
  local machine_name

  kernel_name="$(uname -s)"
  if [[ "${kernel_name}" != "Linux" ]]; then
    fail "This script must run on Linux."
  fi

  machine_name="$(uname -m)"
  if [[ "${machine_name}" != "aarch64" ]]; then
    fail "This script is intended for Jetson aarch64 systems."
  fi
}

# Warn operators when they forgot to activate the project virtual environment.
warn_if_no_venv() {
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' "Warning: no virtual environment detected. Run mungidev before continuing." >&2
  fi
}

# Select the build mode from LLAMA_CPP_MODE and LLAMA_CPP_VERSION.
determine_build_mode() {
  if [[ -n "${LLAMA_CPP_MODE+x}" ]]; then
    case "${LLAMA_CPP_MODE}" in
      legacy-0.3.17|discovery|release)
        printf '%s\n' "${LLAMA_CPP_MODE}"
        ;;
      *)
        fail "Unknown LLAMA_CPP_MODE: ${LLAMA_CPP_MODE}" \
          "Expected one of: legacy-0.3.17, discovery, release."
        ;;
    esac
    return
  fi

  if [[ "${LLAMA_CPP_VERSION_WAS_SET}" -eq 0 || "${LLAMA_CPP_VERSION}" == "0.3.17" ]]; then
    printf '%s\n' "legacy-0.3.17"
    return
  fi

  if [[ "${LLAMA_CPP_VERSION}" == "0.3.20" ]]; then
    fail "LLAMA_CPP_VERSION=0.3.20 requires an explicit b8772 build mode." \
      "Set LLAMA_CPP_MODE=discovery or LLAMA_CPP_MODE=release with LLAMA_CPP_PY_SHA."
  fi

  fail "LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION} is not auto-selectable." \
    "Set LLAMA_CPP_MODE explicitly to legacy-0.3.17, discovery, or release."
}

# Validate the release-mode wrapper commit before any clone or checkout.
validate_release_sha() {
  if [[ ! "${LLAMA_CPP_PY_SHA:-}" =~ ^[0-9a-f]{40}$ ]]; then
    fail "LLAMA_CPP_PY_SHA must be a 40-char lowercase hex commit SHA of abetlen/llama-cpp-python."
  fi
}

# Export CUDA and build environment variables shared by all modes.
prepare_common_environment() {
  mkdir -p "${WHEELHOUSE_DIR}"

  export CUDA_HOME
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CMAKE_BUILD_PARALLEL_LEVEL
}

# Verify CUDA exists after CUDA_HOME has been resolved.
ensure_cuda_compiler() {
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    fail "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc"
  fi
}

# Print common environment diagnostics before a build starts.
print_preflight() {
  local build_mode="$1"
  local nvcc_output
  local nvcc_version
  local python_version

  python_version="$("${PYTHON_BIN}" --version 2>&1)"
  nvcc_output="$("${CUDACXX}" --version)"
  nvcc_version="${nvcc_output##*$'\n'}"

  printf '%s\n' "==> Preflight"
  printf 'Build mode: %s\n' "${build_mode}"
  printf 'Python: %s\n' "${python_version}"
  printf 'nvcc: %s\n' "${nvcc_version}"
  printf 'CUDA architectures: %s\n' "${CUDA_ARCHITECTURES}"
  printf 'Wheelhouse: %s\n' "${WHEELHOUSE_DIR}"
}

# join_by_space converts a bash array to a space-separated string. Used only for:
#   1. CMAKE_ARGS env var passed to pip (scikit-build-core shlex-parses it internally).
#   2. Human-readable flag strings emitted into provenance JSON (not shell-evaluated).
# For direct cmake invocations, pass the bash array with "${array[@]}"; not this function.
join_by_space() {
  local IFS=' '

  printf '%s\n' "$*"
}

# Return the SHA256 digest for a file.
sha256_of() {
  local path="$1"

  sha256sum "${path}" | awk '{print $1}'
}

# Resolve an artifact directory relative to the caller's current directory.
resolve_artifact_dir() {
  local candidate="$1"
  local current_dir

  case "${candidate}" in
    /*)
      printf '%s\n' "${candidate}"
      ;;
    *)
      current_dir="$(pwd)"
      printf '%s/%s\n' "${current_dir}" "${candidate}"
      ;;
  esac
}

# Build the original 0.3.17 pip source wheel path without provenance output.
build_legacy() {
  local built_wheel
  local -a built_wheels
  local -a wheel_cmd

  export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"

  echo "llama-cpp-python version: ${LLAMA_CPP_VERSION}"

  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel scikit-build-core

  wheel_cmd=(
    "${PYTHON_BIN}" -m pip wheel
    --no-cache-dir
    --no-binary llama-cpp-python
    --wheel-dir "${WHEELHOUSE_DIR}"
    "llama-cpp-python==${LLAMA_CPP_VERSION}"
    "${extra_pip_args[@]}"
  )
  "${wheel_cmd[@]}"

  shopt -s nullglob
  built_wheels=( "${WHEELHOUSE_DIR}"/llama_cpp_python-"${LLAMA_CPP_VERSION}"-*.whl )
  if (( ${#built_wheels[@]} == 0 )); then
    echo "No built wheel found in ${WHEELHOUSE_DIR}." >&2
    exit 1
  fi

  built_wheel="${built_wheels[0]}"

  "${PYTHON_BIN}" -m pip install --force-reinstall "${built_wheel}"

  "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import llama_cpp

print(f"llama-cpp-python: {version('llama-cpp-python')}")
print(f"llama_cpp module: {llama_cpp.__file__}")
PY

  echo "==> Built wheel SHA256"
  (
    cd "$(dirname "${built_wheel}")"
    sha256sum "$(basename "${built_wheel}")"
  )

  echo "Build and install completed successfully."
}

# Clone llama-cpp-python for discovery or release mode into an isolated workspace.
clone_wrapper_repo() {
  local build_mode="$1"
  local wrapper_dir="$2"

  if [[ "${build_mode}" == "discovery" ]]; then
    git clone --recursive --branch "${LLAMA_CPP_PY_BRANCH}" \
      "${LLAMA_CPP_PY_REMOTE}" "${wrapper_dir}"
    return
  fi

  git clone --recursive "${LLAMA_CPP_PY_REMOTE}" "${wrapper_dir}"
  git -C "${wrapper_dir}" checkout "${LLAMA_CPP_PY_SHA}"
}

# Check out the requested llama.cpp tag or commit inside the wrapper submodule.
checkout_llama_cpp() {
  local wrapper_dir="$1"
  local llama_dir="${wrapper_dir}/vendor/llama.cpp"

  git -C "${llama_dir}" fetch --tags
  git -C "${llama_dir}" checkout "${LLAMA_CPP_BRANCH}"
}

# Resolve a readable git ref string for provenance.
resolve_git_ref() {
  local repo_dir="$1"
  local sha="$2"
  local ref

  ref="$(git -C "${repo_dir}" symbolic-ref -q HEAD || true)"
  if [[ -z "${ref}" ]]; then
    ref="detached HEAD"
  fi

  printf '%s @ %s\n' "${ref}" "${sha}"
}

# Find the newest b8772 wheel produced in the wheelhouse.
find_b8772_wheel() {
  local wheelhouse_dir="$1"

  "${PYTHON_BIN}" - "${wheelhouse_dir}" "${LLAMA_CPP_B8772_VERSION}" <<'PY'
from pathlib import Path
import sys

wheelhouse = Path(sys.argv[1])
version = sys.argv[2]
wheels = list(wheelhouse.glob(f"llama_cpp_python-{version}-*.whl"))
if not wheels:
    raise SystemExit(f"No llama_cpp_python-{version} wheel found in {wheelhouse}")
print(max(wheels, key=lambda path: path.stat().st_mtime))
PY
}

# Build the b8772 Python wheel from the checked-out wrapper repo.
build_b8772_wheel() {
  local wrapper_dir="$1"
  local wheel_flags="$2"
  local wheelhouse_dir="$3"
  local -a wheel_cmd

  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel scikit-build-core

  wheel_cmd=(
    "${PYTHON_BIN}" -m pip wheel
    --no-cache-dir
    --no-binary=llama-cpp-python
    --wheel-dir "${wheelhouse_dir}"
    .
    "${extra_pip_args[@]}"
  )

  (
    cd "${wrapper_dir}"
    CMAKE_ARGS="${wheel_flags}" FORCE_CMAKE=1 "${wheel_cmd[@]}"
  )
}

# Install the freshly built wheel so libllama.so can be hashed deterministically.
install_b8772_wheel() {
  local built_wheel="$1"

  "${PYTHON_BIN}" -m pip install --force-reinstall "${built_wheel}"
}

# Return the installed llama-cpp-python package version.
installed_llama_cpp_python_version() {
  "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

print(version("llama-cpp-python"))
PY
}

# Return the installed wheel's libllama.so path.
installed_libllama_path() {
  "${PYTHON_BIN}" - <<'PY'
import os
import llama_cpp

print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib/libllama.so"))
PY
}

# Build standalone llama.cpp binaries and package build/bin into a tarball.
build_standalone_binaries() {
  local wrapper_dir="$1"
  local standalone_tarball="$2"
  local llama_dir="${wrapper_dir}/vendor/llama.cpp"
  local build_dir="${llama_dir}/build-b8772"
  local bin_dir="${build_dir}/bin"
  local -a standalone_flag_array

  shift 2
  standalone_flag_array=( "$@" )

  cmake -S "${llama_dir}" -B "${build_dir}" "${standalone_flag_array[@]}"
  cmake --build "${build_dir}" --config Release

  if [[ ! -d "${bin_dir}" ]]; then
    fail "Standalone build completed but ${bin_dir} does not exist."
  fi

  tar -czf "${standalone_tarball}" -C "${bin_dir}" .
}

# Return PRETTY_NAME from /etc/os-release when available.
detect_os_release() {
  if [[ -r /etc/os-release ]]; then
    (
      # shellcheck disable=SC1091
      source /etc/os-release
      printf '%s\n' "${PRETTY_NAME:-unknown}"
    )
    return
  fi

  printf '%s\n' "unknown"
}

# Return the L4T release string from /etc/nv_tegra_release when available.
detect_l4t_release() {
  if [[ -r /etc/nv_tegra_release ]]; then
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re

text = Path("/etc/nv_tegra_release").read_text(encoding="utf-8", errors="ignore")
release = re.search(r"R(\d+)", text)
revision = re.search(r"REVISION:\s*([0-9.]+)", text)
if release and revision:
    print(f"R{release.group(1)}.{revision.group(1)}")
else:
    print(text.strip() or "unknown")
PY
    return
  fi

  printf '%s\n' "unknown"
}

# Return the installed JetPack package version when the package database exposes it.
detect_jetpack_version() {
  local jetpack_version

  if jetpack_version="$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)"; then
    printf '%s\n' "${jetpack_version}"
    return
  fi

  printf '%s\n' "unknown"
}

# Return a fully qualified hostname when available.
detect_hostname() {
  local host_name

  if host_name="$(hostname -f 2>/dev/null)"; then
    printf '%s\n' "${host_name}"
    return
  fi

  hostname
}

# Return the Python ABI tag used by CPython wheels.
detect_python_abi() {
  "${PYTHON_BIN}" - <<'PY'
import sys

print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
}

# Return an ISO-8601 timestamp with timezone.
build_timestamp() {
  "${PYTHON_BIN}" - <<'PY'
from datetime import datetime

print(datetime.now().astimezone().isoformat())
PY
}

# Write provenance JSON with Python's JSON encoder.
write_provenance_json() {
  local output_path="$1"
  local build_mode="$2"
  local wrapper_dir="$3"
  local built_wheel="$4"
  local standalone_tarball="$5"
  local wheel_flags="$6"
  local standalone_flags="$7"
  local wrapper_sha="$8"
  local llama_sha="$9"
  local llama_remote="${10}"
  local wheel_filename
  local libllama_path
  local package_version

  wheel_filename="$(basename "${built_wheel}")"
  libllama_path="$(installed_libllama_path)"
  package_version="$(installed_llama_cpp_python_version)"

  export PROVENANCE_BUILD_MODE="${build_mode}"
  export PROVENANCE_LLAMA_CPP_PYTHON_VERSION="${package_version}"
  export PROVENANCE_LLAMA_CPP_PYTHON_REMOTE="${LLAMA_CPP_PY_REMOTE}"
  export PROVENANCE_LLAMA_CPP_PYTHON_BRANCH_REQUESTED="${LLAMA_CPP_PY_BRANCH}"
  export PROVENANCE_LLAMA_CPP_PYTHON_COMMIT_SHA40="${wrapper_sha}"
  export PROVENANCE_LLAMA_CPP_PYTHON_REF_RESOLVED
  PROVENANCE_LLAMA_CPP_PYTHON_REF_RESOLVED="$(resolve_git_ref "${wrapper_dir}" "${wrapper_sha}")"
  export PROVENANCE_LLAMA_CPP_REMOTE="${llama_remote}"
  export PROVENANCE_LLAMA_CPP_TAG="${LLAMA_CPP_BRANCH}"
  export PROVENANCE_LLAMA_CPP_COMMIT_SHA40="${llama_sha}"
  export PROVENANCE_WHEEL_FILENAME="${wheel_filename}"
  export PROVENANCE_WHEEL_SHA256
  PROVENANCE_WHEEL_SHA256="$(sha256_of "${built_wheel}")"
  export PROVENANCE_LIBLLAMA_SO_RELATIVE_PATH="llama_cpp/lib/libllama.so"
  export PROVENANCE_LIBLLAMA_SO_SHA256
  PROVENANCE_LIBLLAMA_SO_SHA256="$(sha256_of "${libllama_path}")"
  export PROVENANCE_STANDALONE_TARBALL_FILENAME="${STANDALONE_TARBALL_FILENAME}"
  export PROVENANCE_STANDALONE_TARBALL_SHA256
  PROVENANCE_STANDALONE_TARBALL_SHA256="$(sha256_of "${standalone_tarball}")"
  export PROVENANCE_CMAKE_FLAGS_WHEEL="${wheel_flags}"
  export PROVENANCE_CMAKE_FLAGS_STANDALONE="${standalone_flags}"
  export PROVENANCE_JETPACK_VERSION
  PROVENANCE_JETPACK_VERSION="$(detect_jetpack_version)"
  export PROVENANCE_L4T_RELEASE
  PROVENANCE_L4T_RELEASE="$(detect_l4t_release)"
  export PROVENANCE_OS_RELEASE
  PROVENANCE_OS_RELEASE="$(detect_os_release)"
  export PROVENANCE_UNAME_A
  PROVENANCE_UNAME_A="$(uname -a)"
  export PROVENANCE_CUDA_VERSION_FULL
  PROVENANCE_CUDA_VERSION_FULL="$("${CUDACXX}" --version)"
  export PROVENANCE_CMAKE_VERSION
  PROVENANCE_CMAKE_VERSION="$(cmake --version | head -n 1)"
  export PROVENANCE_PYTHON_VERSION_FULL
  PROVENANCE_PYTHON_VERSION_FULL="$(
    "${PYTHON_BIN}" -c 'import platform; print(platform.python_version())'
  )"
  export PROVENANCE_PYTHON_ABI
  PROVENANCE_PYTHON_ABI="$(detect_python_abi)"
  export PROVENANCE_CUDA_ARCHITECTURES="87"
  export PROVENANCE_BUILT_AT
  PROVENANCE_BUILT_AT="$(build_timestamp)"
  export PROVENANCE_BUILT_ON_HOSTNAME
  PROVENANCE_BUILT_ON_HOSTNAME="$(detect_hostname)"

  "${PYTHON_BIN}" - "${output_path}" <<'PY'
import json
import os
import sys

data = {
    "build_mode": os.environ["PROVENANCE_BUILD_MODE"],
    "llama_cpp_python_version": os.environ["PROVENANCE_LLAMA_CPP_PYTHON_VERSION"],
    "llama_cpp_python_remote": os.environ["PROVENANCE_LLAMA_CPP_PYTHON_REMOTE"],
    "llama_cpp_python_branch_requested": os.environ[
        "PROVENANCE_LLAMA_CPP_PYTHON_BRANCH_REQUESTED"
    ],
    "llama_cpp_python_commit_sha40": os.environ[
        "PROVENANCE_LLAMA_CPP_PYTHON_COMMIT_SHA40"
    ],
    "llama_cpp_python_ref_resolved": os.environ[
        "PROVENANCE_LLAMA_CPP_PYTHON_REF_RESOLVED"
    ],
    "llama_cpp_remote": os.environ["PROVENANCE_LLAMA_CPP_REMOTE"],
    "llama_cpp_tag": os.environ["PROVENANCE_LLAMA_CPP_TAG"],
    "llama_cpp_commit_sha40": os.environ["PROVENANCE_LLAMA_CPP_COMMIT_SHA40"],
    "wheel_filename": os.environ["PROVENANCE_WHEEL_FILENAME"],
    "wheel_sha256": os.environ["PROVENANCE_WHEEL_SHA256"],
    "libllama_so_relative_path": os.environ["PROVENANCE_LIBLLAMA_SO_RELATIVE_PATH"],
    "libllama_so_sha256": os.environ["PROVENANCE_LIBLLAMA_SO_SHA256"],
    "standalone_tarball_filename": os.environ[
        "PROVENANCE_STANDALONE_TARBALL_FILENAME"
    ],
    "standalone_tarball_sha256": os.environ[
        "PROVENANCE_STANDALONE_TARBALL_SHA256"
    ],
    "cmake_flags_wheel": os.environ["PROVENANCE_CMAKE_FLAGS_WHEEL"],
    "cmake_flags_standalone": os.environ["PROVENANCE_CMAKE_FLAGS_STANDALONE"],
    "jetpack_version": os.environ["PROVENANCE_JETPACK_VERSION"],
    "l4t_release": os.environ["PROVENANCE_L4T_RELEASE"],
    "os_release": os.environ["PROVENANCE_OS_RELEASE"],
    "uname_a": os.environ["PROVENANCE_UNAME_A"],
    "cuda_version_full": os.environ["PROVENANCE_CUDA_VERSION_FULL"],
    "cmake_version": os.environ["PROVENANCE_CMAKE_VERSION"],
    "python_version_full": os.environ["PROVENANCE_PYTHON_VERSION_FULL"],
    "python_abi": os.environ["PROVENANCE_PYTHON_ABI"],
    "cuda_architectures": os.environ["PROVENANCE_CUDA_ARCHITECTURES"],
    "built_at": os.environ["PROVENANCE_BUILT_AT"],
    "built_on_hostname": os.environ["PROVENANCE_BUILT_ON_HOSTNAME"],
}

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
}

# Validate provenance JSON contains required keys and full-length digests.
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

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)
PY
}

# Build discovery or release b8772 artifacts and emit provenance.
build_b8772() {
  local build_mode="$1"
  local build_root
  local wrapper_dir
  local artifact_dir
  local standalone_install_prefix
  local standalone_tarball
  local built_wheel
  local libllama_path
  local llama_dir
  local wrapper_sha
  local llama_sha
  local llama_remote
  local provenance_path
  local wheel_flags
  local standalone_flags
  local wheel_cmake_flags
  local standalone_cmake_flags
  local build_root_cleanup_state="pending"
  local had_errtrace=0

  build_root="$(mktemp -d)"

  cleanup_b8772_build_root() {
    local reason="${1:-failure}"

    if [[ "${build_root_cleanup_state}" == "done" ]]; then
      return
    fi
    build_root_cleanup_state="done"

    # Any non-empty MUNGI_KEEP_BUILD_ROOT value is truthy by POSIX convention.
    if [[ -n "${MUNGI_KEEP_BUILD_ROOT:-}" ]]; then
      printf 'MUNGI_KEEP_BUILD_ROOT set; retaining build workspace after %s: %s\n' \
        "${reason}" "${build_root}" >&2
      return
    fi

    if [[ -d "${build_root}" ]]; then
      printf 'Removing build workspace after %s: %s\n' "${reason}" "${build_root}" >&2
      rm -rf -- "${build_root}"
    fi
  }

  if shopt -qo errtrace; then
    had_errtrace=1
  else
    set -E
  fi

  trap 'cleanup_b8772_build_root "failure"' ERR
  trap 'cleanup_b8772_build_root "interrupt"; trap - ERR EXIT INT TERM; exit 130' INT
  trap 'cleanup_b8772_build_root "termination"; trap - ERR EXIT INT TERM; exit 143' TERM
  trap 'cleanup_b8772_build_root "failure"; trap - ERR EXIT INT TERM' EXIT

  wrapper_dir="${build_root}/llama-cpp-python"
  llama_dir="${wrapper_dir}/vendor/llama.cpp"
  artifact_dir="$(resolve_artifact_dir "${WHEELHOUSE_DIR}")"
  standalone_install_prefix="${build_root}/standalone-install"
  standalone_tarball="${artifact_dir}/${STANDALONE_TARBALL_FILENAME}"

  declare -a wheel_cmake_flag_array=(
    "-DGGML_CUDA=ON"
    "-DGGML_CUDA_F16=ON"
    "-DGGML_CUDA_FA_ALL_QUANTS=ON"
    "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512"
    "-DGGML_NATIVE=OFF"
    "-DLLAVA_BUILD=OFF"
    "-DCMAKE_CUDA_ARCHITECTURES=87"
  )
  declare -a standalone_cmake_flag_array=(
    "${wheel_cmake_flag_array[@]}"
    "-DLLAMA_CURL=ON"
    "-DLLAMA_OPENSSL=ON"
    "-DLLAMA_BUILD_SERVER=ON"
    "-DLLAMA_BUILD_EXAMPLES=ON"
    "-DLLAMA_BUILD_TESTS=OFF"
    "-DCMAKE_INSTALL_PREFIX=${standalone_install_prefix}"
  )

  # Wheel flags (string form): CMAKE_ARGS env for pip and provenance JSON only.
  wheel_cmake_flags="$(join_by_space "${wheel_cmake_flag_array[@]}")"
  # Standalone flags (string form): provenance JSON only; cmake uses the array directly.
  standalone_cmake_flags="$(join_by_space "${standalone_cmake_flag_array[@]}")"
  wheel_flags="${wheel_cmake_flags}"
  standalone_flags="${standalone_cmake_flags}"

  clone_wrapper_repo "${build_mode}" "${wrapper_dir}"
  checkout_llama_cpp "${wrapper_dir}"

  wrapper_sha="$(git -C "${wrapper_dir}" rev-parse HEAD)"
  if [[ "${build_mode}" == "release" && "${wrapper_sha}" != "${LLAMA_CPP_PY_SHA}" ]]; then
    fail "Wrapper SHA sanity check failed." \
      "Expected wrapper SHA: ${LLAMA_CPP_PY_SHA}" \
      "Observed wrapper SHA: ${wrapper_sha}"
  fi

  llama_sha="$(git -C "${llama_dir}" rev-parse HEAD)"
  if llama_remote="$(git -C "${llama_dir}" remote get-url origin 2>/dev/null)"; then
    :
  else
    llama_remote="${LLAMA_CPP_REMOTE}"
  fi

  build_b8772_wheel "${wrapper_dir}" "${wheel_flags}" "${artifact_dir}"
  built_wheel="$(find_b8772_wheel "${artifact_dir}")"
  install_b8772_wheel "${built_wheel}"

  libllama_path="$(installed_libllama_path)"
  if [[ ! -f "${libllama_path}" ]]; then
    fail "Installed libllama.so not found at ${libllama_path}."
  fi

  build_standalone_binaries \
    "${wrapper_dir}" \
    "${standalone_tarball}" \
    "${standalone_cmake_flag_array[@]}"

  if [[ "${build_mode}" == "release" ]]; then
    provenance_path="${artifact_dir}/release-provenance.json"
  else
    provenance_path="${artifact_dir}/discovery-provenance.json"
  fi

  write_provenance_json \
    "${provenance_path}" \
    "${build_mode}" \
    "${wrapper_dir}" \
    "${built_wheel}" \
    "${standalone_tarball}" \
    "${wheel_flags}" \
    "${standalone_flags}" \
    "${wrapper_sha}" \
    "${llama_sha}" \
    "${llama_remote}"
  validate_provenance_json "${provenance_path}"

  printf 'Built wheel: %s\n' "${built_wheel}"
  printf 'Standalone tarball: %s\n' "${standalone_tarball}"
  printf 'Provenance JSON: %s\n' "${provenance_path}"
  printf '%s\n' "Build and install completed successfully."
  if [[ -n "${MUNGI_KEEP_BUILD_ROOT:-}" ]]; then
    printf 'Build workspace retained for inspection: %s\n' "${build_root}"
  fi

  trap - ERR EXIT INT TERM
  if [[ "${had_errtrace}" -eq 0 ]]; then
    set +E
  fi
}

# Script entry point.
main() {
  local build_mode

  ensure_platform
  require_commands "${PYTHON_BIN}"
  validate_python_version

  build_mode="$(determine_build_mode)"

  warn_if_no_venv

  if [[ "${build_mode}" == "release" ]]; then
    validate_release_sha
  fi

  if [[ "${build_mode}" == "legacy-0.3.17" ]]; then
    require_commands "${PYTHON_BIN}" cmake g++ sha256sum
  else
    require_commands "${PYTHON_BIN}" cmake g++ git tar sha256sum mktemp
  fi

  ensure_cuda_compiler
  prepare_common_environment
  print_preflight "${build_mode}"

  case "${build_mode}" in
    legacy-0.3.17)
      build_legacy
      ;;
    discovery|release)
      build_b8772 "${build_mode}"
      ;;
    *)
      fail "Unhandled build mode: ${build_mode}"
      ;;
  esac
}

main "$@"
