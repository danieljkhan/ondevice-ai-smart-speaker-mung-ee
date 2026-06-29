#!/usr/bin/env bash
set -euo pipefail

# Build sherpa-onnx 1.12.38 from source for Jetson Orin Nano Super.
# Target environment: JetPack 6.2, CUDA 12.6, Python 3.10, sm_87.
# Wave 3 T3.1 requires a CUDA-enabled sherpa-onnx wheel using the
# ONNX Runtime C++ Linux aarch64 GPU artifact.
# Preferred install path: `scripts/install_sherpa_onnx.sh --from-release`.
# Use this script as the fallback when a local source build is required.
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   SHERPA_ONNX_VERSION=1.12.38
#   ONNXRUNTIME_VERSION=1.18.1
#   SHERPA_ONNX_CUDA_VERSION=12.6
#   CUDA_HOME=/usr/local/cuda
#   CUDA_ARCHITECTURES=87
#   WHEELHOUSE_DIR=wheelhouse
#   CMAKE_BUILD_PARALLEL_LEVEL=4
#   PIP_EXTRA_ARGS="--index-url ..."

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run on Linux." >&2
  exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is intended for Jetson aarch64 systems." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SHERPA_ONNX_VERSION="${SHERPA_ONNX_VERSION:-1.12.38}"
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.18.1}"
SHERPA_ONNX_CUDA_VERSION="${SHERPA_ONNX_CUDA_VERSION:-12.6}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-wheelhouse}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

required_commands=("$PYTHON_BIN" cmake g++ curl)
for cmd in "${required_commands[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Warning: no virtual environment detected. Run mungidev before continuing." >&2
fi

mkdir -p "${WHEELHOUSE_DIR}"

export CUDA_HOME
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_BUILD_PARALLEL_LEVEL
export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}"
export SHERPA_ONNX_CUDA_VERSION

declare -a extra_pip_args=()
if [[ -n "${PIP_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_pip_args=( ${PIP_EXTRA_ARGS} )
fi

echo "==> Preflight"
echo "Python: $($PYTHON_BIN --version 2>&1)"
echo "nvcc: $(${CUDACXX} --version | tail -n 1)"
echo "sherpa-onnx version: ${SHERPA_ONNX_VERSION}"
echo "onnxruntime version (C++ artifact): ${ONNXRUNTIME_VERSION}"
echo "CUDA runtime version (PEP 440 local tag): ${SHERPA_ONNX_CUDA_VERSION}"
echo "CUDA architectures: ${CUDA_ARCHITECTURES}"
echo "Wheelhouse: ${WHEELHOUSE_DIR}"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel scikit-build-core

# Source tarball must be downloaded and unpacked first because sherpa-onnx's
# setup.py mutates the package version to "${VERSION}+cuda${CUDA_VERSION}"
# when SHERPA_ONNX_CMAKE_ARGS contains -DSHERPA_ONNX_ENABLE_GPU=ON. pip's
# metadata-consistency check rejects the mutated version against the PyPI
# tarball name (see pip "inconsistent version" error). Building from a local
# directory bypasses the check entirely.
src_root="${WHEELHOUSE_DIR}/src"
mkdir -p "${src_root}"
src_tarball="${src_root}/sherpa_onnx-${SHERPA_ONNX_VERSION}.tar.gz"
src_dir="${src_root}/sherpa_onnx-${SHERPA_ONNX_VERSION}"

if [[ ! -f "${src_tarball}" ]]; then
  "$PYTHON_BIN" -m pip download \
    --no-deps \
    --no-binary sherpa-onnx \
    --dest "${src_root}" \
    "sherpa-onnx==${SHERPA_ONNX_VERSION}" \
    "${extra_pip_args[@]}"
fi

if [[ ! -d "${src_dir}" ]]; then
  tar -xzf "${src_tarball}" -C "${src_root}"
fi

wheel_cmd=(
  "$PYTHON_BIN" -m pip wheel
  --no-cache-dir
  --no-deps
  --wheel-dir "${WHEELHOUSE_DIR}"
  "${src_dir}"
)
"${wheel_cmd[@]}"

shopt -s nullglob
built_wheels=( "${WHEELHOUSE_DIR}"/sherpa_onnx-"${SHERPA_ONNX_VERSION}"*-*.whl )
if (( ${#built_wheels[@]} == 0 )); then
  echo "No built wheel found in ${WHEELHOUSE_DIR}." >&2
  exit 1
fi

built_wheel="${built_wheels[0]}"

"$PYTHON_BIN" -m pip install --force-reinstall "${built_wheel}"

"$PYTHON_BIN" -c 'from importlib.metadata import version; import sherpa_onnx; print("sherpa-onnx: {}".format(version("sherpa-onnx"))); print(f"sherpa_onnx module: {sherpa_onnx.__file__}")'

echo "==> Built wheel SHA256"
(
  cd "$(dirname "${built_wheel}")"
  sha256sum "$(basename "${built_wheel}")"
)

echo "Build and install completed successfully."
