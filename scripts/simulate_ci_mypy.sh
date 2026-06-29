#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/simulate_ci_mypy.sh [--keep-venv] [--verbose] [--help]

Simulate the CI mypy job in a clean temporary virtual environment.

Options:
  --keep-venv  Keep the temporary virtual environment for debugging.
  --verbose    Stream pip install and mypy output to stdout.
  --help, -h   Show this help message and exit.
EOF
}

keep_venv=false
verbose=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-venv)
      keep_venv=true
      ;;
    --verbose)
      verbose=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

tmpdir=""
venv_dir=""
install_log=""
mypy_log=""
VENV_PY=""

cleanup() {
  if [[ -n "$tmpdir" && -d "$tmpdir" ]]; then
    if [[ "$keep_venv" == true ]]; then
      printf 'Temp venv preserved at: %s\n' "$venv_dir"
    else
      rm -rf "$tmpdir"
      printf 'Temp venv cleaned up.\n'
    fi
  fi
}

trap cleanup EXIT

tmpdir="$(mktemp -d -t mungi_ci_mypy.XXXXXX)"
venv_dir="$tmpdir/venv"
install_log="$tmpdir/pip-install.log"
mypy_log="$tmpdir/mypy.log"

mypy_targets=(
  core/
  models/
  safety/
  hardware/
  scripts/
  parental/
)

quiet_args=(--quiet)
if [[ "$verbose" == true ]]; then
  quiet_args=()
fi

printf '=== CI mypy simulation ===\n'
printf 'Repo root: %s\n' "$repo_root"
printf 'Temp venv: %s\n' "$venv_dir"

python3 -m venv "$venv_dir"

if [[ -x "$venv_dir/bin/python" ]]; then
  VENV_PY="$venv_dir/bin/python"
elif [[ -x "$venv_dir/Scripts/python.exe" ]]; then
  VENV_PY="$venv_dir/Scripts/python.exe"
else
  echo "ERROR: Could not find venv Python interpreter in $venv_dir" >&2
  exit 1
fi

printf 'Installing requirements-ci.txt...'
if [[ "$verbose" == true ]]; then
  printf '\n'
  "$VENV_PY" -m pip install "${quiet_args[@]}" --upgrade pip wheel 2>&1 | tee "$install_log"
  "$VENV_PY" -m pip install "${quiet_args[@]}" -r requirements-ci.txt 2>&1 | tee -a "$install_log"
else
  "$VENV_PY" -m pip install "${quiet_args[@]}" --upgrade pip wheel >"$install_log" 2>&1
  "$VENV_PY" -m pip install "${quiet_args[@]}" -r requirements-ci.txt >>"$install_log" 2>&1
fi

package_count="$(
  "$VENV_PY" -m pip list --format=freeze 2>/dev/null | wc -l | awk '{print $1}'
)"
printf ' done (%s packages)\n' "$package_count"

printf 'Running mypy on %s...\n\n' "${mypy_targets[*]}"

set +e
if [[ "$verbose" == true ]]; then
  "$VENV_PY" -m mypy "${mypy_targets[@]}" 2>&1 | tee "$mypy_log"
  mypy_exit=${PIPESTATUS[0]}
else
  "$VENV_PY" -m mypy "${mypy_targets[@]}" >"$mypy_log" 2>&1
  mypy_exit=$?
fi
set -e

if [[ "$mypy_exit" -eq 0 ]]; then
  printf '=== RESULT: PASS ===\n'
else
  printf '=== RESULT: FAIL ===\n'
fi
printf 'mypy exit code: %s\n' "$mypy_exit"

if [[ "$mypy_exit" -ne 0 ]]; then
  printf '\nLast 30 lines of mypy output\n'
  tail -n 30 "$mypy_log"
fi

exit "$mypy_exit"
