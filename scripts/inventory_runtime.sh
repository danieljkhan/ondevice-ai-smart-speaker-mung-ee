#!/usr/bin/env bash
set -euo pipefail

LIVE_DIR="${1:-/opt/mungi}"
REPO_DIR="${2:-/opt/mungi-repo}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${3:-./reports/runtime-inventory-${STAMP}}"

if [[ ! -d "$LIVE_DIR" ]]; then
  echo "Live directory not found: $LIVE_DIR" >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo directory not found: $REPO_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Writing inventory to $OUT_DIR"

{
  echo "live_dir=$LIVE_DIR"
  echo "repo_dir=$REPO_DIR"
  echo "generated_at=$(date -Iseconds)"
} > "$OUT_DIR/metadata.txt"

find "$LIVE_DIR" -mindepth 1 \
  ! -path "$LIVE_DIR/.git/*" \
  | sort > "$OUT_DIR/live-tree.txt"

find "$REPO_DIR" -mindepth 1 \
  ! -path "$REPO_DIR/.git/*" \
  | sort > "$OUT_DIR/repo-tree.txt"

diff -qr \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="__pycache__" \
  --exclude="ai_models" \
  "$LIVE_DIR" "$REPO_DIR" > "$OUT_DIR/diff.txt" || true

du -sh "$LIVE_DIR" "$REPO_DIR" > "$OUT_DIR/size.txt" || true

echo "Inventory complete."
echo "Review:"
echo "  $OUT_DIR/metadata.txt"
echo "  $OUT_DIR/live-tree.txt"
echo "  $OUT_DIR/repo-tree.txt"
echo "  $OUT_DIR/diff.txt"
echo "  $OUT_DIR/size.txt"
