#!/usr/bin/env bash
set -e
echo "=== Ruff check ==="
ruff check .
echo "=== Ruff format check ==="
ruff format --check .
echo "=== Mypy ==="
mypy core/ models/ safety/ hardware/ scripts/ parental/
echo "=== Pytest ==="
pytest tests/ -v
echo "=== All checks passed ==="
