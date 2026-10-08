#!/usr/bin/env bash
# Run from any directory; uses the caller's activated isolated environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest "$@"
python -m ruff check .
python -m mypy --follow-imports=silent --check-untyped-defs utils/readout.py utils/inference_config.py
python -m compileall -q -x '(^|/)(\.git|\.venv|venv)(/|$)' .
git diff --check
