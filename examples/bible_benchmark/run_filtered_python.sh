#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
VENV="${HNSW_PYTHON_VENV:-$ROOT_DIR/target/python-benchmark-venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
export NUM_VECTORS="${NUM_VECTORS:-31126}"
export NUM_QUERIES="${NUM_QUERIES:-1000}"
export THREADS="${THREADS:-10}"
export RUNS="${RUNS:-5}"
export EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
export EF_SEARCH="${EF_SEARCH:-80}"
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
if [[ ! -x "$PYTHON" ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Set PYTHON to an interpreter with numpy and hnswlib installed.\n' >&2
  exit 1
fi
exec "$PYTHON" examples/bible_benchmark/filtered_benchmark.py
