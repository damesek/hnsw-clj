#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HNSW_BENCHMARK_SCRIPT="$EXAMPLE_DIR/construction_benchmark.py"
exec bash "$EXAMPLE_DIR/run_python.sh"
