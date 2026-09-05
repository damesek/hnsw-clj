#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HNSW_PYTHON_VENV:-${TMPDIR:-/tmp}/hnsw-clj-bible-benchmark-venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import hnswlib, numpy' >/dev/null 2>&1; then
  if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    "$VENV/bin/python" -m ensurepip --upgrade
  fi
  "$VENV/bin/python" -m pip install -q -r "$EXAMPLE_DIR/requirements.txt"
fi

exec "$VENV/bin/python" "${HNSW_BENCHMARK_SCRIPT:-$EXAMPLE_DIR/benchmark.py}"
