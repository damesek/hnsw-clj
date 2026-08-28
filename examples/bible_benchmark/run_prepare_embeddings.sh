#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HNSW_PYTHON_VENV:-${TMPDIR:-/tmp}/hnsw-clj-bible-benchmark-venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install -q -r "$EXAMPLE_DIR/requirements-embeddings.txt"
exec "$VENV/bin/python" "$EXAMPLE_DIR/prepare_embeddings.py"
