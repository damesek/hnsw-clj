#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HNSW_PYTHON_VENV:-${TMPDIR:-/tmp}/hnsw-clj-bible-benchmark-venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import hnswlib, numpy, sentence_transformers' >/dev/null 2>&1; then
  "$VENV/bin/pip" install -q -r "$EXAMPLE_DIR/requirements-embeddings.txt"
fi

exec "$VENV/bin/python" "$EXAMPLE_DIR/text_search.py" "$@"
