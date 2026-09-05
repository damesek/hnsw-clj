#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$ROOT_DIR/examples/bible_benchmark"
VENV="${HNSW_PYTHON_VENV:-${TMPDIR:-/tmp}/hnsw-clj-bible-benchmark-venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import hnswlib, numpy, sentence_transformers' >/dev/null 2>&1; then
  "$VENV/bin/pip" install -q -r "$EXAMPLE_DIR/requirements-embeddings.txt"
fi

cd "$ROOT_DIR"
"$ROOT_DIR/scripts/compile_vector_kernel.sh"

QUERY_ENCODER_PYTHON="$VENV/bin/python" \
exec clj -J-Xms1g -J-Xmx8g -J--add-modules=jdk.incubator.vector \
  -M:examples -m text-search "$@"
