#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export NUM_VECTORS="${NUM_VECTORS:-31126}"
export NUM_QUERIES="${NUM_QUERIES:-1000}"
export THREADS="${THREADS:-10}"
export RUNS="${RUNS:-5}"
export QUERY_REPEATS="${QUERY_REPEATS:-5}"
export EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
export EF_SEARCH="${EF_SEARCH:-80}"
bash scripts/compile_vector_kernel.sh
exec clojure -J-Xms1g -J-Xmx6g -J--add-modules=jdk.incubator.vector \
  -M:examples -m filtered-benchmark
