#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export NUM_VECTORS="${NUM_VECTORS:-10000}"
export M="${M:-16}"
export EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
export EF_SEARCH="${EF_SEARCH:-80}"
bash scripts/compile_vector_kernel.sh
exec clojure -J-Xms1g -J-Xmx8g -J--add-modules=jdk.incubator.vector \
  -M:examples -m maintenance-benchmark
