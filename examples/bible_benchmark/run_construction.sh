#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
"$ROOT_DIR/scripts/compile_vector_kernel.sh"

exec clojure -J-Xms1g -J-Xmx8g -J--add-modules=jdk.incubator.vector \
  -M:examples -m construction-benchmark
