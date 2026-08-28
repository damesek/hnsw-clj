#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/scripts/compile_vector_kernel.sh"

exec clj -J-Xms1g -J-Xmx8g -J--add-modules=jdk.incubator.vector \
  -Sdeps '{:aliases {:bible-benchmark {:extra-paths ["examples/bible_benchmark"]}}}' \
  -M:bible-benchmark -m bible-benchmark
