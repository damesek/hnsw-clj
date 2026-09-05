#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
bash scripts/compile_vector_kernel.sh
exec clojure -J--add-modules=jdk.incubator.vector -M:test
