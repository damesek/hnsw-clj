#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export M="${M:-16}"
export EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
export EF_SEARCH="${EF_SEARCH:-80}"
export QUERY_SPLIT=validation
exec bash "$EXAMPLE_DIR/run_construction_python.sh"
