#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/src-java/hnsw/search"
OUTPUT_DIR="$ROOT_DIR/target/classes"
VECTOR_CLASS="$OUTPUT_DIR/hnsw/search/VectorDistance.class"
COMPACT_CLASS="$OUTPUT_DIR/hnsw/search/CompactSearch.class"

if [[ ! -f "$VECTOR_CLASS" || ! -f "$COMPACT_CLASS" \
   || "$SOURCE_DIR/VectorDistance.java" -nt "$VECTOR_CLASS" \
   || "$SOURCE_DIR/CompactSearch.java" -nt "$COMPACT_CLASS" ]]; then
  mkdir -p "$OUTPUT_DIR"
  javac --add-modules jdk.incubator.vector -d "$OUTPUT_DIR" "$SOURCE_DIR"/*.java
fi
