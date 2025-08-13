#!/bin/bash

# ============================================================
# 🚀 HNSW-CLJ MULTITHREAD PERFORMANCE BENCHMARK
# ============================================================

echo "🚀 HNSW-CLJ Multithread Performance Benchmark"
echo "============================================================"
echo ""

# Navigate to library root (hnsw-lib)
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/../.." || exit
echo "Working directory: $(pwd)"
echo ""

# Check if Bible data exists
if [ ! -f "data/bible_embeddings_complete.json" ]; then
    echo "⚠️ Bible embeddings not found!"
    echo "Generating embeddings..."
    python scripts/export_complete_bible.py --all
fi

# Run benchmark with proper JVM options
echo "Running multithread benchmark..."
echo ""

clojure \
  -J-Xmx4g \
  -J-XX:+UseG1GC \
  -J-XX:MaxGCPauseMillis=10 \
  -J--add-modules=jdk.incubator.vector \
  -J-XX:+UnlockExperimentalVMOptions \
  -J-XX:+EnableVectorSupport \
  -M -m benchmark.multithread-performance "$@"

echo ""
echo "============================================================"
echo "✅ Benchmark complete!"