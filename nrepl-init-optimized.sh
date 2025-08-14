#!/bin/bash

echo "============================================="
echo "🚀 nREPL - OPTIMIZED FOR 31K VECTORS"
echo "============================================="
echo ""

# CPU és rendszer info
echo "📊 Rendszer információk:"
echo "   CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep 'Model name' | cut -d: -f2)"
echo "   Magok: $(getconf _NPROCESSORS_ONLN)"
echo "   Java: $(java -version 2>&1 | head -n 1)"
echo ""

# SIMD ellenőrzés
echo "🔍 SIMD támogatás ellenőrzése..."

# Ellenőrizzük hogy a Java Vector API elérhető-e
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -ge 17 ]; then
    echo "   ✅ Java $JAVA_VERSION - Vector API támogatott"
    VECTOR_OPTS="-J--add-modules=jdk.incubator.vector -J-XX:+UnlockExperimentalVMOptions -J-XX:+EnableVectorSupport -J-XX:+EnableVectorAggressiveReboxing"
else
    echo "   ⚠️  Java $JAVA_VERSION - Vector API nem elérhető"
    VECTOR_OPTS=""
fi

# JBLAS ellenőrzés
JBLAS_OK=$(clojure -M -e "(try (import 'org.jblas.DoubleMatrix) (print \"OK\") (catch Exception _ (print \"FAIL\")))" 2>/dev/null)
if [ "$JBLAS_OK" = "OK" ]; then
    echo "   ✅ JBLAS natív SIMD elérhető"
    JBLAS_OPTS="-J-Djava.library.path=/usr/local/lib:/opt/homebrew/lib"
else
    echo "   ⚠️  JBLAS nem elérhető"
    JBLAS_OPTS=""
fi

echo ""
echo "🚀 Starting nREPL with optimized settings..."
echo "   Heap: 12GB (for 31k vectors)"
echo "   GC: G1GC with 20ms pause target"
echo "   SIMD: Enabled"
echo ""

# Futtatás NAGY MEMÓRIÁVAL és minden optimalizációval
clojure \
  -J-Xmx12g \
  -J-Xms10g \
  -J-XX:+UseG1GC \
  -J-XX:MaxGCPauseMillis=20 \
  -J-XX:+UseStringDeduplication \
  -J-XX:+UseNUMA \
  -J-XX:+AlwaysPreTouch \
  -J-XX:+UseCompressedOops \
  -J-XX:G1HeapRegionSize=32m \
  $VECTOR_OPTS \
  $JBLAS_OPTS \
  -M:nrepl

echo ""
echo "============================================="
echo "✅ nREPL closed!"
echo "============================================="
