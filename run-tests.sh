#!/usr/bin/env bash

# Comprehensive test runner for HNSW-CLJ
echo "========================================="
echo "HNSW-CLJ Comprehensive Test Suite"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

run_test() {
    local test_name=$1
    local test_cmd=$2
    
    echo -n "Running $test_name... "
    
    if eval "$test_cmd" > /tmp/test_output.txt 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Error output:"
        tail -20 /tmp/test_output.txt | sed 's/^/    /'
        ((FAILED++))
    fi
}

echo "1. Testing core functionality"
echo "------------------------------"

# Run simple tests
run_test "Simple tests" "clojure -Sdeps '{:paths [\"src\" \"test\"]}' -M -e '
(require (quote [clojure.test :refer [run-tests]]))
(require (quote simple-test))
(let [results (run-tests (quote simple-test))]
  (if (and (zero? (:fail results)) (zero? (:error results)))
    (System/exit 0)
    (System/exit 1)))'"

# Run ALL core tests (including performance ones with reduced sizes)
# Since we already reduced the sizes, they should be fast enough
run_test "Core tests" "clojure -Sdeps '{:paths [\"src\" \"test\"]}' -M -e '
(require (quote [clojure.test :refer [run-tests]]))
(require (quote hnsw.core-test))
(let [results (run-tests (quote hnsw.core-test))]
  (if (and (zero? (:fail results)) (zero? (:error results)))
    (System/exit 0)
    (System/exit 1)))'"

echo ""
echo "2. Testing distance functions"
echo "------------------------------"

if clojure -M -e "
(require '[hnsw.simd-optimized :as simd])
(let [v1 (double-array [1 2 3])
      v2 (double-array [4 5 6])]
  (assert (> (simd/euclidean-distance v1 v2) 5.0))
  (assert (< (simd/cosine-distance v1 v2) 1.0))
  (println \"Distance functions: OK\"))" 2>/dev/null; then
    echo -e "${GREEN}✓ Distance functions test passed${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ Distance functions test failed${NC}"
    ((FAILED++))
fi

echo ""
echo "3. Testing implementations"
echo "------------------------------"

# Test Ultra-fast
if clojure -M -e "
(require '[hnsw.ultra-fast :as ultra])
(let [vectors (vec (map-indexed 
                    (fn [idx _]
                      [(str \"v\" idx) (double-array (repeatedly 64 rand))])
                    (range 100)))
      index (ultra/build-index vectors :show-progress? false)
      [_ query] (first vectors)
      results (ultra/search-knn index query 5)]
  (assert (= 5 (count results)))
  (println \"Ultra-fast implementation: OK\"))" 2>/dev/null; then
    echo -e "${GREEN}✓ Ultra-fast test passed${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ Ultra-fast test failed${NC}"
    ((FAILED++))
fi

# Test Graph
if clojure -M -e "
(require '[hnsw.graph :as graph])
(let [g (graph/create-graph)]
  (graph/insert g \"v1\" [1.0 2.0 3.0])
  (graph/insert g \"v2\" [4.0 5.0 6.0])
  (println \"Graph implementation: OK\"))" 2>/dev/null; then
    echo -e "${GREEN}✓ Graph test passed${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ Graph test failed${NC}"
    ((FAILED++))
fi

# Test Data Generator
if clojure -Sdeps '{:paths ["src" "test"]}' -M -e "
(require '[data-generator :as gen])
(let [v1 (gen/generate-dataset 10 64 :format :vector)
      v2 (gen/generate-dataset 10 64 :format :double-array)
      v3 (gen/generate-dataset 10 64 :format :indexed)]
  (assert (= 10 (count v1)))
  (assert (= 10 (count v2)))
  (assert (= 10 (count v3)))
  (println \"Data generator: OK\"))" 2>/dev/null; then
    echo -e "${GREEN}✓ Data generator test passed${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ Data generator test failed${NC}"
    ((FAILED++))
fi

echo ""
echo "========================================="
echo "Test Results Summary"
echo "========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "NOTE: Performance tests now run with reduced dataset sizes:"
    echo "  • 100-1000 vectors (was 1000-10000)"
    echo "  • 128 dimensions (was 768)"
    echo "  • Should complete in ~20-30 seconds"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
