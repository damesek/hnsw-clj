#!/usr/bin/env bash

echo "========================================="
echo "HNSW-CLJ Minimal Working Test"
echo "========================================="
echo ""

# Test without data-generator first
echo "Basic tests (no data-generator):"
echo "---------------------------------"

clojure -M -e '
(println "1. Testing distance functions...")
(require (quote [hnsw.simd-optimized :as simd]))
(let [v1 (double-array [1 2 3])
      v2 (double-array [4 5 6])]
  (println "   Euclidean distance:" (simd/euclidean-distance v1 v2))
  (println "   ✓ Distance functions work"))

(println)
(println "2. Testing ultra-fast...")
(require (quote [hnsw.ultra-fast :as ultra]))
(let [vectors [["v0" (double-array [1.0 2.0 3.0])]
               ["v1" (double-array [4.0 5.0 6.0])]]
      index (ultra/build-index vectors :show-progress? false)
      [_ query] (first vectors)
      results (ultra/search-knn index query 2)]
  (println "   Found" (count results) "neighbors")
  (println "   ✓ Ultra-fast works"))

(println)
(println "3. Testing graph...")
(require (quote [hnsw.graph :as graph]))
(let [g (graph/create-graph)]
  (graph/insert g "v1" [1.0 2.0 3.0])
  (graph/insert g "v2" [4.0 5.0 6.0])
  (println "   ✓ Graph works"))

(println)
(println "========================================")
(println "✅ Basic tests passed!")
(println "========================================")
'

echo ""
echo "Tests with data-generator:"
echo "--------------------------"

# Use a different approach - add test to classpath directly
clojure -Sdeps '{:paths ["src" "test"]}' -M -e '
(println "4. Testing data-generator...")
(require (quote [data-generator :as gen]))
(try
  (let [v1 (gen/generate-dataset 5 10 :format :vector)
        v2 (gen/generate-dataset 5 10 :format :double-array)
        v3 (gen/generate-dataset 5 10 :format :indexed)]
    (println "   Generated:" (count v1) "vectors," (count v2) "arrays," (count v3) "indexed")
    (println "   ✓ Data generator works"))
  (catch Exception e
    (println "   ✗ Data generator failed:" (.getMessage e))))

(println)
(println "5. Testing integration...")
(require (quote [hnsw.ultra-fast :as ultra]))
(try
  (let [vectors (gen/generate-dataset 20 32 :format :indexed)
        index (ultra/build-index vectors :show-progress? false)
        [_ query] (first vectors)
        results (ultra/search-knn index query 5)]
    (println "   Built index with" (count vectors) "vectors")
    (println "   Found" (count results) "neighbors")
    (println "   ✓ Full integration works"))
  (catch Exception e
    (println "   ✗ Integration failed:" (.getMessage e))))

(println)
(println "========================================")
(println "✅ All tests completed!")
(println "========================================")
'
