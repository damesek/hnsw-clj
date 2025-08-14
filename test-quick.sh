#!/usr/bin/env bash

# Quick smoke test for HNSW-CLJ
echo "======================================"
echo "HNSW-CLJ Quick Smoke Test"
echo "======================================"
echo ""

# Run all tests in one Clojure session for speed
# Add test directory to classpath using -Sdeps
clojure -Sdeps '{:paths ["src" "test"]}' -M -e "
(println \"Loading namespaces...\")
(require '[hnsw.simd-optimized :as simd])
(require '[hnsw.ultra-fast :as ultra])
(require '[hnsw.graph :as graph])
(require '[data-generator :as gen])

(def passed (atom 0))
(def failed (atom 0))

(defn run-test [test-name test-fn]
  (print (str \"  \" test-name \"... \"))
  (flush)
  (try
    (test-fn)
    (println \"✓ PASSED\")
    (swap! passed inc)
    (catch Exception e
      (println \"✗ FAILED\")
      (println \"    Error:\" (.getMessage e))
      (swap! failed inc))))

(println \"\n1. Distance Functions\")
(println \"--------------------\")

(run-test \"Euclidean distance\"
  (fn []
    (let [v1 (double-array [1 2 3])
          v2 (double-array [4 5 6])
          dist (simd/euclidean-distance v1 v2)]
      (assert (> dist 5.0)))))

(run-test \"Cosine distance\"
  (fn []
    (let [v1 (double-array [1 2 3])
          v2 (double-array [1 2 3])
          dist (simd/cosine-distance v1 v2)]
      (assert (< dist 0.01)))))

(println \"\n2. Ultra-Fast Implementation\")
(println \"----------------------------\")

(run-test \"Build empty index\"
  (fn []
    (let [index (ultra/build-index [] :show-progress? false)]
      (assert (not (nil? index))))))

(run-test \"Build and search\"
  (fn []
    (let [vectors (gen/generate-dataset 50 32 :format :indexed)
          index (ultra/build-index vectors :show-progress? false)
          [_ query] (first vectors)
          results (ultra/search-knn index query 5)]
      (assert (= 5 (count results))))))

(println \"\n3. Graph Implementation\")
(println \"----------------------\")

(run-test \"Create and insert\"
  (fn []
    (let [g (graph/create-graph)]
      (graph/insert g \"v1\" [1.0 2.0 3.0])
      (graph/insert g \"v2\" [4.0 5.0 6.0])
      (assert (not (nil? g))))))

(println \"\n4. Data Generator\")
(println \"----------------\")

(run-test \"Generate vectors\"
  (fn []
    (let [v1 (gen/generate-dataset 10 64 :format :vector)
          v2 (gen/generate-dataset 10 64 :format :double-array)
          v3 (gen/generate-dataset 10 64 :format :indexed)]
      (assert (= 10 (count v1)))
      (assert (= 10 (count v2)))
      (assert (= 10 (count v3))))))

(println \"\n=====================================\")
(println \"Results Summary\")
(println \"=====================================\")
(println (str \"Passed: \" @passed))
(println (str \"Failed: \" @failed))
(println)

(if (zero? @failed)
  (do
    (println \"✅ All tests passed!\")
    (System/exit 0))
  (do
    (println (str \"❌ \" @failed \" test(s) failed\"))
    (System/exit 1)))
"
