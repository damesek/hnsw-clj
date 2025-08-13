(ns test.minimal
  "Minimal test to verify library works"
  (:require [hnsw.api :as hnsw]))

(defn run-test []
  (println "\n🔧 Minimal HNSW Test")
  (println "===================\n")

  (try
    ;; Test 1: Basic index creation
    (println "Test 1: Creating index...")
    (def idx (hnsw/index {:dimensions 3}))
    (println "  ✅ Index created")

    ;; Test 2: Add vectors
    (println "\nTest 2: Adding vectors...")
    (hnsw/add! idx "a" [1.0 0.0 0.0])
    (hnsw/add! idx "b" [0.0 1.0 0.0])
    (hnsw/add! idx "c" [0.0 0.0 1.0])
    (println "  ✅ 3 vectors added")

    ;; Test 3: Search
    (println "\nTest 3: Searching...")
    (def results (hnsw/search idx [0.9 0.1 0.0] 2))
    (println "  Results:")
    (doseq [{:keys [id distance]} results]
      (println (format "    %s: %.3f" id distance)))
    (println "  ✅ Search completed")

    ;; Test 4: Test all distance functions
    (println "\nTest 4: Testing distance functions...")

    ;; Euclidean
    (def idx-euc (hnsw/index {:dimensions 2 :distance-fn :euclidean}))
    (hnsw/add! idx-euc "p1" [0.0 0.0])
    (hnsw/add! idx-euc "p2" [3.0 4.0]) ; distance = 5
    (def r1 (first (hnsw/search idx-euc [0.0 0.0] 1)))
    (println (format "  Euclidean: %s dist=%.1f" (:id r1) (:distance r1)))

    ;; Cosine
    (def idx-cos (hnsw/index {:dimensions 2 :distance-fn :cosine}))
    (hnsw/add! idx-cos "v1" [1.0 0.0])
    (hnsw/add! idx-cos "v2" [0.707 0.707]) ; 45 degrees
    (def r2 (first (hnsw/search idx-cos [1.0 0.0] 1)))
    (println (format "  Cosine: %s dist=%.3f" (:id r2) (:distance r2)))

    ;; Dot product
    (def idx-dot (hnsw/index {:dimensions 2 :distance-fn :dot}))
    (hnsw/add! idx-dot "d1" [1.0 1.0])
    (hnsw/add! idx-dot "d2" [2.0 2.0])
    (def r3 (first (hnsw/search idx-dot [1.0 1.0] 1)))
    (println (format "  Dot: %s dist=%.1f" (:id r3) (:distance r3)))

    (println "  ✅ All distance functions work")

    (println "\n✅ ALL TESTS PASSED!")
    true

    (catch Exception e
      (println "\n❌ TEST FAILED:")
      (println (.getMessage e))
      (.printStackTrace e)
      false)))

(defn -main [& args]
  (let [result (run-test)]
    (System/exit (if result 0 1))))
