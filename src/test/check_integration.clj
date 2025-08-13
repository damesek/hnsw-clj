(ns test.check-integration
  "Check if HNSW graph integration works"
  (:require [hnsw.ultra-fast :as ultra]
            [hnsw.simd-selector :as simd]))

(defn run-integration-test []
  (println "\n🔍 Checking HNSW Integration")
  (println "============================\n")

  (try
    ;; Test 1: Direct ultra-fast graph creation
    (println "1. Testing direct ultra-fast graph creation...")
    (def graph (ultra/create-ultra-graph
                :M 16
                :ef-construction 200
                :distance-fn ultra/cosine-distance-ultra))
    (println "   ✅ Graph created")

    ;; Test 2: Insert single element
    (println "\n2. Testing single element insertion...")
    (ultra/insert-single graph "test-1" (double-array [1.0 2.0 3.0]))
    (println "   ✅ Element inserted")

    ;; Test 3: Insert more elements
    (println "\n3. Adding more elements...")
    (ultra/insert-single graph "test-2" (double-array [4.0 5.0 6.0]))
    (ultra/insert-single graph "test-3" (double-array [1.1 2.1 3.1]))
    (println "   ✅ Multiple elements added")

    ;; Test 4: Search
    (println "\n4. Testing search...")
    (def results (ultra/search-knn graph (double-array [1.0 2.0 3.0]) 2))
    (println "   Results:")
    (doseq [{:keys [id distance]} results]
      (println (format "     - %s: %.3f" id distance)))
    (println "   ✅ Search works")

    ;; Test 5: Test with SIMD distance function
    (println "\n5. Testing with SIMD distance function...")
    (def graph2 (ultra/create-ultra-graph
                 :M 16
                 :ef-construction 200
                 :distance-fn simd/euclidean-distance))
    (ultra/insert-single graph2 "simd-1" (double-array [1.0 0.0 0.0]))
    (ultra/insert-single graph2 "simd-2" (double-array [0.0 1.0 0.0]))
    (def results2 (ultra/search-knn graph2 (double-array [0.9 0.1 0.0]) 2))
    (println "   Results with SIMD:")
    (doseq [{:keys [id distance]} results2]
      (println (format "     - %s: %.3f" id distance)))
    (println "   ✅ SIMD distance functions work")

    (println "\n✅ ALL INTEGRATION TESTS PASSED!")
    (println "   The HNSW graph is properly integrated!")
    true

    (catch Exception e
      (println "\n❌ Integration test failed:")
      (println (.getMessage e))
      (.printStackTrace e)
      false)))

(defn -main [& args]
  (let [result (run-integration-test)]
    (System/exit (if result 0 1))))
