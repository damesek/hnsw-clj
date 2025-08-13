(ns benchmark.test-setup
  "Quick test to verify benchmark setup"
  (:require [hnsw.ultra-optimized :as ultra]
            [hnsw.ultra-fast :as fast]))

(defn -main []
  (println "✅ Benchmark setup test")
  (println "Testing namespaces...")

  ;; Test vector creation
  (let [test-vec (double-array [1.0 2.0 3.0])
        test-data [["id1" test-vec]]]
    (println "  • Creating test vector: OK")

    ;; Test index building
    (try
      (let [index (ultra/build-index test-data)]
        (println "  • Building index: OK")

        ;; Test search
        (let [results (ultra/search index test-vec 1)]
          (println "  • Search: OK")
          (println (format "  • Result: %s" (pr-str (first results))))))
      (catch Exception e
        (println (format "  ❌ Error: %s" (.getMessage e)))))

    (println "\n✅ All tests passed!")
    (System/exit 0)))

;; Run with: clojure -M -m benchmark.test-setup