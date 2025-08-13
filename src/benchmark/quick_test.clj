(ns benchmark.quick-test
  (:require [hnsw.api :as hnsw]
            [benchmark.core :as bench])
  (:import [java.util Random]))

(defn quick-benchmark
  "Run a quick performance test to verify everything works"
  []
  (println "\n🚀 Quick Performance Test")
  (println "========================\n")

  ;; Test with common dimensions
  (let [test-configs [{:dim 768 :size 1000 :name "BERT-base"}
                      {:dim 1536 :size 1000 :name "OpenAI Ada"}]]

    (doseq [{:keys [dim size name]} test-configs]
      (println (format "Testing %s (%d dimensions, %d vectors):" name dim size))
      (println "----------------------------------------")

      (let [vectors (bench/generate-dataset size dim)
            queries (take 100 vectors)

            ;; Our implementation
            start (System/currentTimeMillis)
            idx (hnsw/index {:dimensions dim
                             :m 16
                             :ef-construction 200
                             :distance-fn :euclidean})]

        ;; Index building
        (doseq [[i v] (map-indexed vector vectors)]
          (hnsw/add! idx (str i) v))

        (let [index-time (- (System/currentTimeMillis) start)

              ;; Search test - reduced iterations for quick test
              start-search (System/currentTimeMillis)
              _ (doseq [q (take 10 queries)]
                  (hnsw/search idx q 10))
              search-time (- (System/currentTimeMillis) start-search)

              total-searches 10
              qps (if (> search-time 0)
                    (/ (* total-searches 1000.0) search-time)
                    0)]

          (println (format "  ✅ Indexing: %.1f vectors/sec"
                           (/ (* size 1000.0) index-time)))
          (println (format "  ✅ Search:   %.0f QPS" qps))
          (println (format "  ✅ Memory:   %.0f MB\n"
                           (bench/get-memory-usage))))))

    (println "\n✨ Quick test complete!")))

(defn -main [& args]
  (quick-benchmark))
