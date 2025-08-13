(ns benchmark.simple-test
  "Simple performance test for HNSW library"
  (:require [hnsw.api :as hnsw]
            [hnsw.simd-selector :as simd]))

(defn generate-random-vector [dim]
  (double-array (repeatedly dim rand)))

(defn run-simple-benchmark []
  (println "\n📊 SIMPLE HNSW BENCHMARK")
  (println "========================\n")

  (let [dimensions 768
        num-vectors 1000
        num-queries 100
        k 10]

    (println (format "Configuration:"))
    (println (format "  Dimensions: %d" dimensions))
    (println (format "  Vectors: %d" num-vectors))
    (println (format "  Queries: %d" num-queries))
    (println (format "  K: %d" k))
    (println)

    ;; Generate test data
    (println "1. Generating test data...")
    (let [vectors (vec (repeatedly num-vectors #(generate-random-vector dimensions)))
          queries (vec (repeatedly num-queries #(generate-random-vector dimensions)))]
      (println "   ✅ Data generated")

      ;; Build index
      (println "\n2. Building HNSW index...")
      (let [start (System/currentTimeMillis)
            idx (hnsw/index {:dimensions dimensions
                             :m 16
                             :ef-construction 200
                             :distance :cosine})]

        ;; Add vectors
        (doseq [[i v] (map-indexed vector vectors)]
          (hnsw/add! idx (str i) v))

        (let [build-time (- (System/currentTimeMillis) start)]
          (println (format "   ✅ Index built in %.2f seconds" (/ build-time 1000.0)))
          (println (format "   Indexing speed: %.0f vectors/sec"
                           (/ (* num-vectors 1000.0) build-time)))

          ;; Search benchmark
          (println "\n3. Search benchmark...")
          (let [search-start (System/currentTimeMillis)]
            (doseq [q queries]
              (hnsw/search idx q k))

            (let [search-time (- (System/currentTimeMillis) search-start)
                  qps (/ (* num-queries 1000.0) search-time)]
              (println (format "   ✅ %d queries in %.2f seconds"
                               num-queries (/ search-time 1000.0)))
              (println (format "   Search speed: %.0f QPS" qps))

              ;; Single query timing
              (println "\n4. Single query timing...")
              (let [single-times (for [i (range 10)]
                                   (let [start (System/nanoTime)]
                                     (hnsw/search idx (first queries) k)
                                     (- (System/nanoTime) start)))]
                (let [avg-time (/ (reduce + single-times) 10.0)]
                  (println (format "   Average time per query: %.3f ms"
                                   (/ avg-time 1000000.0)))))

              (println "\n✅ Benchmark completed successfully!")
              (println "\nSummary:")
              (println (format "  - Build time: %.2f sec" (/ build-time 1000.0)))
              (println (format "  - Queries/sec: %.0f" qps))
              (println (format "  - ms/query: %.2f" (/ search-time (double num-queries)))))))))))

(defn -main [& args]
  (run-simple-benchmark))
