(ns hnsw.benchmark
  "Simple benchmarking utilities for HNSW performance testing"
  (:require [hnsw.filtered-index-io :as io-utils]
            [hnsw.ultra-optimized :as ultra]
            [hnsw.ultra-fast :as base]
            [hnsw.simd-optimized :as simd-opt]
            [hnsw.filtered :as filtered]
            [clojure.java.io :as io]))

;; ===== Quick Load & Test =====

(defn load-index
  "Load an existing index file with specified or default distance function"
  ([filepath]
   (load-index filepath simd-opt/cosine-distance))
  ([filepath distance-fn]
   (println (format "\n📂 Loading index: %s" filepath))
   (let [start (System/currentTimeMillis)
         index (io-utils/load-filtered-index filepath distance-fn)
         elapsed (- (System/currentTimeMillis) start)]
     (println (format "✅ Loaded in %d ms" elapsed))
     index)))

(defn quick-search
  "Perform a quick search test with random queries"
  [index num-queries k]
  (let [;; Generate random query vectors
        queries (repeatedly num-queries
                            #(double-array (repeatedly 384 rand)))

        ;; Warm-up
        _ (dotimes [_ 5]
            (filtered/search-filtered
             index (first queries) k nil))

        ;; Benchmark
        start (System/currentTimeMillis)
        results (doall
                 (map #(filtered/search-filtered index % k nil)
                      queries))
        elapsed (- (System/currentTimeMillis) start)
        avg-ms (/ elapsed (double num-queries))
        qps (/ (* num-queries 1000.0) elapsed)]

    {:num-queries num-queries
     :total-ms elapsed
     :avg-ms avg-ms
     :qps qps
     :results (take 3 results)}))

(defn compare-distances
  "Compare distance function performance"
  [num-tests dimension]
  (let [v1 (double-array (repeatedly dimension rand))
        v2 (double-array (repeatedly dimension rand))]

    (println (format "\n⚡ Distance Performance (%d-dim, %d tests):"
                     dimension num-tests))
    (println "   " (apply str (repeat 40 "-")))

    (doseq [[name fn] [["Base" base/cosine-distance-ultra]
                       ["4x unroll" ultra/fast-cosine-distance]
                       ["8x SIMD" simd-opt/cosine-distance]]]
      (let [start (System/nanoTime)]
        (dotimes [_ num-tests]
          (fn v1 v2))
        (let [us-per-op (/ (- (System/nanoTime) start)
                           (* num-tests 1000.0))]
          (println (format "   %-10s: %.2f μs/op" name us-per-op)))))))

;; ===== Full Benchmark Suite =====

(defn full-benchmark
  "Run complete benchmark on an index"
  [index-path]
  (println "\n" (str (apply str (repeat 50 "="))))
  (println "         HNSW Performance Benchmark")
  (println (str (apply str (repeat 50 "="))))

  ;; 1. Compare distance functions
  (compare-distances 10000 384)

  ;; 2. Load index with different implementations
  (println "\n📊 Index Search Performance:")

  (let [;; Load with base implementation
        index-base (load-index index-path base/cosine-distance-ultra)
        result-base (quick-search index-base 100 10)

        ;; Load with SIMD implementation 
        index-simd (load-index index-path simd-opt/cosine-distance)
        result-simd (quick-search index-simd 100 10)]

    (println "\n📈 Results:")
    (println "   " (apply str (repeat 45 "-")))
    (println "   Implementation | Avg (ms) | QPS    | Speedup")
    (println "   " (apply str (repeat 45 "-")))
    (println (format "   Base           | %8.2f | %6.1f | 1.00x"
                     (:avg-ms result-base)
                     (:qps result-base)))
    (println (format "   SIMD-optimized | %8.2f | %6.1f | %.2fx"
                     (:avg-ms result-simd)
                     (:qps result-simd)
                     (/ (:avg-ms result-base) (:avg-ms result-simd))))

    {:base result-base
     :simd result-simd
     :speedup (/ (:avg-ms result-base) (:avg-ms result-simd))}))

;; ===== Interactive Testing =====

(defn test-with-query
  "Test with a specific text query (requires embeddings)"
  [index query-text embeddings k]
  (if-let [query-vec (get embeddings query-text)]
    (let [start (System/nanoTime)
          results (filtered/search-filtered
                   index
                   (double-array query-vec)
                   k
                   nil)
          elapsed-ms (/ (- (System/nanoTime) start) 1000000.0)]
      (println (format "\n🔍 Query: '%s'" query-text))
      (println (format "   Found %d results in %.2f ms"
                       (count results) elapsed-ms))
      (doseq [[idx [id dist]] (map-indexed vector (take 3 results))]
        (println (format "   %d. %s (distance: %.4f)"
                         (inc idx) id dist)))
      results)
    (println "Query text not found in embeddings")))

;; ===== Convenience Functions =====

(def index-files
  "Common index file paths"
  {:complete "data/bible_index_complete.hnsw"
   :30k "data/bible_index_30k.hnsw"
   :filtered "data/bible_filtered_index.hnsw"})

(defn available-indexes
  "List available index files"
  []
  (println "\n📁 Available indexes:")
  (doseq [[name path] index-files]
    (let [f (io/file path)]
      (if (.exists f)
        (println (format "   ✅ %-10s: %.1f MB - %s"
                         name
                         (/ (.length f) 1048576.0)
                         path))
        (println (format "   ❌ %-10s: NOT FOUND - %s" name path))))))

;; ===== Usage Examples =====

(comment
  ;; List available indexes
  (available-indexes)

  ;; Load an index
  (def my-index (load-index (:30k index-files)))

  ;; Quick performance test
  (quick-search my-index 100 10)

  ;; Compare distance functions
  (compare-distances 10000 384)

  ;; Full benchmark
  (full-benchmark (:30k index-files))

  ;; Test with specific query (if you have embeddings)
  (def embeddings (io-utils/load-embeddings "data/complete_bible_embeddings.edn"))
  (test-with-query my-index "Genesis 1:1" embeddings 5))

(println "\n✅ Benchmark module loaded!")
(println "   Run (available-indexes) to see available index files")
(println "   Run (full-benchmark (:30k index-files)) for complete test")
