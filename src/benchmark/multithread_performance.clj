(ns benchmark.multithread-performance
  "Multithread-focused performance benchmark for HNSW implementations
   Compares our ultra-optimized implementation with competitors using parallel processing"
  (:require [hnsw.ultra-optimized :as ultra]
            [hnsw.ultra-fast :as fast]
            [hnsw.index-io :as io]
            [clojure.data.json :as json]
            [clojure.java.io :as jio]
            [clojure.string :as str])
  (:import [java.util.concurrent Executors TimeUnit Callable]
           [java.util.concurrent.atomic AtomicLong]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; =================== Configuration ===================

(def THREAD_COUNTS [1 5 10 20 50])
(def DEFAULT_DIMENSIONS 768)
(def DEFAULT_VECTORS 31173)
(def DEFAULT_K 10)
(def DEFAULT_QUERIES 1000)

;; =================== Data Generation ===================

(defn generate-random-vectors
  "Generate random float vectors for testing"
  [num-vectors dimensions]
  (println (format "📊 Generating %,d random vectors (%d dimensions)..." num-vectors dimensions))
  (vec (repeatedly num-vectors
                   #(double-array (repeatedly dimensions rand)))))

(defn load-bible-data
  "Load the 31K Bible embeddings if available"
  []
  (let [embeddings-file "data/bible_embeddings_complete.json"]
    (if (.exists (jio/file embeddings-file))
      (do
        (println "📖 Loading 31K Bible embeddings...")
        (with-open [reader (jio/reader embeddings-file)]
          (let [data (json/read reader :key-fn keyword)
                verses (:verses data)]
            (println (format "✅ Loaded %,d verses" (count verses)))
            (mapv (fn [v]
                    [(str (:id v)) (double-array (:embedding v))])
                  verses))))
      (do
        (println "⚠️ Bible embeddings not found, using random data")
        (mapv vector
              (range DEFAULT_VECTORS)
              (generate-random-vectors DEFAULT_VECTORS DEFAULT_DIMENSIONS))))))

;; =================== Our Implementation (Ultra-Optimized) ===================

(defn benchmark-our-ultra-parallel
  "Benchmark our ultra-optimized HNSW with parallel search"
  [vectors queries k num-threads]
  (let [;; Build index (single-threaded for fairness)
        start-build (System/nanoTime)
        index (ultra/build-index vectors)
        build-time-ms (/ (- (System/nanoTime) start-build) 1000000.0)

        ;; Prepare thread pool
        executor (Executors/newFixedThreadPool num-threads)
        completed (AtomicLong. 0)

        ;; Run parallel searches
        start-search (System/nanoTime)
        futures (mapv (fn [query-vec]
                        (.submit executor
                                 ^Callable
                                 (fn []
                                   (let [result (ultra/search index query-vec k)]
                                     (.incrementAndGet completed)
                                     result))))
                      queries)

        ;; Wait for completion
        _ (doseq [f futures]
            (.get f))
        _ (.shutdown executor)
        _ (.awaitTermination executor 1 TimeUnit/MINUTES)

        search-time-ms (/ (- (System/nanoTime) start-search) 1000000.0)
        avg-latency-ms (/ search-time-ms (count queries))
        qps (/ (* 1000.0 (count queries)) search-time-ms)]

    {:implementation "HNSW-CLJ Ultra (Multithread)"
     :threads num-threads
     :build-time-ms build-time-ms
     :total-search-time-ms search-time-ms
     :avg-latency-ms avg-latency-ms
     :qps qps}))

;; =================== Benchmark Runner ===================

(defn run-multithread-benchmark
  "Run comprehensive multithread benchmark"
  [{:keys [num-vectors dimensions num-queries k use-bible-data?]
    :or {num-vectors DEFAULT_VECTORS
         dimensions DEFAULT_DIMENSIONS
         num-queries DEFAULT_QUERIES
         k DEFAULT_K
         use-bible-data? true}}]

  (println "\n" (str/join (repeat 80 "=")))
  (println (format "🚀 MULTITHREAD PERFORMANCE BENCHMARK"))
  (println (str/join (repeat 80 "=")))

  ;; Load or generate data
  (let [vector-data (if use-bible-data?
                      (load-bible-data)
                      (mapv vector
                            (range num-vectors)
                            (generate-random-vectors num-vectors dimensions)))
        vectors-only (mapv second vector-data)
        queries (take num-queries (shuffle vectors-only))]

    (println (format "\n📊 Configuration:"))
    (println (format "  • Vectors: %,d" (count vector-data)))
    (println (format "  • Dimensions: %d" dimensions))
    (println (format "  • Queries: %,d" num-queries))
    (println (format "  • k: %d" k))
    (println (format "  • Thread counts: %s" (str/join ", " THREAD_COUNTS)))

    ;; Warm-up
    (println "\n🔥 Warming up JVM...")
    (let [small-vectors (take 100 vector-data)
          small-index (ultra/build-index small-vectors)]
      (dotimes [_ 50]
        (ultra/search small-index (second (first small-vectors)) k)))

    ;; Run benchmarks for different thread counts
    (println "\n📈 Running benchmarks...")
    (println (str/join (repeat 80 "-")))
    (printf "%-10s %10s %12s %12s %12s\n"
            "Threads" "Build(ms)" "Search(ms)" "Latency(ms)" "QPS")
    (println (str/join (repeat 80 "-")))

    (let [results (doall
                   (for [num-threads THREAD_COUNTS]
                     (let [result (benchmark-our-ultra-parallel
                                   vector-data queries k num-threads)]
                       (printf "%-10d %10.1f %12.1f %12.3f %12.0f\n"
                               num-threads
                               (:build-time-ms result)
                               (:total-search-time-ms result)
                               (:avg-latency-ms result)
                               (:qps result))
                       (flush)
                       result)))]

      (println (str/join (repeat 80 "-")))

      ;; Summary
      (let [single-thread (first results)
            best-result (apply max-key :qps results)]
        (println "\n📊 SUMMARY:")
        (println (format "  • Single thread QPS: %.0f" (:qps single-thread)))
        (println (format "  • Best QPS: %.0f (%d threads)"
                         (:qps best-result) (:threads best-result)))
        (println (format "  • Speedup: %.1fx"
                         (/ (:qps best-result) (:qps single-thread))))
        (println (format "  • Best latency: %.3f ms (%d threads)"
                         (:avg-latency-ms best-result) (:threads best-result))))

      results)))

;; =================== Main Entry Point ===================

(defn -main
  "Main entry point for multithread benchmark"
  [& args]
  (let [use-bible? (not (some #(= % "--random") args))
        num-queries (if-let [q (some #(when (re-matches #"\d+" %) %) args)]
                      (Integer/parseInt q)
                      1000)]

    (println "🎯 HNSW-CLJ Multithread Performance Benchmark")
    (println "Ultra-optimized implementation with SIMD acceleration")

    (run-multithread-benchmark
     {:use-bible-data? use-bible?
      :num-queries num-queries})

    (System/exit 0)))

;; Run with: clojure -M -m benchmark.multithread-performance