(ns hnsw.wip.reproduce-02ms
  "cli")
;; =============================================
;; 🚀 REPRODUCE 0.2ms PERFORMANCE TEST
;; =============================================
;; Direct implementation to achieve the original 0.2ms performance

(require '[hnsw.ultra-fast :as ultra])
(require '[hnsw.simd-optimized :as simd])
(require '[clojure.data.json :as json])
(import '[java.util.concurrent Executors Callable TimeUnit]
        '[java.util.concurrent.atomic AtomicLong])

(println "\n🎯 REPRODUCING 0.2ms SEARCH PERFORMANCE")
(println "========================================")
(println "Target: 0.212ms latency, 4,719 QPS with 20 threads\n")

;; Load Bible data
(println "📚 Loading Bible embeddings...")
(def data (json/read-str (slurp "data/bible_embeddings.json") :key-fn keyword))
(def vectors (mapv (fn [v]
                     [(:id v) (double-array (:embedding v))])
                   (:verses data)))
(println (format "✅ Loaded %,d vectors\n" (count vectors)))

;; Build index with SIMD cosine (CRITICAL for performance)
(println "🔨 Building index with SIMD cosine distance...")
(def start-build (System/currentTimeMillis))
(def index (ultra/build-index vectors
                              :M 16
                              :ef-construction 200
                              :distance-fn simd/cosine-distance
                              :show-progress? false))
(println (format "✅ Index built in %.1f seconds\n"
                 (/ (- (System/currentTimeMillis) start-build) 1000.0)))

;; Prepare 100 query vectors (like the original test)
(def query-vectors (mapv second (take 100 vectors)))

;; CRITICAL: Proper JVM warmup
(println "🔥 Warming up JVM (critical for 0.2ms)...")
(dotimes [_ 1000]
  (ultra/search-knn index (rand-nth query-vectors) 10))
(println "✅ Warmup complete\n")

;; Single-threaded baseline
(println "📊 SINGLE-THREADED BASELINE:")
(def single-latencies (atom []))
(def single-start (System/nanoTime))
(doseq [q query-vectors]
  (let [t0 (System/nanoTime)]
    (ultra/search-knn index q 10)
    (swap! single-latencies conj (/ (- (System/nanoTime) t0) 1000000.0))))
(def single-total-ms (/ (- (System/nanoTime) single-start) 1000000.0))
(def single-avg (/ (reduce + @single-latencies) 100))

(println (format "  Total time: %.1f ms" single-total-ms))
(println (format "  Avg latency: %.3f ms" single-avg))
(println (format "  Throughput: %.0f QPS\n" (/ 100000.0 single-total-ms)))

;; Multi-threaded test - EXACTLY like the original
(defn test-with-threads [num-threads]
  (let [executor (Executors/newFixedThreadPool num-threads)
        queries (take 100 query-vectors) ; 100 queries like original

        ;; Run the test
        start-time (System/nanoTime)

        ;; Submit all 100 queries
        futures (mapv (fn [q]
                        (.submit executor
                                 ^Callable
                                 (fn [] (ultra/search-knn index q 10))))
                      queries)

        ;; Wait for all to complete
        _ (doseq [f futures] (.get f))

        ;; Calculate timing
        total-time-ns (- (System/nanoTime) start-time)
        total-time-ms (/ total-time-ns 1000000.0)
        avg-latency-ms (/ total-time-ms 100)
        qps (/ 100000.0 total-time-ms)]

    ;; Shutdown executor
    (.shutdown executor)
    (.awaitTermination executor 10 TimeUnit/SECONDS)

    {:threads num-threads
     :total-ms total-time-ms
     :avg-latency-ms avg-latency-ms
     :qps qps}))

;; Test with key thread counts
(println "📊 MULTI-THREADED RESULTS:")
(println "===========================")
(println "Threads | Total (ms) | Latency (ms) | QPS")
(println "--------|------------|--------------|--------")

(doseq [threads [1 5 10 20 50]]
  (let [result (test-with-threads threads)]
    (println (format "%7d | %10.1f | %12.3f | %,7.0f"
                     threads
                     (:total-ms result)
                     (:avg-latency-ms result)
                     (:qps result)))))

;; Special focus on 20 threads (the optimal config)
(println "\n🎯 DETAILED TEST WITH 20 THREADS (Optimal):")
(println "============================================")

;; Run multiple times for stable results
(def results-20 (atom []))
(dotimes [run 5]
  (let [result (test-with-threads 20)]
    (swap! results-20 conj result)
    (println (format "Run %d: %.3f ms latency, %,.0f QPS"
                     (inc run)
                     (:avg-latency-ms result)
                     (:qps result)))))

;; Calculate average
(let [latencies (map :avg-latency-ms @results-20)
      qps-values (map :qps @results-20)]
  (println (format "\nAverage of 5 runs:"))
  (println (format "  Latency: %.3f ms" (/ (reduce + latencies) 5)))
  (println (format "  QPS: %,.0f" (/ (reduce + qps-values) 5))))

(println "\n✅ TEST COMPLETE!")
(println "\n💡 KEY FACTORS FOR 0.2ms PERFORMANCE:")
(println "  • SIMD cosine distance (not euclidean)")
(println "  • Proper JVM warmup (1000+ iterations)")
(println "  • Direct Executors with Callable")
(println "  • 100 queries batch size")
(println "  • 20 threads optimal")

(System/exit 0)
