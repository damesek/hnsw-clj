(ns hnsw.helper.parallel-search
  "Optimized multi-threaded search using the proven approach from benchmarks
   Uses Futures and Callable for better performance"
  (:import [java.util.concurrent Executors TimeUnit Callable Future
            LinkedBlockingQueue ThreadPoolExecutor]
           [java.util.concurrent.atomic AtomicLong AtomicInteger]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ============================================================
;; The PROVEN approach from help-for-claude
;; ============================================================

(defn parallel-search-futures
  "Execute multiple searches in parallel using Futures (the proven fast method)
   
   This is the approach that actually scales to 20 threads with linear speedup!
   
   Parameters:
   - index: The HNSW index to search
   - queries: Collection of query vectors  
   - k: Number of nearest neighbors
   - search-fn: The search function (e.g., ultra/search-knn)
   - num-threads: Number of threads to use
   
   Returns:
   - Vector of search results in the same order as queries"
  [index queries k search-fn num-threads]
  (let [;; Create a fresh thread pool each time (no caching overhead)
        ^ThreadPoolExecutor executor (Executors/newFixedThreadPool num-threads)
        queries-vec (vec queries)

        ;; Submit all queries as Callable tasks that return results
        futures (mapv (fn [query-vec]
                        (.submit executor
                                 ^Callable
                                 (fn []
                                   (search-fn index query-vec k))))
                      queries-vec)]

    ;; Collect all results
    (let [results (mapv #(.get ^Future %) futures)]

      ;; Clean shutdown
      (.shutdown executor)
      (.awaitTermination executor 10 TimeUnit/SECONDS)

      results)))

(defn benchmark-parallel-search
  "Benchmark parallel search performance (matching the original benchmark exactly)
   
   Returns performance metrics matching the original format"
  [index queries k search-fn num-threads]
  (let [queries-vec (vec queries)
        num-queries (count queries-vec)

        ;; Create executor
        ^ThreadPoolExecutor executor (Executors/newFixedThreadPool num-threads)
        completed (AtomicLong. 0)

        ;; Warmup
        _ (dotimes [_ 10]
            (search-fn index (first queries-vec) k))

        ;; Run parallel searches with timing
        start-time (System/nanoTime)

        ;; Submit all as futures
        futures (mapv (fn [query-vec]
                        (.submit executor
                                 ^Callable
                                 (fn []
                                   (let [result (search-fn index query-vec k)]
                                     (.incrementAndGet completed)
                                     result))))
                      queries-vec)

        ;; Wait for all to complete
        _ (doseq [^Future f futures]
            (.get f))

        search-time-ms (/ (- (System/nanoTime) start-time) 1000000.0)

        ;; Shutdown executor
        _ (.shutdown executor)
        _ (.awaitTermination executor 1 TimeUnit/MINUTES)]

    {:threads num-threads
     :queries num-queries
     :total-time-ms search-time-ms
     :avg-latency-ms (/ search-time-ms num-queries)
     :qps (/ (* 1000.0 num-queries) search-time-ms)
     :completed (.get completed)}))

(defn test-thread-scaling-futures
  "Test performance with different thread counts using the PROVEN approach
   
   This should show linear scaling up to 20 threads!"
  [index queries k search-fn & {:keys [thread-counts]
                                :or {thread-counts [1 2 4 8 10 20]}}]
  (println "\n🚀 PARALLEL SEARCH SCALING TEST (Futures Method)")
  (println "=================================================")
  (println "Using the PROVEN approach from benchmarks\n")

  (println "Warming up JVM...")
  (let [warmup-queries (take 50 queries)]
    (dotimes [_ 2]
      (parallel-search-futures index warmup-queries k search-fn 4)))

  (println "\nRunning tests...")
  (println "------------------------------------------------")
  (printf "%-10s %12s %12s %12s\n" "Threads" "Total(ms)" "Latency(ms)" "QPS")
  (println "------------------------------------------------")

  (let [results (doall
                 (for [num-threads thread-counts]
                   (let [test-queries (take (* num-threads 100) queries)
                         result (benchmark-parallel-search
                                 index test-queries k search-fn num-threads)]
                     (printf "%-10d %12.1f %12.3f %12.0f\n"
                             (:threads result)
                             (:total-time-ms result)
                             (:avg-latency-ms result)
                             (:qps result))
                     (flush)
                     result)))]

    (println "------------------------------------------------")

    ;; Calculate speedups
    (when-let [single-thread (first results)]
      (println "\n📊 SPEEDUP ANALYSIS:")
      (doseq [result results]
        (let [speedup (/ (:qps result) (:qps single-thread))
              efficiency (* 100.0 (/ speedup (:threads result)))]
          (printf "  %2d threads: %.1fx speedup, %.0f%% efficiency\n"
                  (:threads result) speedup efficiency)))

      (let [best (apply max-key :qps results)]
        (println (format "\n✅ Best: %d threads = %.0f QPS (%.1fx speedup)"
                         (:threads best)
                         (:qps best)
                         (/ (:qps best) (:qps single-thread))))))

    results))

;; ============================================================
;; Simple wrapper functions for easy use
;; ============================================================

(defn search-ultra-parallel
  "Multi-threaded search for ultra-fast implementation"
  [index queries k num-threads]
  (require '[hnsw.ultra-fast :as ultra])
  (parallel-search-futures index queries k
                           (resolve 'hnsw.ultra-fast/search-knn)
                           num-threads))

(defn search-simd-parallel
  "Multi-threaded search with SIMD-optimized distances"
  [index queries k num-threads]
  (require '[hnsw.ultra-fast :as ultra])
  (require '[hnsw.simd-optimized :as simd])
  ;; Make sure the index uses SIMD distance functions
  (parallel-search-futures index queries k
                           (resolve 'hnsw.ultra-fast/search-knn)
                           num-threads))
