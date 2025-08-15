(ns hnsw.wip.search-config
  "Configurable search parameters for HNSW")

;; =============================================
;; 🚀 31K MULTI-THREADED PERFORMANCE TEST
;; =============================================

(require '[hnsw.ultra-fast :as ultra])
(require '[hnsw.simd-optimized :as simd])
(require '[clojure.data.json :as json])
(require '[clojure.java.io :as io])
(import '[java.util.concurrent Executors TimeUnit CountDownLatch]
        '[java.util.concurrent.atomic AtomicLong])

(println "\n🚀 31K MULTI-THREADED PERFORMANCE TEST")
(println "=====================================")
(println)

;; Find dataset
(def embeddings-file
  (first (filter #(.exists (io/file %))
                 ["data/bible_embeddings_complete.json"
                  "data/bible_embeddings_30000.json"
                  "data/bible_embeddings_10000.json"
                  "data/bible_embeddings.json"])))

(println (format "📚 Loading %s..." embeddings-file))

;; Load data
(def start-load (System/currentTimeMillis))
(def data (json/read-str (slurp embeddings-file) :key-fn keyword))
(def vectors (mapv (fn [v]
                     [(:id v) (double-array (:embedding v))])
                   (:verses data)))

(println (format "✅ Loaded %,d vectors in %.2f seconds\n"
                 (count vectors)
                 (/ (- (System/currentTimeMillis) start-load) 1000.0)))

;; BUILD INDEX WITH SIMD EUCLIDEAN
(println "🔨 BUILDING INDEX WITH SIMD EUCLIDEAN")
(println "=====================================")
(println "Parameters: M=16, ef_construction=200")
(println "Distance: SIMD-optimized EUCLIDEAN\n")

(let [start (System/currentTimeMillis)]
  (println "Building index...")
  (def index (ultra/build-index vectors
                                :M 16
                                :ef-construction 200
                                :distance-fn simd/euclidean-distance
                                :show-progress? true))
  (def build-time (- (System/currentTimeMillis) start))
  
  (println (format "\n✅ Build time: %.2f seconds (%,.0f vec/s)"
                   (/ build-time 1000.0)
                   (/ (count vectors) (/ build-time 1000.0)))))

;; SINGLE-THREADED BASELINE
(println "\n⚡ SINGLE-THREADED BASELINE")
(println "===========================")

;; Warmup
(def query-vectors (mapv second (take 100 vectors)))
(doseq [i (range 20)]
  (ultra/search-knn index (nth query-vectors (mod i 100)) 10))

;; Measure single-threaded
(def single-times (atom []))
(doseq [i (range 100)]
  (let [query (nth query-vectors (mod i 100))
        start (System/nanoTime)]
    (ultra/search-knn index query 10)
    (swap! single-times conj (/ (- (System/nanoTime) start) 1000000.0))))

(def single-avg (/ (reduce + @single-times) 100))
(println (format "  Average: %.3f ms" single-avg))
(println (format "  QPS: %,.0f" (/ 1000.0 single-avg)))

;; MULTI-THREADED TEST FUNCTION
(defn test-with-threads [num-threads num-queries]
  (println (format "\n⚡ %d THREADS TEST" num-threads))
  (println "=" (apply str (repeat 30 "=")))
  
  (let [executor (Executors/newFixedThreadPool num-threads)
        latch (CountDownLatch. num-queries)
        total-time (AtomicLong. 0)
        query-count (atom 0)
        start-time (System/currentTimeMillis)]
    
    ;; Submit queries
    (dotimes [i num-queries]
      (.submit executor
               ^Runnable
               (fn []
                 (try
                   (let [query (nth query-vectors (mod i 100))
                         start (System/nanoTime)]
                     (ultra/search-knn index query 10)
                     (let [elapsed (- (System/nanoTime) start)]
                       (.addAndGet total-time elapsed)
                       (swap! query-count inc)))
                   (finally
                     (.countDown latch))))))
    
    ;; Wait for completion
    (.await latch)
    (.shutdown executor)
    (.awaitTermination executor 10 TimeUnit/SECONDS)
    
    (let [wall-time (- (System/currentTimeMillis) start-time)
          avg-time (/ (/ (.get total-time) @query-count) 1000000.0)
          throughput (/ (* num-queries 1000.0) wall-time)]
      
      (println (format "  Queries: %d" num-queries))
      (println (format "  Wall time: %.2f seconds" (/ wall-time 1000.0)))
      (println (format "  Avg latency: %.3f ms" avg-time))
      (println (format "  Throughput: %,.0f QPS" throughput))
      {:threads num-threads
       :latency avg-time
       :qps throughput})))

;; TEST WITH DIFFERENT THREAD COUNTS
(println "\n🏁 MULTI-THREADED PERFORMANCE TESTS")
(println "====================================")

(def results
  [(test-with-threads 1 100)
   (test-with-threads 2 200)
   (test-with-threads 4 400)
   (test-with-threads 8 800)
   (test-with-threads 10 1000)
   (test-with-threads 20 2000)
   (test-with-threads 50 5000)])

;; SUMMARY
(println "\n📊 PERFORMANCE SUMMARY")
(println "======================")
(println "Threads | Latency | QPS")
(println "--------|---------|--------")

(doseq [{:keys [threads latency qps]} results]
  (println (format "%7d | %6.2fms | %,7.0f" threads latency qps)))

(println "\n🎯 SPEEDUP vs SINGLE-THREADED:")
(doseq [{:keys [threads latency qps]} results]
  (println (format "  %2d threads: %.1fx speedup" 
                   threads 
                   (/ single-avg latency))))

(println "\n✅ Test complete!")
(System/exit 0)
