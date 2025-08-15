(ns hnsw.bench
  "Comprehensive benchmark suite for all HNSW implementations"
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [clojure.set :as set]
            ;; All implementations
            [hnsw.ann.graph.pure-hnsw :as pure-hnsw]
            [hnsw.ann.partition.ivf-flat :as ivf-flat]
            [hnsw.ann.hybrid.ivf-hnsw :as ivf-hnsw]
            [hnsw.ann.hash.hybrid-lsh :as lsh]
            [hnsw.ann.partition.partitioned-hnsw :as phnsw]
            [hnsw.ann.partition.lightning :as lightning]
            [hnsw.ann.dimreduct.pcaf :as pcaf] ; P-HNSW with PCA
            [hnsw.ultra-fast :as ultra]
            [hnsw.ultra-optimized :as ultra-opt]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

;; =====================
;; Data Loading
;; =====================

(defn load-bible-data-fn
  "Load the complete Károli Bible dataset
   Uses bible_embeddings_complete.json for full 31,173 verses if available,
   otherwise falls back to bible_embeddings.json"
  []
  (let [;; Try to use complete dataset first, fall back to smaller if needed
        complete-file "data/bible_embeddings_complete.json"
        fallback-file "data/bible_embeddings.json"
        filename (if (.exists (clojure.java.io/file complete-file))
                   complete-file
                   fallback-file)
        _ (println (format "📚 Loading data from: %s" filename))
        data (json/read-str (slurp filename) :key-fn keyword)
        verses (:verses data)
        verse-count (count verses)]
    (println (format "✅ Loaded %,d verses" verse-count))
    {:vectors (mapv (fn [v] [(:id v) (double-array (:embedding v))]) verses)
     :text-map (into {} (map (fn [v] [(:id v) (:text v)]) verses))
     :metadata {:count verse-count
                :dimension (count (:embedding (first verses)))
                :source filename}}))

(def load-bible-data
  "Lazily loaded and cached Bible dataset"
  (delay (load-bible-data-fn)))

(defn prepare-subset
  "Prepare a subset of data for quick testing"
  [data n]
  (let [subset-vectors (vec (take n (:vectors data)))
        subset-ids (set (map first subset-vectors))]
    (assoc data
           :vectors subset-vectors
           :text-map (select-keys (:text-map data) subset-ids)
           :metadata (assoc (:metadata data) :count n))))

(comment
  ;;
  (let [data @load-bible-data
        subset (prepare-subset data 5000)]
    (->> subset :vectors count))
  ;;
  )

;; =====================
;; Ground Truth
;; =====================

(defn compute-exact-knn
  "Compute exact k-nearest neighbors for recall testing"
  [vectors query k]
  (vec (take k
             (sort-by :distance
                      (map (fn [[id v]]
                             (let [vec-v (if (instance? (Class/forName "[D") v) (vec v) v)
                                   vec-q (if (instance? (Class/forName "[D") query) (vec query) query)
                                   dot (reduce + (map * vec-v vec-q))
                                   nv (Math/sqrt (reduce + (map * vec-v vec-v)))
                                   nq (Math/sqrt (reduce + (map * vec-q vec-q)))]
                               {:id id :distance (- 1.0 (/ dot (* nv nq)))}))
                           vectors)))))

(defn calc-recall
  "Calculate recall@k for approximate results vs exact results"
  [approx exact]
  (let [approx-ids (set (map :id approx))
        exact-ids (set (map :id exact))]
    (float (/ (count (set/intersection approx-ids exact-ids))
              (count exact-ids)))))

;; =====================
;; Performance Metrics
;; =====================

(defn measure-build-time
  "Measure index build time"
  [build-fn vectors & args]
  (let [start (System/currentTimeMillis)
        index (apply build-fn vectors args)
        time (- (System/currentTimeMillis) start)]
    {:index index
     :time-ms time
     :vectors-per-sec (int (/ (count vectors) (/ time 1000.0)))}))

(defn measure-search-latency
  "Measure search latency with percentiles"
  [search-fn index queries k & args]
  (let [times (for [q queries]
                (let [start (System/nanoTime)]
                  (apply search-fn index q k args)
                  (/ (- (System/nanoTime) start) 1000000.0)))
        sorted-times (vec (sort times))
        n (count sorted-times)]
    {:min (first sorted-times)
     :p50 (nth sorted-times (int (* n 0.50)))
     :p95 (nth sorted-times (int (* n 0.95)))
     :p99 (nth sorted-times (int (* n 0.99)))
     :max (last sorted-times)
     :avg (/ (reduce + sorted-times) n)}))

(defn measure-recall
  "Measure recall@k for an implementation"
  [search-fn index test-vectors k & args]
  (let [recalls (for [[id vec-data] test-vectors]
                  (let [query (if (instance? (Class/forName "[D") vec-data) vec-data (double-array vec-data))
                        exact (compute-exact-knn test-vectors vec-data k)
                        approx (apply search-fn index query k args)]
                    (calc-recall approx exact)))]
    (* 100 (/ (reduce + recalls) (count recalls)))))

;; =====================
;; Index Builders
;; =====================

(defn build-index-with-interrupt
  "Build index with interrupt capability - checks *interrupt* atom periodically
   Usage: 
     (def *interrupt* (atom false))
     (future (build-index-with-interrupt :pure-hnsw data *interrupt*))
     ;; To interrupt: (reset! *interrupt* true)"
  [method vectors interrupt-atom & {:keys [show-progress?] :or {show-progress? true}}]

  (when show-progress?
    (println (format "\n🔨 Building %s index (interruptible)..." (name method))))

  ;; Check interrupt before starting
  (when @interrupt-atom
    (println "⛔ Build interrupted before start!")
    (throw (InterruptedException. "Build cancelled")))

  (let [check-interval 100 ; Check every 100 vectors
        counter (atom 0)

        ;; Wrap the data to check for interrupts during iteration
        interruptible-data
        (map (fn [item]
               (swap! counter inc)
               (when (and (zero? (mod @counter check-interval))
                          @interrupt-atom)
                 (println "⛔ Build interrupted!")
                 (throw (InterruptedException. "Build cancelled")))
               item)
             vectors)]

    ;; Build with the wrapped data
    (case method
      :pure-hnsw (pure-hnsw/build-index interruptible-data
                                        :M 16
                                        :ef-construction 200
                                        :show-progress? show-progress?)
      :multithread-pure-hnsw (pure-hnsw/build-index interruptible-data
                                                    :M 16
                                                    :ef-construction 200
                                                    :show-progress? show-progress?)
      :lightning (lightning/build-index vectors
                                        :num-partitions 24
                                        :show-progress? show-progress?)
      :ivf-flat (ivf-flat/build-index vectors
                                      :num-partitions 24
                                      :show-progress? show-progress?)
      (throw (IllegalArgumentException. (str "Unknown method: " method))))))

(defn build-all-indexes
  "Build all index types and return map of indexes with timing"
  [vectors & {:keys [show-progress?] :or {show-progress? true}}]
  (let [results (atom {})]

    ;; 1. Hybrid LSH (fastest build)
    (when show-progress? (println "\n🔨 Building Hybrid LSH..."))
    (let [{:keys [index time-ms vectors-per-sec]}
          (measure-build-time lsh/build-index vectors :show-progress? false)]
      (swap! results assoc :lsh
             {:index index :build-time time-ms :build-rate vectors-per-sec}))

    ;; 2. IVF-FLAT
    (when show-progress? (println "🔨 Building IVF-FLAT..."))
    (let [{:keys [index time-ms vectors-per-sec]}
          (measure-build-time ivf-flat/build-index vectors
                              :num-partitions 24 :show-progress? false)]
      (swap! results assoc :ivf-flat
             {:index index :build-time time-ms :build-rate vectors-per-sec}))

    ;; 3. Partitioned HNSW  
    (when show-progress? (println "🔨 Building Partitioned HNSW..."))
    (let [{:keys [index time-ms vectors-per-sec]}
          (measure-build-time phnsw/build-index vectors
                              :num-partitions 24 :shuffle? true
                              :search-mode :lightning :show-progress? false)]
      (swap! results assoc :part-hnsw
             {:index index :build-time time-ms :build-rate vectors-per-sec}))

    ;; 4. Lightning Index
    (when show-progress? (println "🔨 Building Lightning Index..."))
    (let [{:keys [index time-ms vectors-per-sec]}
          (measure-build-time lightning/build-index vectors
                              :num-partitions 24 :smart-partition? false
                              :show-progress? false)]
      (swap! results assoc :lightning
             {:index index :build-time time-ms :build-rate vectors-per-sec}))

    ;; 5. P-HNSW (PCAF) - PCA-based dimension reduction
    (when show-progress? (println "🔨 Building P-HNSW (PCAF)..."))
    (let [{:keys [index time-ms vectors-per-sec]}
          (measure-build-time pcaf/build-index vectors
                              :n-components 100 :k-filter 32
                              :show-progress? false)]
      (swap! results assoc :pcaf
             {:index index :build-time time-ms :build-rate vectors-per-sec}))

    ;; 6. IVF-HNSW (slowest build, skip for large datasets)
    (when (and show-progress? (< (count vectors) 10000))
      (println "🔨 Building IVF-HNSW (for small dataset only)...")
      (let [{:keys [index time-ms vectors-per-sec]}
            (measure-build-time ivf-hnsw/build-index vectors
                                :num-partitions 16 :M 8 :ef-construction 100
                                :show-progress? false)]
        (swap! results assoc :ivf-hnsw
               {:index index :build-time time-ms :build-rate vectors-per-sec})))

    ;; 7. Multi-threaded Pure HNSW (NEW)
    (when show-progress? (println "🔨 Building Multi-threaded Pure HNSW..."))
    (let [{:keys [index time-ms vectors-per-sec]}
          (measure-build-time pure-hnsw/build-index vectors
                              :M 16 :ef-construction 200
                              :show-progress? false)]
      (swap! results assoc :multithread-pure-hnsw
             {:index index :build-time time-ms :build-rate vectors-per-sec}))

    @results))

;; =====================
;; Benchmark Runners
;; =====================

(defn run-search-benchmark
  "Run search benchmark for all implementations"
  [indexes queries k]
  (let [results (atom {})]

    ;; Test each implementation
    (doseq [[impl-key {:keys [index]}] indexes]
      (let [search-fn (case impl-key
                        :lsh lsh/search-knn
                        :ivf-flat ivf-flat/search-knn
                        :part-hnsw phnsw/search-knn
                        :lightning lightning/search-knn
                        :pcaf pcaf/search-knn ; P-HNSW
                        :ivf-hnsw ivf-hnsw/search-knn
                        :multithread-pure-hnsw pure-hnsw/search-knn ; Multi-threaded Pure HNSW
                        nil)
            mode (if (#{:ivf-flat :part-hnsw :lightning :ivf-hnsw :pcaf :multithread-pure-hnsw} impl-key)
                   :balanced
                   nil)]
        (when search-fn
          (let [query-vecs (map second queries)
                latency (if mode
                          (measure-search-latency search-fn index query-vecs k mode)
                          (measure-search-latency search-fn index query-vecs k))]
            (swap! results assoc impl-key latency)))))

    @results))

(defn run-recall-benchmark
  "Run recall benchmark for all implementations"
  [indexes test-vectors k]
  (let [results (atom {})]

    (doseq [[impl-key {:keys [index]}] indexes]
      (let [search-fn (case impl-key
                        :lsh lsh/search-knn
                        :ivf-flat ivf-flat/search-knn
                        :part-hnsw phnsw/search-knn
                        :lightning lightning/search-knn
                        :pcaf pcaf/search-knn ; P-HNSW
                        :ivf-hnsw ivf-hnsw/search-knn
                        :multithread-pure-hnsw pure-hnsw/search-knn ; Multi-threaded Pure HNSW
                        nil)
            mode (if (#{:ivf-flat :part-hnsw :lightning :ivf-hnsw :pcaf :multithread-pure-hnsw} impl-key)
                   :balanced
                   nil)]
        (when search-fn
          (let [recall (if mode
                         (measure-recall search-fn index test-vectors k mode)
                         (measure-recall search-fn index test-vectors k))]
            (swap! results assoc impl-key recall)))))

    @results))

;; =====================
;; Main Benchmark Functions
;; =====================

(defn quick-benchmark
  "Quick benchmark with 1000 vectors"
  []
  (println "\n⚡ QUICK BENCHMARK (1000 vectors)")
  (println "====================================")

  (let [data @load-bible-data
        subset (prepare-subset data 1000) ; Changed from 31000 to 1000
        vectors (:vectors subset)]

    (println (format "\n📊 Dataset: %d vectors, %d dimensions"
                     (count vectors)
                     (-> subset :metadata :dimension)))

    ;; Build indexes
    (let [indexes (build-all-indexes vectors :show-progress? true)]

      ;; Print build time comparison
      (println "\n📊 BUILD TIME COMPARISON:")
      (println "--------------------------------------------")
      (println "Implementation | Build Time | Vectors/Second")
      (println "---------------|------------|---------------")
      (doseq [[k v] (sort-by #(-> % second :build-time) indexes)]
        (println (format "%-14s | %9.2fs | %,d vec/s"
                         (name k)
                         (/ (:build-time v) 1000.0)
                         (:build-rate v))))

      ;; Search performance
      (let [queries (take 50 vectors)
            search-results (run-search-benchmark indexes queries 10)]

        (println "\n📊 SEARCH PERFORMANCE (50 queries):")
        (println "---------------------------------------------------")
        (println "Implementation | Avg Time | P50    | P95    | P99")
        (println "---------------|----------|--------|--------|--------")
        (doseq [[k v] (sort-by #(-> % second :avg) search-results)]
          (println (format "%-14s | %7.2fms | %6.2fms | %6.2fms | %6.2fms"
                           (name k) (:avg v) (:p50 v) (:p95 v) (:p99 v)))))

      ;; Recall test
      (let [test-vectors (take 20 vectors)
            recall-results (run-recall-benchmark indexes test-vectors 10)]

        (println "\n🎯 RECALL COMPARISON:")
        (println "------------------------")
        (println "Implementation | Recall@10")
        (println "---------------|----------")
        (doseq [[k v] (sort-by second > recall-results)]
          (println (format "%-14s | %8.1f%%"
                           (name k) v)))))))

(defn full-benchmark
  "Full benchmark with complete Bible dataset"
  []
  (println "\n📊 FULL BENCHMARK (31K vectors)")
  (println "==================================")

  (let [data @load-bible-data
        vectors (:vectors data)]

    (println (format "\n📊 Dataset: %d vectors, %d dimensions"
                     (count vectors)
                     (-> data :metadata :dimension)))

    ;; Build only fast implementations for full dataset
    (println "\n🔨 Building indexes (skipping slow implementations)...")
    (let [fast-vectors (mapv (fn [[id v]] [id v]) vectors)
          indexes (atom {})]

      ;; Build only the fast ones
      (println "  Building Hybrid LSH...")
      (let [{:keys [index time-ms vectors-per-sec]}
            (measure-build-time lsh/build-index fast-vectors :show-progress? false)]
        (swap! indexes assoc :lsh
               {:index index :build-time time-ms :build-rate vectors-per-sec}))

      (println "  Building IVF-FLAT...")
      (let [{:keys [index time-ms vectors-per-sec]}
            (measure-build-time ivf-flat/build-index fast-vectors
                                :num-partitions 24 :show-progress? false)]
        (swap! indexes assoc :ivf-flat
               {:index index :build-time time-ms :build-rate vectors-per-sec}))

      (println "  Building Lightning Index...")
      (let [{:keys [index time-ms vectors-per-sec]}
            (measure-build-time lightning/build-index fast-vectors
                                :num-partitions 24 :show-progress? false)]
        (swap! indexes assoc :lightning
               {:index index :build-time time-ms :build-rate vectors-per-sec}))

      ;; Results
      (println "\n✅ Build complete!")
      (println "\n📊 BUILD TIME COMPARISON:")
      (println "--------------------------------------------")
      (println "Implementation | Build Time | Vectors/Second")
      (println "---------------|------------|---------------")
      (doseq [[k v] (sort-by #(-> % second :build-time) @indexes)]
        (println (format "%-14s | %9.2fs | %,d vec/s"
                         (name k)
                         (/ (:build-time v) 1000.0)
                         (:build-rate v))))

      ;; Search test with fewer queries for speed
      (let [queries (take 20 vectors)
            search-results (run-search-benchmark @indexes queries 10)]

        (println "\n📊 SEARCH PERFORMANCE (20 queries):")
        (println "---------------------------------------------------")
        (println "Implementation | Avg Time | P50    | P95    | P99")
        (println "---------------|----------|--------|--------|--------")
        (doseq [[k v] (sort-by #(-> % second :avg) search-results)]
          (println (format "%-14s | %7.2fms | %6.2fms | %6.2fms | %6.2fms"
                           (name k) (:avg v) (:p50 v) (:p95 v) (:p99 v))))))))

(defn semantic-search-demo
  "Demo semantic search with actual Bible texts
   method can be: :lightning, :pcaf, :phnsw, :ivf-flat, :ivf-hnsw, :lsh, :pure-hnsw, :multithread-pure-hnsw
   subset-size: number of verses to load (e.g., 1000, 5000, 31000)"
  ([method subset-size]
   (println "\n📖 SEMANTIC SEARCH DEMO")
   (println "=========================")

   (let [data @load-bible-data
         subset (prepare-subset data subset-size)
         vectors (:vectors subset)
         text-map (:text-map subset)]

     (println (format "\n📚 Dataset: %,d Bible verses" (count vectors)))
     (println (format "🔍 Method: %s\n" (name method)))

     ;; Build index based on method
     (println (format "🔨 Building %s index..." (name method)))
     (let [start-build (System/currentTimeMillis)
           index (case method
                   :lightning (lightning/build-index vectors :num-partitions 24 :show-progress? false)
                   :lightning-multithread (do
                                            (println "   ⚡ Multi-threaded Lightning configuration:")
                                            (println "   • 32 partitions for better parallelization")
                                            (println "   • Parallel search enabled")
                                            (println "   • Random partitioning (fast build)")
                                            (lightning/build-index vectors
                                                                   :num-partitions 32
                                                                   :smart-partition? false ; Default: false for speed
                                                                   :show-progress? false))
                   :lightning-multithread-smart (do
                                                  (println "   ⚡ Multi-threaded Lightning with Smart Partitioning:")
                                                  (println "   • 32 partitions for better parallelization")
                                                  (println "   • Parallel search enabled")
                                                  (println "   • K-means++ partitioning (slower build, better recall)")
                                                  (lightning/build-index vectors
                                                                         :num-partitions 32
                                                                         :smart-partition? true ; Smart k-means++ partitioning
                                                                         :show-progress? false))
                   :pcaf (pcaf/build-index vectors
                                           :n-components (min 200 (int (/ (count (second (first vectors))) 4)))
                                           :k-filter 32
                                           :num-threads 4
                                           :show-progress? false)
                   :phnsw (phnsw/build-index vectors
                                             :num-partitions (min 32 (max 4 (int (/ (count vectors) 1000))))
                                             :shuffle? true
                                             :search-mode :lightning
                                             :show-progress? false)
                   :ivf-flat (ivf-flat/build-index vectors
                                                   :num-partitions (min 32 (max 4 (int (/ (count vectors) 1000))))
                                                   :show-progress? false)
                   :ivf-hnsw (ivf-hnsw/build-index vectors
                                                   :num-partitions (min 16 (max 4 (int (/ (count vectors) 2000))))
                                                   :M 8
                                                   :ef-construction 100
                                                   :show-progress? false)
                   :lsh (lsh/build-index vectors :show-progress? false)
                   :pure-hnsw (do
                                (when (> (count vectors) 1000)
                                  (println "⚠️  WARNING: Pure HNSW is VERY slow for large datasets!")
                                  (println "   This may take several minutes..."))
                                (pure-hnsw/build-index vectors :M 8 :ef-construction 100 :show-progress? false))
                   :multithread-pure-hnsw (do
                                            (when (> (count vectors) 10000)
                                              (println "⚠️  WARNING: Pure HNSW can be slow for very large datasets!")
                                              (println "   Consider using Lightning or IVF-FLAT for datasets > 10k vectors.")
                                              (println "   This may take several minutes..."))
                                            (println "🚀 Using Multi-threaded Pure HNSW for faster performance!")
                                            (pure-hnsw/build-index vectors
                                                                   :M 16
                                                                   :ef-construction 200
                                                                   :show-progress? false))
                   (throw (IllegalArgumentException. (str "Unknown method: " method))))
           build-time (- (System/currentTimeMillis) start-build)]

       (println (format "✅ Index built in %.2fs\n" (/ build-time 1000.0)))

       ;; Test with Genesis 1:1 or first available verse
       (let [query-id (if (some #(= (first %) "Ter_1:1") vectors)
                        "Ter_1:1"
                        (first (first vectors)))
             query-vec (second (first (filter #(= (first %) query-id) vectors)))
             query-text (get text-map query-id)]

         (println (format "🔍 Query: %s" query-id))
         (println (format "\"%s\"\n" query-text))

         ;; Search with appropriate function for each method
         (let [search-fn (case method
                           :lightning lightning/search-knn
                           :lightning-multithread lightning/search-knn
                           :lightning-multithread-smart lightning/search-knn
                           :pcaf pcaf/search-knn
                           :phnsw phnsw/search-knn
                           :ivf-flat ivf-flat/search-knn
                           :ivf-hnsw ivf-hnsw/search-knn
                           :lsh lsh/search-knn
                           :pure-hnsw pure-hnsw/search-knn
                           :multithread-pure-hnsw pure-hnsw/search-knn)

               ;; Configure search parameters based on method
               search-args (cond
                             (#{:lightning-multithread :lightning-multithread-smart} method)
                             [index query-vec 10 0.25 true] ; 25% partitions, parallel

                             (#{:lightning :pcaf :phnsw :ivf-flat :ivf-hnsw :pure-hnsw :multithread-pure-hnsw} method)
                             [index query-vec 10 :balanced]

                             :else
                             [index query-vec 10])

               start (System/nanoTime)
               results (apply search-fn search-args)
               search-time (/ (- (System/nanoTime) start) 1000000.0)]

           (println (format "⚡ Search time: %.2fms\n" search-time))
           (println "📚 Top 10 similar verses:")
           (println "--------------------------------")

           (doseq [[i r] (map-indexed vector results)]
             (let [text (get text-map (:id r) "N/A")
                   similarity (* 100 (- 1.0 (:distance r)))]
               (println (format "\n%2d. [%s] - Similarity: %.1f%%"
                                (inc i) (:id r) similarity))
               (println (format "    \"%s\""
                                (subs text 0 (min 100 (count text)))))))))

       ;; Clean up for methods that need it
       (when (= method :pcaf)
         (pcaf/cleanup index))

       (println "\n✨ Semantic search demo completed!"))))

  ;; Backward compatibility - default to lightning with 31000 verses
  ([]
   (semantic-search-demo :lightning 31000))

  ;; Old signature for compatibility - ignore index-fn parameter
  ([index-fn]
   (semantic-search-demo :lightning 31000)))

(defn semantic-search-demo-safe
  "Safe version of semantic-search-demo with timeout protection
   Automatically falls back to faster method if timeout occurs"
  [method subset-size & {:keys [timeout-seconds] :or {timeout-seconds 30}}]
  (let [future-result (future
                        (try
                          (semantic-search-demo method subset-size)
                          (catch Exception e
                            {:error (.getMessage e)})))]

    ;; Wait for result with timeout
    (let [result (deref future-result (* timeout-seconds 1000) :timeout)]
      (if (= result :timeout)
        (do
          (println (format "\n⚠️  TIMEOUT: %s build exceeded %d seconds!"
                           (name method) timeout-seconds))
          (println "🔄 Falling back to Lightning (fast) implementation...")
          (future-cancel future-result)

          ;; Fallback to Lightning which is always fast
          (semantic-search-demo :lightning subset-size))
        result))))

(defn semantic-search-demo-detailed
  "Enhanced semantic search demo with detailed analysis
   Shows more context about the results and performance metrics
   Supports PCAF multi-threaded mode"
  ([method subset-size]
   (println "\n📖 ENHANCED SEMANTIC SEARCH DEMO")
   (println "==================================")

   (let [data @load-bible-data
         subset (prepare-subset data subset-size)
         vectors (:vectors subset)
         text-map (:text-map subset)]

     (println (format "\n📚 Dataset: %,d Bible verses" (count vectors)))
     (println (format "🔍 Method: %s\n" (name method)))

     ;; Build index based on method
     (println (format "🔨 Building %s index..." (name method)))
     (let [start-build (System/currentTimeMillis)
           index (case method
                   :lightning (lightning/build-index vectors :num-partitions 24 :show-progress? false)
                   :lightning-multithread (do
                                            (println "   ⚡ Multi-threaded Lightning configuration:")
                                            (println "   • 32 partitions for better parallelization")
                                            (println "   • Parallel search enabled")
                                            (println "   • Random partitioning (fast build)")
                                            (lightning/build-index vectors
                                                                   :num-partitions 32
                                                                   :smart-partition? false
                                                                   :show-progress? false))
                   :lightning-multithread-smart (do
                                                  (println "   ⚡ Multi-threaded Lightning with Smart Partitioning:")
                                                  (println "   • 32 partitions for better parallelization")
                                                  (println "   • Parallel search enabled")
                                                  (println "   • K-means++ partitioning (slower build, better recall)")
                                                  (lightning/build-index vectors
                                                                         :num-partitions 32
                                                                         :smart-partition? true
                                                                         :show-progress? false))
                   :pcaf (pcaf/build-index vectors
                                           :n-components (min 200 (int (/ (count (second (first vectors))) 4)))
                                           :k-filter 32
                                           :num-threads 4
                                           :show-progress? false)
                   :pcaf-multithread (do
                                       (println "   🚀 Using 8 threads for parallel search")
                                       (println "   📉 Dimension reduction: 768 -> 200")
                                       (pcaf/build-index vectors
                                                         :n-components 200
                                                         :k-filter 48 ; More candidates for better accuracy
                                                         :num-threads 8 ; More threads
                                                         :show-progress? false))
                   :phnsw (phnsw/build-index vectors
                                             :num-partitions (min 32 (max 4 (int (/ (count vectors) 1000))))
                                             :shuffle? true
                                             :search-mode :lightning
                                             :show-progress? false)
                   :ivf-flat (ivf-flat/build-index vectors
                                                   :num-partitions (min 32 (max 4 (int (/ (count vectors) 1000))))
                                                   :show-progress? false)
                   :lsh (lsh/build-index vectors :show-progress? false)
                   (throw (IllegalArgumentException. (str "Unknown method: " method))))
           build-time (- (System/currentTimeMillis) start-build)]

       (println (format "✅ Index built in %.2fs" (/ build-time 1000.0)))
       (println (format "📊 Build performance: %.0f vectors/second\n"
                        (/ (count vectors) (/ build-time 1000.0))))

       ;; Test with multiple queries for better analysis
       (println "🔬 TESTING WITH MULTIPLE QUERIES:")
       (println "===================================\n")

       ;; Test queries from different parts of the Bible
       (let [test-queries [["Ter_1:1" "Genesis 1:1 - Creation"]
                           ["Zsolt_23:1" "Psalm 23:1 - The Lord is my shepherd"]
                           ["Jn_3:16" "John 3:16 - God so loved the world"]
                           [(first (first vectors)) "First verse in dataset"]]
             available-queries (filter #(some (fn [[id _]] (= id (first %))) vectors) test-queries)]

         (doseq [[query-id description] (take 2 available-queries)] ; Test 2 queries
           (when-let [query-entry (first (filter #(= (first %) query-id) vectors))]
             (let [query-vec (second query-entry)
                   query-text (get text-map query-id)]

               (println (format "📍 Query: %s (%s)" query-id description))
               (println (format "   Text: \"%s\"\n"
                                (subs query-text 0 (min 100 (count query-text)))))

               ;; Search with appropriate function for each method
               (let [search-fn (case method
                                 :lightning lightning/search-knn
                                 :lightning-multithread lightning/search-knn
                                 :lightning-multithread-smart lightning/search-knn
                                 :pcaf pcaf/search-knn
                                 :pcaf-multithread pcaf/search-knn
                                 :phnsw phnsw/search-knn
                                 :ivf-flat ivf-flat/search-knn
                                 :lsh lsh/search-knn)

                     ;; Configure search parameters for multi-threaded versions
                     search-args (cond
                                   (#{:lightning-multithread :lightning-multithread-smart} method)
                                   [index query-vec 10 0.25 true] ; 25% partitions, parallel=true

                                   (#{:lightning :pcaf :pcaf-multithread :phnsw :ivf-flat} method)
                                   [index query-vec 10 :balanced]

                                   :else
                                   [index query-vec 10])

                     ;; Time multiple searches for better average
                     search-times (for [_ (range 5)]
                                    (let [start (System/nanoTime)]
                                      (apply search-fn search-args)
                                      (/ (- (System/nanoTime) start) 1000000.0)))

                     avg-search-time (/ (reduce + search-times) (count search-times))

                     ;; Get actual results
                     results (apply search-fn search-args)]

                 (println (format "   ⚡ Average search time: %.3fms (5 runs)" avg-search-time))
                 (println (format "   🎯 QPS potential: %.0f queries/sec\n" (/ 1000.0 avg-search-time)))
                 (println "   Top 5 similar verses:")
                 (println "   ----------------------")

                 (doseq [[i r] (map-indexed vector (take 5 results))]
                   (let [text (get text-map (:id r) "N/A")
                         similarity (* 100 (- 1.0 (:distance r)))]
                     (println (format "   %d. [%s] - Similarity: %.1f%%"
                                      (inc i) (:id r) similarity))
                     (println (format "      \"%s\""
                                      (subs text 0 (min 80 (count text)))))))

                 (println))))))

       ;; Performance summary
       (println "\n📈 PERFORMANCE SUMMARY:")
       (println "========================")
       (println (format "  Method: %s" (name method)))
       (when (= method :lightning-multithread)
         (println "  Special features:")
         (println "    • 32 partitions for better parallelization")
         (println "    • Parallel search across partitions")
         (println "    • Random partitioning (fast build)")
         (println "    • 25% partition sampling for speed"))
       (when (= method :lightning-multithread-smart)
         (println "  Special features:")
         (println "    • 32 partitions for better parallelization")
         (println "    • Parallel search across partitions")
         (println "    • K-means++ smart partitioning (slower build, better recall)")
         (println "    • 25% partition sampling for balanced search"))
       (when (= method :pcaf-multithread)
         (println "  Special features:")
         (println "    • 8 parallel threads for search")
         (println "    • SIMD-optimized distance computation")
         (println "    • Two-phase search algorithm"))
       (println (format "  Dataset size: %,d vectors" (count vectors)))
       (println (format "  Build time: %.2fs" (/ build-time 1000.0)))
       (println (format "  Build rate: %.0f vectors/sec"
                        (/ (count vectors) (/ build-time 1000.0))))

       ;; Clean up for methods that need it
       (when (#{:pcaf :pcaf-multithread} method)
         (pcaf/cleanup index))

       (println "\n✨ Enhanced semantic search demo completed!"))))

  ;; Backward compatibility
  ([]
   (semantic-search-demo-detailed :lightning 31000)))

;; =====================
;; Interactive REPL Functions
;; =====================

(defn test-multiprobe-lsh
  "Test the new multi-probe LSH implementation"
  []
  (println "\n🧪 TESTING MULTI-PROBE LSH")
  (println "============================")

  (let [data @load-bible-data
        subset (prepare-subset data 1000)
        vectors (:vectors subset)]

    (println "\n📊 Building LSH index...")
    (let [index (lsh/build-index vectors :show-progress? false)
          test-vec (second (first vectors))]

      ;; Compare normal vs multi-probe
      (println "\n⚖️  Comparing search methods:")
      (println "------------------------------")

      ;; Original search
      (let [start (System/nanoTime)
            results-orig (lsh/search-hybrid index test-vec 10 :num-probes 6)
            time-orig (/ (- (System/nanoTime) start) 1000000.0)]
        (println (format "Original LSH:     %.2fms - Found %d results"
                         time-orig (count results-orig))))

      ;; Multi-probe search
      (let [start (System/nanoTime)
            results-multi (lsh/search-hybrid-multiprobe index test-vec 10
                                                        :num-probes 6 :probe-radius 2)
            time-multi (/ (- (System/nanoTime) start) 1000000.0)]
        (println (format "Multi-probe LSH:  %.2fms - Found %d results"
                         time-multi (count results-multi))))

      ;; Test recall improvement
      (println "\n🎯 Recall improvement test:")
      (let [exact (compute-exact-knn vectors test-vec 10)
            test-configs [{:name "Original (2 probes)"
                           :fn lsh/search-hybrid
                           :args [:num-probes 2]}
                          {:name "Original (6 probes)"
                           :fn lsh/search-hybrid
                           :args [:num-probes 6]}
                          {:name "Multi-probe (2 probes, radius 1)"
                           :fn lsh/search-hybrid-multiprobe
                           :args [:num-probes 2 :probe-radius 1]}
                          {:name "Multi-probe (6 probes, radius 2)"
                           :fn lsh/search-hybrid-multiprobe
                           :args [:num-probes 6 :probe-radius 2]}
                          {:name "Multi-probe (8 probes, radius 3)"
                           :fn lsh/search-hybrid-multiprobe
                           :args [:num-probes 8 :probe-radius 3]}]]

        (println "Configuration                        | Recall@10 | Search Time")
        (println "-------------------------------------|-----------|------------")
        (doseq [{:keys [name fn args]} test-configs]
          (let [start (System/nanoTime)
                results (apply fn index test-vec 10 args)
                time (/ (- (System/nanoTime) start) 1000000.0)
                recall (* 100 (calc-recall results exact))]
            (println (format "%-36s | %8.1f%% | %7.2fms" name recall time))))))))

(defn test-pcaf
  "Test the P-HNSW (PCAF) implementation with PCA dimension reduction"
  []
  (println "\n🧮 TESTING P-HNSW (PCAF) IMPLEMENTATION")
  (println "=========================================")

  (let [data @load-bible-data
        subset (prepare-subset data 1000)
        vectors (:vectors subset)]

    (println (format "\n📊 Dataset: %d vectors, 768 dimensions" (count vectors)))

    ;; Build P-HNSW index
    (println "\n🔨 Building P-HNSW index with PCA...")
    (println "   • Original dimensions: 768")
    (println "   • Target dimensions: 100 (7.68x reduction)")
    (println "   • k-filter: 32")

    (let [start (System/currentTimeMillis)
          index (pcaf/build-index vectors
                                  :n-components 100
                                  :k-filter 32
                                  :show-progress? true)
          build-time (- (System/currentTimeMillis) start)]

      (println (format "\n✅ Index built in %.2fs" (/ build-time 1000.0)))

      ;; Test search performance
      (println "\n⚡ Testing search performance (20 queries):")
      (let [query-indices (take 20 (shuffle (range (count vectors))))
            query-vecs (map #(second (nth vectors %)) query-indices)

            ;; Test different modes
            modes [:turbo :fast :balanced :accurate :precise]]

        (println "Mode       | Avg Time | Min    | Max    | k-filter")
        (println "-----------|----------|--------|--------|----------")

        (doseq [mode modes]
          (let [times (for [q query-vecs]
                        (let [start (System/nanoTime)]
                          (pcaf/search-knn index q 10 mode)
                          (/ (- (System/nanoTime) start) 1000000.0)))
                sorted-times (sort times)
                avg-time (/ (reduce + times) (count times))
                k-filter (case mode
                           :turbo 16
                           :fast 24
                           :balanced 32
                           :accurate 48
                           :precise 64)]
            (println (format "%-10s | %7.2fms | %6.2fms | %6.2fms | %d"
                             (name mode) avg-time
                             (first sorted-times)
                             (last sorted-times)
                             k-filter)))))

      ;; Test recall
      (println "\n🎯 Testing recall (10 queries):")
      (let [test-vectors (take 10 vectors)
            recalls (for [[id vec-data] test-vectors]
                      (let [exact (compute-exact-knn vectors vec-data 10)
                            approx (pcaf/search-knn index vec-data 10 :balanced)]
                        (calc-recall approx exact)))
            avg-recall (* 100 (/ (reduce + recalls) (count recalls)))]

        (println (format "Average Recall@10: %.1f%%" avg-recall)))

      ;; Compare with other methods
      (println "\n📊 Comparison with other methods:")
      (println "Building indexes for comparison...")

      (let [;; Build other indexes for comparison
            lsh-start (System/currentTimeMillis)
            lsh-idx (lsh/build-index vectors :show-progress? false)
            lsh-time (- (System/currentTimeMillis) lsh-start)

            ivf-start (System/currentTimeMillis)
            ivf-idx (ivf-flat/build-index vectors :num-partitions 24 :show-progress? false)
            ivf-time (- (System/currentTimeMillis) ivf-start)]

        (println "\n📊 BUILD TIME COMPARISON:")
        (println "Method     | Build Time | Vectors/sec")
        (println "-----------|------------|-------------")
        (println (format "P-HNSW     | %9.2fs | %,d"
                         (/ build-time 1000.0)
                         (int (/ 1000 (/ build-time 1000.0)))))
        (println (format "Hybrid LSH | %9.2fs | %,d"
                         (/ lsh-time 1000.0)
                         (int (/ 1000 (/ lsh-time 1000.0)))))
        (println (format "IVF-FLAT   | %9.2fs | %,d"
                         (/ ivf-time 1000.0)
                         (int (/ 1000 (/ ivf-time 1000.0)))))

        ;; Search comparison
        (println "\n⚡ SEARCH TIME COMPARISON (10 queries):")
        (let [test-query (second (first vectors))]
          (println "Method     | Search Time")
          (println "-----------|------------")

          (let [pcaf-time (let [start (System/nanoTime)]
                            (pcaf/search-knn index test-query 10 :balanced)
                            (/ (- (System/nanoTime) start) 1000000.0))
                lsh-time (let [start (System/nanoTime)]
                           (lsh/search-knn lsh-idx test-query 10)
                           (/ (- (System/nanoTime) start) 1000000.0))
                ivf-time (let [start (System/nanoTime)]
                           (ivf-flat/search-knn ivf-idx test-query 10 :balanced)
                           (/ (- (System/nanoTime) start) 1000000.0))]

            (println (format "P-HNSW     | %7.2fms" pcaf-time))
            (println (format "Hybrid LSH | %7.2fms" lsh-time))
            (println (format "IVF-FLAT   | %7.2fms" ivf-time)))))))

  (println "\n✨ P-HNSW (PCAF) test completed!"))

(defn test-multithread-pure-hnsw
  "Test multi-threaded Pure HNSW with safe dataset sizes"
  [& {:keys [max-size] :or {max-size 5000}}]
  (println "\n🧪 TESTING MULTI-THREADED PURE HNSW")
  (println "=====================================")

  (let [data @load-bible-data
        safe-size (min max-size (count (:vectors data)))
        subset (prepare-subset data safe-size)
        vectors (:vectors subset)]

    (println (format "\n📊 Dataset: %,d vectors (limited from %,d)"
                     safe-size (count (:vectors data))))
    (println "⚠️  Note: Pure HNSW is limited to prevent timeouts\n")

    ;; Test different sizes
    (doseq [test-size [100 500 1000 (min 5000 safe-size)]]
      (when (<= test-size safe-size)
        (println (format "\n--- Testing with %,d vectors ---" test-size))
        (let [test-subset (prepare-subset data test-size)
              test-vectors (:vectors test-subset)]

          ;; Build and test single-threaded
          (println "Single-threaded Pure HNSW:")
          (let [start (System/currentTimeMillis)
                index (pure-hnsw/build-index test-vectors
                                             :M 8
                                             :ef-construction 100
                                             :show-progress? false)
                build-time (- (System/currentTimeMillis) start)
                query (second (first test-vectors))
                search-start (System/nanoTime)
                results (pure-hnsw/search-knn index query 10 :balanced)
                search-time (/ (- (System/nanoTime) search-start) 1000000.0)]
            (println (format "  Build: %.2fs, Search: %.2fms"
                             (/ build-time 1000.0) search-time)))

          ;; Build and test multi-threaded
          (println "Multi-threaded Pure HNSW:")
          (let [start (System/currentTimeMillis)
                index (pure-hnsw/build-index test-vectors
                                             :M 16
                                             :ef-construction 200
                                             :show-progress? false)
                build-time (- (System/currentTimeMillis) start)
                query (second (first test-vectors))
                search-start (System/nanoTime)
                results (pure-hnsw/search-knn index query 10 :balanced)
                search-time (/ (- (System/nanoTime) search-start) 1000000.0)]
            (println (format "  Build: %.2fs, Search: %.2fms"
                             (/ build-time 1000.0) search-time))))))

    (println "\n✨ Multi-threaded Pure HNSW test completed!")
    (println "For larger datasets, use Lightning or IVF-FLAT implementations.")))

;; =====================
;; Main Entry Point
;; =====================

(defn -main
  "Main entry point for benchmarking"
  [& args]
  (let [mode (first args)]
    (case mode
      "quick" (quick-benchmark)
      "full" (full-benchmark)
      "demo" (if (= (count args) 3)
               ;; New format: demo method subset-size
               (let [method (keyword (second args))
                     subset-size (Integer/parseInt (nth args 2))]
                 (semantic-search-demo method subset-size))
               ;; Default to lightning with 5000 verses
               (semantic-search-demo :lightning 5000))
      "multiprobe" (test-multiprobe-lsh)
      "pcaf" (test-pcaf) ; Test P-HNSW
      "multithread" (semantic-search-demo :multithread-pure-hnsw 5000) ; Test multi-threaded Pure HNSW
      ;; Default
      (do
        (println "\n🚀 HNSW Benchmark Suite")
        (println "Usage: clj -M -m hnsw.bench [mode] [options]")
        (println "\nAvailable modes:")
        (println "  quick      - Quick test with 1000 vectors")
        (println "  full       - Full benchmark with 31K vectors")
        (println "  demo [method] [size] - Semantic search demonstration")
        (println "                         methods: lightning, pcaf, phnsw, ivf-flat, ivf-hnsw, lsh, pure-hnsw, multithread-pure-hnsw")
        (println "                         size: number of verses (e.g., 1000, 5000, 31000)")
        (println "  multiprobe - Test multi-probe LSH")
        (println "  pcaf       - Test P-HNSW (PCAF) with PCA")
        (println "  multithread - Test multi-threaded Pure HNSW with 5000 vectors")
        (println "\nExamples:")
        (println "  clj -M -m hnsw.bench demo lightning 5000")
        (println "  clj -M -m hnsw.bench demo pcaf 1000")
        (println "  clj -M -m hnsw.bench demo multithread-pure-hnsw 31000")
        (println "  clj -M -m hnsw.bench multithread")
        (println "\nRunning quick benchmark by default...")
        (quick-benchmark)))))

(comment
  ;; Interactive REPL usage:

  ;; Load data
  (def data @load-bible-data)
  (->> data :vectors count)

  ;; Quick test
  (quick-benchmark)

  ;; === SAFE USAGE FOR nREPL (Java 21+ compatible) ===

  ;; Use semantic-search-demo-safe with timeout protection
  (semantic-search-demo-safe :multithread-pure-hnsw 31070 :timeout-seconds 30)
  (semantic-search-demo-safe :pure-hnsw 5000 :timeout-seconds 20)

  ;; Interruptible index building
  (def *interrupt* (atom false))
  (def build-future
    (future
      (try
        (build-index-with-interrupt :pure-hnsw
                                    (vec (take 10000 (:vectors @load-bible-data)))
                                    *interrupt*)
        (catch InterruptedException e
          (println "Build was interrupted!")))))
  ;; To interrupt: (reset! *interrupt* true)

  ;; === FAST METHODS (no timeout needed) ===

  (semantic-search-demo :lightning 31070) ;; Fast build, bruteforce
  (semantic-search-demo :pcaf 31070) ;; P-HNSW with PCA
  (semantic-search-demo :phnsw 31070) ;; Partitioned HNSW
  (semantic-search-demo :ivf-flat 31070) ;; IVF-FLAT
  (semantic-search-demo :lsh 100) ;; Without HNSW

  ;; === ENHANCED DEMOS WITH DETAILED ANALYSIS ===

  ;; Lightning with multi-threading (super fast!)
  (semantic-search-demo-detailed :lightning 31070)
  (semantic-search-demo-detailed :lightning-multithread 31070) ; Fast build, parallel search
  (semantic-search-demo-detailed :lightning-multithread-smart 31070) ; Slow build, better recall

  ;; PCAF with multi-threading (fast + accurate)
  (semantic-search-demo-detailed :pcaf 31070)
  (semantic-search-demo-detailed :pcaf-multithread 31070) ; 8 threads, more candidates

  ;; Compare different methods with detailed metrics
  (semantic-search-demo-detailed :lightning 5000)
  (semantic-search-demo-detailed :lightning-multithread 5000) ; Random partitioning
  (semantic-search-demo-detailed :lightning-multithread-smart 5000) ; Smart partitioning
  (semantic-search-demo-detailed :ivf-flat 5000)
  (semantic-search-demo-detailed :pcaf 5000)
  (semantic-search-demo-detailed :lsh 5000)

  ;; === SLOW METHODS (use with caution) ===

  ;; Pure HNSW - ALWAYS use safe version or small datasets!
  (semantic-search-demo :pure-hnsw 1000) ;; OK - small dataset
  (semantic-search-demo-safe :multithread-pure-hnsw 31070 :timeout-seconds 30) ;; Safe with timeout

  ;; SAFE test for multi-threaded Pure HNSW
  (test-multithread-pure-hnsw :max-size 20)

  ;; Full benchmark (takes longer)
  (full-benchmark)

  ;; Test P-HNSW (PCAF)
  (test-pcaf))