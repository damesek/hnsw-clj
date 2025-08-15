(ns hnsw.wip.partitioned
  "Optimized partitioned HNSW for production - focused on search performance"
  (:require [hnsw.ultra-fast :as ultra])
  (:import [java.util.concurrent Executors Callable Future]
           [java.util ArrayList Collections]))

;; ===== Core Data Structure =====

(defrecord PartitionedIndex [partitions ; Vector of sub-indexes
                             partition-size ; Size of each partition
                             num-partitions ; Number of partitions
                             total-elements ; Total number of elements
                             distance-fn]) ; Distance function

;; ===== Build Functions =====

(defn build-partition
  "Build a single partition optimized for fast search"
  [partition-id data distance-fn show-progress?]
  (when show-progress?
    (println (format "  Building partition %d with %d vectors..."
                     partition-id (count data))))

  (let [start (System/currentTimeMillis)]
    (try
      (let [index (ultra/build-index data
                                     :distance-fn distance-fn
                                     :M 8 ; Lower M for faster build
                                     :ef-construction 100 ; Balanced for speed
                                     :show-progress? false)]
        (when show-progress?
          (println (format "  ✅ Partition %d complete in %.1f seconds"
                           partition-id
                           (/ (- (System/currentTimeMillis) start) 1000.0))))
        index)
      (catch Exception e
        (println (format "  ❌ Error in partition %d: %s"
                         partition-id (.getMessage e)))
        (ultra/create-ultra-graph :M 8
                                  :ef-construction 100
                                  :distance-fn distance-fn)))))

(defn build-partitioned-index
  "Build partitioned HNSW index with parallel construction"
  [data & {:keys [num-partitions distance-fn show-progress? num-threads shuffle?]
           :or {num-partitions 8
                distance-fn ultra/cosine-distance-ultra
                show-progress? true
                num-threads (.availableProcessors (Runtime/getRuntime))
                shuffle? true}}] ; Alapértelmezetten shuffle-öljük

  (let [;; Opcionális shuffle a jobb eloszlásért
        shuffled-data (if shuffle?
                        (shuffle data)
                        data)
        total-elements (count shuffled-data)
        partition-size (int (Math/ceil (/ total-elements num-partitions)))
        partitioned-data (partition-all partition-size shuffled-data)]

    (when show-progress?
      (println (format "\n🔨 Building PARTITIONED INDEX"))
      (println (format "   Total vectors: %d" total-elements))
      (println (format "   Partitions: %d" num-partitions))
      (println (format "   Partition size: ~%d" partition-size))
      (println (format "   Threads: %d" num-threads))
      (when shuffle?
        (println "   Data: SHUFFLED for better distribution")))

    (let [start-time (System/currentTimeMillis)
          executor (Executors/newFixedThreadPool num-threads)

          ;; Submit all partition builds in parallel
          futures (doall
                   (map-indexed
                    (fn [idx partition]
                      (let [partition-vec (vec partition)]
                        (.submit executor
                                 ^Callable
                                 (fn []
                                   (build-partition idx
                                                    partition-vec
                                                    distance-fn
                                                    show-progress?)))))
                    partitioned-data))

          ;; Wait for all partitions to complete
          partitions (mapv #(.get ^Future %) futures)]

      (.shutdown executor)

      (let [build-time (- (System/currentTimeMillis) start-time)]
        (when show-progress?
          (println (format "\n✅ All partitions built in %.2f seconds"
                           (/ build-time 1000.0)))
          (println (format "   Rate: %.0f vectors/second"
                           (/ total-elements (/ build-time 1000.0)))))

        (->PartitionedIndex partitions
                            partition-size
                            (count partitions)
                            total-elements
                            distance-fn)))))

;; ===== Search Functions - OPTIMIZED FOR <0.6ms =====

;; Pre-allocated thread pool for minimal overhead
(def search-executor (Executors/newFixedThreadPool 8))

(defn search-partition-inline
  "Inline partition search for maximum speed"
  [partition query-vec k]
  (ultra/search-knn partition query-vec k))

(defn search-partitioned-index-ultra
  "Ultra-optimized search targeting <0.6ms latency"
  [^PartitionedIndex p-index query-vec k]
  (let [partitions (:partitions p-index)
        num-partitions (count partitions)
        ;; EXTREME OPTIMIZATION: Only top 1-2 from each partition
        k-per-partition (if (<= k 5) 1 2)

        ;; Pre-allocate with exact size
        ^ArrayList results (ArrayList. (* num-partitions k-per-partition))

        ;; Batch submit for minimum overhead
        futures (into []
                      (map (fn [partition]
                             (.submit search-executor
                                      ^Callable
                                      (fn []
                                        (ultra/search-knn partition query-vec k-per-partition))))
                           partitions))]

    ;; Tight collection loop
    (loop [i 0]
      (when (< i num-partitions)
        (.addAll results (.get ^Future (nth futures i)))
        (recur (inc i))))

    ;; In-place sort
    (Collections/sort results
                      (reify java.util.Comparator
                        (compare [_ a b]
                          (Double/compare (:distance a) (:distance b)))))

    ;; Direct subList for top k
    (if (<= (.size results) k)
      (vec results)
      (vec (.subList results 0 k)))))

(defn search-partitioned-index-lightning
  "Lightning-fast search with partition-aware result fetching"
  [^PartitionedIndex p-index query-vec k]
  (let [partitions (:partitions p-index)
        num-partitions (count partitions)
        ;; Adaptív k_per_partition a partíciók számától függően
        k-per-partition (cond
                          (<= num-partitions 8) 2 ; 8 vagy kevesebb: 2 eredmény
                          (<= num-partitions 12) 3 ; 12 partíció: 3 eredmény
                          :else 1) ; 16+: 1 eredmény

        ;; Pre-allocate result collection
        results (ArrayList. (* num-partitions k-per-partition))

        ;; Submit all searches in parallel
        futures (mapv
                 (fn [partition]
                   (java.util.concurrent.CompletableFuture/supplyAsync
                    (reify java.util.function.Supplier
                      (get [_]
                        (vec (take k-per-partition
                                   (ultra/search-knn partition query-vec k-per-partition)))))
                    search-executor))
                 partitions)]

    ;; Collect results
    (doseq [^java.util.concurrent.CompletableFuture f futures]
      (when-let [partition-results (.get f)]
        (.addAll results partition-results)))

    ;; Sort and return top k
    (Collections/sort results
                      (reify java.util.Comparator
                        (compare [_ a b]
                          (Double/compare (:distance a) (:distance b)))))

    (vec (take k results))))

(defn search-partitioned-index-hyper
  "Hyper-optimized search for absolute minimum latency <0.5ms"
  [^PartitionedIndex p-index query-vec k]
  (let [partitions (:partitions p-index)
        num-partitions (count partitions)

        ;; Use regular Future for even lower overhead
        futures (object-array num-partitions)]

    ;; Submit all searches without any abstraction
    (dotimes [i num-partitions]
      (aset futures i
            (.submit search-executor
                     ^Callable
                     (fn []
                       (first (ultra/search-knn (nth partitions i) query-vec 1))))))

    ;; Collect results directly
    (let [results (ArrayList. num-partitions)]
      (dotimes [i num-partitions]
        (when-let [result (.get ^Future (aget futures i))]
          (.add results result)))

      ;; If we need more results, do a second pass
      (when (< (.size results) k)
        (let [futures2 (object-array num-partitions)]
          (dotimes [i (min 4 num-partitions)] ; Only first 4 partitions
            (aset futures2 i
                  (.submit search-executor
                           ^Callable
                           (fn []
                             (vec (take 3 (ultra/search-knn (nth partitions i) query-vec 3)))))))

          (dotimes [i (min 4 num-partitions)]
            (when-let [more-results (.get ^Future (aget futures2 i))]
              (.addAll results more-results)))))

      ;; Final sort
      (Collections/sort results
                        (reify java.util.Comparator
                          (compare [_ a b]
                            (Double/compare (:distance a) (:distance b)))))

      (vec (take k results)))))

(defn search-partitioned-index-turbo
  "Turbo mode using parallel streams - backup option"
  [^PartitionedIndex p-index query-vec k]
  (let [partitions (:partitions p-index)
        k-per-partition (min k 5)]

    (-> partitions
        vec
        (.parallelStream)
        (.map (reify java.util.function.Function
                (apply [_ partition]
                  (ultra/search-knn partition query-vec k-per-partition))))
        (.flatMap (reify java.util.function.Function
                    (apply [_ results]
                      (.stream ^java.util.Collection results))))
        (.sorted (reify java.util.Comparator
                   (compare [_ a b]
                     (Double/compare (:distance a) (:distance b)))))
        (.limit k)
        (.collect (java.util.stream.Collectors/toList))
        vec)))

;; ===== Public API =====

(defn search-knn
  "Main search function - uses lightning-fast path"
  [index query-vec k]
  (search-partitioned-index-lightning index query-vec k))

(defn index-info
  "Get index statistics"
  [^PartitionedIndex p-index]
  {:num-partitions (:num-partitions p-index)
   :total-elements (:total-elements p-index)
   :partition-size (:partition-size p-index)})