(ns hnsw.ann.hybrid.ivf-hnsw
  "IVF-HNSW: Inverted File Index with HNSW graphs per partition
   Combines the benefits of partitioning with HNSW's fast graph search
   
   Performance targets:
   - Build: 30-60s for 31k vectors
   - Search: 2-3ms with 90-95% recall
   - Memory: 2-3x of IVF-FLAT"
  (:require [hnsw.graph :as graph]
            [hnsw.ultra-fast :as ultra]
            [hnsw.lightning :as lightning])
  (:import [java.util ArrayList Collections Random HashMap]
           [java.util.concurrent ConcurrentHashMap ForkJoinPool CompletableFuture]
           [java.util.concurrent.atomic AtomicInteger]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ============================================================
;; IVF-HNSW Index Structure
;; ============================================================

(defrecord IVFHNSWIndex [partitions ; Vector of partition HNSW graphs
                         centroids ; Partition centroids for routing
                         data-map ; Global id -> vector map
                         distance-fn ; Distance function
                         vector-norms ; Pre-computed norms
                         hnsw-params]) ; HNSW parameters per partition

;; ============================================================
;; K-means++ Initialization
;; ============================================================

(defn- kmeans-plus-plus-init
  "Initialize centroids using k-means++ algorithm"
  [vectors num-centroids distance-fn]
  (let [n (count vectors)
        centroids (ArrayList.)
        rng (Random. 42)]

    ;; First centroid: random selection
    (.add centroids (second (nth vectors (.nextInt rng n))))

    ;; Remaining centroids: weighted by distance
    (dotimes [_ (dec num-centroids)]
      (let [distances (double-array n)]
        ;; Compute min distance to existing centroids
        (dotimes [i n]
          (let [vec (second (nth vectors i))
                min-dist (reduce min Double/MAX_VALUE
                                 (map #(distance-fn vec %) centroids))]
            (aset distances i (double min-dist))))

        ;; Select next centroid proportional to squared distance
        (let [sum (areduce distances i s 0.0 (+ s (* (aget distances i)
                                                     (aget distances i))))
              r (* (.nextDouble rng) sum)]
          (loop [i 0 cumsum 0.0]
            (let [dist-sq (* (aget distances i) (aget distances i))]
              (if (>= (+ cumsum dist-sq) r)
                (.add centroids (second (nth vectors i)))
                (recur (inc i) (+ cumsum dist-sq))))))))

    (vec centroids)))

;; ============================================================
;; Partitioning Functions
;; ============================================================

(defn- assign-to-nearest-centroid
  "Assign vector to nearest centroid"
  ^long [^doubles vector centroids distance-fn]
  (loop [i 0
         min-dist Double/MAX_VALUE
         best-idx 0]
    (if (< i (count centroids))
      (let [dist (double (distance-fn vector (nth centroids i)))]
        (if (< dist min-dist)
          (recur (inc i) dist i)
          (recur (inc i) min-dist best-idx)))
      best-idx)))

(defn- compute-centroid
  "Compute centroid of vectors"
  ^doubles [vectors]
  (let [num-vectors (count vectors)
        dim (alength ^doubles (first vectors))
        centroid (double-array dim)]
    (doseq [^doubles v vectors]
      (dotimes [i dim]
        (aset centroid i (+ (aget centroid i) (aget v i)))))
    (dotimes [i dim]
      (aset centroid i (/ (aget centroid i) num-vectors)))
    centroid))

(defn- partition-vectors
  "Partition vectors using k-means"
  [data num-partitions distance-fn max-iterations]
  (let [vectors (vec data)
        n (count vectors)

        ;; Initialize centroids with k-means++
        centroids (atom (kmeans-plus-plus-init vectors num-partitions distance-fn))

        ;; Lloyd's algorithm iterations
        _ (dotimes [iter max-iterations]
            (let [;; Assignment step
                  assignments (int-array n)
                  _ (dotimes [i n]
                      (let [vec (second (nth vectors i))]
                        (aset assignments i
                              (assign-to-nearest-centroid vec @centroids distance-fn))))

                  ;; Update step
                  partition-lists (vec (repeatedly num-partitions #(ArrayList.)))
                  _ (dotimes [i n]
                      (.add ^ArrayList (nth partition-lists (aget assignments i))
                            (second (nth vectors i))))

                  ;; Compute new centroids
                  new-centroids (mapv #(if (empty? %)
                                         (nth @centroids %2) ; Keep old if empty
                                         (compute-centroid %))
                                      partition-lists
                                      (range num-partitions))]

              ;; Check convergence (simplified - could track centroid movement)
              (reset! centroids new-centroids)))

        ;; Final assignment
        assignments (int-array n)
        _ (dotimes [i n]
            (let [vec (second (nth vectors i))]
              (aset assignments i
                    (assign-to-nearest-centroid vec @centroids distance-fn))))

        ;; Build final partitions
        partition-lists (vec (repeatedly num-partitions #(ArrayList.)))
        _ (dotimes [i n]
            (.add ^ArrayList (nth partition-lists (aget assignments i))
                  (nth vectors i)))]

    [(mapv vec partition-lists) @centroids]))

;; ============================================================
;; HNSW Graph Building
;; ============================================================

(defn- build-partition-hnsw
  "Build HNSW graph for a single partition"
  [partition-data hnsw-params show-progress? partition-idx]
  (when (and show-progress? (zero? (mod partition-idx 4)))
    (println (format "   Building HNSW for partition %d/%d..."
                     (inc partition-idx)
                     (:total-partitions hnsw-params))))

  (if (empty? partition-data)
    (graph/create-graph hnsw-params)
    (let [graph (graph/create-graph hnsw-params)]
      ;; Insert each vector individually to the graph
      (reduce (fn [g [id vector]]
                (graph/insert g id vector))
              graph
              partition-data))))

;; ============================================================
;; Main Build Function
;; ============================================================

(defn build-ivf-hnsw-index
  "Build IVF-HNSW index with parallel HNSW construction"
  [data & {:keys [num-partitions
                  distance-fn
                  show-progress?
                  M ; HNSW M parameter
                  ef-construction ; HNSW construction parameter
                  max-iterations ; K-means iterations
                  parallel-build?]
           :or {num-partitions 24
                distance-fn ultra/cosine-distance-ultra
                show-progress? true
                M 16
                ef-construction 200
                max-iterations 10
                parallel-build? true}}]

  (let [start-time (System/currentTimeMillis)
        total-vectors (count data)
        data-vec (vec data)]

    (when show-progress?
      (println "\n🚀 IVF-HNSW INDEX BUILD")
      (println (format "   Vectors: %,d" total-vectors))
      (println (format "   Partitions: %d" num-partitions))
      (println (format "   HNSW M: %d, ef-construction: %d" M ef-construction))
      (println "   Phase 1: Computing vector norms..."))

    ;; Pre-compute norms
    (let [vector-norms (ConcurrentHashMap. total-vectors)
          data-map (ConcurrentHashMap. total-vectors)
          ^ForkJoinPool pool (ForkJoinPool. 8)]

      ;; Parallel norm computation
      (.submit pool
               ^Runnable
               (fn []
                 (doseq [[id ^doubles vector] data-vec]
                   (.put data-map id vector)
                   (let [norm (Math/sqrt
                               (areduce vector i sum 0.0
                                        (+ sum (* (aget vector i)
                                                  (aget vector i)))))]
                     (.put vector-norms id norm)))))
      (.shutdown pool)
      (.awaitTermination pool 10 java.util.concurrent.TimeUnit/SECONDS)

      (when show-progress?
        (println "   Phase 2: K-means clustering..."))

      ;; Partition data with k-means
      (let [[partitions centroids] (partition-vectors data-vec
                                                      num-partitions
                                                      distance-fn
                                                      max-iterations)]

        (when show-progress?
          (println (format "   Phase 3: Building %d HNSW graphs..." num-partitions)))

        ;; Build HNSW graph for each partition
        (let [hnsw-params (assoc (graph/default-params)
                                 :M M
                                 :ef-construction ef-construction
                                 :distance-fn distance-fn
                                 :total-partitions num-partitions)

              partition-graphs (if parallel-build?
                                ;; Parallel HNSW construction
                                 (let [^ForkJoinPool pool (ForkJoinPool. 4)
                                       futures (mapv (fn [idx partition]
                                                       (CompletableFuture/supplyAsync
                                                        (reify java.util.function.Supplier
                                                          (get [_]
                                                            (build-partition-hnsw
                                                             partition
                                                             hnsw-params
                                                             show-progress?
                                                             idx)))
                                                        pool))
                                                     (range num-partitions)
                                                     partitions)]
                                   (mapv #(.get %) futures))

                                ;; Sequential HNSW construction
                                 (mapv #(build-partition-hnsw %1 hnsw-params show-progress? %2)
                                       partitions
                                       (range num-partitions)))

              build-time (- (System/currentTimeMillis) start-time)]

          (when show-progress?
            (println (format "\n✅ IVF-HNSW Index built in %.2f seconds!"
                             (/ build-time 1000.0)))
            (println (format "   Build rate: %.0f vectors/second"
                             (/ total-vectors (/ build-time 1000.0))))
            (println (format "   Avg partition size: %.0f vectors"
                             (double (/ total-vectors num-partitions)))))

          (->IVFHNSWIndex partition-graphs
                          centroids
                          data-map
                          distance-fn
                          vector-norms
                          hnsw-params))))))

;; ============================================================
;; Search Functions
;; ============================================================

(defn search-ivf-hnsw
  "Search IVF-HNSW index with configurable modes"
  [^IVFHNSWIndex index ^doubles query-vec k
   & {:keys [mode num-probes ef-search]
      :or {mode :balanced}}]

  (let [;; Mode configurations
        mode-configs {:turbo {:num-probes 1 :ef-search 50} ; 1-2ms, 85-90% recall
                      :fast {:num-probes 2 :ef-search 100} ; 2-3ms, 90-95% recall
                      :balanced {:num-probes 3 :ef-search 150} ; 3-4ms, 95-97% recall
                      :accurate {:num-probes 4 :ef-search 200} ; 4-5ms, 97-99% recall
                      :precise {:num-probes 5 :ef-search 300}} ; 5-7ms, 99%+ recall

        config (or (get mode-configs mode)
                   {:num-probes (or num-probes 3)
                    :ef-search (or ef-search 150)})

        ;; Get fields from index
        partition-graphs (.partitions index)
        centroids (.centroids index)
        distance-fn (.distance-fn index)

        ;; Find nearest partitions using centroids
        partition-distances (map-indexed
                             (fn [idx centroid]
                               {:idx idx
                                :dist (distance-fn query-vec centroid)})
                             centroids)
        sorted-partitions (sort-by :dist partition-distances)
        selected-partitions (take (:num-probes config) sorted-partitions)

        ;; Search in selected partitions using HNSW
        results (ArrayList.)]

    ;; Search each selected partition's HNSW graph
    (doseq [{:keys [idx]} selected-partitions]
      (let [partition-graph (nth partition-graphs idx)
            ;; Set ef parameter for search
            graph-with-ef (assoc-in partition-graph [:params :ef] (:ef-search config))
            partition-results (graph/search-knn graph-with-ef query-vec (* k 2))]
        (.addAll results partition-results)))

    ;; Sort all results and return top-k
    (Collections/sort results
                      (reify java.util.Comparator
                        (compare [_ a b]
                          (Double/compare (:distance a) (:distance b)))))
    (vec (take k results))))

;; ============================================================
;; API Functions
;; ============================================================

(defn build-index
  "Build IVF-HNSW index (API compatible)"
  [data & opts]
  (apply build-ivf-hnsw-index data opts))

(defn search-knn
  "Search IVF-HNSW index (API compatible)
   
   Modes:
   - :turbo    - 1-2ms, 85-90% recall
   - :fast     - 2-3ms, 90-95% recall  
   - :balanced - 3-4ms, 95-97% recall (default)
   - :accurate - 4-5ms, 97-99% recall
   - :precise  - 5-7ms, 99%+ recall"
  ([index query-vec k]
   (search-ivf-hnsw index query-vec k :mode :balanced))
  ([index query-vec k mode]
   (if (keyword? mode)
     (search-ivf-hnsw index query-vec k :mode mode)
     ;; Backward compatibility with search-percent
     (let [num-probes (int (* 24 mode))]
       (search-ivf-hnsw index query-vec k :num-probes num-probes)))))

(defn index-info
  "Get information about the index"
  [^IVFHNSWIndex index]
  (let [partitions (.partitions index)
        total-nodes (reduce + (map #(count (:nodes %)) partitions))]
    {:type "IVF-HNSW Index"
     :vectors (.size ^ConcurrentHashMap (.data-map index))
     :partitions (count partitions)
     :avg-partition-size (/ total-nodes (count partitions))
     :hnsw-params (select-keys (.hnsw-params index) [:M :ef-construction])}))
