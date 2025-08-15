(ns hnsw.wip.parallel-build
  "Highly optimized parallel HNSW implementation for ultra-fast index building"
  (:import [java.util PriorityQueue HashMap HashSet ArrayList Collections Random]
           [java.util.concurrent ConcurrentHashMap ThreadLocalRandom ForkJoinPool
            ConcurrentSkipListSet CompletableFuture Executors TimeUnit CountDownLatch
            ConcurrentLinkedQueue]
           [java.util.concurrent.atomic AtomicInteger AtomicReference AtomicLong]
           [java.util.function BiFunction Consumer Function Supplier]
           [java.lang.invoke MethodHandles VarHandle]))

;; ===== Performance Configuration =====
(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

;; ===== SIMD Configuration =====
 ;; ===== Configuration =====
(def VECTOR_UNROLL_SIZE 4)

;; ===== Forward Declarations =====
(declare batch-search-layer batch-distance-calculation parallel-prune-connections)

;; ===== SIMD Distance Functions =====

(defn ^double fast-cosine-distance
  "Fast cosine distance with loop unrolling"
  [^floats v1 ^floats v2]
  (let [len (alength v1)
        chunks (quot len 4)
        remainder (rem len 4)]
    (loop [i 0
           dot 0.0
           norm1 0.0
           norm2 0.0]
      (if (< i chunks)
        ;; Process 4 elements at once
        (let [idx (* i 4)
              a0 (aget v1 idx)
              b0 (aget v2 idx)
              a1 (aget v1 (+ idx 1))
              b1 (aget v2 (+ idx 1))
              a2 (aget v1 (+ idx 2))
              b2 (aget v2 (+ idx 2))
              a3 (aget v1 (+ idx 3))
              b3 (aget v2 (+ idx 3))]
          (recur (inc i)
                 (+ dot (* a0 b0) (* a1 b1) (* a2 b2) (* a3 b3))
                 (+ norm1 (* a0 a0) (* a1 a1) (* a2 a2) (* a3 a3))
                 (+ norm2 (* b0 b0) (* b1 b1) (* b2 b2) (* b3 b3))))
        ;; Handle remainder
        (let [[final-dot final-norm1 final-norm2]
              (loop [j (* chunks 4)
                     d dot
                     n1 norm1
                     n2 norm2]
                (if (< j len)
                  (let [a (aget v1 j)
                        b (aget v2 j)]
                    (recur (inc j)
                           (+ d (* a b))
                           (+ n1 (* a a))
                           (+ n2 (* b b))))
                  [d n1 n2]))]
          (if (and (> final-norm1 0.0) (> final-norm2 0.0))
            (- 1.0 (/ final-dot (* (Math/sqrt final-norm1)
                                   (Math/sqrt final-norm2))))
            1.0))))))

;; ===== Node Structure =====

(deftype ParallelNode [^String id
                       ^floats vector ; Use floats for SIMD
                       ^long level
                       ^objects neighbors ; Array of ConcurrentSkipListSet
                       ^AtomicLong version]) ; For lock-free updates

(deftype ParallelGraph [^ConcurrentHashMap nodes
                        ^AtomicReference entry-point
                        ^long M
                        ^long max-M
                        ^long ef-construction
                        ^double ml
                        distance-fn
                        ^AtomicInteger element-count
                        ^ForkJoinPool pool]) ; Dedicated thread pool

;; ===== Graph Creation =====

(defn create-parallel-graph
  [& {:keys [M ef-construction distance-fn num-threads]
      :or {M 16
           ef-construction 200
           distance-fn fast-cosine-distance
           num-threads (.availableProcessors (Runtime/getRuntime))}}]
  (ParallelGraph. (ConcurrentHashMap.)
                  (AtomicReference. nil)
                  M
                  (* 2 M)
                  ef-construction
                  (/ 1.0 (Math/log 2.0))
                  distance-fn
                  (AtomicInteger. 0)
                  (ForkJoinPool. num-threads)))

;; ===== Batch Distance Calculation =====

(defn batch-distance-calculation
  "Calculate distances for multiple vectors in parallel using SIMD"
  [distance-fn ^floats query-vec vectors]
  (let [^ForkJoinPool pool (ForkJoinPool/commonPool)
        tasks (map (fn [[id ^floats vec]]
                     (CompletableFuture/supplyAsync
                      (reify java.util.function.Supplier
                        (get [_]
                          [id (distance-fn query-vec vec)]))
                      pool))
                   vectors)]
    ;; Wait for all calculations to complete
    (into {} (map #(.get ^CompletableFuture %) tasks))))

;; ===== Parallel Layer Search =====

(defn batch-search-layer
  "Parallel search across multiple entry points"
  [^ParallelGraph graph ^floats query-vec entry-points num-closest level]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        distance-fn (.distance-fn graph)
        ^ConcurrentSkipListSet visited (ConcurrentSkipListSet.)
        ^ConcurrentLinkedQueue results (ConcurrentLinkedQueue.)
        ^CountDownLatch latch (CountDownLatch. (count entry-points))]

    ;; Search from multiple entry points in parallel
    (doseq [entry entry-points]
      (.execute (.pool graph)
                (fn []
                  (try
                    (when-let [^ParallelNode node (.get nodes entry)]
                      (let [^ConcurrentSkipListSet neighbors
                            (when (<= level (.level node))
                              (aget ^objects (.neighbors node) level))]
                        (when neighbors
                          ;; Process neighbors in batch
                          (let [neighbor-vectors
                                (reduce (fn [acc ^String nid]
                                          (when-not (.contains visited nid)
                                            (.add visited nid)
                                            (when-let [^ParallelNode n (.get nodes nid)]
                                              (conj acc [nid (.vector n)]))))
                                        []
                                        neighbors)]
                            ;; Batch distance calculation
                            (when (seq neighbor-vectors)
                              (let [distances (batch-distance-calculation
                                               distance-fn query-vec neighbor-vectors)]
                                (doseq [[id dist] distances]
                                  (.add results {:id id :distance dist}))))))))
                    (finally
                      (.countDown latch))))))

    ;; Wait for all searches to complete
    (.await latch)

    ;; Get top-k results
    (->> results
         (into [])
         (sort-by :distance)
         (take num-closest)
         (mapv :id))))

;; ===== Parallel Insertion =====

(defn insert-parallel
  "Insert element with parallel neighbor search - FIXED DEADLOCK"
  [^ParallelGraph graph ^String id ^floats vector]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        level (long (* (.ml graph) (- (Math/log (.nextDouble (ThreadLocalRandom/current))))))
        neighbors (make-array Object (inc level))]

    ;; Initialize concurrent neighbor sets
    (dotimes [l (inc level)]
      (aset neighbors l (ConcurrentSkipListSet.)))

    (let [new-node (ParallelNode. id vector level neighbors (AtomicLong. 0))]

      ;; Add to graph atomically
      (.put nodes id new-node)
      (.incrementAndGet (.element-count graph))

      ;; Handle entry point
      (.compareAndSet (.entry-point graph) nil id)

      ;; Connect to existing graph - SIMPLIFIED TO AVOID DEADLOCK
      (when (> (.size nodes) 1)
        (when-let [entry-point (.get (.entry-point graph))]
          (when-let [^ParallelNode entry-node (.get nodes entry-point)]
            (let [entry-level (.level entry-node)
                  M (.M graph)
                  ef-construction (.ef-construction graph)]

              ;; Sequential layer processing to avoid deadlock
              (doseq [lc (range (inc (min level entry-level)))]
                (let [search-ef (if (> lc 0) 1 ef-construction)
                      ;; Use simpler search without parallel execution
                      candidates (if (zero? (.size nodes))
                                   []
                                   (take search-ef
                                         (sort-by
                                          #((.distance-fn graph) vector (.vector ^ParallelNode (.get nodes %)))
                                          (filter #(not= % id) (.keySet nodes)))))
                      m (if (= lc 0) (.max-M graph) M)]

                  ;; Connect bidirectionally
                  (doseq [^String neighbor (take m candidates)]
                    (when-let [^ParallelNode neighbor-node (.get nodes neighbor)]
                      (when (<= lc (.level neighbor-node))
                        (.add ^ConcurrentSkipListSet (aget neighbors lc) neighbor)
                        (.add ^ConcurrentSkipListSet
                         (aget ^objects (.neighbors neighbor-node) lc) id))))))))))

      ;; Update entry point if necessary
      (when-let [^ParallelNode entry-node (.get nodes (.get (.entry-point graph)))]
        (when (> level (.level entry-node))
          (.set (.entry-point graph) id)))))

  graph)

;; ===== Batch Parallel Insertion =====

(defn insert-batch-parallel
  "Ultra-fast parallel batch insertion"
  [^ParallelGraph graph elements & {:keys [show-progress? batch-size]
                                    :or {show-progress? true
                                         batch-size 100}}]
  (let [total (count elements)
        ^AtomicInteger counter (AtomicInteger. 0)
        batches (partition-all batch-size elements)]

    (when show-progress?
      (println (format "🚀 Parallel insertion: %d elements in %d batches..."
                       total (count batches))))

    ;; Process batches in parallel
    (doseq [batch batches]
      (let [^CountDownLatch latch (CountDownLatch. (count batch))]
        (doseq [[id vector] batch]
          (.execute (.pool graph)
                    (fn []
                      (try
                        (let [vec-array (if (instance? (Class/forName "[F") vector)
                                          vector
                                          (float-array vector))]
                          (insert-parallel graph id vec-array)
                          (let [processed (.incrementAndGet counter)]
                            (when (and show-progress? (zero? (mod processed 100)))
                              (println (format "Progress: %d/%d (%.1f%%)"
                                               processed total
                                               (* 100.0 (/ processed total)))))))
                        (finally
                          (.countDown latch))))))
        ;; Wait for batch to complete
        (.await latch)))

    (when show-progress?
      (println (format "✅ Completed parallel insertion of %d elements" total)))

    graph))

;; ===== Public API =====

(defn build-parallel-index
  "Build HNSW index with parallel processing"
  [data & {:keys [M ef-construction num-threads show-progress?]
           :or {M 16
                ef-construction 200
                num-threads (.availableProcessors (Runtime/getRuntime))
                show-progress? true}}]
  (let [graph (create-parallel-graph :M M
                                     :ef-construction ef-construction
                                     :num-threads num-threads)]
    (insert-batch-parallel graph data :show-progress? show-progress?)))

(defn search-parallel-knn
  "Parallel k-NN search"
  [^ParallelGraph graph ^floats query-vec ^long k]
  (if (or (nil? (.get (.entry-point graph)))
          (zero? (.size (.nodes graph))))
    []
    (let [entry-point (.get (.entry-point graph))
          ^ParallelNode entry-node (.get (.nodes graph) entry-point)
          entry-level (.level entry-node)
          ef (max k 50)]

      ;; Multi-layer parallel search
      (loop [level entry-level
             nearest [entry-point]]
        (if (< level 0)
          ;; Return top-k results
          (let [^ConcurrentHashMap nodes (.nodes graph)
                distance-fn (.distance-fn graph)]
            (take k
                  (sort-by :distance
                           (map (fn [^String id]
                                  {:id id
                                   :distance (distance-fn query-vec
                                                          (.vector ^ParallelNode (.get nodes id)))})
                                nearest))))

          (recur (dec level)
                 (batch-search-layer graph query-vec nearest
                                     (if (> level 0) 1 ef) level)))))))

;; ===== Graph Info =====

(defn parallel-graph-info
  [^ParallelGraph graph]
  {:num-elements (.get (.element-count graph))
   :entry-point (.get (.entry-point graph))
   :M (.M graph)
   :ef-construction (.ef-construction graph)
   :pool-size (.getParallelism (.pool graph))})
