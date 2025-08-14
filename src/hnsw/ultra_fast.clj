(ns hnsw.ultra-fast
  "Ultra-fast HNSW implementation with maximum performance optimizations"
  (:import [java.util PriorityQueue HashSet ArrayList Collections Random]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic AtomicInteger AtomicReference]
           [java.util.function Consumer]))

;; ===== Performance Configuration =====
(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

;; ===== Forward Declarations =====
(declare prune-connections-ultra)

;; ===== Ultra-Fast Distance Functions =====

(defmacro unroll-loop-4
  "Macro to unroll loops by factor of 4 for SIMD optimization"
  [len v1 v2 body-fn combine-fn init]
  `(let [len# ~len
         v1# ~v1
         v2# ~v2
         chunks# (quot len# 4)
         remainder# (rem len# 4)]
     (loop [i# 0
            acc# ~init]
       (if (< i# chunks#)
         (let [idx# (* i# 4)
               r0# (~body-fn (aget v1# idx#) (aget v2# idx#))
               r1# (~body-fn (aget v1# (+ idx# 1)) (aget v2# (+ idx# 1)))
               r2# (~body-fn (aget v1# (+ idx# 2)) (aget v2# (+ idx# 2)))
               r3# (~body-fn (aget v1# (+ idx# 3)) (aget v2# (+ idx# 3)))]
           (recur (inc i#)
                  (~combine-fn acc# r0# r1# r2# r3#)))
         ;; Handle remainder
         (loop [j# (* chunks# 4)
                acc2# acc#]
           (if (< j# len#)
             (recur (inc j#)
                    (+ acc2# (~body-fn (aget v1# j#) (aget v2# j#))))
             acc2#))))))

(defn ^double euclidean-distance-ultra
  "Ultra-fast Euclidean distance with loop unrolling"
  ^double [^doubles v1 ^doubles v2]
  (let [len (alength v1)
        sum (unroll-loop-4 len v1 v2
                           (fn [a b] (let [d (- a b)] (* d d)))
                           (fn [acc r0 r1 r2 r3] (+ acc r0 r1 r2 r3))
                           0.0)]
    (Math/sqrt sum)))

(defn ^double cosine-distance-ultra
  "Ultra-fast cosine distance with SIMD-friendly operations"
  ^double [^doubles v1 ^doubles v2]
  (let [len (alength v1)
        chunks (quot len 4)
        remainder (rem len 4)]
    (loop [i 0
           dot 0.0
           norm1 0.0
           norm2 0.0]
      (if (< i chunks)
        (let [idx (* i 4)
              ;; Process 4 elements at once
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

;; ===== Optimized Node Structure =====

(deftype UltraNode [^String id
                    ^doubles vector
                    ^long level
                    ^objects neighbors]) ; Array of HashSets for each level

(deftype UltraGraph [^ConcurrentHashMap nodes
                     ^AtomicReference entry-point
                     ^long M
                     ^long max-M
                     ^long ef-construction
                     ^double ml
                     distance-fn
                     ^AtomicInteger element-count])

;; ===== Fast Priority Queue Wrapper =====

(deftype Candidate [^String id ^double distance]
  Comparable
  (compareTo [this other]
    (Double/compare distance (.distance ^Candidate other))))

;; ===== Graph Creation =====

(defn create-ultra-graph
  [& {:keys [M ef-construction distance-fn seed]
      :or {M 16
           ef-construction 200
           distance-fn euclidean-distance-ultra
           seed 42}}]
  (UltraGraph. (ConcurrentHashMap.)
               (AtomicReference. nil)
               M
               (* 2 M)
               ef-construction
               (/ 1.0 (Math/log 2.0))
               distance-fn
               (AtomicInteger. 0)))

;; ===== Level Assignment with Cached Random =====

(def ^ThreadLocal tl-random
  (proxy [ThreadLocal] []
    (initialValue [] (Random.))))

(defn assign-level-ultra
  "Assign level to new node"
  ^long [^double ml]
  (let [^Random rnd (.get tl-random)]
    (long (* ml (- (Math/log (.nextDouble rnd)))))))

;; ===== Ultra-Fast Search =====

(defn search-layer-ultra
  "Ultra-optimized layer search with minimal allocations"
  [graph ^doubles query-vec entry-points num-closest level]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        distance-fn (.distance-fn graph)
        ^HashSet visited (HashSet. (* 2 num-closest))
        ^PriorityQueue candidates (PriorityQueue. num-closest)
        ^PriorityQueue nearest (PriorityQueue. num-closest (Collections/reverseOrder))
        ^ArrayList result (ArrayList. num-closest)]

    ;; Initialize with entry points
    (doseq [^String point entry-points]
      (when-let [node (.get nodes point)]
        (let [dist (distance-fn query-vec (.vector node))]
          (.add visited point)
          (.add candidates (Candidate. point dist))
          (.add nearest (Candidate. point dist)))))

    ;; Main search loop with optimizations
    (while (not (.isEmpty candidates))
      (let [^Candidate current (.poll candidates)
            current-dist (.distance current)]

        ;; Process if within search bounds (fixed logic)
        (when (or (< (.size nearest) num-closest)
                  (<= current-dist (if (.isEmpty nearest)
                                     Double/MAX_VALUE
                                     (.distance ^Candidate (.peek nearest)))))

          (when-let [node (.get nodes (.id current))]
            (when (<= level (.level node)) ;; Check level bounds
              (let [^HashSet level-neighbors (aget ^objects (.neighbors node) level)]
                (when (and level-neighbors (not (.isEmpty level-neighbors)))
                  ;; Process neighbors
                  (.forEach level-neighbors
                            (reify Consumer
                              (accept [_ neighbor]
                                (let [^String neighbor neighbor]
                                  (when-not (.contains visited neighbor)
                                    (.add visited neighbor)
                                    (when-let [neighbor-node (.get nodes neighbor)]
                                      (let [dist (distance-fn query-vec (.vector neighbor-node))]

                                      ;; Only add if improving result
                                        (when (or (< (.size nearest) num-closest)
                                                  (< dist (if (.isEmpty nearest)
                                                            Double/MAX_VALUE
                                                            (.distance ^Candidate (.peek nearest)))))
                                          (.add candidates (Candidate. neighbor dist))
                                          (.add nearest (Candidate. neighbor dist))

                                          ;; Maintain size limit
                                          (when (> (.size nearest) num-closest)
                                            (.poll nearest)))))))))))))))))

    ;; Extract results
    (.forEach nearest
              (reify Consumer
                (accept [_ item]
                  (.add result (.id ^Candidate item)))))

    (vec result)))

;; ===== Single Element Insertion =====

(defn insert-single
  "Optimized single element insertion"
  [graph ^String id ^doubles vector]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        level (assign-level-ultra (.ml graph))
        neighbors (make-array Object (inc level))]

    ;; Initialize neighbor arrays
    (dotimes [l (inc level)]
      (aset neighbors l (HashSet.)))

    (let [new-node (UltraNode. id vector level neighbors)]

      ;; Add to graph
      (.put nodes id new-node)
      (.incrementAndGet (.element-count graph))

      ;; Handle entry point
      (when (nil? (.get (.entry-point graph)))
        (.set (.entry-point graph) id))

      ;; Connect to existing graph
      (when (> (.size nodes) 1)
        (let [entry-point (.get (.entry-point graph))
              entry-node (.get nodes entry-point)]
          (when entry-node
            (let [entry-level (.level entry-node)
                  M (.M graph)
                  ef-construction (.ef-construction graph)]

              ;; Find nearest neighbors at each level
              (loop [lc (min level entry-level)
                     nearest [entry-point]]
                (when (>= lc 0)
                  (let [candidates (search-layer-ultra graph vector nearest
                                                       (if (> lc 0) 1 ef-construction) lc)
                        m (if (= lc 0) (.max-M graph) M)]

                    ;; Connect bidirectionally
                    (doseq [^String neighbor (take m candidates)]
                      (when-let [neighbor-node (.get nodes neighbor)]
                        ;; Check if neighbor has this level
                        (when (<= lc (.level neighbor-node))
                          ;; Add edges
                          (.add ^HashSet (aget neighbors lc) neighbor)
                          (.add ^HashSet (aget ^objects (.neighbors neighbor-node) lc) id)

                          ;; Prune if needed
                          (let [^HashSet neighbor-conns (aget ^objects (.neighbors neighbor-node) lc)]
                            (when (> (.size neighbor-conns) m)
                              (prune-connections-ultra graph neighbor lc m))))))

                    (recur (dec lc) candidates)))))))

        ;; Update entry point if necessary
        (let [entry-node (.get nodes (.get (.entry-point graph)))]
          (when (and entry-node (> level (.level entry-node)))
            (.set (.entry-point graph) id)))))

    graph))

;; ===== Connection Pruning =====

(defn prune-connections-ultra
  "Prune connections using heuristic"
  [graph ^String node-id level max-conns]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        node (.get nodes node-id)
        ^HashSet connections (aget ^objects (.neighbors node) level)]

    (when (> (.size connections) max-conns)
      (let [distance-fn (.distance-fn graph)
            node-vec (.vector node)
            ;; Calculate distances and sort
            sorted-neighbors (sort-by
                              (fn [^String nid]
                                (distance-fn node-vec
                                             (.vector (.get nodes nid))))
                              (vec connections))]

        ;; Keep only closest neighbors
        (.clear connections)
        (doseq [^String neighbor (take max-conns sorted-neighbors)]
          (.add connections neighbor))))))

;; ===== Batch Insertion =====

(defn insert-batch
  "Sequential batch insertion - parallel version had issues"
  [graph elements & {:keys [show-progress? num-threads batch-size]
                     :or {show-progress? true
                          num-threads (.availableProcessors (Runtime/getRuntime))
                          batch-size 1000}}]
  (let [total (count elements)]
    (when show-progress?
      (println (format "Inserting %d elements..." total)))

    ;; Sequential insertion for now - parallel had issues
    (loop [idx 0
           remaining elements
           g graph]
      (if (seq remaining)
        (let [[id vector] (first remaining)
              vec-array (if (instance? (Class/forName "[D") vector)
                          vector
                          (double-array vector))]

          (when (and show-progress? (zero? (mod idx 500)))
            (println (format "Progress: %d/%d (%.1f%%)"
                             idx total (* 100.0 (/ idx total)))))

          (recur (inc idx)
                 (rest remaining)
                 (insert-single g id vec-array)))
        g))))

;; ===== Public API =====

(defn build-index
  "Build HNSW index from data"
  [data & {:keys [M ef-construction distance-fn show-progress?]
           :or {M 16
                ef-construction 200
                distance-fn cosine-distance-ultra
                show-progress? true}}]
  (let [graph (create-ultra-graph :M M
                                  :ef-construction ef-construction
                                  :distance-fn distance-fn)]
    (insert-batch graph data :show-progress? show-progress?)))

(defn search-knn
  "Search k nearest neighbors"
  [graph ^doubles query-vec ^long k]
  (if (or (nil? (.get (.entry-point graph)))
          (zero? (.size (.nodes graph))))
    []
    (let [entry-point (.get (.entry-point graph))
          entry-node (.get (.nodes graph) entry-point)
          entry-level (.level entry-node)
          ef (max k 50)]

      ;; Multi-layer search
      (loop [level entry-level
             nearest [entry-point]]
        (if (< level 0)
          ;; Calculate final distances and return
          (let [distance-fn (.distance-fn graph)
                ^ConcurrentHashMap nodes (.nodes graph)]
            (take k
                  (sort-by :distance
                           (map (fn [^String id]
                                  {:id id
                                   :distance (distance-fn query-vec
                                                          (.vector (.get nodes id)))})
                                nearest))))

          (recur (dec level)
                 (search-layer-ultra graph query-vec nearest
                                     (if (> level 0) 1 ef) level))))))) ; Use ef at all levels

;; ===== Utility Functions =====

(defn graph-info
  "Get graph statistics"
  [graph]
  {:num-elements (.get (.element-count graph))
   :entry-point (.get (.entry-point graph))
   :M (.M graph)
   :ef-construction (.ef-construction graph)})

(defn vector->doubles
  "Convert Clojure vector to double array"
  ^doubles [v]
  (double-array v))

(defn warmup-jvm
  "Warm up JVM for benchmarking"
  []
  (println "Warming up JVM...")
  (let [test-data (repeatedly 100
                              (fn [] [(str (java.util.UUID/randomUUID))
                                      (double-array (repeatedly 128 rand))]))]
    (dotimes [_ 3]
      (let [g (build-index test-data :show-progress? false)]
        (dotimes [_ 100]
          (search-knn g (double-array (repeatedly 128 rand)) 10)))))
  (println "Warmup complete!"))