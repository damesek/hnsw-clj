(ns hnsw.wip.vector
  "HNSW implementation with REAL Java Vector API (SIMD) support"
  (:import [java.util PriorityQueue HashMap HashSet ArrayList Collections Random]
           [java.util.concurrent ConcurrentHashMap ThreadLocalRandom ForkJoinPool]
           [java.util.concurrent.atomic AtomicInteger AtomicReference]
           [jdk.incubator.vector FloatVector VectorSpecies VectorOperators VectorMask]
           [java.nio ByteOrder]))

;; ===== Performance Configuration =====
(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

;; ===== Java Vector API Configuration =====
(def ^VectorSpecies SPECIES FloatVector/SPECIES_PREFERRED)
(def VECTOR_LENGTH (.length SPECIES))

(println (format "🚀 Java Vector API enabled! SIMD width: %d floats" VECTOR_LENGTH))

;; ===== SIMD Distance Functions using REAL Vector API =====

(defn ^float simd-euclidean-distance
  "Euclidean distance using Java Vector API"
  [^floats v1 ^floats v2]
  (let [len (alength v1)
        upper-bound (.loopBound SPECIES len)]
    (loop [i 0
           sum-vec (FloatVector/zero SPECIES)]
      (if (< i upper-bound)
        ;; SIMD processing
        (let [va (FloatVector/fromArray SPECIES v1 i)
              vb (FloatVector/fromArray SPECIES v2 i)
              diff (.sub va vb)]
          (recur (+ i VECTOR_LENGTH)
                 (.add sum-vec (.mul diff diff))))
        ;; Handle remainder and return result
        (let [sum (.reduceLanes sum-vec VectorOperators/ADD)]
          (if (< i len)
            ;; Process remaining elements
            (loop [j i
                   acc sum]
              (if (< j len)
                (let [d (- (aget v1 j) (aget v2 j))]
                  (recur (inc j) (+ acc (* d d))))
                (Math/sqrt acc)))
            (Math/sqrt sum)))))))

(defn ^float simd-cosine-distance
  "Cosine distance using Java Vector API - REAL SIMD!"
  [^floats v1 ^floats v2]
  (let [len (alength v1)
        upper-bound (.loopBound SPECIES len)]
    (loop [i 0
           dot-vec (FloatVector/zero SPECIES)
           norm1-vec (FloatVector/zero SPECIES)
           norm2-vec (FloatVector/zero SPECIES)]
      (if (< i upper-bound)
        ;; SIMD processing with Vector API
        (let [va (FloatVector/fromArray SPECIES v1 i)
              vb (FloatVector/fromArray SPECIES v2 i)]
          (recur (+ i VECTOR_LENGTH)
                 (.add dot-vec (.mul va vb))
                 (.add norm1-vec (.mul va va))
                 (.add norm2-vec (.mul vb vb))))
        ;; Reduce and handle remainder
        (let [dot (.reduceLanes dot-vec VectorOperators/ADD)
              norm1 (.reduceLanes norm1-vec VectorOperators/ADD)
              norm2 (.reduceLanes norm2-vec VectorOperators/ADD)]
          ;; Process remaining elements
          (if (< i len)
            (loop [j i
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
                (if (and (> n1 0.0) (> n2 0.0))
                  (- 1.0 (/ d (* (Math/sqrt n1) (Math/sqrt n2))))
                  1.0)))
            (if (and (> norm1 0.0) (> norm2 0.0))
              (- 1.0 (/ dot (* (Math/sqrt norm1) (Math/sqrt norm2))))
              1.0)))))))

;; ===== SIMD Batch Distance Calculation =====

(defn simd-batch-distances
  "Calculate distances to multiple vectors using SIMD"
  [^floats query vectors]
  (let [results (make-array Float/TYPE (count vectors))]
    (doall
     (map-indexed
      (fn [idx [_ ^floats v]]
        (aset results idx (simd-cosine-distance query v)))
      vectors))
    results))

;; ===== Node and Graph structures (reuse from ultra-fast) =====

(deftype VectorNode [^String id
                     ^floats vector
                     ^long level
                     ^objects neighbors])

(deftype VectorGraph [^ConcurrentHashMap nodes
                      ^AtomicReference entry-point
                      ^long M
                      ^long max-M
                      ^long ef-construction
                      ^double ml
                      distance-fn
                      ^AtomicInteger element-count])

;; ===== Graph Creation =====

(defn create-vector-graph
  [& {:keys [M ef-construction distance-fn]
      :or {M 16
           ef-construction 200
           distance-fn simd-cosine-distance}}]
  (VectorGraph. (ConcurrentHashMap.)
                (AtomicReference. nil)
                M
                (* 2 M)
                ef-construction
                (/ 1.0 (Math/log 2.0))
                distance-fn
                (AtomicInteger. 0)))

;; ===== Level Assignment =====

(defn assign-level
  ^long [^double ml]
  (long (* ml (- (Math/log (.nextDouble (ThreadLocalRandom/current)))))))

;; ===== Search Layer with SIMD =====

(defn search-layer-simd
  [^VectorGraph graph ^floats query-vec entry-points num-closest level]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        distance-fn (.distance-fn graph)
        visited (HashSet.)
        candidates (PriorityQueue.)
        w (PriorityQueue. (Collections/reverseOrder))]

    ;; Initialize with entry points
    (doseq [point entry-points]
      (when-let [^VectorNode node (.get nodes point)]
        (let [dist (distance-fn query-vec (.vector node))]
          (.add visited point)
          (.add candidates [dist point])
          (.add w [dist point]))))

    ;; Search
    (while (not (.isEmpty candidates))
      (let [[current-dist current-id] (.poll candidates)]
        (when (< current-dist (if (.isEmpty w)
                                Double/MAX_VALUE
                                (first (.peek w))))
          (when-let [^VectorNode node (.get nodes current-id)]
            (let [^HashSet neighbors (aget ^objects (.neighbors node) level)]
              (doseq [neighbor neighbors]
                (when-not (.contains visited neighbor)
                  (.add visited neighbor)
                  (when-let [^VectorNode n-node (.get nodes neighbor)]
                    (let [dist (distance-fn query-vec (.vector n-node))]
                      (when (or (< (.size w) num-closest)
                                (< dist (first (.peek w))))
                        (.add candidates [dist neighbor])
                        (.add w [dist neighbor])
                        (when (> (.size w) num-closest)
                          (.poll w))))))))))))

    ;; Return sorted results
    (mapv second (sort w))))

;; ===== Insert with SIMD optimization =====

(defn insert-vector
  [^VectorGraph graph ^String id ^floats vector]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        level (assign-level (.ml graph))
        neighbors (make-array Object (inc level))]

    ;; Initialize neighbor sets
    (dotimes [l (inc level)]
      (aset neighbors l (HashSet.)))

    (let [new-node (VectorNode. id vector level neighbors)]

      ;; Add to graph
      (.put nodes id new-node)
      (.incrementAndGet (.element-count graph))

      ;; Handle entry point
      (when (nil? (.get (.entry-point graph)))
        (.set (.entry-point graph) id))

      ;; Connect to graph
      (when (> (.size nodes) 1)
        (let [entry-point (.get (.entry-point graph))
              ^VectorNode entry-node (.get nodes entry-point)]
          (when entry-node
            (let [M (.M graph)
                  ef-construction (.ef-construction graph)]

              ;; Find neighbors at each level
              (loop [lc (min level (.level entry-node))
                     nearest [entry-point]]
                (when (>= lc 0)
                  (let [candidates (search-layer-simd graph vector nearest
                                                      (if (> lc 0) 1 ef-construction) lc)
                        m (if (= lc 0) (.max-M graph) M)]

                    ;; Connect bidirectionally
                    (doseq [neighbor (take m candidates)]
                      (when-let [^VectorNode n-node (.get nodes neighbor)]
                        (.add ^HashSet (aget neighbors lc) neighbor)
                        (.add ^HashSet (aget ^objects (.neighbors n-node) lc) id)))

                    (recur (dec lc) candidates)))))))))

    graph))

;; ===== Batch Insertion =====

(defn insert-batch-vector
  [^VectorGraph graph elements & {:keys [show-progress?]
                                  :or {show-progress? true}}]
  (let [total (count elements)]
    (when show-progress?
      (println (format "Inserting %d elements with SIMD optimization..." total)))

    (loop [idx 0
           remaining elements]
      (when (seq remaining)
        (let [[id vector] (first remaining)
              vec-array (if (instance? (Class/forName "[F") vector)
                          vector
                          (float-array vector))]

          (when (and show-progress? (zero? (mod idx 500)))
            (println (format "Progress: %d/%d (%.1f%%)"
                             idx total (* 100.0 (/ idx total)))))

          (insert-vector graph id vec-array)
          (recur (inc idx) (rest remaining)))))

    graph))

;; ===== Public API =====

(defn build-vector-index
  "Build HNSW index with REAL Java Vector API optimization"
  [data & {:keys [M ef-construction show-progress?]
           :or {M 16
                ef-construction 200
                show-progress? true}}]
  (println "\n🚀 Building index with Java Vector API (SIMD)...")
  (println (format "   SIMD width: %d floats" VECTOR_LENGTH))

  (let [graph (create-vector-graph :M M
                                   :ef-construction ef-construction
                                   :distance-fn simd-cosine-distance)]
    (insert-batch-vector graph data :show-progress? show-progress?)))

(defn search-vector-knn
  "Search k nearest neighbors using SIMD"
  [^VectorGraph graph ^floats query-vec ^long k]
  (if (or (nil? (.get (.entry-point graph)))
          (zero? (.size (.nodes graph))))
    []
    (let [entry-point (.get (.entry-point graph))
          ^VectorNode entry-node (.get (.nodes graph) entry-point)
          entry-level (.level entry-node)
          ef (max k 50)]

      ;; Multi-layer search
      (loop [level entry-level
             nearest [entry-point]]
        (if (< level 0)
          ;; Return final results
          (let [distance-fn (.distance-fn graph)
                ^ConcurrentHashMap nodes (.nodes graph)]
            (take k
                  (sort-by :distance
                           (map (fn [id]
                                  {:id id
                                   :distance (distance-fn query-vec
                                                          (.vector ^VectorNode (.get nodes id)))})
                                nearest))))

          (recur (dec level)
                 (search-layer-simd graph query-vec nearest
                                    (if (> level 0) 1 ef) level)))))))

(defn vector-graph-info
  [^VectorGraph graph]
  {:num-elements (.get (.element-count graph))
   :entry-point (.get (.entry-point graph))
   :M (.M graph)
   :ef-construction (.ef-construction graph)
   :simd-width VECTOR_LENGTH
   :simd-enabled true})
