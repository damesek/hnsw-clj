(ns hnsw.ultra-fast
  "Ultra-fast HNSW implementation with maximum performance optimizations"
  (:import [java.util PriorityQueue Set ArrayList Arrays Collections Random Iterator]
           [java.util.concurrent ConcurrentHashMap Executors Callable Future]
           [java.util.concurrent.atomic AtomicInteger AtomicReference]))

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

(defn cosine-distance-normalized
  "Cosine distance for vectors normalized to unit length."
  ^double [^doubles v1 ^doubles v2]
  (let [len (alength v1)
        chunks (quot len 4)]
    (loop [chunk 0
           dot 0.0]
      (if (< chunk chunks)
        (let [idx (* chunk 4)]
          (recur (inc chunk)
                 (+ dot
                    (* (aget v1 idx) (aget v2 idx))
                    (* (aget v1 (+ idx 1)) (aget v2 (+ idx 1)))
                    (* (aget v1 (+ idx 2)) (aget v2 (+ idx 2)))
                    (* (aget v1 (+ idx 3)) (aget v2 (+ idx 3))))))
        (loop [idx (* chunks 4)
               final-dot dot]
          (if (< idx len)
            (recur (inc idx)
                   (+ final-dot (* (aget v1 idx) (aget v2 idx))))
            (- 1.0 final-dot)))))))

(defn normalize-vector
  "Return a normalized copy of a double vector."
  ^doubles [^doubles vector]
  (let [len (alength vector)
        norm-squared (loop [idx 0
                            total 0.0]
                       (if (< idx len)
                         (let [value (aget vector idx)]
                           (recur (inc idx) (+ total (* value value))))
                         total))
        normalized (double-array len)]
    (if (pos? norm-squared)
      (let [scale (/ 1.0 (Math/sqrt norm-squared))]
        (dotimes [idx len]
          (aset-double normalized idx (* (aget vector idx) scale))))
      (System/arraycopy vector 0 normalized 0 len))
    normalized))

(definterface NormalizedDistance
  (^double distance [^doubles left ^doubles right]))

(definterface CompactSearchEngine
  (^"[Ljava.lang.Object;" search [^doubles query ^long k ^long ef-search]))

(deftype ScalarNormalizedDistance []
  NormalizedDistance
  (distance [_ left right]
    (cosine-distance-normalized left right)))

(defn create-normalized-distance-engine
  "Create the fastest normalized cosine engine available in this JVM."
  []
  (try
    (Class/forName "hnsw.search.VectorDistance")
    ((requiring-resolve 'hnsw.search.vector/create-distance-engine))
    (catch Throwable _
      (ScalarNormalizedDistance.))))

(definterface VisitedState
  (^void begin [^long capacity])
  (^boolean visit [^long ordinal]))

(deftype GenerationVisited [^:unsynchronized-mutable marks
                            ^ints state]
  VisitedState
  (begin [_ capacity]
    (when (< (alength ^ints marks) capacity)
      (set! marks (int-array capacity)))
    (let [generation (aget state 0)]
      (if (= generation Integer/MAX_VALUE)
      (do
          (Arrays/fill ^ints marks 0)
          (aset-int state 0 1))
        (aset-int state 0 (unchecked-inc-int generation)))))
  (visit [_ ordinal]
    (let [index (int ordinal)
          _ (when (>= index (alength ^ints marks))
              (set! marks (Arrays/copyOf
                           ^ints marks
                           (max (inc index) (* 2 (alength ^ints marks))))))
          generation (aget state 0)]
      (if (= (aget ^ints marks index) generation)
        false
        (do
          (aset-int ^ints marks index generation)
          true)))))

(def ^ThreadLocal tl-visited
  (proxy [ThreadLocal] []
    (initialValue [] (GenerationVisited. (int-array 1024) (int-array 1)))))

;; ===== Optimized Node Structure =====

(deftype UltraNode [^String id
                    ^doubles vector
                    ^long level
                    ^long ordinal
                    ^objects neighbors])

(deftype UltraGraph [^ConcurrentHashMap nodes
                     ^AtomicReference entry-point
                     ^long M
                     ^long max-M
                     ^long ef-construction
                     ^long ef-search
                     ^double ml
                     distance-fn
                     ^NormalizedDistance normalized-distance
                     ^AtomicReference search-index
                     ^AtomicInteger element-count
                     ^AtomicInteger next-ordinal
                     ^Random random-state
                     ^boolean normalize-vectors
                     compact-precision])

;; ===== Fast Priority Queue Wrapper =====

(deftype Candidate [^String id ^double distance]
  Comparable
  (compareTo [this other]
    (Double/compare distance (.distance ^Candidate other))))

;; ===== Graph Creation =====

(defn create-ultra-graph
  [& {:keys [M ef-construction ef-search distance-fn seed compact-precision]
      :or {M 16
           ef-construction 200
           ef-search 50
           distance-fn euclidean-distance-ultra
           seed 42
           compact-precision :float64}}]
  (when (< M 2)
    (throw (IllegalArgumentException. "M must be at least 2")))
  (when (< ef-construction 1)
    (throw (IllegalArgumentException. "ef-construction must be positive")))
  (when (< ef-search 1)
    (throw (IllegalArgumentException. "ef-search must be positive")))
  (when-not (#{:float32 :float64} compact-precision)
    (throw (IllegalArgumentException.
            "compact-precision must be :float32 or :float64")))
  (let [normalize-vectors (identical? distance-fn cosine-distance-ultra)
        internal-distance-fn (if normalize-vectors
                               cosine-distance-normalized
                               distance-fn)
        normalized-distance (when normalize-vectors
                              (create-normalized-distance-engine))]
    (UltraGraph. (ConcurrentHashMap.)
                 (AtomicReference. nil)
                 M
                 (* 2 M)
                 (max M ef-construction)
                 ef-search
                 (/ 1.0 (Math/log (double M)))
                 internal-distance-fn
                 normalized-distance
                 (AtomicReference. nil)
                 (AtomicInteger. 0)
                 (AtomicInteger. 0)
                 (Random. (long seed))
                 normalize-vectors
                 compact-precision)))

;; ===== Level Assignment with Cached Random =====

(def ^ThreadLocal tl-random
  (proxy [ThreadLocal] []
    (initialValue [] (Random.))))

(defn assign-level-ultra
  "Assign level to new node"
  (^long [^double ml]
   (assign-level-ultra ml (.get tl-random)))
  (^long [^double ml ^Random random]
   (let [uniform (max Double/MIN_VALUE (.nextDouble random))]
     (long (* ml (- (Math/log uniform)))))))

(defn- graph-distance
  ^double [^UltraGraph graph ^doubles left ^doubles right]
  (if (.normalize-vectors graph)
    (.distance ^NormalizedDistance (.normalized-distance graph) left right)
    (double ((.distance-fn graph) left right))))

(defn- new-neighbor-set
  ^Set []
  (ConcurrentHashMap/newKeySet))

;; ===== Ultra-Fast Search =====

(defn- candidates-nearest-first
  [^PriorityQueue nearest]
  (let [size (.size nearest)
        result (object-array size)]
    (loop [idx (dec size)]
      (when (>= idx 0)
        (aset result idx (.poll nearest))
        (recur (dec idx))))
    (vec result)))

(defn- search-layer-candidates
  [^UltraGraph graph ^doubles query-vec entry-points num-closest level]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        capacity (int (max 1 num-closest))
        ^VisitedState visited (.get tl-visited)
        ^PriorityQueue candidates (PriorityQueue. capacity)
        ^PriorityQueue nearest (PriorityQueue. capacity (Collections/reverseOrder))]
    (.begin visited (.get (.next-ordinal graph)))
    (doseq [^String point entry-points]
      (when-let [^UltraNode node (.get nodes point)]
        (when (.visit visited (.ordinal node))
          (let [dist (graph-distance graph query-vec (.vector node))
                candidate (Candidate. point dist)]
            (.add candidates candidate)
            (.add nearest candidate)))))

    (loop []
      (when-not (.isEmpty candidates)
        (let [^Candidate current (.poll candidates)
              worst-distance (if (.isEmpty nearest)
                               Double/MAX_VALUE
                               (.distance ^Candidate (.peek nearest)))]
          (when-not (and (>= (.size nearest) num-closest)
                         (> (.distance current) worst-distance))
            (when-let [^UltraNode node (.get nodes (.id current))]
              (when (<= level (.level node))
                (let [^Set level-neighbors (aget ^objects (.neighbors node) level)
                      ^Iterator iterator (.iterator level-neighbors)]
                  (while (.hasNext iterator)
                    (let [^String neighbor (.next iterator)]
                      (when-let [^UltraNode neighbor-node (.get nodes neighbor)]
                        (when (.visit visited (.ordinal neighbor-node))
                          (let [dist (graph-distance graph query-vec (.vector neighbor-node))
                                current-worst (if (.isEmpty nearest)
                                                Double/MAX_VALUE
                                                (.distance ^Candidate (.peek nearest)))]
                            (when (or (< (.size nearest) num-closest)
                                      (< dist current-worst))
                              (let [candidate (Candidate. neighbor dist)]
                                (.add candidates candidate)
                                (.add nearest candidate)
                                (when (> (.size nearest) num-closest)
                                  (.poll nearest))))))))))))
            (recur)))))

    (candidates-nearest-first nearest)))

(defn search-layer-ultra
  "Search one HNSW layer and return nearest ids first."
  [graph ^doubles query-vec entry-points num-closest level]
  (let [prepared-query (if (.normalize-vectors ^UltraGraph graph)
                         (normalize-vector query-vec)
                         query-vec)]
    (mapv (fn [^Candidate candidate] (.id candidate))
          (search-layer-candidates graph prepared-query entry-points num-closest level))))

(defn- greedy-search-level
  [^UltraGraph graph ^doubles query-vec ^String entry-id ^long level]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        ^UltraNode entry-node (.get nodes entry-id)]
    (if-not entry-node
      entry-id
      (loop [current-id entry-id
             current-distance (graph-distance graph query-vec (.vector entry-node))]
        (let [^UltraNode current-node (.get nodes current-id)
              ^Set level-neighbors (aget ^objects (.neighbors current-node) level)
              [best-id best-distance]
              (loop [^Iterator iterator (.iterator level-neighbors)
                     best-id current-id
                     best-distance current-distance]
                (if (.hasNext iterator)
                  (let [^String neighbor (.next iterator)
                        ^UltraNode neighbor-node (.get nodes neighbor)
                        candidate-distance (graph-distance
                                            graph query-vec (.vector neighbor-node))]
                    (if (< candidate-distance best-distance)
                      (recur iterator neighbor candidate-distance)
                      (recur iterator best-id best-distance)))
                  [best-id best-distance]))]
          (if (= best-id current-id)
            current-id
            (recur best-id best-distance)))))))

(defn- select-neighbors-heuristic
  [^UltraGraph graph candidates ^long max-neighbors]
  (if (< (count candidates) max-neighbors)
    (vec candidates)
    (let [^ConcurrentHashMap nodes (.nodes graph)
          ^ArrayList selected (ArrayList. (int max-neighbors))]
      (doseq [^Candidate candidate candidates
              :while (< (.size selected) max-neighbors)]
        (let [^UltraNode candidate-node (.get nodes (.id candidate))]
          (when candidate-node
            (let [good? (loop [^Iterator iterator (.iterator selected)]
                          (if (.hasNext iterator)
                            (let [^Candidate accepted (.next iterator)
                                  ^UltraNode accepted-node (.get nodes (.id accepted))
                                  inter-distance (graph-distance
                                                  graph
                                                  (.vector candidate-node)
                                                  (.vector accepted-node))]
                              (if (< inter-distance (.distance candidate))
                                false
                                (recur iterator)))
                            true))]
              (when good?
                (.add selected candidate))))))
      (vec selected))))

(defn- add-reverse-connection!
  [^UltraGraph graph ^UltraNode target-node ^String source-id
   level max-connections]
  (locking target-node
    (let [^Set connections (aget ^objects (.neighbors target-node) level)]
      (when-not (.contains connections source-id)
        (if (< (.size connections) max-connections)
          (.add connections source-id)
          (let [^ConcurrentHashMap nodes (.nodes graph)
                ^UltraNode source-node (.get nodes source-id)
                target-vector (.vector target-node)
                source-distance (graph-distance graph target-vector (.vector source-node))
                furthest-distance
                (loop [^Iterator iterator (.iterator connections)
                       furthest Double/NEGATIVE_INFINITY]
                  (if (.hasNext iterator)
                    (let [^String neighbor (.next iterator)
                          ^UltraNode neighbor-node (.get nodes neighbor)]
                      (recur iterator
                             (max furthest
                                  (graph-distance graph
                                                  target-vector
                                                  (.vector neighbor-node)))))
                    furthest))]
            (when (< source-distance furthest-distance)
              (.add connections source-id)
              (prune-connections-ultra
               graph (.id target-node) level max-connections))))))))

(defn- update-entry-point!
  [^UltraGraph graph ^String node-id ^long node-level]
  (let [^AtomicReference entry-point (.entry-point graph)
        ^ConcurrentHashMap nodes (.nodes graph)]
    (loop []
      (let [current-id (.get entry-point)]
        (if (nil? current-id)
          (when-not (.compareAndSet entry-point nil node-id)
            (recur))
          (let [^UltraNode current-node (.get nodes current-id)]
            (when (and current-node (> node-level (.level current-node)))
              (when-not (.compareAndSet entry-point current-id node-id)
                (recur)))))))))

(defn- insert-single-at-level
  [^UltraGraph graph ^String id ^doubles vector ^long level]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        stored-vector (if (.normalize-vectors graph)
                        (normalize-vector vector)
                        vector)
        ordinal (.getAndIncrement (.next-ordinal graph))
        neighbors (make-array Object (inc level))]
    (dotimes [layer (inc level)]
      (aset neighbors layer (new-neighbor-set)))
    (let [new-node (UltraNode. id stored-vector level ordinal neighbors)]
      (when (.putIfAbsent nodes id new-node)
        (throw (IllegalArgumentException. (str "Node with id " id " already exists"))))
      (.set (.search-index graph) nil)
      (.incrementAndGet (.element-count graph))
      (let [^AtomicReference entry-reference (.entry-point graph)]
        (if (.compareAndSet entry-reference nil id)
          graph
          (let [entry-point (.get entry-reference)
                ^UltraNode entry-node (.get nodes entry-point)
                entry-level (.level entry-node)
                nearest-entry
                (loop [layer entry-level
                       nearest entry-point]
                  (if (> layer level)
                    (recur (dec layer)
                           (greedy-search-level graph stored-vector nearest layer))
                    nearest))]
            (loop [layer (min level entry-level)
                   nearest nearest-entry]
              (when (>= layer 0)
                (let [candidates (search-layer-candidates
                                  graph stored-vector [nearest] (.ef-construction graph) layer)
                      selected (select-neighbors-heuristic graph candidates (.M graph))
                      max-connections (if (zero? layer) (.max-M graph) (.M graph))
                      next-nearest (if (seq candidates)
                                     (.id ^Candidate (first candidates))
                                     nearest)]
                  (doseq [^Candidate candidate selected]
                    (let [neighbor (.id candidate)]
                      (when-not (= id neighbor)
                        (when-let [^UltraNode neighbor-node (.get nodes neighbor)]
                          (add-reverse-connection!
                           graph new-node neighbor layer max-connections)
                          (add-reverse-connection!
                           graph neighbor-node id layer max-connections)))))
                  (recur (dec layer) next-nearest))))
            (update-entry-point! graph id level)
            graph))))))

;; ===== Single Element Insertion =====

(defn insert-single
  "Optimized single element insertion"
  [^UltraGraph graph ^String id ^doubles vector]
  (insert-single-at-level
   graph id vector (assign-level-ultra (.ml graph) (.random-state graph))))

;; ===== Connection Pruning =====

(defn prune-connections-ultra
  "Prune an overflowing link list with the HNSW diversity heuristic."
  [^UltraGraph graph ^String node-id ^long level ^long max-conns]
  (let [^ConcurrentHashMap nodes (.nodes graph)
        ^UltraNode node (.get nodes node-id)]
    (when node
      (locking node
        (let [^Set connections (aget ^objects (.neighbors node) level)]
          (when (> (.size connections) max-conns)
            (let [node-vector (.vector node)
                  candidates (->> connections
                                  (map (fn [^String neighbor]
                                         (let [^UltraNode neighbor-node (.get nodes neighbor)]
                                           (Candidate.
                                            neighbor
                                            (graph-distance graph
                                                            node-vector
                                                            (.vector neighbor-node))))))
                                  sort
                                  vec)
                  selected (select-neighbors-heuristic graph candidates max-conns)]
              (.clear connections)
              (doseq [^Candidate candidate selected]
                (.add connections (.id candidate))))))))))

;; ===== Batch Insertion =====

(defn- indexed-element?
  [element]
  (and (sequential? element)
       (= 2 (count element))
       (let [vector (second element)]
         (or (sequential? vector)
             (instance? (Class/forName "[D") vector)))))

(defn- prepare-elements
  [elements]
  (mapv (fn [index element]
          (if (indexed-element? element)
            element
            [(str index) element]))
        (range)
        elements))

(defn insert-batch
  "Build one HNSW graph concurrently with dynamic work distribution."
  [^UltraGraph graph elements & {:keys [show-progress? num-threads batch-size]
                     :or {show-progress? true
                          num-threads (.availableProcessors (Runtime/getRuntime))
                          batch-size 4}}]
  (when-not (pos? (long num-threads))
    (throw (IllegalArgumentException. "num-threads must be positive")))
  (when-not (pos? (long batch-size))
    (throw (IllegalArgumentException. "batch-size must be positive")))
  (let [data (prepare-elements elements)
        total (count data)
        worker-count (int (max 1 (min num-threads total)))
        levels (int-array total)]
    (dotimes [idx total]
      (aset-int levels idx
                (int (assign-level-ultra (.ml graph) (.random-state graph)))))
    (when show-progress?
      (println (format "Inserting %d elements with %d thread%s..."
                       total worker-count (if (= worker-count 1) "" "s"))))
    (when (pos? total)
      (letfn [(insert-index! [idx]
                (let [[id vector] (nth data idx)
                      vec-array (if (instance? (Class/forName "[D") vector)
                                  vector
                                  (double-array vector))]
                  (insert-single-at-level
                   graph (str id) vec-array (aget levels idx))))
              (report-progress! [done]
                (when (and show-progress?
                           (or (= done total) (zero? (mod done 500))))
                  (println (format "Progress: %d/%d (%.1f%%)"
                                   done total (* 100.0 (/ done total))))))
              (process-range! [start end completed]
                (loop [idx start]
                  (when (< idx end)
                    (insert-index! idx)
                    (report-progress! (.incrementAndGet ^AtomicInteger completed))
                    (recur (inc idx)))))]
        (insert-index! 0)
        (report-progress! 1)
        (if (or (= worker-count 1) (= total 1))
          (dotimes [offset (dec total)]
            (let [idx (inc offset)]
              (insert-index! idx)
              (report-progress! (inc idx))))
          (let [^AtomicInteger cursor (AtomicInteger. 1)
                ^AtomicInteger completed (AtomicInteger. 1)
                executor (Executors/newFixedThreadPool worker-count)
                tasks (mapv
                       (fn [_]
                         (reify Callable
                           (call [_]
                             (loop []
                               (let [start (.getAndAdd cursor batch-size)]
                                 (when (< start total)
                                   (process-range!
                                    start (min total (+ start batch-size)) completed)
                                   (recur))))
                             nil)))
                       (range worker-count))]
            (try
              (let [futures (.invokeAll executor tasks)]
                (doseq [^Future future futures]
                  (.get future)))
              (finally
                (.shutdown executor)))))))
    graph))

;; ===== Public API =====

(defn optimize-search!
  "Build an immutable primitive-array search snapshot when the Vector API kernel is available."
  [^UltraGraph graph]
  (if (or (not (.normalize-vectors graph))
          (zero? (.size (.nodes graph))))
    graph
    (try
      (Class/forName "hnsw.search.CompactSearch")
      (when-let [engine ((requiring-resolve
                          'hnsw.search.compact/create-search-engine)
                         graph (.compact-precision graph))]
        (.set (.search-index graph) engine))
      graph
      (catch ClassNotFoundException _
        graph)
      (catch LinkageError _
        graph))))

(defn- compact-search-engine
  [^UltraGraph graph]
  (or (.get (.search-index graph))
      (do
        (optimize-search! graph)
        (.get (.search-index graph)))))

(defn build-index
  "Build HNSW index from data"
  [data & {:keys [M ef-construction ef-search distance-fn show-progress? seed
                  num-threads batch-size compact-precision]
           :or {M 16
                ef-construction 200
                ef-search 50
                distance-fn cosine-distance-ultra
                show-progress? true
                seed 42
                num-threads (.availableProcessors (Runtime/getRuntime))
                batch-size 4
                compact-precision :float64}}]
  (let [graph (create-ultra-graph :M M
                                  :ef-construction ef-construction
                                  :ef-search ef-search
                                  :distance-fn distance-fn
                                  :seed seed
                                  :compact-precision compact-precision)]
    (insert-batch graph data
                  :show-progress? show-progress?
                  :num-threads num-threads
                  :batch-size batch-size)
    (optimize-search! graph)))

(defn search-knn
  "Search k nearest neighbors"
  ([^UltraGraph graph query-vec ^long k]
   (search-knn graph query-vec k (.ef-search graph)))
  ([^UltraGraph graph query ^long k ^long ef-search]
   (when-not (pos? ef-search)
     (throw (IllegalArgumentException. "ef-search must be positive")))
   (let [^doubles query-vec (if (instance? (Class/forName "[D") query)
                              query
                              (double-array query))]
     (if (or (not (pos? k))
             (nil? (.get (.entry-point graph)))
             (zero? (.size (.nodes graph))))
       []
       (if-let [^CompactSearchEngine engine (compact-search-engine graph)]
         (vec (.search engine query-vec k (max k ef-search)))
         (let [prepared-query (if (.normalize-vectors graph)
                                (normalize-vector query-vec)
                                query-vec)
               entry-point (.get (.entry-point graph))
               ^UltraNode entry-node (.get (.nodes graph) entry-point)
               entry-level (.level entry-node)
               ef (max k ef-search)
               base-entry (loop [level entry-level
                                 nearest entry-point]
                            (if (pos? level)
                              (recur (dec level)
                                     (greedy-search-level graph prepared-query nearest level))
                              nearest))
               candidates (search-layer-candidates graph prepared-query [base-entry] ef 0)]
           (mapv (fn [^Candidate candidate]
                   {:id (.id candidate)
                    :distance (.distance candidate)})
                 (take k candidates))))))))

(defn search-knn-batch
  "Search many queries in parallel while preserving input order."
  [^UltraGraph graph queries k
   & {:keys [ef-search num-threads]
      :or {ef-search (.ef-search graph)
           num-threads (.availableProcessors (Runtime/getRuntime))}}]
  (when-not (pos? (long num-threads))
    (throw (IllegalArgumentException. "num-threads must be positive")))
  (let [queries (vec queries)
        total (count queries)
        worker-count (int (max 1 (min num-threads total)))
        results (object-array total)]
    (if (or (zero? total) (= worker-count 1))
      (dotimes [index total]
        (aset results index
              (search-knn graph (nth queries index) k ef-search)))
      (let [chunk-size (int (Math/ceil (/ (double total) worker-count)))
            executor (Executors/newFixedThreadPool worker-count)
            tasks (mapv
                   (fn [start]
                     (let [end (min total (+ start chunk-size))]
                       (reify Callable
                         (call [_]
                           (loop [index start]
                             (when (< index end)
                               (aset results index
                                     (search-knn graph (nth queries index) k ef-search))
                               (recur (inc index))))
                           nil))))
                   (range 0 total chunk-size))]
        (try
          (doseq [^Future future (.invokeAll executor tasks)]
            (.get future))
          (finally
            (.shutdown executor)))))
    (vec results)))

;; ===== Utility Functions =====

(defn graph-info
  "Get graph statistics"
  [graph]
  {:num-elements (.get (.element-count graph))
   :entry-point (.get (.entry-point graph))
   :M (.M graph)
   :ef-construction (.ef-construction graph)
   :ef-search (.ef-search graph)
   :compact-precision (.compact-precision graph)
   :vector-api? (and (.normalize-vectors graph)
                     (not (instance? ScalarNormalizedDistance
                                     (.normalized-distance graph))))
   :compact-search? (some? (.get (.search-index graph)))
   :normalized-cosine? (.normalize-vectors graph)
   :max-level (if-let [entry-point (.get (.entry-point graph))]
                (if-let [^UltraNode node (.get (.nodes graph) entry-point)]
                  (.level node)
                  -1)
                -1)})

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
