(ns hnsw.search.compact
  "Read-optimized immutable HNSW snapshot backed by primitive arrays."
  (:import [hnsw.search CompactSearch CompactSearch$SearchResult]
           [hnsw.ultra_fast CompactSearchEngine UltraGraph UltraNode]
           [java.util HashMap Iterator Set]))

(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

(deftype CompactEngine [^CompactSearch search-index]
  CompactSearchEngine
  (search [_ query k ef-search]
    (let [^CompactSearch$SearchResult result
          (.search search-index query (int k) (int ef-search))
          size (.size result)
          neighbors (object-array size)]
      (dotimes [index size]
        (aset neighbors index
              {:id (.idAt result index)
               :distance (.distanceAt result index)}))
      neighbors)))

(defn- build-level-storage
  [nodes ^HashMap id->index level]
  (let [node-count (count nodes)
        offsets (int-array (inc node-count))]
    (dotimes [index node-count]
      (let [^UltraNode node (nth nodes index)
            degree (if (<= level (.level node))
                     (.size ^Set (aget ^objects (.neighbors node) level))
                     0)]
        (aset-int offsets (inc index) (+ (aget offsets index) degree))))
    (let [neighbors (int-array (aget offsets node-count))]
      (dotimes [index node-count]
        (let [^UltraNode node (nth nodes index)]
          (when (<= level (.level node))
            (loop [^Iterator iterator
                   (.iterator ^Set (aget ^objects (.neighbors node) level))
                   position (aget offsets index)]
              (when (.hasNext iterator)
                (let [neighbor-id ^String (.next iterator)
                      neighbor-index (.get id->index neighbor-id)]
                  (when (nil? neighbor-index)
                    (throw (IllegalStateException.
                            (str "Missing HNSW neighbor " neighbor-id))))
                  (aset-int neighbors position (int neighbor-index))
                  (recur iterator (inc position))))))))
      [offsets neighbors])))

(defn create-search-engine
  [^UltraGraph graph compact-precision]
  (when-not (#{:float32 :float64} compact-precision)
    (throw (IllegalArgumentException.
            "compact-precision must be :float32 or :float64")))
  (let [nodes (vec (sort-by #(.ordinal ^UltraNode %)
                            (vals (into {} (.nodes graph)))))]
    (when (seq nodes)
      (let [node-count (count nodes)
            ^UltraNode first-node (first nodes)
            dimension (alength ^doubles (.vector first-node))
            ids (make-array String node-count)
            vectors (double-array (Math/multiplyExact node-count dimension))
            id->index (HashMap. node-count)
            max-level (long (reduce max 0 (map #(.level ^UltraNode %) nodes)))]
        (dotimes [index node-count]
          (let [^UltraNode node (nth nodes index)
                vector (.vector node)]
            (when-not (= dimension (alength ^doubles vector))
              (throw (IllegalArgumentException. "All indexed vectors must have the same dimension")))
            (aset ids index (.id node))
            (.put id->index (.id node) index)
            (System/arraycopy vector 0 vectors (* index dimension) dimension)))
        (let [offsets-by-level (make-array (class (int-array 0)) (inc max-level))
              neighbors-by-level (make-array (class (int-array 0)) (inc max-level))]
          (dotimes [level (inc max-level)]
            (let [[offsets neighbors]
                  (build-level-storage nodes id->index level)]
              (aset offsets-by-level level offsets)
              (aset neighbors-by-level level neighbors)))
          (let [entry-index (.get id->index (.get (.entry-point graph)))]
            (when (nil? entry-index)
              (throw (IllegalStateException. "HNSW entry point is missing")))
            (CompactEngine.
             (CompactSearch. ids vectors dimension (int entry-index) (int max-level)
                             offsets-by-level neighbors-by-level
                             (= :float32 compact-precision)))))))))
