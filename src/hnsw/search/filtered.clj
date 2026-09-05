(ns hnsw.search.filtered
  "Exact subset search and lazy configurable metadata postings for the managed API."
  (:require [hnsw.ultra-fast :as ultra])
  (:import [hnsw.ultra_fast UltraGraph UltraNode NormalizedDistance]
           [java.util Arrays Comparator PriorityQueue]
           [java.util.concurrent ConcurrentHashMap]))

(defn normalize-filter-keys [filter-keys]
  (when-not (or (set? filter-keys) (sequential? filter-keys))
    (throw (IllegalArgumentException. "filter-keys must be a set or sequence of metadata keys")))
  (set filter-keys))

(defn validate-filter! [criteria filter-keys]
  (when-not (and (map? criteria)
                (seq criteria)
                (every? #(contains? filter-keys %) (keys criteria)))
    (throw (IllegalArgumentException.
            "Filter must be a nonempty map using only this index's configured filter-keys"))))

(defn build-postings [metadata filter-keys]
  (let [identifiers (vec (keys metadata))
        fields (reduce-kv
                 (fn [fields position identifier]
                   (let [attributes (get metadata identifier)]
                     (if (map? attributes)
                       (reduce (fn [fields key]
                                 (if (contains? attributes key)
                                   (update-in fields [key (get attributes key)] (fnil conj []) (int position))
                                   fields)) fields filter-keys)
                       fields))) {} identifiers)]
    {:identifiers identifiers
     :fields (into {} (map (fn [[key values]]
                            [key (into {} (map (fn [[value positions]] [value (int-array positions)]) values))])
                          fields))}))

(def ^:private empty-positions (int-array 0))

(defn- intersect-positions ^ints [^ints left ^ints right]
  (let [left-count (alength left)
        right-count (alength right)
        matches (int-array (min left-count right-count))]
    (loop [left-position 0 right-position 0 matched 0]
      (if (or (= left-position left-count) (= right-position right-count))
        (Arrays/copyOf matches (int matched))
        (let [left-value (aget left left-position)
              right-value (aget right right-position)]
          (cond
            (< left-value right-value) (recur (inc left-position) right-position matched)
            (> left-value right-value) (recur left-position (inc right-position) matched)
            :else (do
                    (aset-int matches matched left-value)
                    (recur (inc left-position) (inc right-position) (inc matched)))))))))

(defn matching-ids [{:keys [identifiers fields]} criteria]
  (let [postings (sort-by #(alength ^ints %)
                         (map (fn [[key value]] (get-in fields [key value] empty-positions)) criteria))
        ^ints positions (reduce (fn [^ints matches ^ints candidates]
                                  (if (zero? (alength matches))
                                    (reduced matches)
                                    (intersect-positions matches candidates)))
                                (first postings) (rest postings))]
    (loop [position 0 matches (transient [])]
      (if (= position (alength positions))
        (persistent! matches)
        (recur (inc position) (conj! matches (nth identifiers (aget positions position))))))))

(deftype ScoredId [^String identifier ^double distance])

(defn- compare-score ^long
  [^double left-distance ^String left-id ^double right-distance ^String right-id]
  (let [order (Double/compare left-distance right-distance)]
    (if (zero? order) (.compareTo left-id right-id) order)))

(def ^:private worst-first
  (reify Comparator
    (compare [_ left right]
      (let [^ScoredId left left
            ^ScoredId right right]
        (int (compare-score (.distance right) (.identifier right)
                            (.distance left) (.identifier left)))))))

(defn search
  "Scan only matching external IDs, resolving live float64 vectors under the API read lock."
  [^UltraGraph graph ^doubles query top-k identifiers]
  (if (empty? identifiers)
    []
    (let [limit (int (min top-k (count identifiers)))
          heap (PriorityQueue. limit ^Comparator worst-first)
          ^NormalizedDistance engine (.normalized-distance graph)
          ^ConcurrentHashMap nodes (.nodes graph)
          distance-fn (.distance-fn graph)
          ^doubles prepared (if (.normalize-vectors graph) (ultra/normalize-vector query) query)]
      (doseq [identifier identifiers]
        (when-let [^UltraNode node (.get nodes identifier)]
          (let [distance (double (if engine
                                   (.distance engine prepared (.vector node))
                                   (distance-fn prepared (.vector node))))
                ^ScoredId worst (.peek heap)]
            (when (or (< (.size heap) limit)
                      (neg? (compare-score distance identifier (.distance worst) (.identifier worst))))
              (when (= (.size heap) limit) (.poll heap))
              (.add heap (ScoredId. identifier distance))))))
      (loop [neighbors ()]
        (if-let [^ScoredId neighbor (.poll heap)]
          (recur (conj neighbors {:id (.identifier neighbor) :distance (.distance neighbor)}))
          (vec neighbors))))))
