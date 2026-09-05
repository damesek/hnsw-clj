(ns hnsw.api
  "Managed, thread-safe HNSW indexes. Other namespaces are implementation details."
  (:require [clojure.edn :as edn]
            [hnsw.ultra-fast :as ultra]
            [hnsw.search.filtered :as filtered]
            [hnsw.helper.index-io :as index-io])
  (:import [java.lang.ref WeakReference]
           [java.util HashSet]
           [java.util.concurrent.locks ReentrantReadWriteLock Lock]))

(declare close!)

(defrecord HNSWIndex [state ^ReentrantReadWriteLock lock]
  java.io.Closeable
  (close [this] (close! this)))

(defn- access [index write? operation]
  (let [^Lock lock (if write? (.writeLock ^ReentrantReadWriteLock (:lock index))
                             (.readLock ^ReentrantReadWriteLock (:lock index)))]
    (.lock lock)
    (try
      (let [state @(:state index)]
        (when (:closed? state)
          (throw (IllegalStateException. "Index is closed")))
        (operation state))
      (finally (.unlock lock)))))

(defn- positive-integer! [label value]
  (when-not (and (integer? value) (<= 1 value Integer/MAX_VALUE))
    (throw (IllegalArgumentException. (str label " must be a positive integer")))))

(def ^:private ^Class double-array-class (Class/forName "[D"))
(def ^:private ^Class float-array-class (Class/forName "[F"))

(defn- copy-vector ^doubles [values]
  (cond
    (.isInstance double-array-class values) (aclone ^doubles values)
    (.isInstance float-array-class values)
    (let [^floats source values
          length (alength source)
          copy (double-array length)]
      (dotimes [position length]
        (aset-double copy position (double (aget source position))))
      copy)
    :else (double-array values)))

(defn- checked-vector [values dimensions metric]
  (when-not (or (sequential? values)
                (instance? java.util.List values)
                (and (some? values) (.isArray (class values))))
    (throw (IllegalArgumentException. "A vector must be a sequence or array")))
  (let [^doubles vector (copy-vector values)
        length (alength vector)]
    (when (or (zero? length) (and dimensions (not= dimensions length)))
      (throw (IllegalArgumentException. "Vector dimension mismatch or empty vector")))
    (loop [position 0 norm 0.0]
      (if (< position length)
        (let [value (aget vector position)]
          (when-not (Double/isFinite value)
            (throw (IllegalArgumentException. "Vectors must contain only finite numbers")))
          (recur (inc position) (+ norm (* value value))))
        (when (or (not (Double/isFinite norm))
                  (and (= metric :cosine) (zero? norm)))
          (throw (IllegalArgumentException. "Vector norm is zero or outside supported numeric range")))))
    vector))

(defn- entries [data dimensions metric existing]
  (loop [remaining (seq data) result (transient []) seen (HashSet.) dimension dimensions ordinal 0]
    (if-not remaining
      [(persistent! result) dimension]
      (let [item (first remaining)
            keyed? (and (sequential? item) (<= 2 (count item) 3)
                        (not (number? (second item))))
            [identifier values metadata] (if keyed? item [(str ordinal) item])
            identifier (if (nil? identifier)
                         (throw (IllegalArgumentException. "IDs cannot be nil"))
                         (str identifier))
            vector (checked-vector values dimension metric)]
        (when (or (contains? existing identifier) (not (.add ^HashSet seen identifier)))
          (throw (IllegalArgumentException. (str "Duplicate ID: " identifier))))
        (recur (next remaining) (conj! result [identifier vector metadata])
               seen (or dimension (alength vector)) (inc ordinal))))))

(defn- build [data options]
  (apply ultra/build-index
         (mapv (fn [[identifier vector]] [identifier vector]) data)
         (mapcat identity
                 (assoc (dissoc options :metric :dimensions :filter-keys)
                        :distance-fn (case (:metric options)
                                       :cosine ultra/cosine-distance-ultra
                                       :euclidean ultra/euclidean-distance-ultra)))))

(defn index
  "Create an index from vectors or [id vector metadata?] entries. Copies input arrays."
  [data & {:as supplied}]
  (let [defaults {:metric :euclidean :M 16 :ef-construction 200 :ef-search 50
                  :seed 42 :num-threads (.availableProcessors (Runtime/getRuntime))
                  :batch-size 4 :compact-precision :float64
                  :construction-precision :float64 :show-progress? false :filter-keys #{}}
        options (update (merge defaults supplied) :filter-keys filtered/normalize-filter-keys)]
    (when-let [unknown (seq (remove (conj (set (keys defaults)) :dimensions) (keys supplied)))]
      (throw (IllegalArgumentException. (str "Unknown options: " unknown))))
    (when-not (#{:cosine :euclidean} (:metric options))
      (throw (IllegalArgumentException. "metric must be :cosine or :euclidean")))
    (doseq [key [:M :ef-construction :ef-search :num-threads :batch-size]]
      (positive-integer! key (get options key)))
    (when (:dimensions options) (positive-integer! :dimensions (:dimensions options)))
    (let [[data dimensions] (entries data (:dimensions options) (:metric options) {})]
      (->HNSWIndex (atom {:graph (build data options) :options options :dimensions dimensions
                         :metadata (into {} (map (fn [[identifier _ metadata]] [identifier metadata]) data))})
                   (ReentrantReadWriteLock. true)))))

(defn- results [state neighbors]
  (reduce-kv (fn [result position neighbor]
               (if-some [metadata (get (:metadata state) (:id neighbor))]
                 (assoc result position (assoc neighbor :metadata metadata))
                 result))
             neighbors neighbors))

(defn search
  "Return up to k neighbors, ordered by distance. ef-search trades recall for work."
  ([index query k] (search index query k nil))
  ([index query k ef-search]
   (access index false
     (fn [{:keys [graph dimensions options] :as state}]
       (positive-integer! :k k)
       (let [ef (or ef-search (:ef-search options))
             query (checked-vector query dimensions (:metric options))]
         (positive-integer! :ef-search ef)
         (results state (ultra/search-knn graph query k ef)))))))

(defn search-batch
  "Search concurrently, preserving query order."
  [index queries k & {:keys [ef-search num-threads]
                      :or {num-threads (.availableProcessors (Runtime/getRuntime))}}]
  (access index false
    (fn [{:keys [graph dimensions options] :as state}]
      (positive-integer! :k k)
      (positive-integer! :num-threads num-threads)
      (let [ef (or ef-search (:ef-search options))
            queries (mapv #(checked-vector % dimensions (:metric options)) queries)]
        (positive-integer! :ef-search ef)
        (mapv #(results state %)
              (ultra/search-knn-batch graph queries k :ef-search ef :num-threads num-threads))))))

(defn- filter-postings [index metadata filter-keys]
  (let [state (:state index)
        current? (fn [cache]
                   (and cache (identical? metadata (.get ^WeakReference (:source cache)))))]
    (if-let [cache (let [cache (:metadata-filter-cache @state)] (when (current? cache) cache))]
      cache
      (locking state
        (let [cache (:metadata-filter-cache @state)]
          (if (current? cache)
            cache
            (let [cache (assoc (filtered/build-postings metadata filter-keys) :source (WeakReference. metadata))]
              (swap! state assoc :metadata-filter-cache cache)
              cache)))))))

(defn search-filtered
  "Exact top-k matching all metadata criteria on configured filter-keys. Does not use ef-search.
   Separate from search/search-batch; postings are initialized lazily on first use."
  [index query top-k criteria]
  (access index false
    (fn [{:keys [graph dimensions options metadata] :as state}]
      (positive-integer! :k top-k)
      (filtered/validate-filter! criteria (:filter-keys options))
      (let [query (checked-vector query dimensions (:metric options))
            identifiers (filtered/matching-ids (filter-postings index metadata (:filter-keys options)) criteria)]
        (results state (filtered/search graph query top-k identifiers))))))

(defn info [index]
  (access index false
    (fn [state] (assoc (ultra/graph-info (:graph state))
                      :dimensions (:dimensions state) :filter-keys (get-in state [:options :filter-keys])))))

(defn size [index]
  (access index false #(count (:metadata %))))

(defn add-vectors!
  "Validate the entire batch before inserting. Blocks searches during graph mutation."
  [index data]
  (access index true
    (fn [{:keys [graph dimensions options metadata] :as state}]
      (let [[data dimensions] (entries data dimensions (:metric options) metadata)]
        (apply ultra/insert-batch graph (mapv #(subvec % 0 2) data)
               (mapcat identity (select-keys options [:num-threads :batch-size :show-progress?])))
        (ultra/optimize-search! graph)
        (reset! (:state index)
                (assoc state :dimensions dimensions
                       :metadata (into metadata (map (fn [[identifier _ value]] [identifier value]) data)))))))
  index)

(defn add-vector!
  ([index identifier vector] (add-vector! index identifier vector nil))
  ([index identifier vector metadata]
   (add-vectors! index [[identifier vector metadata]])))

(defn update-vector!
  "Repair the old neighborhood and reinsert locally, retaining the node's level and slot."
  ([index identifier vector]
   (access index true
     (fn [state]
       (update-vector! index identifier vector (get (:metadata state) (str identifier))))))
  ([index identifier vector metadata]
   (access index true
     (fn [state]
       (let [identifier (str identifier)]
         (when-not (contains? (:metadata state) identifier)
           (throw (IllegalArgumentException. (str "Unknown ID: " identifier))))
         (let [vector (checked-vector vector (:dimensions state) (get-in state [:options :metric]))
               graph (:graph state)]
           (ultra/replace-node! graph identifier vector)
           (reset! (:state index) (assoc-in state [:metadata identifier] metadata))
           (ultra/optimize-search! graph)))))
   index))

(defn delete-vectors!
  "Remove nodes with local edge repair and one snapshot refresh; unknown IDs are ignored."
  [index identifiers]
  (access index true
    (fn [state]
      (let [deleted (set (map str identifiers))]
        (when (some #(contains? (:metadata state) %) deleted)
          (ultra/delete-nodes! (:graph state) (sort deleted))
          (reset! (:state index) (update state :metadata #(apply dissoc % deleted)))
          (ultra/optimize-search! (:graph state))))))
  index)

(defn delete-vector! [index identifier]
  (delete-vectors! index [identifier]))

(defn clear!
  "Remove all vectors, retaining options and dimensionality."
  [index]
  (access index true
    (fn [state]
      (reset! (:state index) (assoc state :graph (build [] (:options state)) :metadata {}))))
  index)

(defn close!
  "Release references owned by the index. Idempotent; subsequent operations fail."
  [index]
  (let [^Lock lock (.writeLock ^ReentrantReadWriteLock (:lock index))]
    (.lock lock)
    (try (reset! (:state index) {:closed? true})
         (finally (.unlock lock))))
  nil)

(defn save-index
  "Atomically save topology, options and EDN metadata. Do not mutate metadata externally."
  [index filepath]
  (access index false
    (fn [state]
      (let [data {:format :hnsw-clj :version 1 :options (:options state)
                  :dimensions (:dimensions state) :metadata (:metadata state)
                  :graph (index-io/index-data (:graph state))}]
        (binding [*print-length* nil *print-level* nil]
          (edn/read-string (pr-str data)))
        (index-io/write-data! filepath data))))
  index)

(defn load-index
  "Load a trusted managed index. Optional :filter-keys overrides the saved filter configuration."
  [filepath & {:as supplied}]
  (when-let [unknown (seq (remove #{:filter-keys} (keys supplied)))]
    (throw (IllegalArgumentException. (str "Unknown load options: " unknown))))
  (let [data (edn/read-string (slurp filepath))]
    (when-not (and (= :hnsw-clj (:format data)) (= 1 (:version data)))
      (throw (IllegalArgumentException. "Unsupported HNSW file format/version")))
    (let [options (assoc (dissoc (:options data) :construction-codec :pq-subvector-dim
                                :pq-training-samples :pq-iterations)
                         :filter-keys (filtered/normalize-filter-keys
                                        (get supplied :filter-keys (get (:options data) :filter-keys #{}))))]
      (->HNSWIndex (atom {:graph (index-io/from-data (:graph data) nil)
                         :options options :dimensions (:dimensions data) :metadata (:metadata data)})
                   (ReentrantReadWriteLock. true)))))

(defn tune
  "Offline cosine tuning with disjoint calibration and held-out validation queries.
   Returns measurements and recommended options only if both sets meet the recall target.
   Does not modify an existing index or guarantee recall on future queries."
  [data calibration-queries validation-queries & {:as options}]
  ((requiring-resolve 'hnsw.tuning/tune) data calibration-queries validation-queries options))
