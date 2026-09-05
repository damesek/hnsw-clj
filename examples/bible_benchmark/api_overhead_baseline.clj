(ns api-overhead-baseline
  "Frozen pre-optimization API preprocessing for controlled, same-process benchmarks.")

(defn checked-vector [values dimensions metric]
  (when-not (or (sequential? values)
                (instance? java.util.List values)
                (and (some? values) (.isArray (class values))))
    (throw (IllegalArgumentException. "A vector must be a sequence or array")))
  (let [vector (double-array values)
        length (alength vector)]
    (when (or (zero? length) (and dimensions (not= dimensions length)))
      (throw (IllegalArgumentException. "Vector dimension mismatch or empty vector")))
    (doseq [value vector]
      (when-not (Double/isFinite value)
        (throw (IllegalArgumentException. "Vectors must contain only finite numbers"))))
    (let [norm (reduce (fn [sum value] (+ sum (* value value))) 0.0 vector)]
      (when (or (not (Double/isFinite norm))
                (and (= metric :cosine) (zero? norm)))
        (throw (IllegalArgumentException. "Vector norm is zero or outside supported numeric range"))))
    vector))

(defn entries [data dimensions metric existing]
  (loop [remaining (seq data) result [] seen (set (if (map? existing) (keys existing) existing)) dimension dimensions ordinal 0]
    (if-not remaining
      [result dimension]
      (let [item (first remaining)
            keyed? (and (sequential? item) (<= 2 (count item) 3)
                        (not (number? (second item))))
            [identifier values metadata] (if keyed? item [(str ordinal) item])
            identifier (if (nil? identifier)
                         (throw (IllegalArgumentException. "IDs cannot be nil"))
                         (str identifier))
            vector (checked-vector values dimension metric)]
        (when (contains? seen identifier)
          (throw (IllegalArgumentException. (str "Duplicate ID: " identifier))))
        (recur (next remaining) (conj result [identifier vector metadata])
               (conj seen identifier) (or dimension (alength vector)) (inc ordinal))))))

(defn results [state neighbors]
  (mapv (fn [neighbor]
          (let [metadata (get (:metadata state) (:id neighbor))]
            (if (some? metadata)
              (assoc neighbor :metadata metadata)
              neighbor)))
        neighbors))
