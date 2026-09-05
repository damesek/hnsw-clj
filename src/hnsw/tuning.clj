(ns hnsw.tuning
  "Offline empirical calibration. Never treats calibration recall as a guarantee."
  (:require [clojure.set :as sets]
            [hnsw.api :as api]
            [hnsw.ultra-fast :as ultra])
  (:import [hnsw.ultra_fast NormalizedDistance]
           [java.util PriorityQueue Comparator]))

(defn- seconds [started] (/ (- (System/nanoTime) started) 1.0e9))

(defn- median [values]
  (let [ordered (vec (sort values)) middle (quot (count ordered) 2)]
    (if (odd? (count ordered)) (nth ordered middle)
        (/ (+ (nth ordered (dec middle)) (nth ordered middle)) 2.0))))

(defn- exact-neighbors [index queries neighbor-count]
  (let [graph (:graph @(:state index))
        nodes (vec (sort-by #(.id %) (vals (.nodes graph))))
        ^NormalizedDistance engine (ultra/create-normalized-distance-engine)
        comparator (reify Comparator
                     (compare [_ left right]
                       (let [distance-order (Double/compare (double (second right)) (double (second left)))]
                         (if (zero? distance-order) (compare (first right) (first left)) distance-order))))]
    (api/search-batch index queries neighbor-count :num-threads 1)
    (mapv (fn [query]
            (let [query (ultra/normalize-vector (double-array query))
                  nearest (PriorityQueue. (int neighbor-count) comparator)]
              (doseq [node nodes]
                (let [candidate [(.id node) (.distance engine query (.vector node))]]
                  (when (or (< (.size nearest) neighbor-count)
                            (pos? (.compare comparator candidate (.peek nearest))))
                    (.add nearest candidate)
                    (when (> (.size nearest) neighbor-count) (.poll nearest)))))
              (set (map first nearest))))
          queries)))

(defn- measure-search [index queries expected neighbor-count ef-search query-runs]
  (doseq [query (take 20 queries)] (api/search index query neighbor-count ef-search))
  (let [measurements (mapv (fn [_]
                             (let [started (System/nanoTime)
                                   results (mapv #(api/search index % neighbor-count ef-search) queries)
                                   duration (seconds started)
                                   hits (reduce + (map (fn [neighbors truth]
                                                         (count (sets/intersection truth (set (map :id neighbors)))))
                                                       results expected))]
                               {:query-ms (* 1000.0 (/ duration (count queries)))
                                :recall (/ hits (double (* (count queries) neighbor-count)))}))
                           (range query-runs))]
    {:query-ms (median (map :query-ms measurements))
     :recall (:recall (first measurements))}))

(defn tune
  [data calibration-queries validation-queries
   {:keys [configurations ef-search-values target-recall k build-runs query-runs
           max-build-seconds max-query-ms warmup-size objective]
    :as supplied
    :or {configurations [{:construction-precision :float32 :ef-construction 100}
                         {:construction-precision :float32 :ef-construction 200}]
         ef-search-values [50 80 120] target-recall 0.98 k 10
         build-runs 3 query-runs 3 warmup-size 512 objective :build-time}}]
  (when-let [unknown (seq (remove #{:configurations :ef-search-values :target-recall :k :build-runs
                                   :query-runs :max-build-seconds :max-query-ms :warmup-size :objective}
                                 (keys supplied)))]
    (throw (IllegalArgumentException. (str "Unknown tuning options: " unknown))))
  (doseq [value [k build-runs query-runs]]
    (when-not (and (integer? value) (pos? value))
      (throw (IllegalArgumentException. "k and run counts must be positive integers"))))
  (when-not (and (integer? warmup-size) (not (neg? warmup-size)))
    (throw (IllegalArgumentException. "warmup-size must be a nonnegative integer")))
  (when-not (and (number? target-recall) (< 0.0 target-recall) (<= target-recall 1.0))
    (throw (IllegalArgumentException. "target-recall must be in (0,1]")))
  (when-not (#{:build-time :query-time} objective)
    (throw (IllegalArgumentException. "objective must be :build-time or :query-time")))
  (doseq [limit [max-build-seconds max-query-ms]]
    (when (and limit (not (and (number? limit) (pos? limit) (Double/isFinite (double limit)))))
      (throw (IllegalArgumentException. "Timing limits must be positive and finite"))))
  (when (or (empty? configurations) (empty? ef-search-values)
            (empty? calibration-queries) (empty? validation-queries) (< (count data) k))
    (throw (IllegalArgumentException. "Nonempty configurations/query sets and at least k vectors required")))
  (doseq [ef ef-search-values]
    (when-not (and (integer? ef) (>= ef k))
      (throw (IllegalArgumentException. "ef-search-values must contain integers at least k"))))
  (when (seq (sets/intersection (set (map vec calibration-queries)) (set (map vec validation-queries))))
    (throw (IllegalArgumentException. "Calibration and validation queries must be disjoint")))
  (let [data (vec data)
        calibration-queries (vec calibration-queries)
        validation-queries (vec validation-queries)
        configurations (mapv #(merge {:metric :cosine :compact-precision :float32 :show-progress? false} %)
                             configurations)
        _ (when (some #(not= :cosine (:metric %)) configurations)
            (throw (IllegalArgumentException. "Tuning currently supports cosine only")))
        truth (atom nil)
        samples (atom [])]
    (when (pos? warmup-size)
      (doseq [options configurations]
        (with-open [index (apply api/index (subvec data 0 (min warmup-size (count data))) (mapcat identity options))]
          (api/size index))))
    (doseq [run (range build-runs)
            configuration-id (if (even? run) (range (count configurations))
                                 (reverse (range (count configurations))))]
      (let [options (nth configurations configuration-id)]
        (System/gc)
        (let [started (System/nanoTime)]
          (with-open [index (apply api/index data (mapcat identity options))]
            (let [build-seconds (seconds started)]
              (when-not @truth
                (reset! truth {:calibration (exact-neighbors index calibration-queries k)
                               :validation (exact-neighbors index validation-queries k)}))
              (doseq [ef (if (even? run) ef-search-values (reverse ef-search-values))]
                (swap! samples conj
                       (merge {:configuration-id configuration-id :run (inc run) :ef-search ef
                               :build-seconds build-seconds}
                              (measure-search index calibration-queries (:calibration @truth) k ef query-runs)))))))))
    (let [rows (mapv (fn [[[configuration-id ef] samples]]
                       {:options (assoc (nth configurations configuration-id) :ef-search ef)
                        :build-seconds (median (map :build-seconds samples))
                        :query-ms (median (map :query-ms samples))
                        :calibration-recall (median (map :recall samples))
                        :min-calibration-recall (apply min (map :recall samples))})
                     (sort-by key (group-by (juxt :configuration-id :ef-search) @samples)))
          feasible? (fn [row]
                      (and (>= (:min-calibration-recall row) target-recall)
                           (or (nil? max-build-seconds) (<= (:build-seconds row) max-build-seconds))
                           (or (nil? max-query-ms) (<= (:query-ms row) max-query-ms))))
          selected (first (sort-by (if (= objective :build-time)
                                    (juxt :build-seconds :query-ms)
                                    (juxt :query-ms :build-seconds)) (filter feasible? rows)))
          validation (when selected
                       (let [options (:options selected)]
                         (System/gc)
                         (let [started (System/nanoTime)]
                           (with-open [index (apply api/index data (mapcat identity options))]
                             (let [build-seconds (seconds started)]
                               (assoc (measure-search index validation-queries (:validation @truth) k
                                                      (:ef-search options) query-runs)
                                      :build-seconds build-seconds))))))
          accepted? (boolean (and validation (feasible? (assoc validation :min-calibration-recall (:recall validation)))))]
      {:target-recall target-recall :objective objective :k k
       :calibration-query-count (count calibration-queries) :validation-query-count (count validation-queries)
       :configurations configurations :build-runs build-runs :query-runs query-runs
       :samples @samples :results rows :selected selected :validation validation
       :accepted? accepted? :recommended-options (when accepted? (:options selected))
       :status (cond accepted? :validated selected :validation-failed :else :no-feasible-configuration)})))
