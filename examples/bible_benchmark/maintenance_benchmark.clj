(ns maintenance-benchmark
  (:require [bible-benchmark :as bible]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.set :as sets]
            [hnsw.api :as api]
            [hnsw.helper.index-io :as index-io]
            [hnsw.tuning :as tuning]
            [hnsw.ultra-fast :as ultra])
  (:import [java.util ArrayList Collections Random]
           [java.util.concurrent.locks ReentrantReadWriteLock]))

(defn- elapsed [started] (/ (- (System/nanoTime) started) 1.0e9))

(defn- env-long [name default]
  (Long/parseLong (or (System/getenv name) (str default))))

(defn- median [values]
  (let [ordered (vec (sort values)) middle (quot (count ordered) 2)]
    (if (odd? (count ordered)) (nth ordered middle)
        (/ (+ (nth ordered middle) (nth ordered (dec middle))) 2.0))))

(defn- clone-index [state topology]
  (api/->HNSWIndex (atom (assoc state :graph (index-io/from-data topology nil)))
                   (ReentrantReadWriteLock. true)))

(defn- rebuild-mutation! [index deleted replacement]
  (let [state @(:state index)
        data (into []
                   (comp (remove #(or (contains? deleted (.id %)) (= (first replacement) (.id %))))
                         (map #(vector (.id %) (.vector %) (get (:metadata state) (.id %)))))
                   (sort-by #(.ordinal %) (vals (.nodes (:graph state)))))
        data (cond-> data replacement (conj replacement))
        graph (#'api/build data (:options state))]
    (reset! (:state index) (assoc state :graph graph
                                :metadata (into {} (map (fn [[identifier _ value]] [identifier value]) data))))))

(defn- measure-mutation [index operation]
  (let [snapshot-seconds (atom 0.0)
        optimize ultra/optimize-search!
        started (System/nanoTime)]
    (with-redefs [ultra/optimize-search! (fn [graph]
                                        (let [started (System/nanoTime)
                                              result (optimize graph)]
                                          (swap! snapshot-seconds + (elapsed started))
                                          result))]
      (operation))
    (let [total (elapsed started)]
      {:total-seconds total :snapshot-seconds @snapshot-seconds
       :mutation-seconds (- total @snapshot-seconds)
       :size (api/size index)})))

(defn- quality [index queries truth]
  (let [results (api/search-batch index queries 10 :num-threads 10)]
    (/ (reduce + (map (fn [neighbors expected]
                       (count (sets/intersection expected (set (map :id neighbors))))) results truth))
       (double (* 10 (count queries))))))

(defn- run-pair [base topology queries operations run]
  (with-open [local (clone-index base topology)
              rebuilt (clone-index base topology)]
    (let [indexes {:local local :rebuild rebuilt}]
      (vec (mapcat
             (fn [{:keys [deleted replacement cycle operation]}]
               (let [measurements
                     (into {} (for [mode (if (odd? run) [:local :rebuild] [:rebuild :local])]
                                (let [index (get indexes mode)
                                      action (if (= mode :rebuild)
                                               #(rebuild-mutation! index deleted replacement)
                                               #(if replacement
                                                  (apply api/update-vector! index replacement)
                                                  (api/delete-vectors! index deleted)))]
                                  (System/gc)
                                  [mode (measure-mutation index action)])))
                     truth (#'tuning/exact-neighbors rebuilt queries 10)]
                 (mapv (fn [mode]
                         (merge {:run run :cycle cycle :operation operation :mode mode
                                 :ann-recall-at-10 (quality (get indexes mode) queries truth)}
                                (get measurements mode))) [:local :rebuild])))
             operations)))))

(defn -main [& _]
  (let [environment (assoc (#'bible/environment) :num-queries (env-long "QUALITY_QUERIES" 200))
        _ (when-not (.isFile (io/file (:embedding-dir environment) "manifest.json"))
            (throw (IllegalArgumentException. "Prepare the shared Bible embeddings first")))
        {:keys [vectors queries source dimensions]} (#'bible/load-data environment)
        data (mapv (fn [ordinal vector] [(str ordinal) vector]) (range) vectors)
        queries (->> queries (map (comp vec :vector)) distinct (mapv double-array))
        runs (env-long "BUILD_RUNS" 3)
        cycles (env-long "CHURN_CYCLES" 3)
        delete-count (env-long "DELETE_COUNT" (max 1 (quot (count data) 100)))
        _ (when (or (not (every? pos? [runs cycles delete-count]))
                    (< (- (count data) (* cycles delete-count)) 10) (empty? queries))
            (throw (IllegalArgumentException. "Positive runs/cycles/deletes and ten surviving vectors required")))
        shuffled (doto (ArrayList. (mapv first data)) (Collections/shuffle (Random. 91)))
        update-id (last shuffled)
        update-vector (second (nth data (Long/parseLong update-id)))
        operations (vec (mapcat (fn [cycle]
                                  [{:cycle (inc cycle) :operation :delete-batch
                                    :deleted (set (subvec (vec shuffled) (* cycle delete-count)
                                                          (* (inc cycle) delete-count)))}
                                   {:cycle (inc cycle) :operation :update-one :deleted #{}
                                    :replacement [update-id (double-array (map #(* (if (even? cycle) -1.0 1.0) %)
                                                                               update-vector))]}])
                                (range cycles)))
        options {:metric :cosine :M (:M environment) :seed (:seed environment)
                 :ef-construction (:ef-construction environment) :ef-search (:ef-search environment)
                 :num-threads (:threads environment) :compact-precision :float32
                 :construction-precision :float32 :show-progress? false}]
    (with-open [warmup (apply api/index (subvec data 0 (min 2000 (count data))) (mapcat identity options))]
      (api/delete-vectors! warmup (map first (take (min 10 delete-count) data)))
      (api/update-vector! warmup (first (last (subvec data 0 (min 2000 (count data))))) (second (first data))))
    (with-open [base (apply api/index data (mapcat identity options))]
      (let [state @(:state base)
            topology (index-io/index-data (:graph state))
            truth (#'tuning/exact-neighbors base queries 10)
            initial-recall (quality base queries truth)
            samples (vec (mapcat #(run-pair state topology queries operations %) (range 1 (inc runs))))
            summaries (mapv (fn [[[mode operation cycle] rows]]
                              {:mode mode :operation operation :cycle cycle
                               :median-total-seconds (median (map :total-seconds rows))
                               :median-mutation-seconds (median (map :mutation-seconds rows))
                               :median-snapshot-seconds (median (map :snapshot-seconds rows))
                               :median-recall-at-10 (median (map :ann-recall-at-10 rows))})
                            (sort-by key (group-by (juxt :mode :operation :cycle) samples)))]
        (println (json/write-str
                   {:benchmark :local-maintenance :java (System/getProperty "java.version")
                    :architecture (System/getProperty "os.arch") :source source :dimensions dimensions
                    :vectors (count data) :queries (count queries) :options options :runs runs
                    :cycles cycles :delete-count-per-cycle delete-count :initial-recall-at-10 initial-recall
                    :method "Identical starting topology; alternating local/legacy full rebuild; public snapshot included; ground truth recomputed after every mutation outside timing. First local delete includes reverse-edge cache initialization."
                    :summaries summaries :samples samples}))))
    (shutdown-agents)))
