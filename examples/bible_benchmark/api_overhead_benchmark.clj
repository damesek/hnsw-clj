(ns api-overhead-benchmark
  (:require [api-overhead-baseline :as baseline]
            [bible-benchmark :as bible]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.set :as sets]
            [hnsw.api :as api]
            [hnsw.tuning :as tuning]
            [hnsw.ultra-fast :as ultra]
            [tuned-benchmark :as tuned])
  (:import [com.sun.management ThreadMXBean]
           [java.lang.management ManagementFactory]))

(defn- env-long [name default]
  (Long/parseLong (or (System/getenv name) (str default))))

(defn- median [values]
  (let [ordered (vec (sort values)) middle (quot (count ordered) 2)]
    (if (odd? (count ordered)) (nth ordered middle)
        (/ (+ (nth ordered middle) (nth ordered (dec middle))) 2.0))))

(defn- allocation-counter []
  (let [bean (ManagementFactory/getThreadMXBean)]
    (when (and (instance? ThreadMXBean bean)
               (.isThreadAllocatedMemorySupported ^ThreadMXBean bean))
      (.setThreadAllocatedMemoryEnabled ^ThreadMXBean bean true)
      (fn [] (.getThreadAllocatedBytes ^ThreadMXBean bean (.getId (Thread/currentThread)))))))

(defn- measure [operation allocation]
  (let [allocated (when allocation (allocation))
        started (System/nanoTime)
        value (operation)
        seconds (/ (- (System/nanoTime) started) 1.0e9)]
    (cond-> {:seconds seconds :value value}
      allocation (assoc :caller-allocated-bytes (- (allocation) allocated)))))

(defn- in-mode [mode operation]
  (with-redefs-fn (if (= mode :baseline)
                   {#'api/checked-vector baseline/checked-vector
                    #'api/entries baseline/entries
                    #'api/results baseline/results}
                   {})
    operation))

(defn- modes [run]
  (if (odd? run) [:baseline :optimized] [:optimized :baseline]))

(defn- preprocessing-samples [data runs allocation]
  (let [existing (into {} (map (fn [[identifier]] [identifier nil]) data))
        addition [["new-item" (second (first data))]]]
    (vec (for [run (range 1 (inc runs)) mode (modes run)]
           (in-mode mode
             (fn []
               (System/gc)
               (let [preparation (measure #(#'api/entries data nil :cosine {}) allocation)
                     incremental (measure #(dotimes [_ 100]
                                             (#'api/entries addition nil :cosine existing)) allocation)]
                 (assert (= (count data) (count (first (:value preparation)))))
                 {:mode mode :run run
                  :prepare (dissoc preparation :value)
                  :prepare-one-add-100-calls (dissoc incremental :value)})))))))

(defn- build-samples [data options runs]
  (vec (for [run (range 1 (inc runs)) mode (modes run)]
         (in-mode mode
           (fn []
             (let [prepare #'api/entries
                   original-prepare @prepare
                   original-build ultra/build-index
                   phases (atom {})]
               (System/gc)
               (let [result (measure
                              #(with-redefs-fn
                                 {prepare (fn [& arguments]
                                            (let [result (measure (fn [] (apply original-prepare arguments)) nil)]
                                              (swap! phases assoc :prepare-seconds (:seconds result))
                                              (:value result)))
                                  #'ultra/build-index
                                  (fn [& arguments]
                                    (let [result (measure (fn [] (apply original-build arguments)) nil)]
                                      (swap! phases assoc :graph-and-snapshot-seconds (:seconds result))
                                      (:value result)))}
                                 (fn [] (apply api/index data (mapcat identity options)))) nil)]
                 (with-open [index (:value result)]
                   (assert (= (count data) (api/size index))))
                 (merge {:mode mode :run run :total-seconds (:seconds result)} @phases))))))))

(defn- search-samples [index queries threads runs]
  (let [expected (api/search-batch index queries 10 :num-threads threads)]
    (doseq [mode [:baseline :optimized]]
      (in-mode mode #(dotimes [_ 2]
                       (doseq [query queries] (api/search index query 10))
                       (api/search-batch index queries 10 :num-threads threads))))
    (vec (for [run (range 1 (inc runs)) mode (modes run)]
           (in-mode mode
             (fn []
               (let [serial (measure #(mapv (fn [query] (api/search index query 10)) queries) nil)
                     parallel (measure #(api/search-batch index queries 10 :num-threads threads) nil)]
                 (assert (= expected (:value serial) (:value parallel)))
                 {:mode mode :run run
                  :serial-ms (* 1000.0 (/ (:seconds serial) (count queries)))
                  :parallel-qps (/ (count queries) (:seconds parallel))})))))))

(defn- summarize [preparation builds searches]
  (mapv (fn [mode]
          (let [select (fn [samples] (filter #(= mode (:mode %)) samples))
                preparation (select preparation) builds (select builds) searches (select searches)]
            (cond-> {:mode mode
                     :prepare-seconds (median (map #(get-in % [:prepare :seconds]) preparation))
                     :prepare-one-add-ms (median (map #(* 10.0 (get-in % [:prepare-one-add-100-calls :seconds])) preparation))
                     :build-total-seconds (median (map :total-seconds builds))
                     :build-prepare-seconds (median (map :prepare-seconds builds))
                     :build-graph-and-snapshot-seconds (median (map :graph-and-snapshot-seconds builds))
                     :serial-ms (median (map :serial-ms searches))
                     :parallel-qps (median (map :parallel-qps searches))}
              (get-in (first preparation) [:prepare :caller-allocated-bytes])
              (assoc :prepare-allocated-bytes
                     (median (map #(get-in % [:prepare :caller-allocated-bytes]) preparation))))))
        [:baseline :optimized]))

(defn -main [& _]
  (let [environment (assoc (#'bible/environment) :num-queries (env-long "QUALITY_QUERIES" 1000))
        _ (when-not (.isFile (io/file (:embedding-dir environment) "manifest.json"))
            (throw (IllegalArgumentException. "Prepare the shared Bible embeddings first")))
        {:keys [vectors queries dimensions source]} (#'bible/load-data environment)
        unique (->> queries (map (comp vec :vector)) distinct (mapv double-array))
        queries (subvec unique (quot (count unique) 2))
        data (mapv (fn [ordinal vector] [(str ordinal) vector]) (range) vectors)
        runs (env-long "BUILD_RUNS" 5)
        _ (when (or (not (pos? runs)) (< (count data) 10) (< (count unique) 2))
            (throw (IllegalArgumentException. "Positive runs, ten vectors and two distinct queries required")))
        options {:metric :cosine :M (:M environment) :seed (:seed environment)
                 :ef-construction (:ef-construction environment) :ef-search (:ef-search environment)
                 :num-threads (:threads environment) :construction-precision :float32
                 :compact-precision :float32 :show-progress? false}
        allocation (allocation-counter)]
    (doseq [mode [:baseline :optimized]]
      (in-mode mode
        #(do
           (dotimes [_ 5] (#'api/entries data nil :cosine {}))
           (with-open [index (apply api/index (subvec data 0 (min 2000 (count data))) (mapcat identity options))]
             (api/size index)))))
    (let [preparation (preprocessing-samples data runs allocation)
          _ (binding [*out* *err*] (println "API preparation measurements complete"))
          builds (build-samples data options runs)
          _ (binding [*out* *err*] (println "Paired full-build measurements complete"))]
      (with-open [index (apply api/index data (mapcat identity options))]
        (let [truth (#'tuning/exact-neighbors index queries 10)
              results (api/search-batch index queries 10 :num-threads (:threads environment))
              hits (reduce + (map (fn [neighbors expected]
                                    (count (sets/intersection expected (set (map :id neighbors))))) results truth))
              searches (search-samples index queries (:threads environment) runs)]
          (println (json/write-str
                     {:benchmark :api-overhead :java (System/getProperty "java.version")
                      :architecture (System/getProperty "os.arch") :vectors (count data)
                      :dimensions dimensions :source source :queries (count queries)
                      :query-split :validation :query-sha256 (#'tuned/query-hash queries dimensions)
                      :options options :runs runs :search-results-identical true
                      :ann-recall-at-10 (/ hits (double (* 10 (count queries))))
                      :method "Frozen pre-optimization validation/entries/results vs current API in one JVM, alternating order. Fresh parallel builds; identical graph for all query trials. Loading, embeddings, exact ground truth, warmup and explicit GC excluded. Allocation counters cover only caller-thread preprocessing."
                      :summary (summarize preparation builds searches)
                      :preparation-samples preparation :build-samples builds :search-samples searches})))))
    (shutdown-agents)))
