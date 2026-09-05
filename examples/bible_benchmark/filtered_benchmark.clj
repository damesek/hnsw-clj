(ns filtered-benchmark
  (:require [bible-benchmark :as bible]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [hnsw.api :as api]
            [text-search :as text])
  (:import [java.util Arrays]
           [java.util.concurrent Callable Executors ExecutorService Future ThreadPoolExecutor TimeUnit]))

(defn- env-long [name default]
  (Long/parseLong (or (System/getenv name) (str default))))

(defn- median [values]
  (let [ordered (vec (sort values)) middle (quot (count ordered) 2)]
    (if (odd? (count ordered)) (nth ordered middle)
        (/ (+ (nth ordered middle) (nth ordered (dec middle))) 2.0))))

(defn- invoke-search [index top-k {:keys [vector criteria]}]
  (if criteria
    (api/search-filtered index vector top-k criteria)
    (api/search index vector top-k)))

(defn- elapsed-call [operation]
  (let [started (System/nanoTime)
        result (operation)]
    {:milliseconds (/ (- (System/nanoTime) started) 1.0e6)
     :returned (count result)}))

(defn- request-sets [queries items top-k]
  (let [books (frequencies (map :book items))
        chapters (frequencies (map (juxt :book :chapter) items))
        definitions [{:name :unfiltered-ann :criteria (constantly nil)}
                     {:name :john-3-exact :criteria (constantly {:book "John" :chapter 3})}
                     {:name :john-exact :criteria (constantly {:book "John"})}
                     {:name :psalms-exact :criteria (constantly {:book "Psalms"})}
                     {:name :mixed-chapters-exact :criteria #(select-keys % [:book :chapter])}
                     {:name :mixed-books-exact :criteria #(select-keys % [:book])}]]
    (mapv (fn [{:keys [name criteria]}]
            (let [requests (mapv (fn [{:keys [vector expected]}]
                                   (let [criteria (criteria (nth items expected))
                                         eligible (cond
                                                    (nil? criteria) (count items)
                                                    (:chapter criteria) (get chapters ((juxt :book :chapter) criteria) 0)
                                                    :else (get books (:book criteria) 0))]
                                     {:vector vector :criteria criteria :eligible eligible})) queries)
                  sizes (mapv :eligible requests)]
              {:name name :requests requests
               :eligible {:min (apply min sizes) :max (apply max sizes)
                          :mean (/ (reduce + sizes) (double (count sizes)))}
               :expected-checksum (reduce + (map #(min top-k %) sizes))})) definitions)))

(defn- validate-results! [index top-k workloads]
  (doseq [{:keys [requests]} workloads
          {:keys [criteria eligible] :as request} requests]
    (let [neighbors (invoke-search index top-k request)]
      (assert (= (min top-k eligible) (count neighbors)))
      (when criteria
        (assert (every? #(= criteria (select-keys (:metadata %) (keys criteria))) neighbors))))))

(defn- serial-once [index requests top-k repeats]
  (let [request-count (count requests)
        total (* request-count repeats)
        latencies (double-array total)
        started (System/nanoTime)
        checksum (loop [position 0 checksum 0]
                   (if (= position total)
                     checksum
                     (let [request (nth requests (mod position request-count))
                           query-start (System/nanoTime)
                           neighbors (invoke-search index top-k request)]
                       (aset-double latencies position (/ (- (System/nanoTime) query-start) 1.0e6))
                       (recur (inc position) (+ checksum (count neighbors))))))
        milliseconds (/ (- (System/nanoTime) started) 1.0e6)]
    (Arrays/sort latencies)
    {:serial-ms (/ milliseconds total)
     :p50-ms (aget latencies (quot total 2))
     :p95-ms (aget latencies (dec (int (Math/ceil (* total 0.95)))))
     :checksum checksum}))

(defn- parallel-once [^ExecutorService executor index requests top-k repeats threads]
  (let [request-count (count requests)
        total (* request-count repeats)
        chunk-size (int (Math/ceil (/ (double total) threads)))
        started (System/nanoTime)
        futures (mapv (fn [start]
                        (let [end (min total (+ start chunk-size))]
                          (.submit executor
                                   ^Callable
                                   (reify Callable
                                     (call [_]
                                       (loop [position start checksum 0]
                                         (if (= position end)
                                           checksum
                                           (let [request (nth requests (mod position request-count))]
                                             (recur (inc position)
                                                    (+ checksum (count (invoke-search index top-k request))))))))))))
                      (range 0 total chunk-size))
        checksum (reduce (fn [total ^Future future] (+ total (long (.get future)))) 0 futures)
        seconds (/ (- (System/nanoTime) started) 1.0e9)]
    {:parallel-qps (/ total seconds) :checksum checksum}))

(defn- measure-workloads [executor index workloads top-k repeats threads runs]
  (vec
    (for [run (range runs)
          workload (let [offset (mod run (count workloads))
                         rotated (concat (drop offset workloads) (take offset workloads))]
                     (if (odd? run) (reverse rotated) rotated))]
      (let [{:keys [name requests expected-checksum]} workload
            serial (serial-once index requests top-k repeats)
            parallel (parallel-once executor index requests top-k repeats threads)]
        (assert (= (* repeats expected-checksum) (:checksum serial) (:checksum parallel)))
        (binding [*out* *err*]
          (println "Run" (inc run) name "ms/query" (:serial-ms serial) "parallel QPS" (:parallel-qps parallel)))
        (merge {:run (inc run) :name name} (dissoc serial :checksum) (dissoc parallel :checksum))))))

(defn -main [& _]
  (let [{:keys [embedding-dir threads runs k] :as config} (#'bible/environment)
        repeats (env-long "QUERY_REPEATS" 5)
        _ (doseq [value [threads runs k repeats (:num-vectors config) (:num-queries config)]]
            (when-not (pos? value) (throw (IllegalArgumentException. "Benchmark counts must be positive"))))
        _ (when-not (.isFile (io/file embedding-dir "manifest.json"))
            (throw (IllegalArgumentException. "Prepare the shared Bible embeddings first")))
        {:keys [items vectors dimensions]} (#'text/load-data config)
        manifest (with-open [reader (io/reader (io/file embedding-dir "manifest.json"))]
                   (json/read reader :key-fn keyword))
        source (#'bible/load-language embedding-dir manifest (:query-language config) Long/MAX_VALUE)
        queries (#'bible/aligned-queries {:items items} source (:num-queries config))
        _ (when (empty? queries) (throw (IllegalArgumentException. "No aligned queries")))
        entries (mapv (fn [item vector] [(:id item) vector item]) items vectors)
        workloads (request-sets queries items k)
        options {:metric :cosine :M (:M config) :ef-construction (:ef-construction config)
                 :ef-search (:ef-search config) :num-threads threads :seed (:seed config)
                 :construction-precision :float32 :compact-precision :float32
                 :filter-keys #{:book :chapter}}
        executor (Executors/newFixedThreadPool threads)]
    (.prestartAllCoreThreads ^ThreadPoolExecutor executor)
    (try
      (with-open [index (apply api/index entries (mapcat identity options))]
        (let [request (first (:requests (second workloads)))
              first-call (elapsed-call #(invoke-search index k request))]
          (dotimes [_ 2]
            (doseq [{:keys [requests]} workloads]
              (serial-once index requests k 1)
              (parallel-once executor index requests k 1 threads)))
          (validate-results! index k workloads)
          (let [rebuilds (mapv (fn [_]
                                 (swap! (:state index) dissoc :metadata-filter-cache)
                                 (elapsed-call #(invoke-search index k request))) (range runs))
                samples (measure-workloads executor index workloads k repeats threads runs)
                summary (mapv (fn [{:keys [name eligible]}]
                                (let [selected (filter #(= name (:name %)) samples)]
                                  (merge {:name name :eligible eligible}
                                         (into {} (map (fn [key] [key (median (map key selected))])
                                                       [:serial-ms :p50-ms :p95-ms :parallel-qps]))))) workloads)]
            (println
              (json/write-str
                {:benchmark :book-chapter-filtered :filter-implementation :generic-sorted-postings
                 :timestamp (str (java.time.Instant/now))
                 :java (System/getProperty "java.version") :architecture (System/getProperty "os.arch")
                 :available-processors (.availableProcessors (Runtime/getRuntime))
                 :vectors (count vectors) :dimensions dimensions :queries (count queries)
                 :query-repeats repeats :requests-per-trial (* repeats (count queries))
                 :runs runs :threads threads :top-k k :options options
                 :index-language (:index-language config) :query-language (:query-language config)
                 :model (:model manifest) :vector-api? (:vector-api? (api/info index))
                 :method "One full managed index with verse metadata; identical deterministic aligned English queries across workloads. Warmup and validation excluded; five rotated/reversed workload rounds by default. Serial mean, p50 and p95 are medians of per-round values. Parallel QPS uses a prestarted shared executor calling individual managed APIs, including locking, validation and result metadata, not search-batch. Fixed-book workloads reuse the same subset; mixed workloads choose each query's aligned book/chapter. No MPNet inference, loading, graph build or console printing in query timings. Filtered search is exact float64; unfiltered ANN is float32 efSearch as configured, so these are different search scopes/accuracy, not an algorithm speedup comparison."
                 :cold-first-filtered-call first-call
                 :warm-jit-postings-rebuild-ms (median (map :milliseconds rebuilds))
                 :postings-rebuild-note "Only the derived cache is dropped between rebuild samples. This measures the next cache-miss query, not total add/update/delete latency."
                 :postings-rebuild-samples rebuilds :summary summary :samples samples})))))
      (finally
        (.shutdown executor)
        (.awaitTermination executor 1 TimeUnit/MINUTES)
        (shutdown-agents)))))
