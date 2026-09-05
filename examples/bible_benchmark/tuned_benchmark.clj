(ns tuned-benchmark
  (:require [bible-benchmark :as bible]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.set :as sets]
            [hnsw.api :as api]
            [hnsw.tuning :as tuning])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.security MessageDigest]
           [java.util HexFormat]))

(defn- median [values]
  (let [ordered (vec (sort values)) middle (quot (count ordered) 2)]
    (if (odd? (count ordered)) (nth ordered middle)
        (/ (+ (nth ordered (dec middle)) (nth ordered middle)) 2.0))))

(defn- elapsed [started] (/ (- (System/nanoTime) started) 1.0e9))

(defn- query-hash [queries dimensions]
  (let [digest (MessageDigest/getInstance "SHA-256")
        buffer (doto (ByteBuffer/allocate (* dimensions Float/BYTES)) (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [query queries]
      (.clear buffer)
      (doseq [value query] (.putFloat buffer (float value)))
      (.update digest (.array buffer)))
    (.formatHex (HexFormat/of) (.digest digest))))

(defn -main [& _]
  (let [environment (assoc (#'bible/environment) :num-queries (Long/parseLong (or (System/getenv "QUALITY_QUERIES") "1000")))
        _ (when-not (.isFile (io/file (:embedding-dir environment) "manifest.json"))
            (throw (IllegalArgumentException. "Prepare the shared Bible embeddings first")))
        {:keys [vectors queries source dimensions]} (#'bible/load-data environment)
        selected-queries (count queries)
        unique (->> queries (map (comp vec :vector)) distinct (mapv double-array))
        queries (subvec unique (quot (count unique) 2))
        data (mapv (fn [position vector] [(str position) vector]) (range) vectors)
        runs (Long/parseLong (or (System/getenv "BUILD_RUNS") "5"))
        _ (when (or (< (count unique) 2) (< (count data) 10) (not (pos? runs)))
            (throw (IllegalArgumentException. "Positive runs, two distinct queries and ten vectors required")))
        options {:metric :cosine :M (:M environment) :seed (:seed environment)
                 :ef-construction (:ef-construction environment) :ef-search (:ef-search environment)
                 :num-threads (:threads environment) :compact-precision :float32
                 :construction-precision :float32 :show-progress? false}
        truth (atom nil)
        search-results (atom nil)]
    (when-not (= 10 (:k environment))
      (throw (IllegalArgumentException. "This benchmark requires K=10")))
    (with-open [index (apply api/index (subvec data 0 (min 2000 (count data))) (mapcat identity options))]
      (api/size index))
    (let [samples (mapv
                    (fn [run]
                      (System/gc)
                      (let [started (System/nanoTime)]
                        (with-open [index (apply api/index data (mapcat identity options))]
                          (let [duration (elapsed started)]
                            (when-not @truth (reset! truth (#'tuning/exact-neighbors index queries 10)))
                            (let [neighbors (api/search-batch index queries 10 :num-threads (:threads environment))
                                  hits (reduce + (map (fn [result expected]
                                                        (count (sets/intersection expected (set (map :id result)))))
                                                      neighbors @truth))]
                              (when (= run runs)
                                (doseq [query (take 200 queries)] (api/search index query 10))
                                (let [serial (#'tuning/measure-search index queries @truth 10 (:ef-search environment) 5)
                                      parallel (mapv (fn [_]
                                                       (let [started (System/nanoTime)
                                                             result (api/search-batch index queries 10 :num-threads (:threads environment))
                                                             duration (elapsed started)]
                                                         (assert (= (count queries) (count result)))
                                                         (/ (count queries) duration)))
                                                     (range 5))]
                                  (reset! search-results {:serial_latency_ms (:query-ms serial)
                                                          :parallel_qps (median parallel)})))
                              {:run run :total_seconds duration
                               :ann_recall_at_10 (/ hits (double (* 10 (count queries))))})))))
                    (range 1 (inc runs)))]
      (println (json/write-str
                 (merge {:runtime "clojure" :java (System/getProperty "java.version")
                         :architecture (System/getProperty "os.arch") :seed (:seed environment)
                         :source source :vectors (count data) :dimensions dimensions :queries (count queries)
                         :selected_queries selected-queries :unique_queries (count unique)
                         :query_split "validation" :query_sha256 (query-hash queries dimensions)
                         :ground_truth_tie_break "lexicographic-id" :threads (:threads environment)
                         :M (:M environment) :ef_construction (:ef-construction environment)
                         :ef_search (:ef-search environment) :k 10 :build_runs runs
                         :construction_precision "float32" :compact_precision "float32"
                         :measurement "managed-api-including-validation-and-snapshot"
                         :samples samples :median_total_seconds (median (map :total_seconds samples))
                         :median_ann_recall_at_10 (median (map :ann_recall_at_10 samples))}
                        @search-results))))
    (shutdown-agents)))
