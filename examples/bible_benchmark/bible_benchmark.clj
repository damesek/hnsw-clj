(ns bible-benchmark
  "Comparable Clojure search benchmark over shared bilingual Bible embeddings."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [hnsw.ultra-fast :as hnsw])
  (:import [java.nio ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Path Paths StandardOpenOption]
           [java.util Locale Random]
           [java.util.concurrent Callable Executors ExecutorService Future TimeUnit]))

(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

(defn- root-format
  [pattern & arguments]
  (String/format Locale/ROOT pattern (to-array arguments)))

(defn- env-long
  [name default]
  (long (if-let [value (System/getenv name)]
          (Long/parseLong value)
          default)))

(defn- environment
  []
  {:embedding-dir (or (System/getenv "EMBEDDING_DIR")
                      "examples/bible_benchmark/data")
   :index-language (keyword (or (System/getenv "INDEX_LANGUAGE") "hu"))
   :query-language (keyword (or (System/getenv "QUERY_LANGUAGE") "en"))
   :num-vectors (env-long "NUM_VECTORS" 31173)
   :dimensions (env-long "DIMENSIONS" 768)
   :num-queries (env-long "NUM_QUERIES" 1000)
   :runs (env-long "RUNS" 5)
   :threads (env-long "THREADS" (.availableProcessors (Runtime/getRuntime)))
   :k (env-long "K" 10)
   :M (env-long "M" 16)
   :ef-construction (env-long "EF_CONSTRUCTION" 200)
   :ef-search (env-long "EF_SEARCH" 50)
   :seed (env-long "SEED" 42)})

(defn- load-float-vectors
  [path count dimensions]
  (let [^Path nio-path (Paths/get (str path) (make-array String 0))
        expected-bytes (* count dimensions Float/BYTES)]
    (with-open [channel (FileChannel/open
                         nio-path
                         (into-array StandardOpenOption [StandardOpenOption/READ]))]
      (when (> expected-bytes (.size channel))
        (throw (ex-info "Unexpected float32 vector file size"
                        {:path (str path)
                         :minimum expected-bytes
                         :actual (.size channel)})))
      (let [buffer (doto (.map channel FileChannel$MapMode/READ_ONLY 0 expected-bytes)
                     (.order ByteOrder/LITTLE_ENDIAN))
            floats (.asFloatBuffer buffer)]
        (mapv (fn [_]
                (let [vector (double-array dimensions)]
                  (dotimes [dimension dimensions]
                    (aset-double vector dimension (double (.get floats))))
                  vector))
              (range count))))))

(defn- load-language
  [embedding-dir manifest language max-vectors]
  (let [descriptor (get-in manifest [:languages language])
        count (min (long (:count descriptor)) max-vectors)
        dimensions (long (:dimensions manifest))
        items (subvec (vec (:items descriptor)) 0 count)
        vectors (load-float-vectors
                 (io/file embedding-dir (:vectors descriptor)) count dimensions)]
    {:items items
     :vectors vectors
     :dimensions dimensions
     :translation (:translation descriptor)}))

(defn- aligned-queries
  [target source requested]
  (let [target-labels (into {}
                            (map-indexed (fn [label item] [(:id item) label])
                                         (:items target)))
        aligned (into []
                      (keep-indexed
                       (fn [row item]
                         (when-let [label (get target-labels (:id item))]
                           {:vector (nth (:vectors source) row)
                            :expected label}))
                       (:items source)))
        query-count (min requested (count aligned))]
    (mapv (fn [index]
            (nth aligned (mod (* index 7919) (count aligned))))
          (range query-count))))

(defn- synthetic-data
  [count dimensions query-count seed]
  (let [random (Random. seed)
        vectors (mapv (fn [_]
                        (let [vector (double-array dimensions)]
                          (dotimes [dimension dimensions]
                            (aset-double vector dimension (.nextDouble random)))
                          vector))
                      (range count))
        queries (mapv (fn [index]
                        (let [label (mod (* index 7919) count)]
                          {:vector (nth vectors label)
                           :expected label}))
                      (range (min query-count count)))]
    {:vectors vectors
     :queries queries
     :dimensions dimensions
     :source "synthetic-shape"}))

(defn- load-data
  [{:keys [embedding-dir index-language query-language num-vectors
           dimensions num-queries seed]}]
  (let [manifest-file (io/file embedding-dir "manifest.json")]
    (if-not (.exists manifest-file)
      (synthetic-data num-vectors dimensions num-queries seed)
      (let [manifest (with-open [reader (io/reader manifest-file)]
                       (json/read reader :key-fn keyword))
            target (load-language embedding-dir manifest index-language num-vectors)
            source (if (= index-language query-language)
                     target
                     (load-language embedding-dir manifest query-language Long/MAX_VALUE))]
        {:vectors (:vectors target)
         :queries (aligned-queries target source num-queries)
         :dimensions (:dimensions target)
         :source (str (name query-language) "->" (name index-language)
                      ":" (:model manifest))}))))

(defn- result-metrics
  [seconds query-count]
  {:seconds seconds
   :latency-ms (/ (* seconds 1000.0) query-count)
   :qps (/ query-count seconds)})

(defn- serial-once
  [index queries k ef-search]
  (let [started (System/nanoTime)
        checksum (reduce (fn [total {:keys [vector]}]
                           (+ total (count (hnsw/search-knn
                                           index vector k ef-search))))
                         0
                         queries)
        seconds (/ (- (System/nanoTime) started) 1.0e9)]
    (assoc (result-metrics seconds (count queries)) :checksum checksum)))

(defn- parallel-once
  [^ExecutorService executor index queries k ef-search threads]
  (let [query-count (count queries)
        chunk-size (int (Math/ceil (/ (double query-count) threads)))
        started (System/nanoTime)
        futures (mapv (fn [start]
                        (let [end (min query-count (+ start chunk-size))]
                          (.submit executor
                                   ^Callable
                                   (reify Callable
                                     (call [_]
                                       (loop [position start
                                              total 0]
                                         (if (< position end)
                                           (recur (inc position)
                                                  (+ total
                                                     (count
                                                      (hnsw/search-knn
                                                       index
                                                       (:vector (nth queries position))
                                                       k ef-search))))
                                           total)))))))
                      (range 0 query-count chunk-size))
        checksum (reduce (fn [total ^Future future]
                           (+ total (long (.get future))))
                         0
                         futures)
        seconds (/ (- (System/nanoTime) started) 1.0e9)]
    (assoc (result-metrics seconds (count queries)) :checksum checksum)))

(defn- median-result
  [results query-count]
  (let [seconds (nth (vec (sort (map :seconds results)))
                     (quot (count results) 2))]
    (result-metrics seconds query-count)))

(defn- print-result
  [mode threads result]
  (println (root-format
            "RESULT runtime=clojure mode=%s threads=%d seconds=%.6f latency_ms=%.6f qps=%.1f"
            mode threads (:seconds result) (:latency-ms result) (:qps result))))

(defn- print-quality
  [index queries threads ef-search]
  (let [executor (Executors/newFixedThreadPool threads)]
    (try
      (let [futures (mapv (fn [{:keys [vector expected]}]
                            (.submit executor
                                     ^Callable
                                     (reify Callable
                                       (call [_]
                                         [expected (hnsw/search-knn
                                                    index vector 10 ef-search)]))))
                          queries)
            results (mapv #(.get ^Future %) futures)
            hits (fn [limit]
                   (/ (count (filter (fn [[expected neighbors]]
                                       (some #(= (str expected) (:id %))
                                             (take limit neighbors)))
                                     results))
                      (double (count results))))]
        (println (root-format
                  "QUALITY exact_verse_recall_at_1=%.4f at_5=%.4f at_10=%.4f"
                  (hits 1) (hits 5) (hits 10))))
      (finally
        (.shutdown executor)
        (.awaitTermination executor 1 TimeUnit/MINUTES)))))

(defn -main
  [& _]
  (let [{:keys [runs threads k M ef-construction ef-search seed] :as config}
        (environment)]
    (when (some #(not (pos? (long %)))
                [(:num-vectors config) (:num-queries config) runs threads k])
      (throw (IllegalArgumentException. "Benchmark sizes, runs, threads and k must be positive")))
    (let [{:keys [vectors queries dimensions source]} (load-data config)
          indexed-vectors (mapv (fn [label vector] [(str label) vector])
                                (range (count vectors)) vectors)]
      (println (str "Runtime: Clojure " (clojure-version)
                    " / Java " (System/getProperty "java.version")))
      (println (str "Dataset: " source))
      (println (root-format
                "Vectors=%,d dimensions=%d queries=%,d M=%d efConstruction=%d efSearch=%d k=%d"
                (count vectors) dimensions (count queries) M ef-construction ef-search k))
      (System/gc)
      (let [started (System/nanoTime)
            index (hnsw/build-index indexed-vectors
                                    :M M
                                    :ef-construction ef-construction
                                    :ef-search ef-search
                                    :seed seed
                                    :num-threads threads
                                    :batch-size 4
                                    :show-progress? false)
            build-seconds (/ (- (System/nanoTime) started) 1.0e9)
            warmup (subvec queries 0 (min 200 (count queries)))]
        (println (root-format "BUILD runtime=clojure threads=%d seconds=%.6f"
                              threads build-seconds))
        (serial-once index warmup k ef-search)
        (let [executor (Executors/newFixedThreadPool threads)]
          (try
        (parallel-once executor index warmup k ef-search threads)
            (let [serial-results (repeatedly
                                  runs #(serial-once index queries k ef-search))
                  parallel-results (repeatedly
                                    runs #(parallel-once
                                           executor index queries k ef-search threads))]
              (print-result "serial" 1 (median-result serial-results (count queries)))
              (print-result "parallel" threads
                            (median-result parallel-results (count queries))))
            (finally
              (.shutdown executor)
              (.awaitTermination executor 1 TimeUnit/MINUTES))))
        (print-quality index queries threads ef-search)))))
