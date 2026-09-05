(ns construction-benchmark
  "Repeated construction measurements with exact cosine top-k ground truth."
  (:require [bible-benchmark :as bible]
            [clojure.java.io :as io]
            [clojure.set :as sets]
            [clojure.string :as string]
            [hnsw.ultra-fast :as hnsw])
  (:import [hnsw.ultra_fast NormalizedDistance]
           [java.util PriorityQueue]))

(defn- env-long [name default]
  (Long/parseLong (or (System/getenv name) (str default))))

(defn- elapsed [started]
  (/ (- (System/nanoTime) started) 1.0e9))

(defn- median [values]
  (let [ordered (vec (sort values))
        middle (quot (count ordered) 2)]
    (if (odd? (count ordered))
      (nth ordered middle)
      (/ (+ (nth ordered (dec middle)) (nth ordered middle)) 2.0))))

(defn- exact-neighbors [vectors queries neighbor-count]
  (let [normalized (mapv hnsw/normalize-vector vectors)
        ^NormalizedDistance engine (hnsw/create-normalized-distance-engine)]
    (mapv (fn [{:keys [vector]}]
            (let [query (hnsw/normalize-vector vector)
                  nearest (PriorityQueue.
                           (int neighbor-count)
                           (reify java.util.Comparator
                             (compare [_ left right]
                               (Double/compare (double (second right))
                                               (double (second left))))))]
              (doseq [position (range (count normalized))]
                (let [distance (.distance engine query (nth normalized position))]
                  (when (or (< (.size nearest) neighbor-count)
                            (< distance (double (second (.peek nearest)))))
                    (.add nearest [(str position) distance])
                    (when (> (.size nearest) neighbor-count)
                      (.poll nearest)))))
              (set (map first nearest))))
          queries)))

(defn- quality [index queries ground-truth neighbor-count ef-search]
  (/ (reduce +
             (map (fn [{:keys [vector]} expected]
                    (count (sets/intersection
                            expected
                            (set (map :id (hnsw/search-knn index vector neighbor-count ef-search))))))
                  queries ground-truth))
     (double (* (count queries) neighbor-count))))

(defn- measure [data queries ground-truth config precision]
  (System/gc)
  (let [started (System/nanoTime)
        graph (hnsw/create-ultra-graph
               :M (:M config)
               :ef-construction (:ef-construction config)
               :ef-search (:ef-search config)
               :seed (:seed config)
               :distance-fn hnsw/cosine-distance-ultra
               :compact-precision :float32
               :construction-precision precision)
        _ (hnsw/insert-batch graph data :num-threads (:threads config)
                             :batch-size 4 :show-progress? false)
        construction-seconds (elapsed started)
        snapshot-start (System/nanoTime)
        _ (hnsw/optimize-search! graph)
        snapshot-seconds (elapsed snapshot-start)]
    {:construction-seconds construction-seconds
     :snapshot-seconds snapshot-seconds
     :total-seconds (+ construction-seconds snapshot-seconds)
     :ann-recall-at-10 (quality graph queries ground-truth 10 (:ef-search config))}))

(defn -main [& _]
  (let [config (assoc (#'bible/environment) :num-queries (env-long "QUALITY_QUERIES" 100))
        _ (when-not (.isFile (io/file (:embedding-dir config) "manifest.json"))
            (throw (IllegalArgumentException.
                    "Prepare the shared Bible embeddings before running construction benchmarks")))
        runs (env-long "BUILD_RUNS" 3)
        precisions (mapv keyword (string/split
                                 (or (System/getenv "BUILD_PRECISIONS") "float64,float32")
                                 #","))
        {:keys [vectors queries dimensions source]} (#'bible/load-data config)
        data (mapv (fn [position vector] [(str position) vector]) (range) vectors)
        _ (when (or (not (pos? runs)) (empty? queries) (< (count vectors) 10))
            (throw (IllegalArgumentException. "Positive runs/queries and at least 10 vectors required")))
        ground-truth (exact-neighbors vectors queries 10)]
    (println (pr-str {:benchmark :construction
                      :java (System/getProperty "java.version")
                      :os (System/getProperty "os.name")
                      :architecture (System/getProperty "os.arch")
                      :source source :vectors (count data) :dimensions dimensions
                      :queries (count queries) :runs runs :threads (:threads config)
                      :M (:M config) :ef-construction (:ef-construction config)
                      :ef-search (:ef-search config) :k 10 :compact-precision :float32}))
    (doseq [precision precisions]
      (hnsw/build-index (subvec data 0 (min 2000 (count data)))
                        :construction-precision precision :compact-precision :float32
                        :num-threads (:threads config) :show-progress? false))
    (let [results (reduce (fn [collected run]
                            (reduce (fn [collected precision]
                                      (let [result (measure data queries ground-truth config precision)]
                                        (println (pr-str (assoc result :run run :precision precision)))
                                        (flush)
                                        (update collected precision (fnil conj []) result)))
                                    collected (if (odd? run) precisions (reverse precisions))))
                          {} (range 1 (inc runs)))]
      (doseq [precision precisions
              :let [samples (get results precision)]]
        (println (pr-str {:summary precision
                          :construction-seconds (median (map :construction-seconds samples))
                          :snapshot-seconds (median (map :snapshot-seconds samples))
                          :total-seconds (median (map :total-seconds samples))
                          :ann-recall-at-10 (median (map :ann-recall-at-10 samples))
                          :min-recall (apply min (map :ann-recall-at-10 samples))
                          :max-recall (apply max (map :ann-recall-at-10 samples))}))))))
