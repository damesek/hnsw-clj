(ns hnsw.ultra-fast-test
  (:require [clojure.edn :as edn]
            [clojure.set :as set]
            [clojure.test :refer :all]
            [hnsw.api :as api]
            [hnsw.helper.index-io :as index-io]
            [hnsw.ultra-fast :as hnsw])
  (:import [java.nio.file Files]
           [java.util Set Random]))

(defn- random-data
  [size dimensions seed]
  (let [random (Random. seed)]
    (mapv (fn [idx]
            [(str idx)
             (double-array
              (repeatedly dimensions #(.nextDouble random)))])
          (range size))))

(defn- graph-levels
  [graph]
  (into {}
        (map (fn [[id node]] [id (.level node)]))
        (.nodes graph)))

(defn- exact-neighbor-ids
  [data query k]
  (set (map first
            (take k
                  (sort-by (fn [[_ vector]]
                             (hnsw/cosine-distance-ultra query vector))
                           data)))))

(defn- degree-violations
  [graph M]
  (for [node (vals (into {} (.nodes graph)))
        level (range (inc (.level node)))
        :let [degree (.size ^Set
                            (aget ^objects (.neighbors node) level))
              limit (if (zero? level) (* 2 M) M)]
        :when (> degree limit)]
    [(.id node) level degree limit]))

(deftest reference-level-distribution
  (let [random (Random. 42)
        ml (/ 1.0 (Math/log 16.0))
        levels (repeatedly 4096 #(hnsw/assign-level-ultra ml random))
        upper-level-count (count (filter pos? levels))]
    (is (< 180 upper-level-count 340))
    (is (< (apply max levels) 6))))

(deftest deterministic-build-and-graph-invariants
  (let [data (random-data 500 32 7)
        first-index (hnsw/build-index data
                                      :M 8
                                      :ef-construction 100
                                      :seed 99
                                      :show-progress? false)
        second-index (hnsw/build-index data
                                       :M 8
                                       :ef-construction 100
                                       :seed 99
                                       :show-progress? false)
        violations (degree-violations first-index 8)]
    (is (= (graph-levels first-index) (graph-levels second-index)))
    (is (empty? violations))
    (is (= 500 (:num-elements (hnsw/graph-info first-index))))
    (is (:normalized-cosine? (hnsw/graph-info first-index)))))

(deftest parallel-build-preserves-graph-invariants
  (let [data (random-data 1200 32 123)
        index (hnsw/build-index data
                                :M 12
                                :ef-construction 150
                                :seed 123
                                :num-threads 8
                                :batch-size 4
                                :show-progress? false)
        nodes (.nodes index)]
    (is (= 1200 (.size nodes)))
    (is (empty? (degree-violations index 12)))
    (is (every? (fn [node]
                  (every? (fn [level]
                            (every? #(.containsKey nodes %)
                                    (aget ^objects (.neighbors node) level)))
                          (range (inc (.level node)))))
                (vals (into {} nodes))))
    (doseq [[id query] (take 20 data)]
      (is (= id (:id (first (hnsw/search-knn index query 1))))))))

(deftest rejects-invalid-parallel-options
  (is (thrown-with-msg? IllegalArgumentException
                        #"num-threads"
                        (hnsw/build-index [] :num-threads 0
                                          :show-progress? false)))
  (is (thrown-with-msg? IllegalArgumentException
                        #"batch-size"
                        (hnsw/build-index [] :batch-size 0
                                          :show-progress? false)))
  (is (thrown-with-msg? IllegalArgumentException
                        #"compact-precision"
                        (hnsw/build-index [] :compact-precision :float16
                                          :show-progress? false))))

(deftest configurable-ef-search-preserves-safe-default
  (let [data (random-data 300 24 81)
        index (hnsw/build-index data
                                :ef-search 50
                                :seed 81
                                :show-progress? false)
        [_ query] (first data)]
    (is (= 50 (:ef-search (hnsw/graph-info index))))
    (is (= (hnsw/search-knn index query 10)
           (hnsw/search-knn index query 10 50)))
    (is (= 10 (count (hnsw/search-knn index query 10 5))))
    (is (thrown-with-msg? IllegalArgumentException
                          #"ef-search"
                          (hnsw/search-knn index query 10 0)))))

(deftest compact-search-matches-mutable-search
  (when (try
          (Class/forName "hnsw.search.CompactSearch")
          true
          (catch Throwable _ false))
    (let [data (random-data 500 48 314)
          index (hnsw/build-index data
                                  :M 12
                                  :ef-construction 150
                                  :seed 314
                                  :show-progress? false)
          compact-results (mapv (fn [[_ query]]
                                  (hnsw/search-knn index query 10 50))
                                (take 30 data))
          mutable-results
          (with-redefs [hnsw/optimize-search! identity]
            (.set (.search-index index) nil)
            (mapv (fn [[_ query]]
                    (hnsw/search-knn index query 10 50))
                  (take 30 data)))]
      (is (= (mapv #(mapv :id %) compact-results)
             (mapv #(mapv :id %) mutable-results)))
      (is (every? #(< % 1.0e-12)
                  (mapcat (fn [compact mutable]
                            (map (fn [left right]
                                   (Math/abs (- (:distance left)
                                                (:distance right))))
                                 compact mutable))
                          compact-results mutable-results)))
      (hnsw/optimize-search! index)
      (is (:compact-search? (hnsw/graph-info index))))))

(deftest float32-compact-search-preserves-neighbors
  (when (try
          (Class/forName "hnsw.search.CompactSearch")
          true
          (catch Throwable _ false))
    (let [data (random-data 500 48 2718)
          float64-index (hnsw/build-index data
                                          :seed 2718
                                          :num-threads 1
                                          :compact-precision :float64
                                          :show-progress? false)
          float32-index (hnsw/build-index data
                                          :seed 2718
                                          :num-threads 1
                                          :compact-precision :float32
                                          :show-progress? false)
          queries (mapv second (take 30 data))
          float64-results (mapv #(hnsw/search-knn float64-index % 10) queries)
          float32-results (mapv #(hnsw/search-knn float32-index % 10) queries)
          overlaps (map (fn [float64 float32]
                          (/ (count (set/intersection
                                     (set (map :id float64))
                                     (set (map :id float32))))
                             10.0))
                        float64-results float32-results)
          distance-deltas
          (mapcat (fn [float64 float32]
                    (let [float32-by-id (into {} (map (juxt :id :distance) float32))]
                      (keep (fn [{:keys [id distance]}]
                              (when-let [float32-distance (get float32-by-id id)]
                                (Math/abs (- distance float32-distance))))
                            float64)))
                  float64-results float32-results)]
      (is (= :float64 (:compact-precision (hnsw/graph-info float64-index))))
      (is (= :float32 (:compact-precision (hnsw/graph-info float32-index))))
      (is (>= (/ (reduce + overlaps) (count overlaps)) 0.99))
      (is (every? #(< % 1.0e-5) distance-deltas)))))

(deftest insertion-invalidates-and-lazily-rebuilds-compact-search
  (when (try
          (Class/forName "hnsw.search.CompactSearch")
          true
          (catch Throwable _ false))
    (let [index (hnsw/build-index
                 [["x" (double-array [1.0 0.0])]
                  ["y" (double-array [0.0 1.0])]]
                 :show-progress? false)]
      (is (:compact-search? (hnsw/graph-info index)))
      (hnsw/insert-single index "new" (double-array [0.9 0.1]))
      (is (not (:compact-search? (hnsw/graph-info index))))
      (is (= "new" (:id (first (hnsw/search-knn
                                  index (double-array [0.9 0.1]) 1)))))
      (is (:compact-search? (hnsw/graph-info index))))))

(deftest parallel-batch-search-preserves-order
  (let [data (random-data 400 32 721)
        index (hnsw/build-index data :seed 721 :show-progress? false)
        queries (mapv second (take 80 data))
        serial (mapv #(hnsw/search-knn index % 5) queries)
        parallel (hnsw/search-knn-batch index queries 5 :num-threads 8)]
    (is (= serial parallel))
    (is (thrown-with-msg? IllegalArgumentException
                          #"num-threads"
                          (hnsw/search-knn-batch index queries 5 :num-threads 0)))))

(deftest cosine-search-preserves-distance-semantics
  (let [data [["x" (double-array [10.0 0.0])]
              ["diagonal" (double-array [2.0 2.0])]
              ["y" (double-array [0.0 4.0])]]
        query (double-array [3.0 0.0])
        index (hnsw/build-index data :seed 42 :show-progress? false)
        results (hnsw/search-knn index query 3)]
    (is (= ["x" "diagonal" "y"] (mapv :id results)))
    (is (< (Math/abs (:distance (first results))) 1.0e-12))
    (is (< (Math/abs (- (:distance (second results))
                        (hnsw/cosine-distance-ultra
                         query (second (second data)))))
           1.0e-12))))

(deftest recall-regression
  (let [data (random-data 750 48 42)
        index (hnsw/build-index data
                                :M 16
                                :ef-construction 200
                                :seed 42
                                :show-progress? false)
        recalls (for [[_ query] (take 20 data)
                      :let [expected (exact-neighbor-ids data query 10)
                            actual (set (map :id
                                             (hnsw/search-knn index query 10)))]]
                  (/ (count (set/intersection expected actual)) 10.0))]
    (is (>= (/ (reduce + recalls) (count recalls)) 0.95))
    (is (>= (apply min recalls) 0.8))))

(deftest rejects-duplicate-identifiers
  (let [index (hnsw/create-ultra-graph)]
    (hnsw/insert-single index "same" (double-array [1.0 0.0]))
    (is (thrown-with-msg? IllegalArgumentException
                          #"already exists"
                          (hnsw/insert-single
                           index "same" (double-array [0.0 1.0]))))))

(deftest normalized-index-roundtrip
  (let [data (random-data 100 16 12)
        index (hnsw/build-index data
                                :seed 12
                                :compact-precision :float32
                                :show-progress? false)
        [_ query] (first data)
        path (Files/createTempFile "hnsw-clj" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (index-io/save-index index (str path))
      (let [loaded (index-io/load-index
                    (str path) hnsw/cosine-distance-ultra)]
        (is (:normalized-cosine? (hnsw/graph-info loaded)))
        (is (= 50 (:ef-search (hnsw/graph-info loaded))))
        (is (= :float32 (:compact-precision (hnsw/graph-info loaded))))
        (is (= (hnsw/search-knn index query 10)
               (hnsw/search-knn loaded query 10))))
      (finally
        (Files/deleteIfExists path)))))

(deftest float32-distance-kernels-handle-tails-and-zero-vectors
  (let [engine (hnsw/create-float-distance-engine)
        scalar (hnsw/->ScalarFloatNormalizedDistance)]
    (doseq [dimensions [1 3 4 7 15 16 17 31 32 33 63 64 65 768 769]]
      (let [random (Random. dimensions)
            left (hnsw/normalize-vector
                  (double-array (repeatedly dimensions #(- (.nextDouble random) 0.5))))
            right (hnsw/normalize-vector
                   (double-array (repeatedly dimensions #(- (.nextDouble random) 0.5))))
            float-left (hnsw/construction-vector left :float32)
            float-right (hnsw/construction-vector right :float32)
            expected (hnsw/cosine-distance-normalized left right)]
        (is (< (Math/abs (- expected (.distanceFloat engine float-left float-right))) 2.0e-6))
        (is (< (Math/abs (- expected (.distanceFloat scalar float-left float-right))) 2.0e-6))
        (is (= 1.0 (.distanceFloat engine float-left (float-array dimensions))))
        (is (< (Math/abs (.distanceFloat engine float-left float-left)) 2.0e-6))))))

(deftest float32-construction-preserves-recall-and-graph-invariants
  (let [data (random-data 750 65 1024)
        queries (mapv second (random-data 40 65 8192))
        exact (mapv #(exact-neighbor-ids data % 10) queries)
        build (fn [precision]
                (hnsw/build-index data :construction-precision precision
                                  :num-threads 1 :show-progress? false))
        double-index (build :float64)
        float-index (build :float32)
        recall (fn [index]
                 (/ (reduce + (map (fn [query expected]
                                     (count (set/intersection
                                             expected
                                             (set (map :id (hnsw/search-knn index query 10 100))))))
                                   queries exact))
                    (* 10.0 (count queries))))]
    (is (= :float64 (:construction-precision (hnsw/graph-info double-index))))
    (is (= :float32 (:construction-precision (hnsw/graph-info float-index))))
    (is (= (graph-levels double-index) (graph-levels float-index)))
    (is (empty? (degree-violations float-index 16)))
    (is (>= (recall float-index) 0.95))
    (is (< (Math/abs (- (recall float-index) (recall double-index))) 0.02))
    (let [double-node (.get (.nodes double-index) "0")
          float-node (.get (.nodes float-index) "0")]
      (is (identical? (.vector double-node) (.build-vector double-node)))
      (is (instance? (Class/forName "[F") (.build-vector float-node)))
      (is (= (vec (.vector double-node)) (vec (.vector float-node)))))))

(deftest float32-parallel-construction-and-portable-search
  (let [data (random-data 800 33 16)
        index (hnsw/build-index data :construction-precision :float32
                                    :num-threads 4 :batch-size 8 :show-progress? false)]
    (is (= 800 (.size (.nodes index))))
    (is (empty? (degree-violations index 16)))
    (is (every? (fn [node]
                  (every? (fn [level]
                            (every? #(.containsKey (.nodes index) %)
                                    (aget ^objects (.neighbors node) level)))
                          (range (inc (.level node)))))
                (vals (into {} (.nodes index)))))
    (with-redefs [hnsw/optimize-search! identity]
      (.set (.search-index index) nil)
      (doseq [[id query] (take 20 data)]
        (is (= id (:id (first (hnsw/search-knn index query 1))))))))
  (let [index (with-redefs [hnsw/create-float-distance-engine hnsw/->ScalarFloatNormalizedDistance]
                (hnsw/build-index [["first" (double-array [1.0 0.0])]]
                                  :construction-precision :float32 :show-progress? false))]
    (hnsw/insert-single index "added" (double-array [0.0 2.0]))
    (is (false? (:construction-vector-api? (hnsw/graph-info index))))
    (is (= "added" (:id (first (hnsw/search-knn index [0.0 1.0] 1)))))))

(deftest construction-precision-validation
  (is (thrown-with-msg? IllegalArgumentException #"construction-precision"
                        (hnsw/build-index [] :construction-precision :float16)))
  (is (thrown-with-msg? IllegalArgumentException #"cosine"
                        (api/index [] :metric :euclidean :construction-precision :float32)))
  (is (thrown-with-msg? IllegalArgumentException #"cosine"
                        (hnsw/build-index [] :construction-precision :float32
                                          :distance-fn (fn [_ _] 0.0)))))

(deftest public-persistence-preserves-metric-and-construction-precision
  (doseq [[metric precision] [[:euclidean :float64] [:cosine :float32]]]
    (let [index (api/index [["horizontal" [20.0 0.0]] ["diagonal" [1.0 1.0]]]
                           :metric metric :construction-precision precision
                           :compact-precision :float32 :ef-search 80 :show-progress? false)
          path (Files/createTempFile "hnsw-public" ".edn"
                                     (make-array java.nio.file.attribute.FileAttribute 0))]
      (try
        (api/save-index index (str path))
        (let [loaded (api/load-index (str path))]
          (is (= 2 (api/size loaded)))
          (is (= (= metric :cosine) (:normalized-cosine? (api/info loaded))))
          (is (= precision (:construction-precision (api/info loaded))))
          (is (= 80 (:ef-search (api/info loaded))))
          (is (= (api/search index [1.0 0.0] 2) (api/search loaded [1.0 0.0] 2)))
          (api/add-vector! loaded "new" [-1.0 -1.0])
          (is (= "new" (:id (first (api/search loaded [-1.0 -1.0] 1))))))
        (finally (Files/deleteIfExists path))))))

(deftest legacy-edn-defaults-to-double-construction
  (let [index (hnsw/build-index (random-data 80 17 991) :show-progress? false)
        path (Files/createTempFile "hnsw-legacy" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (index-io/save-index index (str path))
      (spit (str path) (pr-str (dissoc (edn/read-string (slurp (str path)))
                                      :metric :construction-precision)))
      (let [loaded (index-io/load-index (str path))
            query (second (first (random-data 1 17 81)))]
        (is (= :float64 (:construction-precision (hnsw/graph-info loaded))))
        (is (= (hnsw/search-knn index query 10) (hnsw/search-knn loaded query 10))))
      (finally (Files/deleteIfExists path)))))

(deftest legacy-unnormalized-cosine-roundtrip
  (let [distance (fn [left right] (hnsw/cosine-distance-ultra left right))
        index (hnsw/build-index [["axis" [20.0 0.0]] ["diagonal" [1.0 1.0]]]
                                :distance-fn distance :show-progress? false)
        path (Files/createTempFile "hnsw-raw-legacy" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (index-io/save-index index (str path))
      (spit (str path) (pr-str (dissoc (edn/read-string (slurp (str path)))
                                      :metric :normalize-vectors :construction-precision)))
      (let [loaded (index-io/load-index (str path))]
        (is (= (hnsw/search-knn index [1.0 0.0] 2)
               (hnsw/search-knn loaded [1.0 0.0] 2)))
        (is (= (.ml index) (.ml loaded)))
        (index-io/save-index loaded (str path))
        (is (= (hnsw/search-knn index [1.0 0.0] 2)
               (hnsw/search-knn (index-io/load-index (str path)) [1.0 0.0] 2))))
      (finally (Files/deleteIfExists path)))))

(deftest custom-distance-roundtrip-requires-explicit-function
  (let [distance (fn [^doubles left ^doubles right]
                   (Math/abs (- (aget left 0) (aget right 0))))
        index (hnsw/build-index [["first" [1.0 0.0]] ["second" [3.0 100.0]]]
                                :distance-fn distance :show-progress? false)
        path (Files/createTempFile "hnsw-custom" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (index-io/save-index index (str path))
      (is (thrown-with-msg? IllegalArgumentException #"custom metric"
                            (index-io/load-index (str path))))
      (is (= (hnsw/search-knn index [2.9 0.0] 2)
             (hnsw/search-knn (index-io/load-index (str path) distance) [2.9 0.0] 2)))
      (finally (Files/deleteIfExists path)))))
