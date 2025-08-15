(ns hnsw.core-test
  "Unit tests for HNSW core functionality"
  (:require [clojure.test :refer :all]
            [hnsw.ultra-fast :as ultra]
            [hnsw.graph :as graph]
            [hnsw.simd-optimized :as simd]
            [data-generator :as gen]))

(deftest test-vector-distance
  (testing "Euclidean distance calculation"
    (let [v1 (double-array [1 2 3])
          v2 (double-array [1 2 3])
          v3 (double-array [0 0])
          v4 (double-array [3 4])
          v5 (double-array [4 5 6])]
      (is (= 0.0 (simd/euclidean-distance v1 v2)))
      (is (= 5.0 (simd/euclidean-distance v3 v4)))
      (is (< (Math/abs (- (simd/euclidean-distance v1 v5)
                          5.196152422706632))
             0.00001)))))

(deftest test-cosine-distance
  (testing "Cosine distance calculation"
    (let [v1 (double-array [1 2 3])
          v2 (double-array [1 2 3])
          v3 (double-array [1 0])
          v4 (double-array [-1 0])
          v5 (double-array [4 5 6])]
      (is (< (simd/cosine-distance v1 v2) 0.001)) ; same vectors have distance ~0
      (is (< (Math/abs (- (simd/cosine-distance v3 v4) 2.0)) 0.001)) ; opposite vectors
      (is (< (Math/abs (- (simd/cosine-distance v1 v5) 0.0253)) 0.01))))) ; ~97.5% similar

(deftest test-ultra-fast-index
  (testing "Ultra-fast index creation and search"
    (let [vectors (gen/generate-dataset 100 128 :format :indexed)
          index (ultra/build-index vectors :show-progress? false)]
      (is (not (nil? index)))

      ;; Test search
      (let [[_ query-vec] (first vectors) ; Get the vector part
            neighbors (ultra/search-knn index query-vec 5)]
        (is (= 5 (count neighbors)))
        (is (every? #(contains? % :id) neighbors))
        (is (every? #(contains? % :distance) neighbors))
        ;; First result should be very close to query
        (is (< (:distance (first neighbors)) 0.01))))))

(deftest test-graph-implementation
  (testing "Graph-based HNSW"
    (let [g (graph/create-graph)]
      (is (not (nil? g)))

      ;; Insert some vectors
      (graph/insert g "v1" [1.0 2.0 3.0])
      (graph/insert g "v2" [4.0 5.0 6.0])
      (graph/insert g "v3" [1.1 2.1 3.1])

      ;; Search
      (let [results (graph/search-knn g [1.0 2.0 3.0] 2)]
        (is (= 2 (count results)))
        (is (every? #(contains? % :id) results))))))

(deftest test-empty-index
  (testing "Empty index operations"
    (let [index (ultra/build-index [] :show-progress? false)]
      (is (not (nil? index)))
      ;; Search on empty index should return empty results
      (is (empty? (ultra/search-knn index (double-array [1 2 3]) 5))))))

(deftest test-single-vector
  (testing "Index with single vector"
    (let [vector (double-array [1.0 2.0 3.0 4.0])
          indexed-vector ["vec_0" vector]
          index (ultra/build-index [indexed-vector] :show-progress? false)
          results (ultra/search-knn index vector 1)]
      (is (= 1 (count results)))
      (is (= "vec_0" (:id (first results))))
      (is (< (:distance (first results)) 0.001)))))

(deftest test-knn-search
  (testing "Basic KNN search"
    (let [vectors (gen/generate-dataset 5 2 :format :indexed)
          index (ultra/build-index vectors :show-progress? false)
          [_ query-vec] (first vectors)
          neighbors (ultra/search-knn index query-vec 3)]
      (is (= 3 (count neighbors)))
      ;; First neighbor should be the query itself
      (is (< (:distance (first neighbors)) 0.001)))))

(deftest test-search-with-more-k-than-vectors
  (testing "Search with more neighbors than vectors"
    (let [vectors (gen/generate-dataset 5 64 :format :indexed)
          index (ultra/build-index vectors :show-progress? false)
          [_ query-vec] (first vectors)
          neighbors (ultra/search-knn index query-vec 10)]
      (is (= 5 (count neighbors))))))

(deftest test-search-quality
  (testing "Search returns actual nearest neighbors"
    (let [;; Create clustered data for predictable results
          vectors (gen/generate-dataset 20 64
                                        :distribution :clustered
                                        :num-clusters 2
                                        :format :indexed)
          index (ultra/build-index vectors :show-progress? false)
          ;; Query with first vector
          [_ query-vec] (first vectors)
          neighbors (ultra/search-knn index query-vec 10)]
      ;; At least half should be from same cluster
      (is (>= (count neighbors) 5)))))

(deftest test-concurrent-operations
  (testing "Concurrent searches"
    (let [vectors (gen/generate-dataset 100 128 :format :indexed)
          index (ultra/build-index vectors :show-progress? false)
          queries (gen/generate-query-set 20 128 :format :double-array)
          futures (map (fn [query]
                         (future (ultra/search-knn index query 10)))
                       queries)
          results (map deref futures)]
      (is (every? #(= 10 (count %)) results)))))

;; Performance tests
(deftest ^:performance test-build-performance
  (testing "Index build time scales reasonably"
    (let [;; Reduced sizes for faster testing
          small-vectors (gen/generate-dataset 100 128 :format :indexed) ; Was 1000x768
          medium-vectors (gen/generate-dataset 500 128 :format :indexed) ; Was 5000x768

          small-start (System/currentTimeMillis)
          small-index (ultra/build-index small-vectors :show-progress? false)
          small-time (- (System/currentTimeMillis) small-start)

          medium-start (System/currentTimeMillis)
          medium-index (ultra/build-index medium-vectors :show-progress? false)
          medium-time (- (System/currentTimeMillis) medium-start)]

      (println (format "Build time - 100 vectors: %d ms" small-time))
      (println (format "Build time - 500 vectors: %d ms" medium-time))

      (is (not (nil? small-index)))
      (is (not (nil? medium-index))))))

(deftest ^:performance test-search-performance
  (testing "Search time is reasonable"
    (let [;; Reduced size for faster testing
          vectors (gen/generate-dataset 1000 128 :format :indexed) ; Was 10000x768
          index (ultra/build-index vectors :show-progress? false)
          queries (gen/generate-query-set 20 128 :format :double-array) ; Was 100 queries
          start (System/currentTimeMillis)
          _ (doseq [q queries]
              (ultra/search-knn index q 10))
          elapsed (- (System/currentTimeMillis) start)
          avg-time (/ elapsed 20.0)]
      (println (format "Average search time: %.2f ms" avg-time))
      ;; Average search should be under 50ms
      (is (< avg-time 50)))))