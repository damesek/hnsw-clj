(ns hnsw.integration-test
  "Integration tests for HNSW implementations including Lightning mode"
  (:require [clojure.test :refer :all]
            [hnsw.api :as api]
            [hnsw.ultra-fast :as ultra]
            [hnsw.lightning :as lightning]
            [hnsw.helper.parallel-search :as ps]
            [hnsw.helper.index-io :as io]
            [data-generator :as gen]
            [clojure.java.io :as jio]))

(deftest test-api-integration
  (testing "Full API workflow with real data"
    (let [vectors (gen/generate-dataset 1000 768)
          queries (gen/generate-query-set 10 768)
          index (api/index vectors :metric :euclidean)
          results (map #(api/search index % 5) queries)]
      (is (every? #(= 5 (count %)) results))
      (is (every? #(every? :distance %) results)))))

(deftest test-ultra-fast-implementation
  (testing "Ultra-fast index build and search"
    (let [vectors (gen/generate-dataset 5000 768)
          index (ultra/build-index vectors)
          query (first vectors)
          neighbors (ultra/search-knn index query 10)]
      (is (= 10 (count neighbors)))
      ;; First result should be the query itself with distance 0
      (is (< (:distance (first neighbors)) 0.001)))))

(deftest test-lightning-mode
  (testing "Lightning mode with default settings"
    (let [vectors (gen/generate-dataset 10000 768)
          index (lightning/build-index vectors)
          query (gen/add-noise (first vectors) 0.1)
          neighbors (lightning/search-knn index query 10)]
      (is (= 10 (count neighbors)))
      (is (every? #(contains? % :distance) neighbors))))

  (testing "Lightning mode with custom partitions"
    (let [vectors (gen/generate-dataset 10000 768)
          index (lightning/build-index vectors
                                       :num-partitions 32
                                       :smart-partition? true)
          query (first vectors)
          neighbors (lightning/search-knn index query 10 0.25 true)]
      (is (= 10 (count neighbors))))))

(deftest test-parallel-search
  (testing "Parallel search with futures"
    (let [vectors (gen/generate-dataset 5000 768)
          index (ultra/build-index vectors)
          queries (gen/generate-query-set 20 768)
          search-fn (fn [idx q k] (ultra/search-knn idx q k))
          results (ps/parallel-search-futures index queries 10 search-fn 4)]
      (is (= 20 (count results)))
      (is (every? #(= 10 (count %)) results))))

  (testing "Parallel search with core.async"
    (let [vectors (gen/generate-dataset 5000 768)
          index (ultra/build-index vectors)
          queries (gen/generate-query-set 20 768)
          search-fn (fn [idx q k] (ultra/search-knn idx q k))
          results (ps/parallel-search-async index queries 10 search-fn 4)]
      (is (= 20 (count results)))
      (is (every? #(= 10 (count %)) results)))))

(deftest test-index-persistence
  (testing "Save and load ultra-fast index"
    (let [vectors (gen/generate-dataset 1000 512)
          original-index (ultra/build-index vectors)
          filepath "/tmp/test-ultra-index.hnsw"
          _ (io/save-index original-index filepath)
          loaded-index (io/load-index filepath)
          query (first vectors)
          original-results (ultra/search-knn original-index query 5)
          loaded-results (ultra/search-knn loaded-index query 5)]
      (is (= (map :id original-results) (map :id loaded-results)))))

  (testing "Save and load lightning index"
    (let [vectors (gen/generate-dataset 1000 512)
          original-index (lightning/build-index vectors :num-partitions 8)
          filepath "/tmp/test-lightning-index.hnsw"
          _ (io/save-index original-index filepath)
          loaded-index (io/load-index filepath)
          query (first vectors)
          original-results (lightning/search-knn original-index query 5)
          loaded-results (lightning/search-knn loaded-index query 5)]
      (is (= (count original-results) (count loaded-results))))))

(deftest test-different-vector-dimensions
  (testing "Small dimensions (384)"
    (let [vectors (gen/generate-dataset 1000 384)
          index (ultra/build-index vectors)
          query (first vectors)
          neighbors (ultra/search-knn index query 10)]
      (is (= 10 (count neighbors)))))

  (testing "Medium dimensions (768)"
    (let [vectors (gen/generate-dataset 1000 768)
          index (ultra/build-index vectors)
          query (first vectors)
          neighbors (ultra/search-knn index query 10)]
      (is (= 10 (count neighbors)))))

  (testing "Large dimensions (1536)"
    (let [vectors (gen/generate-dataset 500 1536)
          index (ultra/build-index vectors)
          query (first vectors)
          neighbors (ultra/search-knn index query 10)]
      (is (= 10 (count neighbors)))))

  (testing "Extra large dimensions (3072)"
    (let [vectors (gen/generate-dataset 200 3072)
          index (ultra/build-index vectors)
          query (first vectors)
          neighbors (ultra/search-knn index query 5)]
      (is (= 5 (count neighbors))))))

(deftest test-clustered-data
  (testing "Search quality on clustered data"
    (let [vectors (gen/generate-dataset 5000 512
                                        :distribution :clustered
                                        :num-clusters 5
                                        :noise-level 0.1)
          index (ultra/build-index vectors)
          ;; Pick a vector from the dataset
          query-idx 100
          query (nth vectors query-idx)
          neighbors (ultra/search-knn index query 20)]
      ;; The query itself should be the first result
      (is (= query-idx (:id (first neighbors))))
      (is (< (:distance (first neighbors)) 0.001))
      ;; Results should be sorted by distance
      (let [distances (map :distance neighbors)]
        (is (= distances (sort distances)))))))

(deftest test-search-accuracy
  (testing "Recall rate for nearest neighbor search"
    (let [vectors (gen/generate-dataset 1000 128)
          index (ultra/build-index vectors)
          query (nth vectors 42)
          k 10
          ;; Get ground truth by brute force
          ground-truth (take k
                             (sort-by #(gen/vector-distance query %)
                                      vectors))
          ground-truth-indices (map #(.indexOf vectors %) ground-truth)
          ;; Get HNSW results
          hnsw-results (ultra/search-knn index query k)
          hnsw-indices (map :id hnsw-results)
          ;; Calculate recall
          recall (/ (count (filter (set ground-truth-indices) hnsw-indices))
                    k)]
      ;; Should have at least 80% recall
      (is (>= recall 0.8)
          (format "Recall rate %.2f is below 0.8" recall)))))

(deftest test-concurrent-mixed-operations
  (testing "Concurrent building and searching"
    (let [initial-vectors (gen/generate-dataset 1000 256)
          index (atom (ultra/build-index initial-vectors))
          additional-vectors (gen/generate-dataset 500 256)
          queries (gen/generate-query-set 50 256)
          ;; Start search threads
          search-futures (repeatedly 5
                                     #(future
                                        (Thread/sleep (rand-int 100))
                                        (ultra/search-knn @index
                                                          (rand-nth queries) 10)))
          ;; Rebuild index with more data
          _ (Thread/sleep 50)
          _ (reset! index (ultra/build-index
                           (concat initial-vectors additional-vectors)))
          ;; More searches
          more-search-futures (repeatedly 5
                                          #(future
                                             (ultra/search-knn @index
                                                               (rand-nth queries) 10)))
          all-results (map deref (concat search-futures more-search-futures))]
      (is (every? #(> (count %) 0) all-results)))))

(deftest ^:performance test-large-scale-performance
  (testing "Performance on large dataset"
    (let [vectors (gen/generate-dataset 30000 768) ; Bible-sized dataset
          _ (println "Building index for 30k vectors...")
          build-start (System/currentTimeMillis)
          index (ultra/build-index vectors)
          build-time (- (System/currentTimeMillis) build-start)
          _ (println (format "Build time: %.2f seconds" (/ build-time 1000.0)))

          queries (gen/generate-query-set 100 768)
          search-start (System/currentTimeMillis)
          results (doall (map #(ultra/search-knn index % 10) queries))
          search-time (- (System/currentTimeMillis) search-start)
          avg-search-time (/ search-time 100.0)]

      (println (format "Average search time: %.2f ms" avg-search-time))
      (is (every? #(= 10 (count %)) results))
      ;; Build should be under 30 seconds for 30k vectors
      (is (< build-time 30000))
      ;; Average search should be under 20ms
      (is (< avg-search-time 20)))))

(deftest ^:performance test-lightning-vs-ultra
  (testing "Compare Lightning and Ultra-fast modes"
    (let [vectors (gen/generate-dataset 10000 768)
          queries (gen/generate-query-set 50 768)

          ;; Ultra-fast
          ultra-build-start (System/currentTimeMillis)
          ultra-index (ultra/build-index vectors)
          ultra-build-time (- (System/currentTimeMillis) ultra-build-start)

          ultra-search-start (System/currentTimeMillis)
          ultra-results (doall (map #(ultra/search-knn ultra-index % 10) queries))
          ultra-search-time (- (System/currentTimeMillis) ultra-search-start)

          ;; Lightning
          lightning-build-start (System/currentTimeMillis)
          lightning-index (lightning/build-index vectors :num-partitions 24)
          lightning-build-time (- (System/currentTimeMillis) lightning-build-start)

          lightning-search-start (System/currentTimeMillis)
          lightning-results (doall (map #(lightning/search-knn lightning-index % 10)
                                        queries))
          lightning-search-time (- (System/currentTimeMillis) lightning-search-start)]

      (println "\nPerformance Comparison (10k vectors, 50 queries):")
      (println (format "Ultra-fast - Build: %.2fs, Search: %.2fms avg"
                       (/ ultra-build-time 1000.0)
                       (/ ultra-search-time 50.0)))
      (println (format "Lightning  - Build: %.2fs, Search: %.2fms avg"
                       (/ lightning-build-time 1000.0)
                       (/ lightning-search-time 50.0)))

      ;; Both should return valid results
      (is (every? #(= 10 (count %)) ultra-results))
      (is (every? #(= 10 (count %)) lightning-results))

      ;; Lightning should be faster for search
      (is (< lightning-search-time (* ultra-search-time 2))
          "Lightning should be significantly faster than Ultra-fast"))))