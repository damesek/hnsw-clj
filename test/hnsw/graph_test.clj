(ns hnsw.graph-test)
  ;; KIKOMMENTELVE A REQUIRE-OK
  ;; (:require [clojure.test :refer :all]
  ;;           [hnsw.graph :as hnsw]))

(defn approx=
  "Check if two floating point numbers are approximately equal"
  [a b tolerance]
  (< (Math/abs (- a b)) tolerance))

(deftest test-distance-functions
  (testing "Euclidean distance"
    (is (approx= 0.0 (hnsw/euclidean-distance [1 2 3] [1 2 3]) 0.001))
    (is (< 1.73 (hnsw/euclidean-distance [1 2 3] [2 3 4]) 1.74)))

  (testing "Cosine similarity"
    (is (approx= 1.0 (hnsw/cosine-similarity [1 0] [1 0]) 0.001))
    (is (approx= 0.0 (hnsw/cosine-similarity [1 0] [0 1]) 0.001)))

  (testing "Cosine distance"
    (is (approx= 0.0 (hnsw/cosine-distance [1 0] [1 0]) 0.001))
    (is (approx= 1.0 (hnsw/cosine-distance [1 0] [0 1]) 0.001))))

(deftest test-graph-creation
  (testing "Creating empty graph"
    (let [graph (hnsw/create-graph)]
      (is (= {} (:nodes graph)))
      (is (nil? (:entry-point graph)))
      (is (= 0 (:element-count graph)))))

  (testing "Creating graph with custom params"
    (let [params (assoc (hnsw/default-params) :M 8)
          graph (hnsw/create-graph params)]
      (is (= 8 (get-in graph [:params :M]))))))

(deftest test-insertion
  (testing "Inserting first element"
    (let [graph (hnsw/create-graph)
          graph (hnsw/insert graph "id1" [1 2 3])]
      (is (= 1 (:element-count graph)))
      (is (= "id1" (:entry-point graph)))
      (is (contains? (:nodes graph) "id1"))))

  (testing "Inserting multiple elements"
    (let [graph (hnsw/create-graph)
          graph (-> graph
                    (hnsw/insert "id1" [1 2 3])
                    (hnsw/insert "id2" [4 5 6])
                    (hnsw/insert "id3" [7 8 9]))]
      (is (= 3 (:element-count graph)))
      (is (= 3 (count (:nodes graph))))))

  (testing "Duplicate insertion throws exception"
    (let [graph (hnsw/create-graph)
          graph (hnsw/insert graph "id1" [1 2 3])]
      (is (thrown? IllegalArgumentException
                   (hnsw/insert graph "id1" [4 5 6]))))))

(deftest test-search
  (testing "Search in empty graph"
    (let [graph (hnsw/create-graph)
          results (hnsw/search-knn graph [1 2 3] 5)]
      (is (empty? results))))

  (testing "Search with one element"
    (let [graph (hnsw/create-graph)
          graph (hnsw/insert graph "id1" [1 2 3])
          results (hnsw/search-knn graph [1 2 3] 1)]
      (is (= 1 (count results)))
      (is (= "id1" (:id (first results))))
      (is (approx= 0.0 (:distance (first results)) 0.001))))

  (testing "Search with multiple elements"
    (let [graph (hnsw/create-graph)
          vectors [["id1" [0 0]]
                   ["id2" [1 0]]
                   ["id3" [0 1]]
                   ["id4" [1 1]]
                   ["id5" [0.5 0.5]]]
          graph (hnsw/insert-batch graph vectors)
          results (hnsw/search-knn graph [0 0] 3)]
      (is (= 3 (count results)))
      (is (= "id1" (:id (first results))))
      (is (< (:distance (first results))
             (:distance (second results))
             (:distance (nth results 2)))))))

(deftest test-batch-operations
  (testing "Batch insertion"
    (let [graph (hnsw/create-graph)
          vectors (map (fn [i] [(str "id" i) [i 0 0]]) (range 10))
          graph (hnsw/insert-batch graph vectors)]
      (is (= 10 (:element-count graph)))
      (is (= 10 (count (:nodes graph)))))))

(deftest test-graph-info
  (testing "Graph info for empty graph"
    (let [graph (hnsw/create-graph)
          info (hnsw/graph-info graph)]
      (is (= 0 (:nodes info)))
      (is (nil? (:entry-point info)))
      (is (= 0 (:levels info)))
      (is (= 0 (:avg-connections info)))))

  (testing "Graph info with elements"
    (let [graph (hnsw/create-graph)
          vectors (map (fn [i] [(str "id" i) [i 0 0]]) (range 5))
          graph (hnsw/insert-batch graph vectors)
          info (hnsw/graph-info graph)]
      (is (= 5 (:nodes info)))
      (is (not (nil? (:entry-point info))))
      (is (> (:levels info) 0))
      (is (> (:avg-connections info) 0)))))

(deftest test-performance
  (testing "Performance with larger dataset"
    (let [graph (hnsw/create-graph)
          num-vectors 100
          dimension 10
          vectors (map (fn [i]
                         [(str "id" i)
                          (vec (repeatedly dimension rand))])
                       (range num-vectors))
          graph (hnsw/insert-batch graph vectors)
          query (vec (repeatedly dimension rand))
          results (hnsw/search-knn graph query 10)]
      (is (= 10 (count results)))
      (is (every? #(contains? % :id) results))
      (is (every? #(contains? % :distance) results))
      (is (apply <= (map :distance results))))))

(deftest test-different-distance-metrics
  (testing "Graph with cosine distance"
    (let [params (assoc (hnsw/default-params)
                        :distance-fn hnsw/cosine-distance)
          graph (hnsw/create-graph params)
          vectors [["id1" [1 0 0]]
                   ["id2" [0 1 0]]
                   ["id3" [0 0 1]]
                   ["id4" [1 1 0]]]
          graph (hnsw/insert-batch graph vectors)
          results (hnsw/search-knn graph [1 0 0] 2)]
      (is (= "id1" (:id (first results))))
      (is (< (:distance (first results)) 0.01)))))
