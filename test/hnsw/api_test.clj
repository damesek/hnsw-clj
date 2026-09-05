(ns hnsw.api-test
  (:require [clojure.edn :as edn]
            [clojure.test :refer :all]
            [hnsw.api :as api])
  (:import [java.nio.file Files]))

(deftest lifecycle
  (with-open [index (api/index [["a" [1 0] {:text "first"}] ["b" [0 1]]]
                               :metric :cosine :num-threads 1)]
    (is (= "a" (:id (first (api/search index [1 0] 1)))))
    (api/update-vector! index "a" [-1 0])
    (is (= {:text "first"} (:metadata (first (api/search index [-1 0] 1)))))
    (api/delete-vector! index "a")
    (is (= ["b"] (mapv :id (api/search index [1 0] 10))))
    (api/delete-vector! index "missing")
    (is (= 1 (api/size index)))
    (api/clear! index)
    (is (= [] (api/search index [1 0] 10)))
    (api/add-vector! index "c" [1 1])
    (is (= 1 (api/size index)))
    (is (thrown? IllegalArgumentException (api/update-vector! index "missing" [1 0])))))

(deftest validates-before-mutation
  (with-open [index (api/index [["a" [1 0]]] :metric :cosine)]
    (doseq [data [[["b" [0 1]] ["a" [1 1]]]
                  [["b" [0 1]] ["c" [1 2 3]]]
                  [["b" [0 1]] ["b" [1 0]]]
                  [["b" [Double/NaN 0]]]
                  [["b" [Double/POSITIVE_INFINITY 0]]]
                  [["b" [0 0]]]]]
      (is (thrown? IllegalArgumentException (api/add-vectors! index data)))
      (is (= 1 (api/size index))))
    (is (thrown? IllegalArgumentException (api/search index [1] 1)))
    (is (thrown? IllegalArgumentException (api/search index 2 1)))
    (is (thrown? IllegalArgumentException (api/search index [1 0] 0)))
    (is (thrown? IllegalArgumentException (api/search-batch index [[1 0]] 1 :num-threads 0))))
  (doseq [options [{:metric :wrong} {:M 1} {:ef-search 0} {:typo 1} {:num-threads 1.5}]]
    (is (thrown? IllegalArgumentException (apply api/index [] (mapcat identity options))))))

(deftest owns-vector-storage
  (let [vector (double-array [1 0])
        index (api/index [["a" vector] ["b" [2 0]]] :metric :euclidean)]
    (aset-double vector 0 1000.0)
    (is (= "a" (:id (first (api/search index [1 0] 1)))))
    (api/close! index)
    (api/close! index)
    (is (thrown? IllegalStateException (api/size index)))))

(deftest primitive-validation-preserves-input-contract
  (doseq [values [(double-array [1 -2 3]) (float-array [1 -2 3])
                  (int-array [1 -2 3]) (long-array [1 -2 3])
                  (object-array [1 -2 3]) [1 -2 3] '(1 -2 3)
                  (map identity [1 -2 3]) (java.util.ArrayList. [1 -2 3])]]
    (let [copy (#'api/checked-vector values 3 :cosine)]
      (is (= [1.0 -2.0 3.0] (vec copy)))
      (is (not (identical? values copy)))))
  (doseq [dimension [1 2 3 7 8 15 16 31 32 768]]
    (let [values (double-array (repeat dimension 0.5))
          copy (#'api/checked-vector values dimension :cosine)]
      (aset-double values 0 42.0)
      (is (= 0.5 (aget copy 0))))
    (doseq [position (distinct [0 (quot dimension 2) (dec dimension)])
            value [Double/NaN Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
      (let [values (double-array (repeat dimension 0.5))]
        (aset-double values position value)
        (is (thrown-with-msg? IllegalArgumentException #"finite numbers"
                             (#'api/checked-vector values dimension :cosine))))))
  (doseq [values [[Double/MAX_VALUE] [1.3e154 1.3e154]]
          metric [:cosine :euclidean]]
    (is (thrown-with-msg? IllegalArgumentException #"Vector norm"
                         (#'api/checked-vector values nil metric))))
  (doseq [values [[] [0.0 -0.0] [Double/MIN_VALUE]]]
    (is (thrown? IllegalArgumentException (#'api/checked-vector values nil :cosine))))
  (is (= [0.0 0.0] (vec (#'api/checked-vector [0 0] nil :euclidean))))
  (is (thrown? IllegalArgumentException (#'api/checked-vector [1 2] 3 :cosine))))

(deftest batch-validation-keeps-existing-ids-and-metadata-safe
  (with-open [index (api/index [["0" [1 0] false] ["1" [0 1] {:value 1}]]
                               :metric :euclidean :num-threads 1)]
    (doseq [batch [[["new" [2 3]] [0 [3 4]]]
                   [["new" [2 3]] ["new" [3 4]]]
                   [["new" [2 3]] [nil [3 4]]]
                   [["new" [2 3]] ["bad" [1.3e154 1.3e154]]]]]
      (is (thrown? IllegalArgumentException (api/add-vectors! index batch)))
      (is (= 2 (api/size index))))
    (let [existing (:metadata @(:state index))]
      (api/add-vectors! index (list ["2" [2 3] {:value 2}] ["3" [3 4]]))
      (is (= {"0" false "1" {:value 1}} existing)))
    (is (= [false {:value 1} {:value 2} nil]
           (mapv (fn [query] (:metadata (first (api/search index query 1))))
                 [[1 0] [0 1] [2 3] [3 4]])))
    (is (= (mapv #(api/search index % 2) [[1 0] [2 3]])
           (api/search-batch index [[1 0] [2 3]] 2 :num-threads 2)))))

(deftest empty-batches-and-dimension-inference
  (with-open [index (api/index [] :metric :euclidean)]
    (api/add-vectors! index [])
    (is (nil? (:dimensions (api/info index))))
    (is (thrown? IllegalArgumentException
                 (api/add-vectors! index [["a" [1 2]] ["b" [1 2 3]]])))
    (is (= 0 (api/size index)))
    (is (nil? (:dimensions (api/info index))))
    (api/add-vector! index "a" [1 2 3])
    (is (= 3 (:dimensions (api/info index))))
    (is (= [] (api/search-batch index [] 10)))))

(deftest absent-metadata-does-not-copy-result-vectors
  (let [neighbors [{:id "a" :distance 0.0} {:id "b" :distance 1.0}]]
    (is (identical? neighbors (#'api/results {:metadata {"a" nil "b" nil}} neighbors)))
    (is (= [{:id "a" :distance 0.0 :metadata false} {:id "b" :distance 1.0}]
           (#'api/results {:metadata {"a" false}} neighbors)))
    (is (= [{:id "a" :distance 0.0} {:id "b" :distance 1.0}] neighbors))))

(deftest managed-persistence
  (let [path (Files/createTempFile "hnsw-api" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (with-open [index (api/index [["a" [1 0] {:text "árvíz"}]] :metric :cosine
                                   :construction-precision :float32)]
        (api/save-index index (str path))
        (with-open [loaded (api/load-index (str path))]
          (is (= (api/search index [1 0] 1) (api/search loaded [1 0] 1)))
          (is (= 2 (:dimensions (api/info loaded))))
          (api/add-vector! loaded "b" [0 1])
          (api/delete-vector! loaded "a")
          (is (= "b" (:id (first (api/search loaded [0 1] 1)))))))
      (spit (str path) "{:format :hnsw-clj :version 99}")
      (is (thrown? IllegalArgumentException (api/load-index (str path))))
      (finally (Files/deleteIfExists path)))))

(deftest concurrent-readers-and-writer
  (with-open [index (api/index [["initial" [1 0]]] :metric :cosine :num-threads 2)]
    (let [readers (doall (repeatedly 4 #(future
                                        (dotimes [_ 30]
                                          (is (seq (api/search index [1 0] 2)))))))
          writer (future (doseq [ordinal (range 20)]
                           (api/add-vector! index (str ordinal) [1 1])))]
      (is (not= ::timeout (deref writer 30000 ::timeout)))
      (doseq [reader readers] (is (not= ::timeout (deref reader 30000 ::timeout))))
      (is (= 21 (api/size index)))
      (is (= (mapv #(api/search index % 2) [[1 0] [0 1]])
             (api/search-batch index [[1 0] [0 1]] 2 :num-threads 2))))))

(deftest failed-save-preserves-existing-file
  (let [path (Files/createTempFile "hnsw-save" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (with-open [index (api/index [["a" [1 0] false]])]
        (api/save-index index (str path))
        (is (false? (:metadata (first (api/search index [1 0] 1)))))
        (let [saved (slurp (str path))]
          (api/add-vector! index "b" [0 1] (Object.))
          (is (thrown? RuntimeException (api/save-index index (str path))))
          (is (= saved (slurp (str path))))))
      (finally (Files/deleteIfExists path)))))

(deftest removed-construction-codecs-are-rejected
  (is (thrown-with-msg? IllegalArgumentException #"Unknown options"
                        (api/index [] :construction-codec :unsupported)))
  (let [path (Files/createTempFile "hnsw-legacy-options" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (with-open [index (api/index [["a" [1 0]]] :metric :cosine)]
        (api/save-index index (str path)))
      (let [saved (edn/read-string (slurp (str path)))
            legacy (assoc-in saved [:options :construction-codec] :none)]
        (spit (str path) (pr-str legacy))
        (with-open [loaded (api/load-index (str path))]
          (is (= 1 (api/size loaded)))
          (is (not (contains? (:options @(:state loaded)) :construction-codec)))
          (api/add-vector! loaded "b" [0 1])
          (api/delete-vector! loaded "a")
          (is (= "b" (:id (first (api/search loaded [0 1] 1))))))
        (spit (str path) (pr-str (assoc-in legacy [:graph :construction-codec] {:removed true})))
        (is (thrown-with-msg? IllegalArgumentException #"removed construction codec"
                              (api/load-index (str path)))))
      (finally (Files/deleteIfExists path)))))
