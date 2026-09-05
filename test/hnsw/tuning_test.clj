(ns hnsw.tuning-test
  (:require [clojure.test :refer :all]
            [hnsw.api :as api]
            [hnsw.tuning :as tuning]))

(def vectors [["first" [1.0 0.0]] ["second" [0.0 1.0]] ["third" [-1.0 0.0]]])
(def options {:configurations [{:num-threads 1 :construction-precision :float32}
                               {:num-threads 1 :construction-precision :float64}]
              :ef-search-values [3] :target-recall 1.0 :k 1
              :build-runs 2 :query-runs 1 :warmup-size 0})

(deftest calibration-and-independent-validation
  (let [report (apply api/tune vectors [[1 0] [0 1]] [[0.9 0.1] [-0.9 0.1]] (mapcat identity options))]
    (is (= :validated (:status report)))
    (is (:accepted? report))
    (is (= 1.0 (get-in report [:validation :recall])))
    (is (= 4 (count (:samples report))))
    (is (= 2 (count (:results report))))
    (is (= 1.0 (get-in report [:selected :min-calibration-recall])))
    (is (= 3 (:ef-search (:recommended-options report))))))

(deftest refuses-infeasible-recommendation
  (let [report (apply api/tune vectors [[1 0]] [[0 1]]
                      (mapcat identity (assoc options :max-build-seconds 1e-12)))]
    (is (= :no-feasible-configuration (:status report)))
    (is (false? (:accepted? report)))
    (is (nil? (:recommended-options report)))
    (is (nil? (:validation report)))))

(deftest rejects-invalid-tuning-inputs
  (is (thrown? IllegalArgumentException
               (apply api/tune vectors [[1 0]] [[1 0]] (mapcat identity options))))
  (doseq [invalid [{:target-recall 1.1} {:k 0} {:build-runs 0} {:query-runs 0}
                   {:configurations []} {:ef-search-values []} {:ef-search-values [0]}
                   {:configurations [{:metric :euclidean}]} {:objective :unknown}
                   {:max-query-ms -1} {:warmup-size -1} {:target-recal 0.5}]]
    (is (thrown? IllegalArgumentException
                 (apply api/tune vectors [[1 0]] [[0 1]] (mapcat identity (merge options invalid)))))))

(deftest failed-held-out-validation-is-not-a-recommendation
  (with-redefs-fn {#'tuning/measure-search
                  (fn [_ queries _ _ _ _]
                    {:query-ms 0.001 :recall (if (= [[0 1]] queries) 0.0 1.0)})}
    (fn []
      (let [report (apply api/tune vectors [[1 0]] [[0 1]] (mapcat identity options))]
        (is (= :validation-failed (:status report)))
        (is (some? (:selected report)))
        (is (false? (:accepted? report)))
        (is (nil? (:recommended-options report)))))))
