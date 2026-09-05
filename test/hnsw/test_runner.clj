(ns hnsw.test-runner
  (:require [clojure.test :as test]
            [hnsw.api-test]
            [hnsw.filtered-test]
            [hnsw.maintenance-test]
            [hnsw.tuning-test]
            [hnsw.ultra-fast-test]))

(defn -main [& _]
  (let [result (test/run-tests 'hnsw.api-test 'hnsw.filtered-test 'hnsw.maintenance-test 'hnsw.tuning-test 'hnsw.ultra-fast-test)]
    (shutdown-agents)
    (when (pos? (+ (:fail result) (:error result)))
      (System/exit 1))))
