(ns tuning-benchmark
  (:require [bible-benchmark :as bible]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [hnsw.api :as api]))

(defn- env-long [name default]
  (Long/parseLong (or (System/getenv name) (str default))))

(defn -main [& _]
  (let [environment (assoc (#'bible/environment) :num-queries (env-long "QUALITY_QUERIES" 1000))
        _ (when-not (.isFile (io/file (:embedding-dir environment) "manifest.json"))
            (throw (IllegalArgumentException. "Prepare the shared Bible embeddings first")))
        {:keys [vectors queries source dimensions]} (#'bible/load-data environment)
        data (mapv (fn [position vector] [(str position) vector]) (range) vectors)
        query-vectors (->> queries (map (comp vec :vector)) distinct (mapv double-array))
        middle (quot (count query-vectors) 2)
        grid (if-let [value (System/getenv "BUILD_CONFIGS_EDN")]
               (edn/read-string value)
               [{:construction-precision :float32 :ef-construction 100}
                {:construction-precision :float32 :ef-construction 200}])
        configurations (mapv #(merge {:M (:M environment) :num-threads (:threads environment)
                                      :seed (:seed environment)} %) grid)
        result (api/tune data (subvec query-vectors 0 middle) (subvec query-vectors middle)
                         :configurations configurations
                         :ef-search-values (edn/read-string (or (System/getenv "EF_SEARCH_VALUES") "[50 80 120]"))
                         :target-recall (Double/parseDouble (or (System/getenv "TARGET_RECALL") "0.98"))
                         :build-runs (env-long "BUILD_RUNS" 3) :query-runs (env-long "QUERY_RUNS" 3)
                         :objective (keyword (or (System/getenv "OBJECTIVE") "build-time")))]
    (prn (assoc result :source source :vectors (count vectors) :dimensions dimensions
                       :query-split :deduplicated-deterministic-half
                       :java (System/getProperty "java.version") :architecture (System/getProperty "os.arch")
                       :measurement :managed-api-including-validation-and-snapshot))
    (shutdown-agents)))
