(ns hnsw.maintenance-test
  (:require [clojure.set :as sets]
            [clojure.test :refer :all]
            [hnsw.api :as api]
            [hnsw.helper.index-io :as index-io]
            [hnsw.ultra-fast :as ultra])
  (:import [java.nio.file Files]
           [java.util Random]))

(defn- random-data [size dimensions seed]
  (let [random (Random. seed)]
    (mapv (fn [ordinal]
            [(str ordinal) (double-array (repeatedly dimensions #(.nextGaussian random)))])
          (range size))))

(defn- graph [index] (:graph @(:state index)))

(defn- edges [graph]
  (set (for [node (vals (.nodes graph))
             level (range (inc (.level node)))
             neighbor (aget (.neighbors node) level)]
         [(.id node) neighbor level])))

(defn- check-graph! [index]
  (let [graph (graph index)
        nodes (into {} (.nodes graph))
        links (edges graph)
        maintenance (.get (.maintenance graph))]
    (is (= (api/size index) (count nodes) (.get (.element-count graph))))
    (is (= (count nodes) (count (set (map #(.ordinal %) (vals nodes))))))
    (is (every? (fn [[source target level]]
                  (and (not= source target) (contains? nodes target)
                       (<= level (.level (get nodes target))))) links))
    (is (every? (fn [node]
                  (every? (fn [level]
                            (<= (count (aget (.neighbors node) level))
                                (if (zero? level) (.max-M graph) (.M graph))))
                          (range (inc (.level node))))) (vals nodes)))
    (if (empty? nodes)
      (is (nil? (.get (.entry-point graph))))
      (is (= (apply max (map #(.level %) (vals nodes)))
             (.level (get nodes (.get (.entry-point graph)))))))
    (when maintenance
      (is (= (set (keys nodes)) (set (keys (:incoming maintenance)))))
      (is (= links (set (for [[target incoming] (:incoming maintenance)
                             level (range (alength incoming))
                             source (aget incoming level)]
                         [source target level]))))
      (is (= (set (map (fn [node] [(.level node) (.id node)]) (vals nodes)))
             (set (for [[level identifiers] (:levels maintenance) identifier identifiers]
                    [level identifier]))))
      (let [free (vec (:free-ordinals maintenance))
            occupied (set (map #(.ordinal %) (vals nodes)))]
        (is (= (count free) (count (set free))))
        (is (empty? (sets/intersection occupied (set free)))))
      (is (= (.get (.next-ordinal graph))
             (+ (count nodes) (count (:free-ordinals maintenance))))))))

(deftest local-mutations-do-not-rebuild
  (doseq [metric [:cosine :euclidean]]
    (with-open [index (api/index (random-data 100 12 7) :metric metric :num-threads 1)]
      (let [original (graph index)
            nodes (into {} (.nodes original))
            capacity (.get (.next-ordinal original))
            snapshots (atom 0)
            optimize ultra/optimize-search!]
        (is (nil? (.get (.maintenance original))))
        (with-redefs [ultra/build-index (fn [& _] (throw (AssertionError. "Unexpected full rebuild")))
                      ultra/optimize-search! (fn [graph] (swap! snapshots inc) (optimize graph))]
          (api/delete-vectors! index ["3" "5" "5" "missing"])
          (is (= 1 @snapshots))
          (api/update-vector! index "7" (repeat 12 -1.0) {:updated true})
          (is (= 2 @snapshots)))
        (is (identical? original (graph index)))
        (is (every? (fn [[identifier node]]
                      (or (#{"3" "5" "7"} identifier)
                          (identical? node (.get (.nodes original) identifier)))) nodes))
        (is (= (.ordinal (get nodes "7")) (.ordinal (.get (.nodes original) "7"))))
        (is (= (.level (get nodes "7")) (.level (.get (.nodes original) "7"))))
        (is (= capacity (.get (.next-ordinal original))))
        (is (= {:updated true} (:metadata (first (api/search index (repeat 12 -1.0) 1)))))
        (check-graph! index)))))

(deftest directed-links-deleted-chains-and-entry-point
  (with-open [index (api/index [["a" [0.0]] ["b" [1.0]] ["c" [2.0]]
                                ["d" [3.0]] ["isolated" [100.0]]]
                               :metric :euclidean :M 2 :num-threads 1)]
    (let [graph (graph index)]
      (doseq [node (vec (vals (.nodes graph)))]
        (.put (.nodes graph) (.id node)
              (ultra/->UltraNode (.id node) (.vector node) 0 (.ordinal node)
                                (object-array [(java.util.concurrent.ConcurrentHashMap/newKeySet)])
                                (.build-vector node))))
      (doseq [[source target] [["a" "b"] ["b" "c"] ["c" "d"]]]
        (.add (aget (.neighbors (.get (.nodes graph) source)) 0) target))
      (.set (.entry-point graph) "a")
      (api/delete-vectors! index ["b" "c"])
      (is (= #{"a" "d"} (set (map :id (api/search index [3.0] 10 20)))))
      (is (empty? (aget (.neighbors (.get (.nodes graph) "isolated")) 0)))
      (check-graph! index)
      (api/delete-vector! index "a")
      (check-graph! index))))

(deftest deletes-upper-entry-and-reuses-slots-after-parallel-add
  (with-open [index (api/index (random-data 200 16 19) :metric :cosine :num-threads 4
                               :construction-precision :float32 :compact-precision :float32)]
    (let [graph (graph index)
          entry (.get (.entry-point graph))
          capacity (.get (.next-ordinal graph))
          deleted (conj (set (map str (range 0 100))) entry)]
      (api/delete-vectors! index deleted)
      (check-graph! index)
      (api/add-vectors! index (mapv (fn [[identifier vector]] [(str "new-" identifier) vector])
                                    (random-data (count deleted) 16 33)))
      (is (= capacity (.get (.next-ordinal graph))))
      (check-graph! index)
      (api/delete-vector! index "new-0")
      (check-graph! index)
      (is (not-any? #(contains? (conj deleted "new-0") (:id %))
                    (api/search index (repeat 16 1.0) 200 200))))))

(deftest invalid-updates-and-no-op-deletes-leave-topology-unchanged
  (with-open [index (api/index (random-data 30 8 9) :metric :cosine :num-threads 1)]
    (let [before (index-io/index-data (graph index))
          snapshot (.get (.search-index (graph index)))]
      (doseq [vector [[1.0] (repeat 8 0.0) (repeat 8 Double/NaN)]]
        (is (thrown? IllegalArgumentException (api/update-vector! index "0" vector))))
      (is (thrown? IllegalArgumentException (api/update-vector! index "missing" (repeat 8 1.0))))
      (api/delete-vectors! index ["missing" "missing"])
      (api/delete-vectors! index [])
      (is (= before (index-io/index-data (graph index))))
      (is (identical? snapshot (.get (.search-index (graph index)))))
      (is (nil? (.get (.maintenance (graph index))))))))

(deftest empty-and-single-node-mutations
  (with-open [index (api/index [["only" [1 0] false]] :metric :cosine :num-threads 1)]
    (dotimes [_ 20] (api/update-vector! index "only" [0 1]))
    (is (= false (:metadata (first (api/search index [0 1] 1)))))
    (is (= 1 (.get (.next-ordinal (graph index)))))
    (check-graph! index)
    (api/delete-vector! index "only")
    (check-graph! index)
    (is (= [] (api/search index [1 0] 10)))
    (api/add-vector! index "again" [1 1])
    (is (= 1 (.get (.next-ordinal (graph index)))))
    (is (= "again" (:id (first (api/search index [1 1] 1)))))
    (check-graph! index)))

(deftest fixed-size-churn-reuses-ordinals-and-keeps-inverse-links
  (with-open [index (api/index (random-data 100 8 28) :metric :cosine :num-threads 4)]
    (dotimes [cycle 8]
      (let [deleted (mapv str (range 10))
            replacement (random-data 10 8 (+ cycle 80))]
        (api/delete-vectors! index deleted)
        (api/add-vectors! index replacement)
        (api/update-vector! index "50" (second (last replacement)))
        (is (= 100 (.get (.next-ordinal (graph index)))))
        (check-graph! index)
        (let [before (edges (graph index))]
          (is (thrown? IllegalArgumentException (api/add-vectors! index replacement)))
          (is (= before (edges (graph index)))))))
    (let [original (graph index)]
      (api/clear! index)
      (is (not (identical? original (graph index))))
      (is (nil? (.get (.maintenance (graph index)))))
      (api/add-vector! index "new" (repeat 8 1.0))
      (check-graph! index))))

(deftest persistence-after-local-mutations
  (let [path (Files/createTempFile "hnsw-local" ".edn" (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (with-open [index (api/index (random-data 80 12 41) :metric :cosine :num-threads 1
                                   :construction-precision :float32)]
        (api/delete-vectors! index (map str (range 20)))
        (api/update-vector! index "40" (repeat 12 -1.0))
        (api/save-index index (str path))
        (with-open [loaded (api/load-index (str path))]
          (is (= (api/search index (repeat 12 -1.0) 10)
                 (api/search loaded (repeat 12 -1.0) 10)))
          (is (nil? (.get (.maintenance (graph loaded)))))
          (api/delete-vector! loaded "21")
          (api/add-vector! loaded "replacement" (repeat 12 1.0))
          (check-graph! loaded)
          (api/delete-vectors! loaded (keys (:metadata @(:state loaded))))
          (is (= [] (api/search loaded (repeat 12 1.0) 10)))
          (check-graph! loaded)))
      (finally (Files/deleteIfExists path)))))

(defn- recall [index data queries metric]
  (let [distance (fn [query vector]
                   (if (= metric :cosine)
                     (ultra/cosine-distance-ultra query vector)
                     (ultra/euclidean-distance-ultra query vector)))]
    (/ (reduce + (for [query queries]
                   (let [expected (set (map first (take 10 (sort-by #(distance query (second %)) data))))
                         actual (set (map :id (api/search index query 10 100)))]
                     (count (sets/intersection expected actual)))))
       (double (* 10 (count queries))))))

(deftest recall-after-repeated-deletions-and-large-vector-moves
  (doseq [metric [:cosine :euclidean]]
    (let [data (random-data 400 16 108)
          queries (mapv second (random-data 40 16 333))
          active (atom (into {} data))]
      (with-open [index (api/index data :metric metric :M 12 :ef-construction 120
                                   :ef-search 100 :num-threads 1)]
        (is (>= (recall index @active queries metric) 0.97))
        (dotimes [cycle 3]
          (let [deleted (mapv str (range (* cycle 30) (+ (* cycle 30) 30)))
                moved (mapv (fn [[identifier vector]] [(str (+ 200 (Long/parseLong identifier))) vector])
                            (random-data 20 16 (+ 600 cycle)))]
            (api/delete-vectors! index deleted)
            (swap! active #(apply dissoc % deleted))
            (doseq [[identifier vector] moved]
              (api/update-vector! index identifier vector)
              (swap! active assoc identifier vector))
            (is (>= (recall index @active queries metric) 0.97))
            (check-graph! index)))))))

(deftest concurrent-readers-with-local-writer
  (with-open [index (api/index (random-data 80 8 63) :metric :cosine :num-threads 2)]
    (let [readers (mapv (fn [_] (future (dotimes [_ 30]
                                       (is (= 10 (count (api/search index (repeat 8 1.0) 10)))))))
                        (range 4))
          writer (future (doseq [identifier (map str (range 10))]
                           (api/update-vector! index identifier (repeat 8 -1.0))
                           (api/delete-vector! index identifier)))]
      (is (not= ::timeout (deref writer 30000 ::timeout)))
      (doseq [reader readers] (is (not= ::timeout (deref reader 30000 ::timeout))))
      (check-graph! index))))
