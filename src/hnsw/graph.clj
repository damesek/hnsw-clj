(ns hnsw.graph
  "HNSW (Hierarchical Navigable Small World) graph implementation"
  (:require [clojure.set :as set]))

;; Distance functions - Using Java Math instead of numeric-tower
(defn euclidean-distance
  "Calculate Euclidean distance between two vectors"
  [v1 v2]
  (Math/sqrt
   (reduce + (map (fn [a b] (Math/pow (- a b) 2)) v1 v2))))

(defn cosine-similarity
  "Calculate cosine similarity between two vectors"
  [v1 v2]
  (let [dot-product (reduce + (map * v1 v2))
        norm1 (Math/sqrt (reduce + (map #(* % %) v1)))
        norm2 (Math/sqrt (reduce + (map #(* % %) v2)))]
    (if (and (> norm1 0) (> norm2 0))
      (/ dot-product (* norm1 norm2))
      0)))

(defn cosine-distance
  "Calculate cosine distance (1 - cosine similarity)"
  [v1 v2]
  (- 1 (cosine-similarity v1 v2)))

;; HNSW Parameters
(defrecord HNSWParams [M ; Number of bi-directional links per node
                       max-M ; Maximum allowed connections
                       ef-construction ; Size of dynamic candidate list
                       ml ; Normalization factor for level assignment
                       seed ; Random seed for level assignment
                       distance-fn]) ; Distance function to use

(defn default-params
  "Create default HNSW parameters"
  []
  (->HNSWParams 16 16 200 (/ 1.0 (Math/log 2.0)) 42 euclidean-distance))

;; Node structure
(defrecord Node [id ; Unique identifier
                 vector ; Feature vector
                 level ; Node level in hierarchy
                 neighbors]) ; Map of level -> set of neighbor ids

;; HNSW Graph structure
(defrecord HNSWGraph [nodes ; Map of id -> Node
                      entry-point ; Entry point node id
                      params ; HNSW parameters
                      element-count ; Number of elements in graph
                      random-state]) ; Random number generator state

(defn create-graph
  "Create a new empty HNSW graph"
  ([]
   (create-graph (default-params)))
  ([params]
   (->HNSWGraph {} nil params 0 (java.util.Random. (:seed params)))))

;; Level assignment
(defn assign-level
  "Randomly assign level for a new node using exponential decay"
  [graph]
  (let [random (:random-state graph)
        ml (get-in graph [:params :ml])]
    (int (* ml (- (Math/log (.nextDouble random)))))))

;; Search utilities
(defn get-neighbors
  "Get neighbors of a node at a specific level"
  [graph node-id level]
  (get-in graph [:nodes node-id :neighbors level] #{}))

(defn distance
  "Calculate distance between two nodes"
  [graph id1 id2]
  (let [dist-fn (get-in graph [:params :distance-fn])
        v1 (get-in graph [:nodes id1 :vector])
        v2 (get-in graph [:nodes id2 :vector])]
    (dist-fn v1 v2)))

(defn search-layer
  "Search for nearest neighbors at a specific layer"
  [graph query-vec entry-points num-closest level]
  (let [dist-fn (get-in graph [:params :distance-fn])
        visited (atom #{})
        candidates (atom {}) ; Will store {id -> distance}
        nearest (atom {}) ; Will store {id -> distance}

        calc-dist (fn [id]
                    (dist-fn query-vec (get-in graph [:nodes id :vector])))]

    ;; Initialize with entry points
    (doseq [point entry-points]
      (let [d (calc-dist point)]
        (swap! candidates assoc point d)
        (swap! nearest assoc point d)
        (swap! visited conj point)))

    ;; Search expansion
    (while (seq @candidates)
      (let [;; Get candidate with minimum distance
            [current current-dist] (apply min-key second @candidates)
            _ (swap! candidates dissoc current)]

        ;; Process if within search bounds
        (when (or (< (count @nearest) num-closest)
                  (<= current-dist (if (empty? @nearest)
                                     Double/MAX_VALUE
                                     (apply max (vals @nearest)))))
          (let [neighbors (get-neighbors graph current level)]
            (doseq [neighbor neighbors]
              (when-not (contains? @visited neighbor)
                (swap! visited conj neighbor)
                (let [d (calc-dist neighbor)]
                  ;; Add to candidates if promising
                  (when (or (< (count @nearest) num-closest)
                            (< d (if (empty? @nearest)
                                   Double/MAX_VALUE
                                   (apply max (vals @nearest)))))
                    (swap! candidates assoc neighbor d)
                    (swap! nearest assoc neighbor d)
                    ;; Prune if we have too many
                    (when (> (count @nearest) num-closest)
                      (let [[furthest _] (apply max-key second @nearest)]
                        (swap! nearest dissoc furthest)))))))))))

    (keys @nearest)))

(defn get-neighbors-heuristic
  "Select M neighbors using a heuristic (pruning)"
  [graph candidates M level extend-candidates?]
  (let [nearest (atom (sorted-set-by
                       (fn [a b]
                         (let [d1 (second a) d2 (second b)]
                           (if (= d1 d2)
                             (compare (first a) (first b))
                             (compare d1 d2))))))
        discarded (atom [])]

    ;; Add candidates to nearest
    (doseq [[id dist] candidates]
      (swap! nearest conj [id dist]))

    ;; Pruning process
    (let [result (atom [])]
      (while (and (not (empty? @nearest)) (< (count @result) M))
        (let [[current-id current-dist] (first @nearest)]
          (swap! nearest disj [current-id current-dist])

          ;; Check if current is closer than existing results
          (if (empty? @result)
            (swap! result conj current-id)
            (let [closer-to-result?
                  (some (fn [res-id]
                          (< (distance graph current-id res-id) current-dist))
                        @result)]
              (if closer-to-result?
                (swap! discarded conj [current-id current-dist])
                (swap! result conj current-id))))))

      ;; Add discarded if we need more neighbors
      (when extend-candidates?
        (doseq [[id dist] @discarded]
          (when (< (count @result) M)
            (swap! result conj id))))

      @result)))

;; Connection management
(defn connect-nodes
  "Create bidirectional connection between two nodes at a level"
  [graph id1 id2 level]
  (-> graph
      (update-in [:nodes id1 :neighbors level] (fnil conj #{}) id2)
      (update-in [:nodes id2 :neighbors level] (fnil conj #{}) id1)))

(defn prune-connections
  "Prune connections of a node if it exceeds max-M"
  [graph node-id level]
  (let [max-M (if (= level 0)
                (* 2 (get-in graph [:params :max-M]))
                (get-in graph [:params :max-M]))
        neighbors (get-neighbors graph node-id level)]

    (if (<= (count neighbors) max-M)
      graph
      (let [node-vec (get-in graph [:nodes node-id :vector])
            dist-fn (get-in graph [:params :distance-fn])
            candidates (map (fn [id] [id (dist-fn node-vec
                                                  (get-in graph [:nodes id :vector]))])
                            neighbors)
            new-neighbors (get-neighbors-heuristic graph candidates max-M level false)]

        ;; Remove old connections and add new ones
        (reduce (fn [g neighbor]
                  (if (contains? (set new-neighbors) neighbor)
                    g
                    (-> g
                        (update-in [:nodes node-id :neighbors level] disj neighbor)
                        (update-in [:nodes neighbor :neighbors level] disj node-id))))
                graph
                neighbors)))))

;; Insertion
(defn insert
  "Insert a new element into the HNSW graph"
  [graph id vector]
  (if (contains? (:nodes graph) id)
    (throw (IllegalArgumentException. (str "Node with id " id " already exists")))

    (let [level (assign-level graph)
          new-node (->Node id vector level
                           (into {} (map (fn [l] [l #{}]) (range (inc level)))))

          ;; If this is the first node
          graph (if (empty? (:nodes graph))
                  (-> graph
                      (assoc :entry-point id)
                      (assoc-in [:nodes id] new-node)
                      (update :element-count inc))

                  ;; Insert into existing graph
                  (let [graph (-> graph
                                  (assoc-in [:nodes id] new-node)
                                  (update :element-count inc))

                        ;; Find nearest neighbors at all levels
                        entry-point (:entry-point graph)
                        dist-fn (get-in graph [:params :distance-fn])
                        ef-construction (get-in graph [:params :ef-construction])
                        M (get-in graph [:params :M])

                        ;; Start from top layer
                        entry-level (get-in graph [:nodes entry-point :level])

                        ;; Search from top to target layer
                        nearest (atom [entry-point])
                        graph (atom graph)]

                    ;; Search phase - from top to layer 0
                    (doseq [lc (reverse (range (inc (min level entry-level))))]
                      (reset! nearest
                              (search-layer @graph vector @nearest
                                            (if (> lc level) 1 ef-construction) lc)))

                    ;; Insert phase - connect at each layer
                    (doseq [lc (range (inc level))]
                      (let [candidates (search-layer @graph vector @nearest M lc)
                            m (if (= lc 0) (* 2 M) M)]

                        ;; Add bidirectional links
                        (doseq [neighbor candidates]
                          (swap! graph connect-nodes id neighbor lc)
                          (swap! graph prune-connections neighbor lc))))

                    ;; Update entry point if necessary
                    (if (> level entry-level)
                      (assoc @graph :entry-point id)
                      @graph)))]
      graph)))

;; Search
(defn search-knn
  "Search for k nearest neighbors"
  [graph query-vec k]
  (if (empty? (:nodes graph))
    []
    (let [entry-point (:entry-point graph)
          entry-level (get-in graph [:nodes entry-point :level])
          ef (max k 50) ; Dynamic ef parameter
          nearest (atom [entry-point])]

      ;; Search from top layer to layer 0
      (doseq [level (reverse (range (inc entry-level)))]
        (reset! nearest
                (search-layer graph query-vec @nearest
                              (if (> level 0) 1 ef) level)))

      ;; Return top k results with distances
      (let [dist-fn (get-in graph [:params :distance-fn])
            results (map (fn [id]
                           {:id id
                            :distance (dist-fn query-vec
                                               (get-in graph [:nodes id :vector]))})
                         @nearest)]
        (take k (sort-by :distance results))))))

;; Batch operations
(defn insert-batch
  "Insert multiple elements into the graph"
  [graph elements]
  (reduce (fn [g [id vec]] (insert g id vec)) graph elements))

;; Utility functions
(defn graph-info
  "Get information about the graph structure"
  [graph]
  {:nodes (count (:nodes graph))
   :entry-point (:entry-point graph)
   :levels (if (empty? (:nodes graph))
             0
             (inc (apply max (map :level (vals (:nodes graph))))))
   :avg-connections (if (empty? (:nodes graph))
                      0
                      (float (/ (reduce + (map (fn [node]
                                                 (reduce + (map count
                                                                (vals (:neighbors node)))))
                                               (vals (:nodes graph))))
                                (count (:nodes graph)))))})