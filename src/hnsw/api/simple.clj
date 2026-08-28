(ns hnsw.api.simple
  "HNSW Library - Clean Public API"
  (:require [hnsw.ultra-fast :as ultra]
            [hnsw.helper.index-io :as io]))

;; Simplified API wrapper for HNSW

(defn index
  [{:keys [dimensions m ef-construction ef-search distance seed]
    :or {m 16
         ef-construction 200
         ef-search 50
         distance :cosine
         seed 42}}]
  (when-not dimensions
    (throw (IllegalArgumentException. "dimensions is required")))

  (let [dist-fn (case distance
                  :cosine ultra/cosine-distance-ultra
                  :euclidean ultra/euclidean-distance-ultra
                  ultra/cosine-distance-ultra)
        graph (ultra/create-ultra-graph
               :M m
               :ef-construction ef-construction
               :ef-search ef-search
               :seed seed
               :distance-fn dist-fn)]

    (atom {:graph graph
           :dimensions dimensions
           :distance-type distance
           :metadata {}
           :size 0})))

(defn add!
  ([index id vector]
   (add! index id vector nil))
  ([index id vector metadata]
   (let [vec-array (if (sequential? vector)
                     (double-array vector)
                     vector)
         id-str (if (keyword? id) (name id) (str id))]
     (swap! index
            (fn [idx]
              (let [graph (:graph idx)]
                (ultra/insert-single graph id-str vec-array)
                (-> idx
                    (update :size inc)
                    (assoc-in [:metadata id-str] metadata)))))
     index)))

(defn add-all!
  "Add [id vector] or [id vector metadata] entries with parallel HNSW construction."
  [index elements & {:keys [num-threads batch-size show-progress?]
                     :or {num-threads (.availableProcessors (Runtime/getRuntime))
                          batch-size 4
                          show-progress? true}}]
  (let [entries (vec elements)
        prepared (mapv (fn [[id vector]]
                         [(if (keyword? id) (name id) (str id))
                          (if (sequential? vector)
                            (double-array vector)
                            vector)])
                       entries)
        graph (:graph @index)]
    (ultra/insert-batch graph prepared
                        :num-threads num-threads
                        :batch-size batch-size
                        :show-progress? show-progress?)
    (ultra/optimize-search! graph)
    (swap! index
           (fn [idx]
             (reduce (fn [result [id _ metadata]]
                       (assoc-in result
                                 [:metadata (if (keyword? id) (name id) (str id))]
                                 metadata))
                     (update idx :size + (count entries))
                     entries)))
    index))

(defn search
  ([index query k]
   (search index query k nil))
  ([index query k ef-search]
   (let [idx @index
         query-array (if (sequential? query)
                       (double-array query)
                       query)
         graph (:graph idx)
         results (if ef-search
                   (ultra/search-knn graph query-array k ef-search)
                   (ultra/search-knn graph query-array k))]
     (map (fn [{:keys [id distance]}]
            {:id id
             :distance distance
             :metadata (get-in idx [:metadata id])})
          results))))

(defn search-batch
  "Search many queries concurrently and preserve their input order."
  [index queries k & {:keys [ef-search num-threads]
                      :or {num-threads (.availableProcessors (Runtime/getRuntime))}}]
  (let [idx @index
        graph (:graph idx)
        query-arrays (mapv #(if (sequential? %) (double-array %) %) queries)
        results (ultra/search-knn-batch
                 graph query-arrays k
                 :ef-search (or ef-search (.ef-search graph))
                 :num-threads num-threads)]
    (mapv (fn [neighbors]
            (mapv (fn [{:keys [id distance]}]
                    {:id id
                     :distance distance
                     :metadata (get-in idx [:metadata id])})
                  neighbors))
          results)))

(defn save
  [index filepath]
  (io/save-index (:graph @index) filepath))

(defn load-index
  "Load an index from disk"
  [filepath]
  (let [graph (io/load-index filepath ultra/cosine-distance-ultra)]
    (atom {:graph graph
           :dimensions nil
           :distance-type :cosine
           :metadata {}
           :size 0})))

(defn info
  [index]
  (let [idx @index
        graph (:graph idx)]
    {:size (:size idx)
     :dimensions (:dimensions idx)
     :distance (:distance-type idx)}))


(comment
  ;; how to use

  (require '[hnsw.api.simple :as hnsw])
  (def idx (hnsw/index {:dimensions 768}))

  ;; Unified API
  (require '[hnsw.api.unified :as api])
  (api/search-knn any-index query 10)

  ;; Specific API
  (require '[hnsw.api.graph.hnsw-search :as pure])
  (pure/build-index data)

  ;;
  )
