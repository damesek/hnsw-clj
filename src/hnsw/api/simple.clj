(ns hnsw.api.simple
  "HNSW Library - Clean Public API"
  (:require [hnsw.ultra-fast :as ultra]
            [hnsw.helper.index-io :as io]))

;; Simplified API wrapper for HNSW

(defn index
  [{:keys [dimensions m ef-construction distance]
    :or {m 16
         ef-construction 200
         distance :cosine}}]
  (when-not dimensions
    (throw (IllegalArgumentException. "dimensions is required")))

  (let [dist-fn (case distance
                  :cosine ultra/cosine-distance-ultra
                  :euclidean ultra/euclidean-distance-ultra
                  ultra/cosine-distance-ultra)
        graph (ultra/create-ultra-graph
               :M m
               :ef-construction ef-construction
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

(defn search
  [index query k]
  (let [idx @index
        query-array (if (sequential? query)
                      (double-array query)
                      query)
        graph (:graph idx)
        results (ultra/search-knn graph query-array k)]
    (map (fn [{:keys [id distance]}]
           {:id id
            :distance distance
            :metadata (get-in idx [:metadata id])})
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