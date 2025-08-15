(ns hnsw.api
  "High-level API for HNSW operations"
  (:require [hnsw.ultra-fast :as ultra]
            [hnsw.graph :as graph]))

(defn index
  "Create an HNSW index from vectors
   Options:
   - :metric - :euclidean or :cosine (default: :euclidean)
   - :M - connectivity parameter (default: 16)
   - :ef-construction - construction parameter (default: 200)"
  [vectors & {:keys [metric M ef-construction]
              :or {metric :euclidean
                   M 16
                   ef-construction 200}}]
  (let [distance-fn (case metric
                      :euclidean ultra/euclidean-distance-ultra
                      :cosine ultra/cosine-distance-ultra
                      ultra/euclidean-distance-ultra)]
    (ultra/build-index vectors
                       :M M
                       :ef-construction ef-construction
                       :distance-fn distance-fn)))

(defn search
  "Search for k nearest neighbors"
  [index query k]
  (ultra/search-knn index query k))

(defn add-vector!
  "Add a vector to an existing index"
  [index id vector]
  (ultra/insert-single index (str id) vector))

(defn size
  "Get the number of vectors in the index"
  [index]
  (count (.-nodes index)))

(defn save-index
  "Save index to a file"
  [index filepath]
  ;; TODO: Implement serialization
  (throw (UnsupportedOperationException. "Not yet implemented")))

(defn load-index
  "Load index from a file"
  [filepath]
  ;; TODO: Implement deserialization
  (throw (UnsupportedOperationException. "Not yet implemented")))