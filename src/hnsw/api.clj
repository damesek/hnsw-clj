(ns hnsw.api
  "High-level API for HNSW operations"
  (:require [hnsw.ultra-fast :as ultra]
            [hnsw.graph :as graph]))

(defn index
  "Create an HNSW index from vectors
   Options:
   - :metric - :euclidean or :cosine (default: :euclidean)
   - :M - connectivity parameter (default: 16)
   - :ef-construction - construction parameter (default: 200)
   - :ef-search - default query candidate pool (default: 50)
   - :seed - deterministic level generation seed (default: 42)
   - :num-threads - parallel construction workers (default: CPU count)
   - :batch-size - dynamically claimed vectors per worker (default: 4)
   - :show-progress? - print construction progress (default: true)"
  [vectors & {:keys [metric M ef-construction ef-search seed num-threads batch-size
                     show-progress?]
              :or {metric :euclidean
                   M 16
                   ef-construction 200
                   ef-search 50
                   seed 42
                   num-threads (.availableProcessors (Runtime/getRuntime))
                   batch-size 4
                   show-progress? true}}]
  (let [distance-fn (case metric
                      :euclidean ultra/euclidean-distance-ultra
                      :cosine ultra/cosine-distance-ultra
                      ultra/euclidean-distance-ultra)]
    (ultra/build-index vectors
                       :M M
                       :ef-construction ef-construction
                       :ef-search ef-search
                       :seed seed
                       :num-threads num-threads
                       :batch-size batch-size
                       :show-progress? show-progress?
                       :distance-fn distance-fn)))

(defn search
  "Search for k nearest neighbors"
  ([index query k]
   (ultra/search-knn index query k))
  ([index query k ef-search]
   (ultra/search-knn index query k ef-search)))

(defn search-batch
  "Search many queries concurrently and preserve their input order."
  [index queries k & {:keys [ef-search num-threads]
                      :or {ef-search (.ef-search index)
                           num-threads (.availableProcessors (Runtime/getRuntime))}}]
  (ultra/search-knn-batch index queries k
                          :ef-search ef-search
                          :num-threads num-threads))

(defn add-vector!
  "Add a vector to an existing index"
  [index id vector]
  (ultra/insert-single index
                       (str id)
                       (if (instance? (Class/forName "[D") vector)
                         vector
                         (double-array vector))))

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
