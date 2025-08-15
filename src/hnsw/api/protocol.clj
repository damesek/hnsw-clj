(ns hnsw.api.protocol
  "Core protocol definition for all ANN implementations
   Provides a unified interface for different index types")

;; ============================================================
;; Core Protocol
;; ============================================================

(defprotocol ANNIndex
  "Protocol for Approximate Nearest Neighbor indexes
   All ANN implementations should extend this protocol"

  (search-knn* [this query k mode]
    "Search for k nearest neighbors
     Parameters:
     - this: The index instance
     - query: Query vector (double array)
     - k: Number of neighbors to find
     - mode: Search mode (:turbo, :fast, :balanced, :accurate, :precise)
     Returns: Vector of {:id :distance} maps")

  (index-info* [this]
    "Get information about the index
     Returns: Map with index-specific information")

  (index-type* [this]
    "Get the type identifier of this index
     Returns: Keyword identifying the index type"))

;; ============================================================
;; Optional Extended Protocol for Advanced Features
;; ============================================================

(defprotocol FilterableIndex
  "Optional protocol for indexes supporting filtered search"

  (search-knn-filtered* [this query k filter-fn mode]
    "Search with a filter predicate
     Parameters:
     - filter-fn: Predicate function (id -> boolean)
     Returns: Vector of {:id :distance} maps"))

(defprotocol PersistableIndex
  "Optional protocol for indexes that can be saved/loaded"

  (save-index* [this filepath]
    "Save index to disk
     Parameters:
     - filepath: Path to save the index
     Returns: true on success")

  (load-index* [filepath]
    "Load index from disk
     Parameters:
     - filepath: Path to load the index from
     Returns: Index instance"))

(defprotocol BatchSearchIndex
  "Optional protocol for indexes supporting batch search"

  (search-batch* [this queries k mode]
    "Search multiple queries in batch
     Parameters:
     - queries: Vector of query vectors
     - k: Number of neighbors per query
     - mode: Search mode
     Returns: Vector of result vectors"))

;; ============================================================
;; Helper Functions for Protocol Users
;; ============================================================

(defn supports-filtering?
  "Check if an index supports filtered search"
  [index]
  (satisfies? FilterableIndex index))

(defn supports-persistence?
  "Check if an index supports save/load"
  [index]
  (satisfies? PersistableIndex index))

(defn supports-batch-search?
  "Check if an index supports batch search"
  [index]
  (satisfies? BatchSearchIndex index))

;; ============================================================
;; Default Implementations
;; ============================================================

(defn default-batch-search
  "Default batch search implementation using sequential search"
  [index queries k mode]
  (mapv #(search-knn* index % k mode) queries))

(defn default-filtered-search
  "Default filtered search implementation using post-filtering"
  [index query k filter-fn mode]
  ;; Search for more candidates and filter
  (let [candidates (search-knn* index query (* 3 k) mode)]
    (vec (take k (filter #(filter-fn (:id %)) candidates)))))