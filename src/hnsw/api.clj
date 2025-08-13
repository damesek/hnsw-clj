(ns hnsw.api
  "HNSW Library - Clean Public API
   
   Simple, production-ready API for vector search.
   Designed to be used as a library dependency.
   
   Usage in deps.edn:
   ```clojure
   {:deps {com.github.user/hnsw-lib 
           {:git/url \"https://github.com/user/hnsw-lib\"
            :git/sha \"abc123\"}}}
   ```
   
   Basic usage:
   ```clojure
   (require '[hnsw.api :as hnsw])
   
   ;; Create index
   (def idx (hnsw/index {:dimensions 384}))
   
   ;; Add vectors  
   (hnsw/add! idx \"id-1\" vector)
   
   ;; Search
   (hnsw/search idx query 10)
   ```"
  (:require [hnsw.ultra-fast :as ultra]
            [hnsw.filtered-index-io :as io]
            [hnsw.simd-selector :as simd]))

;; Initialize SIMD on namespace load
(def ^:private simd-impl (delay (simd/get-implementation)))

;; ========== Index Creation ==========

(defn index
  "Create a new HNSW index.
   
   Options map:
   - :dimensions - vector dimension (required)
   - :m - connections per node (default: 16)
   - :ef-construction - construction parameter (default: 200)
   - :distance - :cosine, :euclidean, or :dot (default: :cosine)
   
   Example:
   ```clojure
   (def my-index (index {:dimensions 384
                          :m 16
                          :distance :cosine}))
   ```"
  [{:keys [dimensions m ef-construction distance]
    :or {m 16
         ef-construction 200
         distance :cosine}}]
  (when-not dimensions
    (throw (IllegalArgumentException. "dimensions is required")))

  ;; Select appropriate distance function
  ;; Use ultra-fast implementations for compatibility with HNSW graph
  (let [dist-fn (case distance
                  :cosine ultra/cosine-distance-ultra
                  :euclidean ultra/euclidean-distance-ultra
                  :dot (fn [^doubles a ^doubles b]
                         (- 1.0 (simd/dot-product a b)))
                  ultra/cosine-distance-ultra)
        ;; Create the actual HNSW graph using ultra-fast implementation
        graph (ultra/create-ultra-graph
               :M m
               :ef-construction ef-construction
               :distance-fn dist-fn)]

    ;; Wrap in atom with metadata
    (atom {:graph graph
           :dimensions dimensions
           :distance-type distance
           :metadata {}
           :size 0})))

;; ========== Vector Operations ==========

(defn add!
  "Add a vector to the index.
   
   Args:
   - index: the HNSW index
   - id: unique identifier (string/keyword)
   - vector: vector data (array or sequence)
   - metadata: optional metadata map
   
   Returns: the index
   
   Example:
   ```clojure
   (add! idx \"doc-123\" [0.1 0.2 0.3 ...])
   (add! idx \"doc-124\" vector {:title \"Document\"})
   ```"
  ([index id vector]
   (add! index id vector nil))
  ([index id vector metadata]
   (let [vec-array (if (sequential? vector)
                     (double-array vector)
                     vector)
         id-str (if (keyword? id) (name id) (str id))]
     ;; Add to HNSW graph
     (swap! index
            (fn [idx]
              (let [graph (:graph idx)]
                ;; Use ultra-fast insert
                (ultra/insert-single graph id-str vec-array)
                (-> idx
                    (update :size inc)
                    (assoc-in [:metadata id-str] metadata)))))
     index)))

(defn add-batch!
  "Add multiple vectors efficiently.
   
   Args:
   - index: the HNSW index
   - items: sequence of [id vector] or [id vector metadata]
   
   Example:
   ```clojure
   (add-batch! idx [[\"id1\" vec1]
                    [\"id2\" vec2 {:meta \"data\"}]])
   ```"
  [index items]
  (let [idx @index
        graph (:graph idx)]
    ;; Prepare vectors for batch insertion
    (let [elements (map (fn [item]
                          (let [[id vec meta] item
                                id-str (if (keyword? id) (name id) (str id))
                                vec-array (if (sequential? vec)
                                            (double-array vec)
                                            vec)]
                            [id-str vec-array meta]))
                        items)]
      ;; Use ultra-fast batch insertion
      (ultra/insert-batch graph (map (fn [[id vec _]] [id vec]) elements))

      ;; Update metadata and size
      (swap! index
             (fn [idx]
               (reduce (fn [idx [id _ meta]]
                         (-> idx
                             (update :size inc)
                             (cond-> meta (assoc-in [:metadata id] meta))))
                       idx
                       elements))))
    index))

;; ========== Search Operations ==========

(defn search
  "Search for k nearest neighbors.
   
   Args:
   - index: the HNSW index
   - query: query vector
   - k: number of results
   
   Returns: sequence of {:id :distance :metadata} maps
   
   Example:
   ```clojure
   (search idx [0.1 0.2 0.3 ...] 10)
   ;; => ({:id \"doc-1\" :distance 0.123 :metadata {...}} ...)
   ```"
  [index query k]
  (let [idx @index
        query-array (if (sequential? query)
                      (double-array query)
                      query)
        graph (:graph idx)
        ;; Use ultra-fast HNSW search
        results (ultra/search-knn graph query-array k)]
    ;; Enrich results with metadata
    (map (fn [{:keys [id distance]}]
           {:id id
            :distance distance
            :metadata (get-in idx [:metadata id])})
         results)))

(defn search-with-filter
  "Search with a filter function.
   
   Args:
   - index: the HNSW index
   - query: query vector
   - k: number of results
   - filter-fn: predicate function on {:id :metadata}
   
   Example:
   ```clojure
   (search-with-filter idx query 10 
     (fn [{:keys [metadata]}]
       (= (:category metadata) \"news\")))
   ```"
  [index query k filter-fn]
  (let [results (search index query (* k 3))] ; Over-fetch for filtering
    (take k (filter filter-fn results))))

;; ========== Persistence ==========

(defn save
  "Save index to disk.
   
   Args:
   - index: the HNSW index
   - filepath: where to save
   
   Example:
   ```clojure
   (save idx \"my-index.hnsw\")
   ```"
  [index filepath]
  (io/save-filtered-index @index filepath))

(defn load
  "Load index from disk.
   
   Args:
   - filepath: path to saved index
   
   Returns: new index atom
   
   Example:
   ```clojure
   (def idx (load \"my-index.hnsw\"))
   ```"
  [filepath]
  (let [loaded (io/load-filtered-index filepath)
        ;; Restore distance function
        dist-fn (case (:distance-type loaded :cosine)
                  :cosine simd/cosine-distance
                  :euclidean simd/euclidean-distance
                  :dot #(- 1.0 (simd/dot-product %1 %2))
                  simd/cosine-distance)]
    (atom (assoc loaded :distance-fn dist-fn))))

;; ========== Index Information ==========

(defn info
  "Get index information.
   
   Returns map with:
   - :size - number of vectors
   - :dimensions - vector dimensions
   - :distance - distance metric
   
   Example:
   ```clojure
   (info idx)
   ;; => {:size 10000 :dimensions 384 :distance :cosine}
   ```"
  [index]
  (let [idx @index
        graph (:graph idx)]
    {:size (:size idx)
     :dimensions (:dimensions idx)
     :distance (:distance-type idx)
     :nodes (when graph (.size (.nodes graph)))
     :entry-point (when graph (.get (.entry-point graph)))}))

(defn set-ef!
  "Set search parameter ef (accuracy/speed tradeoff).
   Higher values = better accuracy, slower search.
   
   Args:
   - index: the HNSW index
   - ef: search parameter (default 50, typical 10-500)
   
   Example:
   ```clojure
   (set-ef! idx 100) ; More accurate but slower
   ```"
  [index ef]
  (swap! index assoc :ef ef)
  index)

;; ========== SIMD Information ==========

(defn simd-info
  "Get SIMD implementation information.
   
   Returns map with implementation details.
   
   Example:
   ```clojure
   (simd-info)
   ;; => {:type :jblas :name \"Native BLAS\" ...}
   ```"
  []
  (let [impl @simd-impl]
    {:type (:type impl)
     :name (:name impl)
     :available-implementations
     {:jblas (simd/jblas-available?)
      :vector-api (simd/java-vector-api-available?)}}))

;; ========== Convenience Functions ==========

(defn similarity-search
  "Search and return similarities instead of distances.
   
   Similarity = 1 - distance (for normalized vectors).
   
   Example:
   ```clojure
   (similarity-search idx query 10)
   ;; => ({:id \"doc-1\" :similarity 0.877 ...} ...)
   ```"
  [index query k]
  (map (fn [result]
         (assoc result
                :similarity (- 1.0 (:distance result))
                :distance nil))
       (search index query k)))
