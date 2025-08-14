(ns hnsw.api.unified
  "Unified API for all ANN implementations
   Provides a consistent interface for all index types"
  (:require [hnsw.api.protocol :refer [ANNIndex]]
            ;; Graph-based implementations
            [hnsw.ann.graph.pure-hnsw :as pure]
            [hnsw.ultra-fast :as ultra]
            ;; Partition-based implementations
            [hnsw.ann.partition.partitioned-hnsw :as phnsw]
            [hnsw.ann.partition.lightning :as lightning]
            [hnsw.ann.partition.ivf-flat :as ivf]
            ;; Hash-based implementations
            [hnsw.ann.hash.hybrid-lsh :as lsh]
            ;; Hybrid implementations
            [hnsw.ann.hybrid.ivf-hnsw :as ivf-hnsw]
            ;; Dimension reduction implementations
            [hnsw.ann.dimreduct.pcaf :as pcaf])
  (:import [hnsw.ann.graph.pure_hnsw PureHNSWIndex]
           [hnsw.ann.partition.partitioned_hnsw PartitionedHNSWIndex]
           [hnsw.ann.partition.lightning LightningIndex]
           [hnsw.ann.partition.ivf_flat IVFFlatIndex]
           [hnsw.ann.hash.hybrid_lsh HybridLSHIndex]
           [hnsw.ann.hybrid.ivf_hnsw IVFHNSWIndex]
           [hnsw.ann.dimreduct.pcaf PCAFIndex]))

;; ============================================================
;; Protocol Extension for All Implementations
;; ============================================================

(extend-type PureHNSWIndex
  ANNIndex
  (search-knn* [this query k mode]
    (pure/search-knn this query k mode))
  (index-info* [this]
    {:type :pure-hnsw
     :graph (.graph this)
     :vectors (count (.data-map this))})
  (index-type* [this] :pure-hnsw))

(extend-type PartitionedHNSWIndex
  ANNIndex
  (search-knn* [this query k mode]
    (phnsw/search-knn this query k mode))
  (index-info* [this]
    {:type :partitioned-hnsw
     :partitions (.num-partitions this)
     :search-mode (.search-mode this)})
  (index-type* [this] :partitioned-hnsw))

(extend-type LightningIndex
  ANNIndex
  (search-knn* [this query k mode]
    (lightning/search-knn this query k mode))
  (index-info* [this]
    {:type :lightning
     :partitions (.num-partitions this)})
  (index-type* [this] :lightning))

(extend-type IVFFlatIndex
  ANNIndex
  (search-knn* [this query k mode]
    (ivf/search-knn this query k mode))
  (index-info* [this]
    {:type :ivf-flat
     :partitions (count (.centroids this))})
  (index-type* [this] :ivf-flat))

(extend-type HybridLSHIndex
  ANNIndex
  (search-knn* [this query k mode]
    (lsh/search-knn this query k))
  (index-info* [this]
    {:type :hybrid-lsh
     :num-tables (.num-tables this)
     :num-bits (.num-bits this)})
  (index-type* [this] :hybrid-lsh))

(extend-type IVFHNSWIndex
  ANNIndex
  (search-knn* [this query k mode]
    (ivf-hnsw/search-knn this query k mode))
  (index-info* [this]
    {:type :ivf-hnsw
     :partitions (count (.centroids this))})
  (index-type* [this] :ivf-hnsw))

(extend-type PCAFIndex
  ANNIndex
  (search-knn* [this query k mode]
    (pcaf/search-knn this query k mode))
  (index-info* [this]
    {:type :pcaf
     :dimension-reduction (.dimension-reduction this)
     :k-filter (.k-filter this)})
  (index-type* [this] :pcaf))

;; ============================================================
;; Helper Functions
;; ============================================================

(defn detect-index-type
  "Automatically detect the type of an index"
  [index]
  (cond
    (instance? PureHNSWIndex index) :pure-hnsw
    (instance? PartitionedHNSWIndex index) :partitioned-hnsw
    (instance? LightningIndex index) :lightning
    (instance? IVFFlatIndex index) :ivf-flat
    (instance? HybridLSHIndex index) :hybrid-lsh
    (instance? IVFHNSWIndex index) :ivf-hnsw
    (instance? PCAFIndex index) :pcaf
    :else :unknown))

(defn format-index-info
  "Format index information for display"
  [index]
  (let [info (index-info* index)
        type (:type info)]
    (str "Index Type: " (name type) "\n"
         (case type
           :pure-hnsw (format "Vectors: %d" (:vectors info))
           :partitioned-hnsw (format "Partitions: %d, Mode: %s"
                                     (:partitions info) (:search-mode info))
           :lightning (format "Partitions: %d" (:partitions info))
           :ivf-flat (format "Partitions: %d" (:partitions info))
           :hybrid-lsh (format "Tables: %d, Bits: %d"
                               (:num-tables info) (:num-bits info))
           :ivf-hnsw (format "Partitions: %d" (:partitions info))
           :pcaf (format "Reduction: %.1fx, K-filter: %d"
                         (:dimension-reduction info) (:k-filter info))
           "Unknown index type"))))

;; ============================================================
;; Public API using Protocol Functions
;; ============================================================

(defn search-knn
  "Unified search function for any index type"
  ([index query k]
   (search-knn* index query k :default))
  ([index query k mode]
   (search-knn* index query k mode)))

(defn index-info
  "Get information about any index type"
  [index]
  (index-info* index))

(defn index-type
  "Get the type of an index"
  [index]
  (index-type* index))