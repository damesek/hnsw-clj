(ns hnsw.simd-wrapper
  "Wrapper module that automatically uses SIMD-optimized distance functions
   for all HNSW implementations.
   
   Usage:
   (require '[hnsw.simd-wrapper :as sw])
   (def index (sw/build-ultra-fast data))  ; Automatically uses SIMD
   (sw/search index query 10)"
  (:require [hnsw.simd-optimized :as simd]
            [hnsw.ultra-fast :as ultra]
            [hnsw.ultra-optimized :as opt]
            [hnsw.hnsw-search :as pure]
            [hnsw.partitioned-hnsw :as phnsw]
            [hnsw.ivf-flat :as ivf-flat]
            [hnsw.ivf-hnsw :as ivf-hnsw]
            [hnsw.hybrid-lsh :as lsh]
            [hnsw.lightning :as lightning]
            [hnsw.pcaf :as pcaf]))

;; ============================================
;; ULTRA-FAST WITH SIMD
;; ============================================

(defn build-ultra-fast
  "Build Ultra-Fast index with SIMD cosine distance"
  [data & {:keys [M ef-construction show-progress?]
           :or {M 16 
                ef-construction 200
                show-progress? true}}]
  (ultra/build-index data
                    :M M
                    :ef-construction ef-construction
                    :distance-fn simd/cosine-distance  ; <- SIMD!
                    :show-progress? show-progress?))

;; ============================================
;; ULTRA-OPTIMIZED WITH SIMD
;; ============================================

(defn build-ultra-optimized
  "Build Ultra-Optimized index with SIMD cosine distance"
  [data & {:keys [M ef-construction show-progress?]
           :or {M 16 
                ef-construction 200
                show-progress? true}}]
  (opt/build-index data
                  :M M
                  :ef-construction ef-construction
                  :distance-fn simd/cosine-distance  ; <- SIMD!
                  :show-progress? show-progress?))

;; ============================================
;; PURE HNSW WITH SIMD
;; ============================================

(defn build-pure-hnsw
  "Build Pure HNSW index with SIMD cosine distance"
  [data & {:keys [M ef-construction show-progress?]
           :or {M 16 
                ef-construction 200
                show-progress? true}}]
  (pure/build-index data
                   :M M
                   :ef-construction ef-construction
                   :distance-fn simd/cosine-distance  ; <- SIMD!
                   :show-progress? show-progress?))

;; ============================================
;; PARTITIONED HNSW WITH SIMD
;; ============================================

(defn build-partitioned
  "Build Partitioned HNSW index with SIMD cosine distance"
  [data & {:keys [num-partitions show-progress? search-mode]
           :or {num-partitions 8
                show-progress? true
                search-mode :lightning}}]
  (phnsw/build-index data
                    :num-partitions num-partitions
                    :distance-fn simd/cosine-distance  ; <- SIMD!
                    :show-progress? show-progress?
                    :search-mode search-mode))

;; ============================================
;; IVF-FLAT WITH SIMD
;; ============================================

(defn build-ivf-flat
  "Build IVF-FLAT index with SIMD cosine distance"
  [data & {:keys [n-lists show-progress?]
           :or {n-lists 100
                show-progress? true}}]
  (ivf-flat/build-index data
                       :n-lists n-lists
                       :distance-fn simd/cosine-distance  ; <- SIMD!
                       :show-progress? show-progress?))

;; ============================================
;; IVF-HNSW WITH SIMD
;; ============================================

(defn build-ivf-hnsw
  "Build IVF-HNSW index with SIMD cosine distance"
  [data & {:keys [n-lists M show-progress?]
           :or {n-lists 100
                M 16
                show-progress? true}}]
  (ivf-hnsw/build-index data
                       :n-lists n-lists
                       :M M
                       :distance-fn simd/cosine-distance  ; <- SIMD!
                       :show-progress? show-progress?))

;; ============================================
;; HYBRID LSH WITH SIMD
;; ============================================

(defn build-lsh
  "Build Hybrid LSH index with SIMD cosine distance"
  [data & {:keys [n-tables show-progress?]
           :or {n-tables 10
                show-progress? true}}]
  (lsh/build-index data
                  :n-tables n-tables
                  :distance-fn simd/cosine-distance  ; <- SIMD!
                  :show-progress? show-progress?))

;; ============================================
;; GENERIC SEARCH (works with any index)
;; ============================================

(defn search
  "Search any index (automatically detects type)"
  [index query k]
  (cond
    ;; Ultra-Fast/Optimized
    (instance? hnsw.ultra_fast.UltraGraph index)
    (ultra/search-knn index query k)
    
    ;; Ultra-Optimized
    (instance? hnsw.ultra_optimized.OptimizedGraph index)
    (opt/search index query k)
    
    ;; Pure HNSW
    (instance? hnsw.hnsw_search.PureHNSWIndex index)
    (pure/search-knn index query k)
    
    ;; Partitioned HNSW
    (instance? hnsw.partitioned_hnsw.PartitionedHNSWIndex index)
    (phnsw/search-knn index query k)
    
    ;; IVF-FLAT
    (instance? hnsw.ivf_flat.IVFFlatIndex index)
    (ivf-flat/search index query k)
    
    ;; IVF-HNSW
    (instance? hnsw.ivf_hnsw.IVFHNSWIndex index)
    (ivf-hnsw/search index query k)
    
    ;; Hybrid LSH
    (instance? hnsw.hybrid_lsh.HybridLSHIndex index)
    (lsh/search index query k)
    
    :else
    (throw (IllegalArgumentException. "Unknown index type"))))

;; ============================================
;; CONVENIENCE FUNCTIONS
;; ============================================

(defn build-best-for-size
  "Automatically choose best implementation based on dataset size"
  [data & {:keys [show-progress?] :or {show-progress? true}}]
  (let [n (count data)]
    (cond
      ;; Small dataset: Ultra-Fast
      (< n 1000)
      (do
        (println (format "📊 Dataset: %d vectors -> Using Ultra-Fast" n))
        (build-ultra-fast data :show-progress? show-progress?))
      
      ;; Medium dataset: Partitioned HNSW
      (< n 10000)
      (do
        (println (format "📊 Dataset: %d vectors -> Using Partitioned HNSW" n))
        (build-partitioned data 
                          :num-partitions (if (< n 5000) 4 8)
                          :show-progress? show-progress?))
      
      ;; Large dataset: IVF-FLAT or IVF-HNSW
      :else
      (do
        (println (format "📊 Dataset: %d vectors -> Using IVF-FLAT" n))
        (build-ivf-flat data 
                       :n-lists (int (Math/sqrt n))
                       :show-progress? show-progress?)))))

(defn benchmark-all
  "Benchmark all implementations with SIMD"
  [data k]
  (let [query (second (first data))
        impls [{:name "Ultra-Fast" :build build-ultra-fast}
               {:name "Ultra-Optimized" :build build-ultra-optimized}
               {:name "Pure HNSW" :build build-pure-hnsw}
               {:name "Partitioned" :build build-partitioned}
               {:name "IVF-FLAT" :build build-ivf-flat}]]
    
    (println "\n🚀 BENCHMARKING ALL IMPLEMENTATIONS WITH SIMD")
    (println "=============================================\n")
    
    (doseq [{:keys [name build]} impls]
      (println (format "Testing %s..." name))
      (let [start (System/currentTimeMillis)
            index (build data :show-progress? false)
            build-time (- (System/currentTimeMillis) start)
            
            ;; Search test
            _ (dotimes [_ 10] (search index query k))
            times (for [_ (range 20)]
                   (let [start (System/nanoTime)]
                     (search index query k)
                     (/ (- (System/nanoTime) start) 1000000.0)))
            avg-time (/ (reduce + times) 20)]
        
        (println (format "  Build: %.2fs, Search: %.3fms, QPS: %,.0f\n"
                        (/ build-time 1000.0)
                        avg-time
                        (/ 1000.0 avg-time)))))))
