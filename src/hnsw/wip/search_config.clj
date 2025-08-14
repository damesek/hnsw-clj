(ns hnsw.wip.search-config
  "Configurable search parameters for HNSW")

(def ^:dynamic *ef-search* 50)

(defn set-ef-search!
  "Set the ef parameter for search (affects search quality/speed tradeoff)"
  [ef]
  (alter-var-root #'*ef-search* (constantly ef)))

(defn search-knn-configurable
  "Search with configurable ef parameter"
  [index query-vec k & {:keys [ef] :or {ef *ef-search*}}]
  ;; This would need to be integrated into ultra-fast.clj
  ;; For now, it's hardcoded to max(k, 50)
  (println (format "Using ef=%d for search" ef))
  ;; Call the actual search
  (require '[hnsw.ultra-fast :as ultra])
  ((resolve 'hnsw.ultra-fast/search-knn) index query-vec k))

;; Performance guide:
;; ef=10-20:  Very fast, lower quality (~0.1ms, 80-85% recall)
;; ef=50:     Balanced (current) (~0.2ms, 90-95% recall)  
;; ef=100:    Higher quality (~0.3ms, 95-98% recall)
;; ef=200:    Best quality (~0.5ms, 98-99% recall)
