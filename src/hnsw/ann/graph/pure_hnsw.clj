(ns hnsw.ann.graph.pure-hnsw
  "Pure HNSW implementation wrapper - Original graph-based approach
   This is the baseline HNSW implementation for comparison
   
   Performance characteristics:
   - Build: ~205 seconds for 31k vectors
   - Search: ~0.2-1ms with 99%+ recall
   - Memory: 2x (graph structure overhead)"
  (:require [hnsw.graph :as graph]
            [hnsw.ultra-fast :as ultra])
  (:import [java.util.concurrent ConcurrentHashMap]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; todo delete

;; ============================================================
;; Pure HNSW Index Structure
;; ============================================================

(defrecord PureHNSWIndex [graph ; The HNSW graph
                          data-map ; id -> vector map
                          vector-norms ; Pre-computed norms
                          params]) ; HNSW parameters

;; ============================================================
;; Build Function
;; ============================================================

(defn build-pure-hnsw-index
  "Build pure HNSW index (original implementation)
   
   Parameters:
   - data: Vector of [id vector] pairs
   - M: Number of bi-directional links (default 16)
   - ef-construction: Size of dynamic candidate list (default 200)
   - distance-fn: Distance function to use
   - show-progress?: Show build progress
   - seed: Random seed for level assignment"
  [data & {:keys [M
                  ef-construction
                  distance-fn
                  show-progress?
                  seed
                  ef]
           :or {M 16
                ef-construction 200
                distance-fn ultra/cosine-distance-ultra
                show-progress? true
                seed 42
                ef 50}}]

  (let [start-time (System/currentTimeMillis)
        total-vectors (count data)
        data-vec (vec data)]

    (when show-progress?
      (println "\n🌐 PURE HNSW INDEX BUILD")
      (println (format "   Vectors: %,d" total-vectors))
      (println (format "   M: %d, ef-construction: %d" M ef-construction))
      (println "   Method: Original graph-based HNSW"))

    ;; Pre-compute vector norms for cosine distance
    (let [vector-norms (ConcurrentHashMap. total-vectors)
          data-map (ConcurrentHashMap. total-vectors)]

      (when show-progress?
        (println "   Phase 1: Computing vector norms..."))

      ;; Store vectors and compute norms
      (doseq [[id ^doubles vector] data-vec]
        (.put data-map id vector)
        (let [norm (Math/sqrt
                    (areduce vector i sum 0.0
                             (+ sum (* (aget vector i) (aget vector i)))))]
          (.put vector-norms id norm)))

      ;; Build HNSW graph
      (when show-progress?
        (println "   Phase 2: Building HNSW graph..."))

      ;; Create HNSW parameters
      (let [params (assoc (graph/default-params)
                          :M M
                          :ef-construction ef-construction
                          :ef ef
                          :distance-fn distance-fn
                          :seed seed)

            ;; Create empty graph
            graph (graph/create-graph params)

            ;; Progress tracking
            progress-interval (max 1 (int (/ total-vectors 20)))
            counter (atom 0)

            ;; Insert all vectors
            final-graph (reduce (fn [g [idx [id vector]]]
                                  (swap! counter inc)
                                  (when (and show-progress?
                                             (zero? (mod @counter progress-interval)))
                                    (println (format "      Progress: %d/%d (%.1f%%)"
                                                     @counter total-vectors
                                                     (* 100.0 (/ @counter total-vectors)))))
                                  (graph/insert g id vector))
                                graph
                                (map-indexed vector data-vec))

            build-time (- (System/currentTimeMillis) start-time)]

        (when show-progress?
          (println (format "\n✅ Pure HNSW Index built in %.2f seconds!"
                           (/ build-time 1000.0)))
          (println (format "   Build rate: %.0f vectors/second"
                           (/ total-vectors (/ build-time 1000.0))))
          (println (format "   Graph nodes: %d" (:element-count final-graph)))
          (println (format "   Entry point: %s" (:entry-point final-graph))))

        (->PureHNSWIndex final-graph
                         data-map
                         vector-norms
                         params)))))

;; ============================================================
;; Search Functions
;; ============================================================

(defn search-pure-hnsw
  "Search pure HNSW index with configurable parameters"
  [^PureHNSWIndex index ^doubles query-vec k
   & {:keys [mode ef]
      :or {mode :balanced}}]

  (let [;; Mode configurations for ef parameter
        mode-configs {:turbo {:ef 50} ; Fastest, lower recall
                      :fast {:ef 100} ; Fast with good recall
                      :balanced {:ef 200} ; Balanced (default)
                      :accurate {:ef 300} ; Higher recall
                      :precise {:ef 500}} ; Maximum recall

        config (or (get mode-configs mode)
                   {:ef (or ef 200)})

        ;; Get graph from index
        graph (.graph index)

        ;; Update ef parameter in graph
        graph-with-ef (assoc-in graph [:params :ef] (:ef config))]

    ;; Use original search-knn
    (graph/search-knn graph-with-ef query-vec k)))

;; ============================================================
;; API Functions (Compatible with IVF-FLAT and IVF-HNSW)
;; ============================================================

(defn build-index
  "Build pure HNSW index (API compatible)"
  [data & opts]
  (apply build-pure-hnsw-index data opts))

(defn search-knn
  "Search pure HNSW index (API compatible)
   
   Modes:
   - :turbo    - ef=50  (fastest, ~95% recall)
   - :fast     - ef=100 (fast, ~97% recall)
   - :balanced - ef=200 (balanced, ~99% recall) [default]
   - :accurate - ef=300 (accurate, ~99.5% recall)
   - :precise  - ef=500 (precise, ~99.9% recall)"
  ([index query-vec k]
   (search-pure-hnsw index query-vec k :mode :balanced))
  ([index query-vec k mode]
   (search-pure-hnsw index query-vec k :mode mode)))

(defn index-info
  "Get information about the index"
  [^PureHNSWIndex index]
  (let [graph (.graph index)
        nodes (:nodes graph)
        total-edges (reduce (fn [sum [_ node]]
                              (+ sum (reduce + (map count (vals (:neighbors node))))))
                            0 nodes)]
    {:type "Pure HNSW Index"
     :vectors (:element-count graph)
     :nodes (count nodes)
     :entry-point (:entry-point graph)
     :avg-edges-per-node (/ total-edges (count nodes))
     :params (select-keys (.params index) [:M :ef-construction :ef])}))
