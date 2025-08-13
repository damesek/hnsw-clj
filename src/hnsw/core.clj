(ns hnsw.core
  "HNSW Library - Main Entry Point
   
   High-performance Hierarchical Navigable Small World implementation
   for approximate nearest neighbor search in Clojure."
  (:require [hnsw.graph :as graph]
            [hnsw.ultra-fast :as ultra]
            [hnsw.ultra-optimized :as opt]
            [clojure.string :as str])
  (:gen-class))

;; ===== Basic Example =====

(defn hello-world-example
  "Simple demonstration of HNSW usage"
  []
  (println "\n🌍 HNSW Hello World Example")
  (println (str/join (repeat 50 "=")))

  ;; Create simple 3D vectors
  (let [vectors [["vec1" [1.0 2.0 3.0]]
                 ["vec2" [4.0 5.0 6.0]]
                 ["vec3" [7.0 8.0 9.0]]
                 ["vec4" [2.0 3.0 4.0]]
                 ["vec5" [5.0 6.0 7.0]]]

        ;; Build index using standard implementation
        _ (println "\n1️⃣ Building index with standard implementation...")
        graph (graph/insert-batch (graph/create-graph) vectors)

        ;; Search for nearest neighbors
        query [1.5 2.5 3.5]
        k 3
        _ (println (format "\n2️⃣ Searching for %d nearest neighbors to %s" k (vec query)))
        results (graph/search-knn graph query k)]

    (println "\n3️⃣ Results:")
    (doseq [{:keys [id distance]} results]
      (println (format "   %s - distance: %.3f" id distance)))

    (println "\n✅ Basic example complete!")))

;; ===== Performance Example =====

(defn performance-example
  "Demonstration with larger dataset and optimized implementation"
  []
  (println "\n⚡ HNSW Performance Example")
  (println (str/join (repeat 50 "=")))

  (let [;; Generate random vectors
        n 1000
        dim 128
        _ (println (format "\n1️⃣ Generating %d random %d-dimensional vectors..." n dim))
        vectors (repeatedly n
                            (fn []
                              [(str (java.util.UUID/randomUUID))
                               (double-array (repeatedly dim rand))]))

        ;; Build with ultra-fast implementation
        _ (println "\n2️⃣ Building index with ultra-fast implementation...")
        start (System/currentTimeMillis)
        index (ultra/build-index vectors
                                 :M 16
                                 :ef-construction 200
                                 :distance-fn ultra/cosine-distance-ultra
                                 :show-progress? false)
        build-time (- (System/currentTimeMillis) start)
        _ (println (format "   Index built in %.2f seconds" (/ build-time 1000.0)))

        ;; Perform searches
        _ (println "\n3️⃣ Performing searches...")
        query (second (first vectors))
        k 10

        ;; Measure search time
        times (repeatedly 100
                          (fn []
                            (let [start (System/nanoTime)]
                              (ultra/search-knn index query k)
                              (/ (- (System/nanoTime) start) 1000000.0))))
        avg-time (/ (reduce + times) (count times))]

    (println (format "   Average search time (k=%d): %.3f ms" k avg-time))
    (println (format "   Queries per second: %.0f" (/ 1000.0 avg-time)))

    (println "\n✅ Performance example complete!")))

;; ===== Comparison Example =====

(defn comparison-example
  "Compare different implementations"
  []
  (println "\n📊 Implementation Comparison")
  (println (str/join (repeat 50 "=")))

  (let [vectors (repeatedly 500
                            (fn []
                              [(str (java.util.UUID/randomUUID))
                               (double-array (repeatedly 128 rand))]))
        query (second (first vectors))
        k 10]

    ;; Standard implementation
    (println "\n1️⃣ Standard Implementation:")
    (let [start (System/currentTimeMillis)
          graph (graph/insert-batch (graph/create-graph) vectors)
          time (- (System/currentTimeMillis) start)]
      (println (format "   Build time: %.2f seconds" (/ time 1000.0))))

    ;; Ultra-fast implementation
    (println "\n2️⃣ Ultra-Fast Implementation:")
    (let [start (System/currentTimeMillis)
          index (ultra/build-index vectors :show-progress? false)
          time (- (System/currentTimeMillis) start)]
      (println (format "   Build time: %.2f seconds" (/ time 1000.0))))

    ;; Optimized implementation
    (println "\n3️⃣ Ultra-Optimized Implementation:")
    (let [start (System/currentTimeMillis)
          index (opt/build-index vectors :show-progress? false)
          time (- (System/currentTimeMillis) start)]
      (println (format "   Build time: %.2f seconds" (/ time 1000.0))))

    (println "\n✅ Comparison complete!")))

;; ===== Main Entry Point =====

(defn -main
  "Main entry point for the HNSW library"
  [& args]
  (println "\n🚀 HNSW Library - High-Performance Vector Search")
  (println (str/join (repeat 60 "=")))

  (let [command (first args)]
    (case command
      "hello" (hello-world-example)
      "perf" (performance-example)
      "compare" (comparison-example)
      "all" (do (hello-world-example)
                (println)
                (performance-example)
                (println)
                (comparison-example))
      ;; Default
      (do
        (println "\nUsage: clj -M:run [command]")
        (println "\nAvailable commands:")
        (println "  hello   - Basic hello world example")
        (println "  perf    - Performance demonstration")
        (println "  compare - Compare implementations")
        (println "  all     - Run all examples")
        (println "\nRunning hello world example by default...")
        (println)
        (hello-world-example))))

  (println "\n👋 Thank you for using HNSW Library!")
  (System/exit 0))
