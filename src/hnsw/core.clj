(ns hnsw.core
  "HNSW Library - Main Entry Point
   
   High-performance Hierarchical Navigable Small World implementation
   for approximate nearest neighbor search in Clojure."
  (:require [hnsw.graph :as graph]
            [hnsw.ultra-fast :as ultra]
            [clojure.string :as str])
  (:gen-class))

;; ===== Main Entry Point =====

(defn -main
  "Main entry point for the HNSW library"
  [& args]
  (println "\n🚀 HNSW Library - High-Performance Vector Search")
  (println (str/join (repeat 60 "=")))

  (println "\n👋 Thank you for using HNSW Library!")
  (System/exit 0))
