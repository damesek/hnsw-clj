#!/usr/bin/env clojure
;; =============================================
;; Build and Save HNSW Index for Complete Bible
;; =============================================

(require '[hnsw.ultra-fast :as ultra])
(require '[hnsw.ultra-optimized :as opt])
(require '[hnsw.index-io :as index-io])
(require '[clojure.data.json :as json])
(require '[clojure.java.io :as io])
(require '[clojure.string :as str])

(println "\n🏗️ HNSW INDEX BUILDER - COMPLETE BIBLE")
(println (str/join (repeat 60 "=")))

(def index-file "data/bible_index_complete.hnsw")
(def embeddings-file "data/bible_embeddings_complete.json")
(def rebuild? (= (first *command-line-args*) "--rebuild"))

;; Check if index already exists
(when (and (index-io/index-exists? index-file) (not rebuild?))
  (println (format "\n✅ Index already exists: %s" index-file))
  (println "   Use --rebuild flag to force rebuild")
  (System/exit 0))

;; Check if data file exists
(when-not (.exists (io/file embeddings-file))
  (println (format "❌ ERROR: %s not found!" embeddings-file))
  (println "   Please run: python scripts/export_complete_bible.py --all")
  (System/exit 1))

;; Load embeddings
(println "\n📚 Loading Complete Bible embeddings...")
(def start-load (System/currentTimeMillis))
(def data
  (with-open [reader (io/reader embeddings-file)]
    (json/read reader :key-fn keyword)))

(def verses (:verses data))
(def load-time (- (System/currentTimeMillis) start-load))

(println (format "✅ Loaded %d verses in %.2f seconds"
                 (count verses) (double (/ load-time 1000.0))))
(println (format "   Embedding dimension: %d"
                 (count (:embedding (first verses)))))
(println (format "   This is the COMPLETE Károli Bible! 📖"))

;; Build HNSW index
(println "\n🔨 Building HNSW index...")
(println "   Parameters: M=16, EF=100")
(println "   This will take about 3-4 minutes...")

(def start-build (System/currentTimeMillis))

(def index
  (opt/build-index
   (mapv (fn [v] [(:id v) (double-array (:embedding v))]) verses)
   :M 16
   :ef-construction 100
   :distance-fn opt/fast-cosine-distance
   :show-progress? true))

(def build-time (- (System/currentTimeMillis) start-build))

(println (format "\n✅ Index built in %.2f seconds" (double (/ build-time 1000.0))))
(println (format "   Indexing speed: %.0f verses/second"
                 (double (/ (* (count verses) 1000.0) build-time))))

;; Save the index
(index-io/save-index index index-file)

;; Test the saved index
(println "\n🧪 Testing saved index...")
(let [test-index (index-io/load-index index-file opt/fast-cosine-distance)]
  (if test-index
    (let [test-verse (first verses)
          query (double-array (:embedding test-verse))
          results (opt/search test-index query 5)]
      (println "   Test search successful!")
      (println (format "   Found %d results" (count results)))
      (println (format "   First result: %s" (:id (first results)))))
    (println "   ❌ Failed to load saved index!")))

(println "\n✅ INDEX BUILD COMPLETE!")
(println (str/join (repeat 60 "=")))
(println (format "Index saved to: %s" index-file))
(println (format "Total verses indexed: %,d" (count verses)))
(println "You can now run tests with the complete Bible!")
(println "\nUsage in tests:")
(println "  (def index (index-io/load-index \"data/bible_index_complete.hnsw\" opt/fast-cosine-distance))")

(System/exit 0)
