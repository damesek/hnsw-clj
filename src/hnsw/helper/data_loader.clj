(ns hnsw.helper.data-loader
  "Optimized data loading for large Bible embeddings"
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io])
  (:import [java.io BufferedReader FileReader]))

(defn load-bible-vectors 
  "Load Bible vectors efficiently from JSON file
   Supports: bible_embeddings.json (1k)
            bible_embeddings_10000.json (10k)
            bible_embeddings_30000.json (30k)
            bible_embeddings_complete.json (31k)"
  [filename]
  (println (format "Loading %s..." filename))
  
  (try
    (let [start (System/currentTimeMillis)
          ;; Read file in chunks to avoid OOM
          data (with-open [rdr (io/reader filename)]
                 (json/read rdr :key-fn keyword))
          verses (:verses data)
          vectors (mapv (fn [v] 
                         [(:id v) (double-array (:embedding v))]) 
                       verses)
          elapsed (- (System/currentTimeMillis) start)]
      
      (println (format "✅ Loaded %d vectors in %.2fs (%.0f vec/s)" 
                      (count vectors)
                      (/ elapsed 1000.0)
                      (/ (count vectors) (/ elapsed 1000.0))))
      
      {:vectors vectors
       :text-map (into {} (map (fn [v] [(:id v) (:text v)]) verses))
       :metadata {:count (count vectors)
                  :load-time elapsed
                  :filename filename}})
    
    (catch OutOfMemoryError e
      (println "❌ OOM Error - need more heap space!")
      (println "   Run with: -J-Xmx10g -J-Xms8g")
      nil)
    
    (catch Exception e
      (println (format "❌ Error loading file: %s" (.getMessage e)))
      nil)))

(defn get-best-available-data
  "Get the largest available Bible dataset that can be loaded"
  []
  (let [files ["data/bible_embeddings_complete.json"
               "data/bible_embeddings_30000.json"
               "data/bible_embeddings_10000.json"
               "data/bible_embeddings.json"]]
    (loop [remaining files]
      (when (seq remaining)
        (let [file (first remaining)]
          (if (.exists (io/file file))
            (if-let [data (load-bible-vectors file)]
              data
              (recur (rest remaining)))
            (recur (rest remaining))))))))
