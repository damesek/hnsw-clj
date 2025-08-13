(ns test.bible-search
  "Bible search test with 31k vectors"
  (:require [hnsw.filtered-index-io :as io]
            [clojure.data.json :as json]
            [clojure.java.io :as jio]
            [clojure.string :as str]
            [hnsw.simd-selector :as simd]))

(defn load-bible-data []
  (println "\n📖 Loading Bible Data")
  (println "=====================\n")

  ;; Load the Bible verses
  (println "1. Loading Bible verses...")
  (let [verses-file "data/KaroliRevid_m.tsv"
        verses (with-open [reader (jio/reader verses-file)]
                 (vec (map #(str/split % #"\t") (line-seq reader))))
        _ (println (format "   ✅ Loaded %d verses" (count verses)))

        ;; Create verse lookup map
        verse-lookup (into {}
                           (map-indexed (fn [idx verse]
                                          [idx {:book (nth verse 0)
                                                :chapter (nth verse 1)
                                                :verse-num (nth verse 2)
                                                :text (nth verse 3 "")}])
                                        verses))]

    ;; Load embeddings - FIX: Parse as array of objects
    (println "\n2. Loading embeddings...")
    (let [embeddings-file "data/bible_embeddings_30000.json"
          embeddings-str (slurp embeddings-file)
          ;; Parse as JSON array
          embeddings-data (json/read-str embeddings-str :key-fn keyword)
          ;; Handle both formats - array of objects or single object
          embeddings-list (if (sequential? embeddings-data)
                            embeddings-data
                            [embeddings-data])
          embedding-map (into {}
                              (map (fn [e]
                                     [(:index e) (double-array (:embedding e))])
                                   embeddings-list))]
      (println (format "   ✅ Loaded %d embeddings" (count embedding-map)))

      {:verses verse-lookup
       :embeddings embedding-map})))

(defn load-or-build-index [{:keys [verses embeddings]}]
  (println "\n3. Loading HNSW index...")

  ;; Try to load existing index
  (let [index-file "data/bible_index_30k.hnsw"]
    (if (.exists (jio/file index-file))
      (do
        (println (format "   Loading from %s..." index-file))
        ;; FIX: Add distance function parameter
        (let [index (io/load-filtered-index index-file simd/cosine-distance)]
          (println "   ✅ Index loaded")
          index))

      ;; Build new index if not found
      (do
        (println "   ⚠️ Index file not found, building new index...")
        (println "   Note: This will take a few minutes...")

        ;; Build using filtered index
        (require '[hnsw.filtered :as filtered])
        (let [dim (alength (first (vals embeddings)))
              _ (println (format "   Dimensions: %d" dim))
              ;; Create filtered graph
              graph (filtered/create-filtered-graph
                     :M 16
                     :ef-construction 200
                     :distance-fn simd/cosine-distance)]

          ;; Add all embeddings with metadata
          (println "   Adding vectors to index...")
          (doseq [[idx-num embedding] embeddings]
            (when (zero? (mod idx-num 1000))
              (println (format "   Progress: %d/%d" idx-num (count embeddings))))
            (filtered/insert-single graph
                                    (str idx-num)
                                    embedding
                                    (get verses idx-num)))

          (println "   ✅ Index built")

          ;; Save for next time
          (println "   Saving index...")
          (io/save-filtered-index graph index-file)
          (println "   ✅ Index saved")

          graph)))))

(defn verse-ref [{:keys [book chapter verse-num]}]
  (format "%s %s:%s" book chapter verse-num))

(defn search-bible [index verses query-text k]
  (println (format "\n🔍 Searching for: \"%s\"" query-text))
  (println (str (apply str (repeat 50 "-"))))

  ;; Find verses containing the text
  (let [matching-verses (filter (fn [[idx verse]]
                                  (str/includes?
                                   (str/lower-case (:text verse ""))
                                   (str/lower-case query-text)))
                                verses)
        match-count (count matching-verses)]

    (if (zero? match-count)
      (println "   No verses found containing this text.")

      (do
        (println (format "   Found %d verses containing \"%s\"" match-count query-text))

        ;; Use first match for similarity search
        (when-let [[idx verse] (first matching-verses)]
          (println (format "\n   Using verse for similarity search:"))
          (println (format "   %s - %s" (verse-ref verse)
                           (subs (:text verse) 0 (min 100 (count (:text verse))))))

          ;; Get the embedding for this verse - check in nodes
          (when-let [node (get (.nodes index) (str idx))]
            (let [embedding (.vector node)]
              (println (format "\n   Searching for %d similar verses..." k))

              ;; Search using filtered search
              (let [results (filtered/search-knn
                             index
                             embedding
                             k
                             (constantly true))] ; no filter

                (println (format "\n   Top %d similar verses:" k))
                (doseq [[i {:keys [id distance]}] (map-indexed vector results)]
                  (let [verse-idx (Integer/parseInt id)
                        verse (get verses verse-idx)]
                    (println (format "\n   %d. [%.3f] %s"
                                     (inc i)
                                     distance
                                     (verse-ref verse)))
                    (println (format "      %s"
                                     (subs (:text verse) 0
                                           (min 150 (count (:text verse))))))))))))))

    (println (str "\n" (apply str (repeat 50 "-"))))))

(defn run-bible-search-test []
  (println "\n📚 BIBLE SEARCH TEST (31K vectors)")
  (println "===================================")

  (try
    ;; Load data
    (let [{:keys [verses embeddings] :as data} (load-bible-data)]

      ;; Check if we have enough embeddings
      (if (< (count embeddings) 1000)
        (do
          (println (format "\n⚠️ Warning: Only %d embeddings found!" (count embeddings)))
          (println "   Expected 30,000 embeddings.")
          (println "   Please check data/bible_embeddings_30000.json")
          false)

        (let [index (load-or-build-index data)]
          ;; Run searches
          (search-bible index verses "szeretet" 5)
          (search-bible index verses "hit" 5)
          (search-bible index verses "gonosz" 5)
          (search-bible index verses "világosság" 5)

          (println "\n✅ Bible search test completed successfully!")
          true)))

    (catch Exception e
      (println "\n❌ Bible search test failed:")
      (println (.getMessage e))
      (.printStackTrace e)
      false)))

(defn -main [& args]
  (let [result (run-bible-search-test)]
    (System/exit (if result 0 1))))
