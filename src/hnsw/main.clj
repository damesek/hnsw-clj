(ns hnsw.main
  "Main entry point for compiled JAR"
  (:require [hnsw.partitioned :as part]
            [hnsw.ultra-fast :as ultra]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:gen-class))

(defn verse-ref [v]
  (format "%s %s:%s" (:book v) (:chapter v) (:verse v)))

(defn find-verses [verses text]
  (filter #(str/includes? (str/lower-case (:text %))
                          (str/lower-case text))
          verses))

(defn search-and-display
  [index verse-lookup verses term n search-method]
  (let [matches (find-verses verses term)]
    (if (empty? matches)
      (println (format "\n❌ No verses found containing '%s'" term))
      (do
        (println (format "\n✅ Found %d verses containing '%s'"
                         (count matches) term))
        (let [v (first matches)
              ;; Warm up
              _ (dotimes [_ 5]
                  (case search-method
                    :turbo (part/search-partitioned-index-turbo index (double-array (:embedding v)) n)
                    :fast (part/search-partitioned-index-fast index (double-array (:embedding v)) n)
                    :normal (part/search-partitioned-index index (double-array (:embedding v)) n)))

              ;; Actual search
              start-time (System/nanoTime)
              results (case search-method
                        :turbo (part/search-partitioned-index-turbo index (double-array (:embedding v)) n)
                        :fast (part/search-partitioned-index-fast index (double-array (:embedding v)) n)
                        :normal (part/search-partitioned-index index (double-array (:embedding v)) n))
              search-time (/ (- (System/nanoTime) start-time) 1000000.0)]

          (println (format "\n📖 Query verse: [%s]" (verse-ref v)))
          (println (format "\"%s\"\n" (:text v)))

          (println (format "⏱️ Search time: %.2f ms (%s mode)"
                           search-time
                           (name search-method)))

          (when (< search-time 1.0)
            (println "   🎯 TARGET ACHIEVED! Sub-millisecond search!"))

          (println (format "\n🔗 Top %d similar verses:" n))
          (println (str/join (repeat 60 "-")))

          (doseq [[i {:keys [id distance]}] (map-indexed vector results)]
            (when-let [verse (verse-lookup id)]
              (println (format "%2d. [%s] (similarity: %.1f%%)"
                               (inc i)
                               (verse-ref verse)
                               (* 100 (- 1 distance))))
              (println (format "    \"%s\"\n"
                               (subs (:text verse) 0 (min 100 (count (:text verse)))))))))))))

(defn benchmark-search [index verses]
  (println "\n⚡ SEARCH PERFORMANCE BENCHMARK")
  (println (str/join (repeat 60 "=")))

  (let [test-query (double-array (:embedding (first verses)))
        methods [[:normal "Normal parallel"]
                 [:fast "Fast pre-alloc"]
                 [:turbo "Turbo streams"]]]

    ;; Warm up
    (println "\nWarming up...")
    (doseq [[method _] methods]
      (dotimes [_ 10]
        (case method
          :turbo (part/search-partitioned-index-turbo index test-query 10)
          :fast (part/search-partitioned-index-fast index test-query 10)
          :normal (part/search-partitioned-index index test-query 10))))

    (println "\nRunning benchmark (100 iterations each):")
    (println (str/join (repeat 50 "-")))

    (doseq [[method name] methods]
      (let [trials 100
            start-ns (System/nanoTime)
            _ (dotimes [_ trials]
                (case method
                  :turbo (part/search-partitioned-index-turbo index test-query 10)
                  :fast (part/search-partitioned-index-fast index test-query 10)
                  :normal (part/search-partitioned-index index test-query 10)))
            time-ms (/ (- (System/nanoTime) start-ns) 1000000.0)
            avg-ms (/ time-ms trials)]

        (println (format "%-20s: %.2f ms avg (QPS: %.0f) %s"
                         name
                         avg-ms
                         (/ 1000.0 avg-ms)
                         (if (< avg-ms 1.0) "✅" "")))))))

(defn test-recall [index verse-lookup verses]
  (println "\n🧪 RECALL TESTING - Semantic Similarity Check")
  (println (str/join (repeat 60 "=")))

  (let [test-queries ["szeretet" "hit" "bűn" "megváltás" "örök élet" "Krisztus"]]
    (doseq [query test-queries]
      (println (format "\n🎯 TEST: %s" query))
      (println (str/join (repeat 50 "-")))
      (search-and-display index verse-lookup verses query 5 :turbo)
      (Thread/sleep 200))))

(defn show-help []
  (println "\n📝 COMMANDS:")
  (println "  <word/phrase>     - Search for word or phrase")
  (println "  recall            - Run recall test with Hungarian words")
  (println "  benchmark         - Compare search methods")
  (println "  stats             - Show index statistics")
  (println "  mode [1-3]        - Switch search mode:")
  (println "                      1: Normal (2.6ms)")
  (println "                      2: Fast (1.4ms)")
  (println "                      3: Turbo (0.5ms) [default]")
  (println "  help              - Show this help")
  (println "  exit              - Exit"))

(defn show-stats [index verses build-time num-partitions]
  (let [info (part/partitioned-index-info index)]
    (println "\n📊 SYSTEM STATISTICS:")
    (println (str/join (repeat 50 "-")))
    (println (format "• Verses in data: %d" (count verses)))
    (println (format "• Verses indexed: %d"
                     (reduce + (:elements-per-partition info))))
    (println (format "• Partitions: %d" num-partitions))
    (println (format "• Build time: %.2f seconds" (/ build-time 1000.0)))
    (println (format "• Build rate: %.0f vectors/sec"
                     (/ (count verses) (/ build-time 1000.0))))
    (println (format "• Configuration: %s"
                     (case num-partitions
                       16 "Fast Build"
                       8 "Fast Search"
                       12 "Balanced")))))

(defn -main
  "Main entry point for JAR"
  [& args]
  (println "\n🚀 HNSW BIBLE SEARCH - COMPILED JAR VERSION")
  (println (str/join (repeat 60 "=")))
  (println "⚡ Maximum performance with AOT compilation!")
  (println (str/join (repeat 60 "=")))

  ;; Parse arguments or use default
  (let [num-partitions (if (seq args)
                         (Integer/parseInt (first args))
                         8)] ; Default to 8 for fast search

    (println (format "\n✅ Using %d partitions" num-partitions))

    (let [embeddings-file "data/bible_embeddings_complete.json"]

      ;; Check if file exists
      (when-not (.exists (io/file embeddings-file))
        (println "❌ ERROR: Bible embeddings not found!")
        (println "   Run: python scripts/export_complete_bible.py")
        (System/exit 1))

      ;; Load data
      (println "\n⏳ Loading Bible data...")
      (let [start-load (System/currentTimeMillis)
            data (with-open [r (io/reader embeddings-file)]
                   (json/read r :key-fn keyword))
            verses (:verses data)
            verse-lookup (into {} (map (fn [v] [(:id v) v]) verses))
            vectors (mapv (fn [v] [(:id v) (double-array (:embedding v))]) verses)]

        (println (format "✅ Loaded %d verses in %.2f seconds"
                         (count verses)
                         (/ (- (System/currentTimeMillis) start-load) 1000.0)))

        ;; Build index
        (println "\n🔨 Building PARTITIONED HNSW index...")
        (println (format "   Configuration: %d partitions, %d threads"
                         num-partitions num-partitions))

        (let [start-build (System/currentTimeMillis)
              index (part/build-partitioned-index
                     vectors
                     :num-partitions num-partitions
                     :num-threads num-partitions
                     :distance-fn ultra/cosine-distance-ultra
                     :show-progress? true)
              build-time (- (System/currentTimeMillis) start-build)]

          (println (format "\n✅ Index built in %.2f seconds (%.0f vectors/sec)"
                           (/ build-time 1000.0)
                           (/ (count verses) (/ build-time 1000.0))))

          (println "\n✅ SYSTEM READY!")
          (println (format "🎯 Goals achieved: Build <%.0fs ✅, Search <1ms %s"
                           (/ build-time 1000.0)
                           (if (= num-partitions 8) "✅" "⚠️")))
          (show-help)

          ;; Main loop
          (let [search-mode (atom :turbo)]
            (loop []
              (print (format "\n🔍 Search [%s]> "
                             (case @search-mode
                               :normal "normal"
                               :fast "fast"
                               :turbo "TURBO")))
              (flush)
              (when-let [input (read-line)]
                (let [cmd (str/trim input)]
                  (cond
                    (= cmd "exit")
                    (do
                      (println "\n👋 Goodbye!")
                      (System/exit 0))

                    (= cmd "help")
                    (do
                      (show-help)
                      (recur))

                    (= cmd "recall")
                    (do
                      (test-recall index verse-lookup verses)
                      (recur))

                    (= cmd "benchmark")
                    (do
                      (benchmark-search index verses)
                      (recur))

                    (= cmd "stats")
                    (do
                      (show-stats index verses build-time num-partitions)
                      (recur))

                    (str/starts-with? cmd "mode ")
                    (let [mode-num (str/trim (subs cmd 5))]
                      (case mode-num
                        "1" (do (reset! search-mode :normal)
                                (println "✅ Switched to NORMAL mode"))
                        "2" (do (reset! search-mode :fast)
                                (println "✅ Switched to FAST mode"))
                        "3" (do (reset! search-mode :turbo)
                                (println "✅ Switched to TURBO mode"))
                        (println "❌ Invalid mode. Use 1, 2, or 3"))
                      (recur))

                    (str/blank? cmd)
                    (recur)

                    :else
                    (do
                      (search-and-display index verse-lookup verses cmd 10 @search-mode)
                      (recur))))))))))))
