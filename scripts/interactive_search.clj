#!/usr/bin/env clojure
;; =============================================
;; Interactive Bible Search - Load once, search many times
;; =============================================

(require '[hnsw.ultra-fast :as ultra])
(require '[hnsw.ultra-optimized :as opt])
(require '[hnsw.index-io :as index-io])
(require '[clojure.data.json :as json])
(require '[clojure.java.io :as io])
(require '[clojure.string :as str])

(println "\n🔍 INTERACTIVE BIBLE SEARCH - COMPLETE KÁROLI BIBLE")
(println (str/join (repeat 60 "=")))

(def index-file "data/bible_index_complete.hnsw")
(def embeddings-file "data/bible_embeddings_complete.json")

;; Check if files exist
(when-not (.exists (io/file embeddings-file))
  (println "❌ ERROR: Complete Bible embeddings not found!")
  (println "   Please run: python scripts/export_complete_bible.py --all")
  (System/exit 1))

(when-not (index-io/index-exists? index-file)
  (println "❌ ERROR: Index not found!")
  (println "   Please run: clojure -M scripts/build_index_complete.clj")
  (System/exit 1))

;; Load everything ONCE
(println "\n⏳ Loading Bible data (this happens only once)...")
(def start-load (System/currentTimeMillis))

(def data
  (with-open [reader (io/reader embeddings-file)]
    (json/read reader :key-fn keyword)))

(def verses (:verses data))
(def verse-lookup (into {} (map (fn [v] [(:id v) v]) verses)))

(println (format "✅ Loaded %,d verses in %.2f seconds"
                 (count verses)
                 (double (/ (- (System/currentTimeMillis) start-load) 1000.0))))

(println "\n📂 Loading search index...")
(def start-index (System/currentTimeMillis))
(def index (index-io/load-index index-file opt/fast-cosine-distance))
(println (format "✅ Index loaded in %.2f seconds"
                 (double (/ (- (System/currentTimeMillis) start-index) 1000.0))))

;; Helper functions
(defn verse-ref [verse]
  (format "%s %s:%s"
          (or (:book_full verse) (:book verse))
          (:chapter verse)
          (:verse verse)))

(defn find-verses-by-text [search-text]
  (filter #(str/includes? (str/lower-case (:text %))
                          (str/lower-case search-text))
          verses))

(defn search-and-display [search-term num-results]
  (let [matching-verses (find-verses-by-text search-term)]
    (if (empty? matching-verses)
      (println (format "\n❌ No verses found containing '%s'" search-term))

      (do
        (println (format "\n✅ Found %d verses containing '%s'"
                         (count matching-verses) search-term))

        (let [first-match (first matching-verses)]
          (println (format "\n📖 First match: [%s]" (verse-ref first-match)))
          (println (format "   \"%s\"" (:text first-match)))

          (println (format "\n🔗 Top %d semantically similar verses:" num-results))
          (println (str/join (repeat 60 "-")))

          (let [query-emb (double-array (:embedding first-match))
                start-time (System/nanoTime)
                results (opt/search index query-emb (inc num-results))
                search-time (/ (- (System/nanoTime) start-time) 1000000.0)]

            (println (format "⏱️ Search completed in %.2f ms\n" search-time))

            ;; Skip first result (it's the query verse itself)
            (doseq [[idx {:keys [id distance]}] (map-indexed vector (rest results))]
              (when-let [verse (get verse-lookup id)]
                (println (format "%2d. [%s] (similarity: %.1f%%)"
                                 (inc idx)
                                 (verse-ref verse)
                                 (* 100 (- 1 distance))))
                (println (format "    \"%s\"\n"
                                 (subs (:text verse) 0 (min 120 (count (:text verse)))))))))

          ;; Show other matching verses if any
          (when (> (count matching-verses) 1)
            (println (format "\n📚 Other verses containing '%s' (%d total):"
                             search-term (count matching-verses)))
            (doseq [v (take 3 (rest matching-verses))]
              (println (format "  • [%s] %s..."
                               (verse-ref v)
                               (subs (:text v) 0 (min 60 (count (:text v)))))))))))))

;; Show help
(defn show-help []
  (println "\n📝 COMMANDS:")
  (println "  <word/phrase> [num]  - Search for word/phrase (optional: number of results)")
  (println "  help                 - Show this help")
  (println "  stats                - Show statistics")
  (println "  books                - List all books")
  (println "  random               - Search from a random verse")
  (println "  exit/quit            - Exit the program")
  (println "\nEXAMPLES:")
  (println "  szeretet            - Search for 'szeretet' (10 results)")
  (println "  szeretet 5          - Search for 'szeretet' (5 results)")
  (println "  örök élet 15        - Search for 'örök élet' (15 results)")
  (println "  Jézus 20            - Search for 'Jézus' (20 results)"))

;; Show stats
(defn show-stats []
  (println "\n📊 BIBLE STATISTICS:")
  (println (str/join (repeat 50 "-")))
  (println (format "• Total verses: %,d" (count verses)))
  (let [books (distinct (map :book verses))
        ot-books #{"Ter" "2Móz" "3Móz" "4Móz" "5Móz" "Józs" "Bír" "Ruth"
                   "1Sám" "2Sám" "1Kir" "2Kir" "1Krón" "2Krón" "Ezsd"
                   "Neh" "Eszt" "Jób" "Zsolt" "Péld" "Préd" "Ének" "Ézs"
                   "Jer" "Siral" "Ezék" "Dán" "Hós" "Jóel" "Ámós" "Abd"
                   "Jón" "Mik" "Náh" "Hab" "Zof" "Hag" "Zak" "Mal"}
        ot-verses (filter #(contains? ot-books (:book %)) verses)
        nt-verses (filter #(not (contains? ot-books (:book %))) verses)]
    (println (format "• Total books: %d" (count books)))
    (println (format "• Old Testament: %,d verses" (count ot-verses)))
    (println (format "• New Testament: %,d verses" (count nt-verses))))
  (println (format "• Index size: %.1f MB"
                   (double (/ (.length (io/file index-file)) (* 1024 1024))))))

;; List books
(defn list-books []
  (println "\n📚 BOOKS IN THE BIBLE:")
  (println (str/join (repeat 50 "-")))
  (let [books-with-counts (->> verses
                               (group-by :book)
                               (map (fn [[book vs]]
                                      [book (count vs)]))
                               (sort-by first))]
    (doseq [[book count] books-with-counts]
      (println (format "  %-10s - %4d verses" book count)))))

;; Random verse search
(defn search-random []
  (let [random-verse (rand-nth verses)]
    (println (format "\n🎲 Random verse: [%s]" (verse-ref random-verse)))
    (println (format "   \"%s\"" (:text random-verse)))

    (println "\n🔗 Similar verses:")
    (println (str/join (repeat 60 "-")))

    (let [query-emb (double-array (:embedding random-verse))
          start-time (System/nanoTime)
          results (opt/search index query-emb 6)
          search-time (/ (- (System/nanoTime) start-time) 1000000.0)]

      (println (format "⏱️ Search completed in %.2f ms\n" search-time))

      (doseq [[idx {:keys [id distance]}] (map-indexed vector (rest results))]
        (when-let [verse (get verse-lookup id)]
          (println (format "%d. [%s] (similarity: %.1f%%)"
                           (inc idx)
                           (verse-ref verse)
                           (* 100 (- 1 distance))))
          (println (format "   \"%s\"\n"
                           (subs (:text verse) 0 (min 100 (count (:text verse)))))))))))

;; Main interactive loop
(println "\n✅ SYSTEM READY! Type 'help' for commands.")
(println (str/join (repeat 60 "=")))

(show-help)

(loop []
  (print "\n🔍 Search> ")
  (flush)

  (when-let [input (read-line)]
    (let [parts (str/split (str/trim input) #"\s+")
          command (str/lower-case (first parts))]

      (cond
        ;; Exit commands
        (contains? #{"exit" "quit" "q"} command)
        (do
          (println "\n👋 Goodbye! Thank you for using Bible Search!")
          (System/exit 0))

        ;; Help
        (= command "help")
        (do
          (show-help)
          (recur))

        ;; Stats
        (= command "stats")
        (do
          (show-stats)
          (recur))

        ;; Books
        (= command "books")
        (do
          (list-books)
          (recur))

        ;; Random
        (= command "random")
        (do
          (search-random)
          (recur))

        ;; Empty input
        (str/blank? input)
        (recur)

        ;; Search
        :else
        (let [search-term (if (> (count parts) 1)
                           ;; Multi-word search, but check if last is a number
                            (let [last-part (last parts)]
                              (if (re-matches #"\d+" last-part)
                                (str/join " " (butlast parts))
                                input))
                            input)
              num-results (if (and (> (count parts) 1)
                                   (re-matches #"\d+" (last parts)))
                            (Integer/parseInt (last parts))
                            10)]

          (println (format "\n🎯 SEARCHING FOR: \"%s\" (top %d results)"
                           search-term num-results))
          (println (str/join (repeat 60 "=")))

          (search-and-display search-term num-results)
          (recur))))))

;; Keep the program running if no input
(println "\n⚠️ Input stream ended. Exiting...")
