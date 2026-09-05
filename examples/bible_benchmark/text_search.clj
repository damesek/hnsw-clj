(ns text-search
  "Interactive text-to-top-10 Bible search using Clojure HNSW."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [hnsw.ultra-fast :as hnsw])
  (:import [java.io BufferedReader BufferedWriter InputStreamReader OutputStreamWriter]
           [java.nio ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Path Paths StandardOpenOption]
           [java.util Locale]
           [java.util.concurrent TimeUnit]))

(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

(defn- env-long
  [name default]
  (long (if-let [value (System/getenv name)]
          (Long/parseLong value)
          default)))

(defn- environment
  []
  {:embedding-dir (or (System/getenv "EMBEDDING_DIR")
                      "examples/bible_benchmark/data")
   :index-language (keyword (or (System/getenv "INDEX_LANGUAGE") "hu"))
   :num-vectors (env-long "NUM_VECTORS" 31173)
   :threads (env-long "THREADS" (.availableProcessors (Runtime/getRuntime)))
   :top-k (env-long "TOP_K" 10)
   :M (env-long "M" 16)
   :ef-construction (env-long "EF_CONSTRUCTION" 200)
   :ef-search (env-long "EF_SEARCH" 50)
   :seed (env-long "SEED" 42)
   :encoder-python (or (System/getenv "QUERY_ENCODER_PYTHON") "python3")})

(defn- load-float-vectors
  [path count dimensions]
  (let [^Path nio-path (Paths/get (str path) (make-array String 0))
        expected-bytes (* count dimensions Float/BYTES)]
    (with-open [channel (FileChannel/open
                         nio-path
                         (into-array StandardOpenOption [StandardOpenOption/READ]))]
      (when (> expected-bytes (.size channel))
        (throw (ex-info "Unexpected float32 vector file size"
                        {:path (str path)
                         :minimum expected-bytes
                         :actual (.size channel)})))
      (let [buffer (doto (.map channel FileChannel$MapMode/READ_ONLY 0 expected-bytes)
                     (.order ByteOrder/LITTLE_ENDIAN))
            floats (.asFloatBuffer buffer)]
        (mapv (fn [_]
                (let [vector (double-array dimensions)]
                  (dotimes [dimension dimensions]
                    (aset-double vector dimension (double (.get floats))))
                  vector))
              (range count))))))

(defn- load-data
  [{:keys [embedding-dir index-language num-vectors]}]
  (let [manifest-file (io/file embedding-dir "manifest.json")]
    (when-not (.exists manifest-file)
      (throw (ex-info "Embedding manifest not found; run run_prepare_embeddings.sh first"
                      {:path (.getAbsolutePath manifest-file)})))
    (let [manifest (with-open [reader (io/reader manifest-file)]
                     (json/read reader :key-fn keyword))
          descriptor (get-in manifest [:languages index-language])]
      (when-not descriptor
        (throw (ex-info "Index language is not present in the manifest"
                        {:language index-language})))
      (let [count (min (long (:count descriptor)) num-vectors)
            dimensions (long (:dimensions manifest))]
        {:manifest manifest
         :translation (:translation descriptor)
         :items (subvec (vec (:items descriptor)) 0 count)
         :vectors (load-float-vectors
                   (io/file embedding-dir (:vectors descriptor)) count dimensions)
         :dimensions dimensions}))))

(defrecord QueryEncoder [^Process process ^BufferedWriter writer ^BufferedReader reader])

(defn- start-query-encoder
  [{:keys [embedding-dir encoder-python]}]
  (let [script (.getAbsolutePath
                (io/file "examples/bible_benchmark/query_encoder.py"))
        builder (ProcessBuilder. (into-array String [encoder-python script]))
        _ (.put (.environment builder) "EMBEDDING_DIR"
                (.getAbsolutePath (io/file embedding-dir)))
        _ (.redirectError builder java.lang.ProcessBuilder$Redirect/INHERIT)
        process (.start builder)]
    (->QueryEncoder
     process
     (BufferedWriter. (OutputStreamWriter. (.getOutputStream process)))
     (BufferedReader. (InputStreamReader. (.getInputStream process))))))

(defn- stop-query-encoder!
  [^QueryEncoder encoder]
  (try
    (.write (.writer encoder) (json/write-str {:command "shutdown"}))
    (.newLine (.writer encoder))
    (.flush (.writer encoder))
    (.readLine (.reader encoder))
    (catch Exception _))
  (try
    (.close (.writer encoder))
    (catch Exception _))
  (try
    (.close (.reader encoder))
    (catch Exception _))
  (when-not (.waitFor (.process encoder) 10 TimeUnit/SECONDS)
    (.destroy (.process encoder))))

(defn- encode-query!
  ^doubles [^QueryEncoder encoder text]
  (.write (.writer encoder) (json/write-str {:text text}))
  (.newLine (.writer encoder))
  (.flush (.writer encoder))
  (let [line (.readLine (.reader encoder))]
    (when-not line
      (throw (ex-info "MPNet encoder terminated unexpectedly"
                      {:exit-code (when-not (.isAlive (.process encoder))
                                    (.exitValue (.process encoder)))})))
    (let [{:keys [vector error]} (json/read-str line :key-fn keyword)]
      (when error
        (throw (ex-info "MPNet query encoding failed" {:error error})))
      (double-array vector))))

(defn- print-results
  [query results items]
  (println (str "\nQuery: \"" query "\""))
  (println (apply str (repeat 80 "=")))
  (doseq [[rank {:keys [id distance]}] (map-indexed vector results)]
    (let [item (nth items (Long/parseLong id))
          similarity (- 1.0 distance)]
      (println (String/format
                Locale/ROOT
                "%2d. %s %d:%d | similarity=%.4f distance=%.4f"
                (to-array [(inc rank) (:book item) (:chapter item) (:verse item)
                           similarity distance])))
      (println (str "    " (:text item)))))
  (println))

(defn- search-once!
  [encoder index items top-k ef-search query]
  (let [vector (encode-query! encoder query)
        results (hnsw/search-knn index vector top-k ef-search)]
    (print-results query results items)))

(defn- interactive-search!
  [encoder index items top-k ef-search]
  (println "Írj be magyar vagy angol szöveget. Kilépés: :q")
  (loop []
    (print "Keresés> ")
    (flush)
    (when-let [query (read-line)]
      (let [query (string/trim query)]
        (when-not (#{":q" "quit" "exit"} query)
          (when (seq query)
            (search-once! encoder index items top-k ef-search query))
          (recur))))))

(defn -main
  [& arguments]
  (let [{:keys [threads top-k M ef-construction ef-search seed] :as config}
        (environment)]
    (when (some #(not (pos? (long %)))
                [(:num-vectors config) threads top-k M ef-construction ef-search])
      (throw (IllegalArgumentException. "Search sizes and HNSW parameters must be positive")))
    (let [encoder (start-query-encoder config)]
      (try
        (let [{:keys [manifest translation items vectors dimensions]} (load-data config)
              indexed-vectors (mapv (fn [label vector] [(str label) vector])
                                    (range (count vectors)) vectors)]
          (println (str "Loading " (:model manifest) " encoder and building "
                        (format "%,d" (count vectors)) "-verse " translation " index..."))
          (let [started (System/nanoTime)
                index (hnsw/build-index indexed-vectors
                                        :M M
                                        :ef-construction ef-construction
                                        :ef-search ef-search
                                        :compact-precision :float32
                                        :seed seed
                                        :num-threads threads
                                        :batch-size 4
                                        :show-progress? false)
                build-seconds (/ (- (System/nanoTime) started) 1.0e9)]
            (println (String/format
                      Locale/ROOT
                      "Clojure HNSW index ready in %.2fs (dimensions=%d, threads=%d, efSearch=%d, topK=%d)"
                      (to-array [build-seconds dimensions threads ef-search top-k])))
            (let [query (string/trim (string/join " " arguments))]
              (if (seq query)
                (search-once! encoder index items top-k ef-search query)
                (interactive-search! encoder index items top-k ef-search)))))
        (finally
          (stop-query-encoder! encoder))))))
