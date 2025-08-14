(ns hnsw.benchmark-test
  "Benchmark tests using generated test data for various embedding model dimensions"
  (:require [clojure.test :refer :all]
            [clojure.java.io :as io]
            [hnsw.ultra-fast :as ultra]
            [hnsw.lightning :as lightning]
            [hnsw.api :as api]
            [data-generator :as gen]))

(defn load-or-generate
  "Load test data from file or generate if not exists"
  [size-key dim-key]
  (let [filename (format "test/data/test_vectors_%s_%s.json"
                         (name size-key) (name dim-key))
        file (io/file filename)]
    (if (.exists file)
      (do
        (println (format "Loading %s..." filename))
        (:vectors (gen/load-dataset filename)))
      (do
        (println (format "Generating %s vectors of dimension %s..."
                         (name size-key) (name dim-key)))
        (let [size (gen/sizes size-key)
              dim (gen/dimensions dim-key)
              vectors (gen/generate-dataset size dim)]
          (gen/save-dataset vectors filename)
          vectors)))))

(deftest ^:performance test-openai-embeddings
  (testing "Performance with OpenAI text-embedding-3-small dimensions (1536)"
    (let [vectors (load-or-generate :medium :large) ; 10k × 1536
          queries (gen/generate-query-set 100 1536)

          ;; Build index
          build-start (System/currentTimeMillis)
          index (ultra/build-index vectors)
          build-time (- (System/currentTimeMillis) build-start)

          ;; Search performance
          search-start (System/currentTimeMillis)
          results (doall (map #(ultra/search-knn index % 10) queries))
          search-time (- (System/currentTimeMillis) search-start)
          avg-search (/ search-time 100.0)]

      (println "\n📊 OpenAI Embeddings (1536 dim) Performance:")
      (println (format "  Dataset: 10,000 vectors"))
      (println (format "  Build time: %.2f seconds" (/ build-time 1000.0)))
      (println (format "  Avg search: %.2f ms" avg-search))
      (println (format "  Throughput: %.0f searches/sec" (/ 100000.0 search-time)))

      (is (< avg-search 15) "Search should be under 15ms for OpenAI embeddings")
      (is (every? #(= 10 (count %)) results)))))

(deftest ^:performance test-bert-embeddings
  (testing "Performance with BERT/Sentence-Transformers dimensions (768)"
    (let [vectors (load-or-generate :bible :medium) ; 30k × 768 (Bible size)
          queries (gen/generate-query-set 200 768)

          ;; Ultra-fast mode
          ultra-build-start (System/currentTimeMillis)
          ultra-index (ultra/build-index vectors)
          ultra-build-time (- (System/currentTimeMillis) ultra-build-start)

          ultra-search-start (System/currentTimeMillis)
          ultra-results (doall (map #(ultra/search-knn ultra-index % 10) queries))
          ultra-search-time (- (System/currentTimeMillis) ultra-search-start)

          ;; Lightning mode
          lightning-build-start (System/currentTimeMillis)
          lightning-index (lightning/build-index vectors
                                                 :num-partitions 32
                                                 :smart-partition? true)
          lightning-build-time (- (System/currentTimeMillis) lightning-build-start)

          lightning-search-start (System/currentTimeMillis)
          lightning-results (doall (map #(lightning/search-knn lightning-index % 10 0.25 true)
                                        queries))
          lightning-search-time (- (System/currentTimeMillis) lightning-search-start)]

      (println "\n📊 BERT Embeddings (768 dim) Performance - Bible-sized dataset:")
      (println (format "  Dataset: 30,000 vectors"))
      (println "\n  Ultra-fast mode:")
      (println (format "    Build: %.2f seconds" (/ ultra-build-time 1000.0)))
      (println (format "    Avg search: %.2f ms" (/ ultra-search-time 200.0)))
      (println (format "    Throughput: %.0f searches/sec" (/ 200000.0 ultra-search-time)))
      (println "\n  Lightning mode:")
      (println (format "    Build: %.2f seconds" (/ lightning-build-time 1000.0)))
      (println (format "    Avg search: %.2f ms" (/ lightning-search-time 200.0)))
      (println (format "    Throughput: %.0f searches/sec" (/ 200000.0 lightning-search-time)))
      (println (format "    Speedup: %.1fx" (/ (double ultra-search-time) lightning-search-time)))

      (is (every? #(= 10 (count %)) ultra-results))
      (is (every? #(= 10 (count %)) lightning-results))
      (is (< lightning-search-time ultra-search-time)
          "Lightning should be faster than Ultra-fast"))))

(deftest ^:performance test-lightweight-embeddings
  (testing "Performance with MiniLM dimensions (384)"
    (let [vectors (load-or-generate :stress :small) ; 50k × 384
          queries (gen/generate-query-set 500 384)

          build-start (System/currentTimeMillis)
          index (ultra/build-index vectors)
          build-time (- (System/currentTimeMillis) build-start)

          search-start (System/currentTimeMillis)
          results (doall (map #(ultra/search-knn index % 10) queries))
          search-time (- (System/currentTimeMillis) search-start)
          avg-search (/ search-time 500.0)]

      (println "\n📊 Lightweight Embeddings (384 dim) Performance:")
      (println (format "  Dataset: 50,000 vectors"))
      (println (format "  Build time: %.2f seconds" (/ build-time 1000.0)))
      (println (format "  Avg search: %.2f ms" avg-search))
      (println (format "  Throughput: %.0f searches/sec" (/ 500000.0 search-time)))

      (is (< avg-search 10) "Search should be very fast for small dimensions")
      (is (< build-time 20000) "Build should be under 20 seconds for 50k small vectors"))))

(deftest ^:performance test-maximum-dimensions
  (testing "Performance with maximum dimensions (3072) - Gemini/GPT-4 scale"
    (let [vectors (load-or-generate :small :max) ; 5k × 3072
          queries (gen/generate-query-set 50 3072)

          build-start (System/currentTimeMillis)
          index (ultra/build-index vectors)
          build-time (- (System/currentTimeMillis) build-start)

          search-start (System/currentTimeMillis)
          results (doall (map #(ultra/search-knn index % 10) queries))
          search-time (- (System/currentTimeMillis) search-start)
          avg-search (/ search-time 50.0)]

      (println "\n📊 Maximum Dimensions (3072) Performance:")
      (println (format "  Dataset: 5,000 vectors"))
      (println (format "  Build time: %.2f seconds" (/ build-time 1000.0)))
      (println (format "  Avg search: %.2f ms" avg-search))
      (println (format "  Memory usage estimate: ~%.0f MB"
                       (* 5000 3072 8 1.5 0.000001))) ; rough estimate

      (is (< avg-search 50) "Search should handle large dimensions reasonably")
      (is (every? #(= 10 (count %)) results)))))

(deftest ^:performance test-scaling-behavior
  (testing "How performance scales with dataset size"
    (let [dimension 768
          sizes [1000 5000 10000 20000 30000]
          results (atom [])]

      (println "\n📊 Scaling Behavior (768 dimensions):")
      (println "Size\tBuild(s)\tSearch(ms)")
      (println "-----\t--------\t----------")

      (doseq [size sizes]
        (let [vectors (gen/generate-dataset size dimension)
              queries (gen/generate-query-set 50 dimension)

              build-start (System/currentTimeMillis)
              index (ultra/build-index vectors)
              build-time (- (System/currentTimeMillis) build-start)

              search-start (System/currentTimeMillis)
              _ (doall (map #(ultra/search-knn index % 10) queries))
              search-time (- (System/currentTimeMillis) search-start)
              avg-search (/ search-time 50.0)]

          (println (format "%dk\t%.2f\t\t%.2f"
                           (/ size 1000)
                           (/ build-time 1000.0)
                           avg-search))
          (swap! results conj {:size size
                               :build-time build-time
                               :search-time avg-search})))

      ;; Check that performance scales sub-linearly
      (let [times @results
            small-search (:search-time (first times))
            large-search (:search-time (last times))
            size-ratio (/ (:size (last times)) (:size (first times)))
            time-ratio (/ large-search small-search)]
        (println (format "\nSize increased %.0fx, search time increased %.1fx"
                         size-ratio time-ratio))
        (is (< time-ratio (Math/sqrt size-ratio))
            "Search time should scale sub-linearly with size")))))

(deftest ^:performance test-recall-quality
  (testing "Search quality (recall) across different configurations"
    (let [vectors (load-or-generate :small :medium) ; 5k × 768
          test-indices (take 10 (range 0 5000 100))
          k 20]

      (println "\n📊 Search Quality (Recall @ 20):")

      (doseq [mode [:ultra :lightning]
              :let [index (case mode
                            :ultra (ultra/build-index vectors)
                            :lightning (lightning/build-index vectors
                                                              :num-partitions 16))
                    recalls (atom [])]]

        (doseq [idx test-indices]
          (let [query (nth vectors idx)
                ;; Get ground truth (brute force)
                ground-truth (set (take k
                                        (map first
                                             (sort-by second
                                                      (map-indexed
                                                       (fn [i v]
                                                         [i (gen/vector-distance query v)])
                                                       vectors)))))
                ;; Get HNSW results
                results (case mode
                          :ultra (ultra/search-knn index query k)
                          :lightning (lightning/search-knn index query k))
                result-ids (set (map :id results))
                recall (/ (count (clojure.set/intersection ground-truth result-ids))
                          (double k))]
            (swap! recalls conj recall)))

        (let [avg-recall (/ (reduce + @recalls) (count @recalls))]
          (println (format "  %s mode: %.1f%% recall"
                           (name mode)
                           (* 100 avg-recall)))
          (is (> avg-recall 0.85)
              (format "%s mode should have >85%% recall" (name mode))))))))

(deftest ^:performance test-api-convenience
  (testing "High-level API performance"
    (let [vectors (load-or-generate :medium :standard) ; 10k × 1024
          queries (gen/generate-query-set 100 1024)

          ;; Test different metrics
          euclidean-index (api/index vectors :metric :euclidean)
          cosine-index (api/index vectors :metric :cosine)

          euclidean-start (System/currentTimeMillis)
          euclidean-results (doall (map #(api/search euclidean-index % 10) queries))
          euclidean-time (- (System/currentTimeMillis) euclidean-start)

          cosine-start (System/currentTimeMillis)
          cosine-results (doall (map #(api/search cosine-index % 10) queries))
          cosine-time (- (System/currentTimeMillis) cosine-start)]

      (println "\n📊 API Performance (10k × 1024):")
      (println (format "  Euclidean - Avg search: %.2f ms" (/ euclidean-time 100.0)))
      (println (format "  Cosine    - Avg search: %.2f ms" (/ cosine-time 100.0)))

      (is (every? #(= 10 (count %)) euclidean-results))
      (is (every? #(= 10 (count %)) cosine-results)))))

(deftest ^:performance test-memory-efficiency
  (testing "Memory usage across different vector sizes"
    (println "\n📊 Memory Efficiency Analysis:")

    (doseq [[size-key dim-key expected-mb] [[:small :small 7] ; 5k × 384
                                            [:medium :medium 59] ; 10k × 768
                                            [:medium :large 118] ; 10k × 1536
                                            [:bible :medium 176]]] ; 30k × 768
      (let [size (gen/sizes size-key)
            dim (gen/dimensions dim-key)
            vectors (gen/generate-dataset size dim)

            ;; Force GC before measurement
            _ (System/gc)
            _ (Thread/sleep 100)

            before-mem (- (.. Runtime getRuntime totalMemory)
                          (.. Runtime getRuntime freeMemory))

            index (ultra/build-index vectors)

            _ (System/gc)
            _ (Thread/sleep 100)

            after-mem (- (.. Runtime getRuntime totalMemory)
                         (.. Runtime getRuntime freeMemory))

            used-mb (/ (- after-mem before-mem) 1048576.0)]

        (println (format "  %s × %s: ~%.0f MB (expected ~%d MB)"
                         (name size-key) (name dim-key) used-mb expected-mb))

        ;; Check memory usage is reasonable (within 2x of theoretical minimum)
        (is (< used-mb (* 2 expected-mb))
            (format "Memory usage should be reasonable for %s × %s"
                    (name size-key) (name dim-key)))))

    (println "\n  Note: Actual memory usage includes JVM overhead and index structures")))