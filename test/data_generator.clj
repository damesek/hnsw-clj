(ns data-generator
  "Test data generator for HNSW testing with various embedding dimensions and dataset sizes"
  (:require [clojure.java.io :as io]
            [clojure.data.json :as json])
  (:import [java.util Random]
           [java.time Instant]))

;; Common embedding dimensions from popular models
(def dimensions
  {:mini 256 ; Reduced dimensions
   :small 384 ; MiniLM, Cohere Light
   :medium 768 ; BERT-based, Gecko, Arctic-M
   :standard 1024 ; Voyage, Mistral, Arctic-L, Cohere
   :large 1536 ; OpenAI embeddings
   :xlarge 2048 ; Voyage-3-large
   :max 3072}) ; OpenAI large, Gemini

;; Dataset sizes for testing
(def sizes
  {:tiny 100 ; Debug only
   :smoke 1000 ; Quick smoke tests (1K)
   :small 5000 ; Fast unit tests (5K)
   :medium 10000 ; Standard integration tests (10K)
   :large 20000 ; Performance validation (20K)
   :bible 30000 ; Real-world size (30K)
   :stress 50000}) ; Large-scale testing (50K)

(defn generate-random-vector
  "Generate a random vector with specified dimension using Gaussian distribution"
  [^Random rng dim]
  (vec (repeatedly dim #(.nextGaussian rng))))

(defn generate-unit-vector
  "Generate a random unit vector (normalized) with specified dimension"
  [^Random rng dim]
  (let [v (generate-random-vector rng dim)
        norm (Math/sqrt (reduce + (map #(* % %) v)))]
    (if (zero? norm)
      v
      (vec (map #(/ % norm) v)))))

(defn generate-clustered-vector
  "Generate a vector near a cluster center with some noise"
  [^Random rng dim center noise-level]
  (vec (map-indexed
        (fn [_idx c]
          (+ c (* noise-level (.nextGaussian rng))))
        center)))

(defn generate-dataset
  "Generate a dataset with specified size and dimension.
   Options:
   - :distribution - :gaussian (default), :uniform, :unit, :clustered
   - :num-clusters - for clustered distribution (default 10)
   - :noise-level - for clustered distribution (default 0.1)
   - :seed - random seed (default 42)
   - :format - :vector (default), :double-array, :indexed
   
   Formats:
   - :vector - returns vectors of doubles
   - :double-array - returns double arrays (for SIMD operations)
   - :indexed - returns [id vector] pairs for ultra-fast implementation"
  [size dim & {:keys [distribution num-clusters noise-level seed format]
               :or {distribution :gaussian
                    num-clusters 10
                    noise-level 0.1
                    seed 42
                    format :vector}}]
  (let [rng (Random. seed)
        generate-fn (case distribution
                      :gaussian #(generate-random-vector rng dim)
                      :uniform #(vec (repeatedly dim (fn [] (- (* 2 (.nextDouble rng)) 1))))
                      :unit #(generate-unit-vector rng dim)
                      :clustered
                      (let [centers (vec (repeatedly num-clusters
                                                     #(generate-random-vector rng dim)))]
                        #(generate-clustered-vector rng dim
                                                    (nth centers (.nextInt rng num-clusters))
                                                    noise-level)))
        vectors (vec (repeatedly size generate-fn))]
    (case format
      :vector vectors
      :double-array (mapv double-array vectors)
      :indexed (vec (map-indexed
                     (fn [idx v]
                       [(str "vec_" idx) (double-array v)])
                     vectors)))))

(defn save-dataset
  "Save dataset to JSON file with metadata"
  [vectors filepath & {:keys [distribution seed]
                       :or {distribution :gaussian
                            seed 42}}]
  (let [metadata {:count (count vectors)
                  :dimension (count (first vectors))
                  :generated (str (Instant/now))
                  :distribution (name distribution)
                  :seed seed}
        data {:metadata metadata
              :vectors (map-indexed
                        (fn [idx v]
                          {:id (str "vec_" idx)
                           :embedding v})
                        vectors)}]
    (io/make-parents filepath)
    (with-open [w (io/writer filepath)]
      (json/write data w))
    (println (format "Saved %d vectors of dimension %d to %s"
                     (:count metadata)
                     (:dimension metadata)
                     filepath))
    metadata))

(defn load-dataset
  "Load dataset from JSON file"
  [filepath]
  (with-open [r (io/reader filepath)]
    (let [data (json/read r :key-fn keyword)]
      {:metadata (:metadata data)
       :vectors (mapv :embedding (:vectors data))})))

(defn generate-test-matrix
  "Generate a complete test matrix with common size/dimension combinations"
  [& {:keys [output-dir]
      :or {output-dir "test/data"}}]
  (let [test-cases
        [;; Quick tests
         [:smoke :small] ; 1K × 384
         [:smoke :medium] ; 1K × 768

         ;; Unit tests
         [:small :medium] ; 5K × 768
         [:small :standard] ; 5K × 1024

         ;; Integration tests
         [:medium :medium] ; 10K × 768
         [:medium :standard] ; 10K × 1024
         [:medium :large] ; 10K × 1536

         ;; Performance tests
         [:large :large] ; 20K × 1536

         ;; Bible dataset equivalent
         [:bible :medium] ; 30K × 768
         [:bible :standard] ; 30K × 1024

         ;; Stress tests
         [:stress :standard] ; 50K × 1024
         [:stress :max]] ; 50K × 3072

        results (atom [])]

    (doseq [[size-key dim-key] test-cases]
      (let [size (sizes size-key)
            dim (dimensions dim-key)
            filename (format "%s/test_vectors_%s_%s.json"
                             output-dir
                             (name size-key)
                             (name dim-key))
            _ (println (format "Generating %s (%d) × %s (%d)..."
                               (name size-key) size
                               (name dim-key) dim))
            vectors (generate-dataset size dim)
            metadata (save-dataset vectors filename)]
        (swap! results conj metadata)))

    @results))

(defn generate-query-set
  "Generate a set of query vectors for testing search operations"
  [num-queries dim & opts]
  (apply generate-dataset num-queries dim opts))

(defn save-query-set
  "Save query vectors to a separate file for consistent testing"
  [queries filepath]
  (save-dataset queries filepath :distribution :gaussian))

;; Utility functions for testing

(defn vector-distance
  "Calculate Euclidean distance between two vectors"
  [v1 v2]
  (Math/sqrt
   (reduce +
           (map (fn [a b]
                  (let [diff (- a b)]
                    (* diff diff)))
                v1 v2))))

(defn cosine-similarity
  "Calculate cosine similarity between two vectors"
  [v1 v2]
  (let [dot-product (reduce + (map * v1 v2))
        norm1 (Math/sqrt (reduce + (map #(* % %) v1)))
        norm2 (Math/sqrt (reduce + (map #(* % %) v2)))]
    (/ dot-product (* norm1 norm2))))

(defn add-noise
  "Add Gaussian noise to a vector (useful for creating near-duplicates)"
  [vector noise-level & {:keys [seed] :or {seed 42}}]
  (let [rng (Random. seed)]
    (vec (map #(+ % (* noise-level (.nextGaussian rng))) vector))))

(comment
  ;; Generate a single test dataset
  (def test-vectors (generate-dataset 1000 768))

  ;; Save to file
  (save-dataset test-vectors "test/data/test_1k_768.json")

  ;; Load from file
  (def loaded (load-dataset "test/data/test_1k_768.json"))

  ;; Generate complete test matrix
  (generate-test-matrix)

  ;; Generate clustered data for testing clustering behavior
  (def clustered (generate-dataset 5000 512
                                   :distribution :clustered
                                   :num-clusters 5))

  ;; Generate query set
  (def queries (generate-query-set 100 768))
  (save-query-set queries "test/data/queries_768.json"))