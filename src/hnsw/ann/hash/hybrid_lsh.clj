(ns hnsw.ann.hash.hybrid-lsh
  "Hybrid LSH + Lazy HNSW implementation for ultra-fast indexing"
  (:require [hnsw.ultra-fast :as ultra])
  (:import [java.util HashMap ArrayList Collections Random]
           [java.util.concurrent ConcurrentHashMap Callable]
           [java.util.concurrent.atomic AtomicReference]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; LSH hyperparameters
(def ^:const NUM-HASH-TABLES 8) ; Number of hash tables
(def ^:const NUM-HASH-BITS 12) ; Bits per hash (4096 buckets per table)
(def ^:const PROJECTION-DIM 64) ; Random projection dimension

(defrecord LSHBucket [^ArrayList ids ^ArrayList vectors])

(defrecord HybridIndex [hash-tables ; Vector of hash tables
                        random-matrices ; Random projection matrices
                        data-map ; id -> vector map
                        distance-fn
                        vector-norms]) ; Pre-computed norms for cosine

(defn generate-random-matrix
  "Generate random projection matrix for LSH"
  [^long input-dim ^long output-dim ^Random rng]
  (let [matrix (make-array Double/TYPE output-dim input-dim)]
    (dotimes [i output-dim]
      (dotimes [j input-dim]
        (aset-double matrix i j (.nextGaussian rng))))
    matrix))

(defn compute-hash-vector
  "Project vector and compute hash"
  [^doubles vector ^"[[D" projection-matrix]
  (let [output-dim (alength projection-matrix)
        hash-bits (long-array output-dim)]
    (dotimes [i output-dim]
      (let [^doubles row (aget projection-matrix i)
            dot (loop [j 0 sum 0.0]
                  (if (< j (alength vector))
                    (recur (inc j) (+ sum (* (aget vector j) (aget row j))))
                    sum))]
        (aset-long hash-bits i (if (>= dot 0.0) 1 0))))
    hash-bits))

(defn hash-to-bucket-id
  "Convert hash bits to bucket ID"
  [^longs hash-bits ^long num-bits]
  (loop [i 0 bucket-id 0]
    (if (< i (min num-bits (alength hash-bits)))
      (recur (inc i)
             (bit-or bucket-id
                     (bit-shift-left (aget hash-bits i) i)))
      bucket-id)))

(defn compute-vector-norm
  "Pre-compute vector norm for cosine distance"
  ^double [^doubles vector]
  (loop [i 0 sum 0.0]
    (if (< i (alength vector))
      (let [v (aget vector i)]
        (recur (inc i) (+ sum (* v v))))
      (Math/sqrt sum))))

(defn build-lsh-index
  "Build LSH index - extremely fast, just hashing"
  [data & {:keys [distance-fn show-progress? num-threads]
           :or {distance-fn ultra/cosine-distance-ultra
                show-progress? true
                num-threads 8}}]

  (let [start-time (System/currentTimeMillis)
        total-vectors (count data)
        data-vec (vec data) ; Convert to vector for indexed access
        first-vector (second (first data-vec))
        vector-dim (alength ^doubles first-vector)

        ;; Initialize random projections (smaller for speed)
        rng (Random. 42)
        random-matrices (vec (repeatedly NUM-HASH-TABLES
                                         #(generate-random-matrix vector-dim
                                                                  PROJECTION-DIM
                                                                  rng)))

        ;; Initialize hash tables with initial capacity
        hash-tables (vec (repeatedly NUM-HASH-TABLES
                                     #(ConcurrentHashMap. (bit-shift-left 1 NUM-HASH-BITS))))

        ;; Data storage with initial capacity
        data-map (ConcurrentHashMap. total-vectors)
        vector-norms (ConcurrentHashMap. total-vectors)]

    (when show-progress?
      (println "\n🚀 HYBRID LSH INDEX BUILD")
      (println (format "   Vectors: %d" total-vectors))
      (println (format "   Hash tables: %d" NUM-HASH-TABLES))
      (println (format "   Buckets per table: %d" (bit-shift-left 1 NUM-HASH-BITS)))) ; Reduced to 1024

    ;; Parallel processing with ForkJoinPool
    (let [pool (java.util.concurrent.ForkJoinPool. num-threads)]
      (.submit pool
               ^Runnable
               (fn []
                 (doseq [[id vector] data-vec]
                   (let [^doubles vec-array vector
                         norm (compute-vector-norm vec-array)]

                     ;; Store vector and norm
                     (.put data-map id vec-array)
                     (.put vector-norms id norm)

                     ;; Hash into each table (optimized)
                     (dotimes [table-idx NUM-HASH-TABLES]
                       (let [^"[[D" proj-matrix (nth random-matrices table-idx)
                             hash-bits (compute-hash-vector vec-array proj-matrix)
                             bucket-id (bit-and (hash-to-bucket-id hash-bits NUM-HASH-BITS)
                                                (dec (bit-shift-left 1 NUM-HASH-BITS))) ; Mask to NUM-HASH-BITS
                             ^ConcurrentHashMap table (nth hash-tables table-idx)]

                         ;; Use computeIfAbsent for thread-safe bucket creation
                         (let [^LSHBucket bucket (.computeIfAbsent table bucket-id
                                                                   (reify java.util.function.Function
                                                                     (apply [_ _]
                                                                       (->LSHBucket (ArrayList.) (ArrayList.)))))]
                           ;; Synchronized add to bucket
                           (locking bucket
                             (.add ^ArrayList (.ids bucket) id)
                             (.add ^ArrayList (.vectors bucket) vec-array)))))))))

      (.shutdown pool)
      (.awaitTermination pool 60 java.util.concurrent.TimeUnit/SECONDS))

    (let [build-time (- (System/currentTimeMillis) start-time)]
      (when show-progress?
        (println (format "\n✅ LSH Index built in %.3f seconds!"
                         (/ build-time 1000.0)))
        (println (format "   Rate: %.0f vectors/second"
                         (/ total-vectors (/ build-time 1000.0)))))

      (->HybridIndex hash-tables
                     random-matrices
                     data-map
                     distance-fn
                     vector-norms))))

(defn search-bucket-brute-force
  "Brute force search within a bucket"
  [^LSHBucket bucket ^doubles query-vec k distance-fn query-norm ^ConcurrentHashMap vector-norms]
  (let [^ArrayList ids (.ids bucket)
        ^ArrayList vectors (.vectors bucket)
        size (.size ids)]
    (if (<= size k)
      ;; Return all if bucket is small
      (map-indexed (fn [i ^doubles vec]
                     {:id (.get ids i)
                      :distance (if query-norm
                                ;; Optimized cosine with pre-computed norms
                                  (let [vec-norm (.get vector-norms (.get ids i))
                                        dot (loop [j 0 sum 0.0]
                                              (if (< j (alength vec))
                                                (recur (inc j)
                                                       (+ sum (* (aget vec j)
                                                                 (aget query-vec j))))
                                                sum))]
                                    (- 1.0 (/ dot (* query-norm vec-norm))))
                                ;; Regular distance
                                  (distance-fn query-vec vec))})
                   vectors)

      ;; Compute all distances and take top-k
      (let [distances (ArrayList. size)]
        (dotimes [i size]
          (let [^doubles vec (.get vectors i)
                dist (if query-norm
                       (let [vec-norm (.get vector-norms (.get ids i))
                             dot (loop [j 0 sum 0.0]
                                   (if (< j (alength vec))
                                     (recur (inc j)
                                            (+ sum (* (aget vec j)
                                                      (aget query-vec j))))
                                     sum))]
                         (- 1.0 (/ dot (* query-norm vec-norm))))
                       (distance-fn query-vec vec))]
            (.add distances {:id (.get ids i) :distance dist})))

        ;; Sort and take top-k
        (Collections/sort distances
                          (reify java.util.Comparator
                            (compare [_ a b]
                              (Double/compare (:distance a) (:distance b)))))

        (vec (take k distances))))))

(defn search-hybrid
  "Search using LSH buckets with parallel processing for better performance"
  [^HybridIndex index ^doubles query-vec k & {:keys [num-probes parallel?]
                                              :or {num-probes 2 parallel? true}}]
  (let [query-norm (compute-vector-norm query-vec)
        distance-fn (.distance-fn index)
        ^ConcurrentHashMap vector-norms (.vector-norms index)
        num-tables (count (.hash-tables index))
        actual-probes (int (Math/min (long num-probes) (long num-tables)))]

    ;; Collect candidates from all hash tables
    (let [candidates (if parallel?
                      ;; PARALLEL VERSION - use ForkJoinPool
                       (let [pool (java.util.concurrent.ForkJoinPool/commonPool)
                             results (java.util.concurrent.ConcurrentLinkedQueue.)
                             futures (vec
                                      (for [table-idx (range actual-probes)]
                                        (.submit pool
                                                 ^Callable
                                                 (fn []
                                                   (let [^"[[D" proj-matrix (nth (.random-matrices index) table-idx)
                                                         hash-bits (compute-hash-vector query-vec proj-matrix)
                                                         bucket-id (bit-and (hash-to-bucket-id hash-bits NUM-HASH-BITS)
                                                                            (dec (bit-shift-left 1 NUM-HASH-BITS)))
                                                         ^ConcurrentHashMap table (nth (.hash-tables index) table-idx)
                                                         ^LSHBucket bucket (.get table bucket-id)]
                                                     (when bucket
                                                       (search-bucket-brute-force
                                                        bucket query-vec (* k 3)
                                                        distance-fn query-norm vector-norms)))))))]
                        ;; Wait for all futures and collect results
                         (doseq [^java.util.concurrent.Future f futures]
                           (when-let [bucket-results (.get f)]
                             (.addAll results bucket-results)))
                         results)

                      ;; SEQUENTIAL VERSION - original
                       (let [results (ArrayList.)]
                         (doseq [table-idx (range actual-probes)]
                           (let [^"[[D" proj-matrix (nth (.random-matrices index) table-idx)
                                 hash-bits (compute-hash-vector query-vec proj-matrix)
                                 bucket-id (bit-and (hash-to-bucket-id hash-bits NUM-HASH-BITS)
                                                    (dec (bit-shift-left 1 NUM-HASH-BITS)))
                                 ^ConcurrentHashMap table (nth (.hash-tables index) table-idx)
                                 ^LSHBucket bucket (.get table bucket-id)]
                             (when bucket
                               (let [bucket-results (search-bucket-brute-force
                                                     bucket query-vec (* k 2)
                                                     distance-fn query-norm vector-norms)]
                                 (.addAll results bucket-results)))))
                         results))]

      ;; Deduplicate and sort
      (let [seen (java.util.HashSet.)
            unique (ArrayList.)]
        (doseq [candidate candidates]
          (when (.add seen (:id candidate))
            (.add unique candidate)))

        (Collections/sort unique
                          (reify java.util.Comparator
                            (compare [_ a b]
                              (Double/compare (:distance a) (:distance b)))))

        (vec (take k unique))))))

(defn search-hybrid-multiprobe
  "Multi-probe LSH - check neighboring buckets for MUCH better recall"
  [^HybridIndex index ^doubles query-vec k & {:keys [num-probes probe-radius parallel?]
                                              :or {num-probes 6 probe-radius 2 parallel? true}}]
  (let [query-norm (compute-vector-norm query-vec)
        distance-fn (.distance-fn index)
        ^ConcurrentHashMap vector-norms (.vector-norms index)
        num-tables (count (.hash-tables index))
        actual-probes (int (Math/min (long num-probes) (long num-tables)))]

    (let [candidates (if parallel?
                      ;; PARALLEL MULTI-PROBE
                       (let [pool (java.util.concurrent.ForkJoinPool/commonPool)
                             results (java.util.concurrent.ConcurrentLinkedQueue.)
                             futures (vec
                                      (for [table-idx (range actual-probes)]
                                        (.submit pool
                                                 ^Callable
                                                 (fn []
                                                   (let [^"[[D" proj-matrix (nth (.random-matrices index) table-idx)
                                                         hash-bits (compute-hash-vector query-vec proj-matrix)
                                                         base-bucket-id (hash-to-bucket-id hash-bits NUM-HASH-BITS)
                                                         ^ConcurrentHashMap table (nth (.hash-tables index) table-idx)
                                                         bucket-mask (dec (bit-shift-left 1 NUM-HASH-BITS))]

                                                    ;; Check main bucket
                                                     (when-let [^LSHBucket main-bucket (.get table (bit-and base-bucket-id bucket-mask))]
                                                       (.addAll results (search-bucket-brute-force
                                                                         main-bucket query-vec (* k 2)
                                                                         distance-fn query-norm vector-norms)))

                                                    ;; Check neighboring buckets by flipping hash bits
                                                     (dotimes [bit-pos (min probe-radius NUM-HASH-BITS)]
                                                       (let [flipped-id (bit-xor base-bucket-id (bit-shift-left 1 bit-pos))
                                                             neighbor-bucket-id (bit-and flipped-id bucket-mask)]
                                                         (when-let [^LSHBucket neighbor (.get table neighbor-bucket-id)]
                                                           (.addAll results (search-bucket-brute-force
                                                                             neighbor query-vec k
                                                                             distance-fn query-norm vector-norms))))))))))]
                        ;; Wait for all futures
                         (doseq [^java.util.concurrent.Future f futures]
                           (.get f))
                         results)

                      ;; SEQUENTIAL MULTI-PROBE
                       (let [results (ArrayList.)]
                         (doseq [table-idx (range actual-probes)]
                           (let [^"[[D" proj-matrix (nth (.random-matrices index) table-idx)
                                 hash-bits (compute-hash-vector query-vec proj-matrix)
                                 base-bucket-id (hash-to-bucket-id hash-bits NUM-HASH-BITS)
                                 ^ConcurrentHashMap table (nth (.hash-tables index) table-idx)
                                 bucket-mask (dec (bit-shift-left 1 NUM-HASH-BITS))]

                            ;; Check main bucket
                             (when-let [^LSHBucket main-bucket (.get table (bit-and base-bucket-id bucket-mask))]
                               (.addAll results (search-bucket-brute-force
                                                 main-bucket query-vec (* k 2)
                                                 distance-fn query-norm vector-norms)))

                            ;; Check neighboring buckets
                             (dotimes [bit-pos (min probe-radius NUM-HASH-BITS)]
                               (let [flipped-id (bit-xor base-bucket-id (bit-shift-left 1 bit-pos))
                                     neighbor-bucket-id (bit-and flipped-id bucket-mask)]
                                 (when-let [^LSHBucket neighbor (.get table neighbor-bucket-id)]
                                   (.addAll results (search-bucket-brute-force
                                                     neighbor query-vec k
                                                     distance-fn query-norm vector-norms)))))))
                         results))]

      ;; Deduplicate and return top-k
      (let [seen (java.util.HashSet.)
            unique (ArrayList.)]
        (doseq [candidate candidates]
          (when (.add seen (:id candidate))
            (.add unique candidate)))

        (Collections/sort unique
                          (reify java.util.Comparator
                            (compare [_ a b]
                              (Double/compare (:distance a) (:distance b)))))

        (vec (take k unique))))))

;; API-compatible wrapper functions
(defn build-index
  "API compatible with hnsw.ultra-fast"
  [data & opts]
  (apply build-lsh-index data opts))

(defn search-knn
  "API compatible with hnsw.ultra-fast - now with multi-probe LSH for better recall"
  ([index query-vec k]
   ;; Use multi-probe search with 6 tables and radius 2 for better recall
   (search-hybrid-multiprobe index query-vec k :num-probes 6 :probe-radius 2 :parallel? true))
  ([index query-vec k mode]
   ;; Support mode parameter for compatibility with different recall/speed tradeoffs
   (case mode
     :turbo (search-hybrid-multiprobe index query-vec k :num-probes 2 :probe-radius 1 :parallel? true)
     :fast (search-hybrid-multiprobe index query-vec k :num-probes 4 :probe-radius 1 :parallel? true)
     :balanced (search-hybrid-multiprobe index query-vec k :num-probes 6 :probe-radius 2 :parallel? true)
     :accurate (search-hybrid-multiprobe index query-vec k :num-probes 8 :probe-radius 3 :parallel? true)
     :precise (search-hybrid-multiprobe index query-vec k :num-probes 8 :probe-radius 4 :parallel? false)
     ;; Default - use balanced mode
     (search-hybrid-multiprobe index query-vec k :num-probes 6 :probe-radius 2 :parallel? true))))

(defn index-info
  [^HybridIndex index]
  (let [total-buckets (reduce + (map #(.size ^ConcurrentHashMap %)
                                     (.hash-tables index)))
        avg-bucket-size (if (> total-buckets 0)
                          (/ (.size ^ConcurrentHashMap (.data-map index))
                             total-buckets)
                          0)]
    {:type "Hybrid LSH Index"
     :vectors (.size ^ConcurrentHashMap (.data-map index))
     :hash-tables NUM-HASH-TABLES
     :buckets-per-table (bit-shift-left 1 NUM-HASH-BITS)
     :total-buckets total-buckets
     :avg-bucket-size avg-bucket-size}))
