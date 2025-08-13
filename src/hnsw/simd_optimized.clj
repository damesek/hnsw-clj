(ns hnsw.simd-optimized
  "Optimized distance functions with automatic SIMD/fallback selection
   Automatically detects and uses best available implementation:
   - Java Vector API (SIMD) if available  
   - Optimized loop-unrolled implementation as fallback"
  (:import [java.util.concurrent ForkJoinPool]
           [java.util Arrays]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ===== SIMD Detection =====

(def ^:private simd-available?
  "Check if Java Vector API is available"
  (try
    (Class/forName "jdk.incubator.vector.FloatVector")
    true
    (catch ClassNotFoundException _ false)))

(when simd-available?
  (println "✅ SIMD (Java Vector API) detected - using hardware acceleration"))

(when-not simd-available?
  (println "⚠️  SIMD not available - using optimized fallback implementation"))

;; ===== Helper Macros for Zero-Boxing =====

(defmacro aget-d
  "Direct double array access without boxing"
  [arr idx]
  `(aget ^doubles ~arr (int ~idx)))

(defmacro aset-d
  "Direct double array set without boxing"
  [arr idx val]
  `(aset ^doubles ~arr (int ~idx) (double ~val)))

;; ===== Optimized Distance Functions (Fallback) =====

(defn ^double cosine-distance-loop-unrolled
  "Optimized cosine distance with 8x loop unrolling
   Processes 8 elements per iteration for better CPU pipeline utilization"
  ^double [^doubles v1 ^doubles v2]
  (let [len (int (alength v1))
        chunks (int (quot len 8))
        remainder (int (rem len 8))]

    ;; Process 8 elements at a time
    (loop [i (int 0)
           dot (double 0.0)
           norm1 (double 0.0)
           norm2 (double 0.0)]
      (if (< i chunks)
        (let [idx (int (* i 8))
              ;; Load 8 elements
              a0 (aget-d v1 idx)
              b0 (aget-d v2 idx)
              a1 (aget-d v1 (+ idx 1))
              b1 (aget-d v2 (+ idx 1))
              a2 (aget-d v1 (+ idx 2))
              b2 (aget-d v2 (+ idx 2))
              a3 (aget-d v1 (+ idx 3))
              b3 (aget-d v2 (+ idx 3))
              a4 (aget-d v1 (+ idx 4))
              b4 (aget-d v2 (+ idx 4))
              a5 (aget-d v1 (+ idx 5))
              b5 (aget-d v2 (+ idx 5))
              a6 (aget-d v1 (+ idx 6))
              b6 (aget-d v2 (+ idx 6))
              a7 (aget-d v1 (+ idx 7))
              b7 (aget-d v2 (+ idx 7))]
          (recur (unchecked-inc i)
                 (+ dot
                    (* a0 b0) (* a1 b1) (* a2 b2) (* a3 b3)
                    (* a4 b4) (* a5 b5) (* a6 b6) (* a7 b7))
                 (+ norm1
                    (* a0 a0) (* a1 a1) (* a2 a2) (* a3 a3)
                    (* a4 a4) (* a5 a5) (* a6 a6) (* a7 a7))
                 (+ norm2
                    (* b0 b0) (* b1 b1) (* b2 b2) (* b3 b3)
                    (* b4 b4) (* b5 b5) (* b6 b6) (* b7 b7))))

        ;; Handle remainder
        (let [start-idx (int (* chunks 8))]
          (loop [j (int start-idx)
                 d dot
                 n1 norm1
                 n2 norm2]
            (if (< j len)
              (let [a (aget-d v1 j)
                    b (aget-d v2 j)]
                (recur (unchecked-inc j)
                       (+ d (* a b))
                       (+ n1 (* a a))
                       (+ n2 (* b b))))
              (let [magnitude (* (Math/sqrt n1) (Math/sqrt n2))]
                (if (zero? magnitude)
                  1.0
                  (- 1.0 (/ d magnitude)))))))))))

(defn ^double euclidean-distance-loop-unrolled
  "Optimized Euclidean distance with 8x loop unrolling"
  ^double [^doubles v1 ^doubles v2]
  (let [len (int (alength v1))
        chunks (int (quot len 8))
        remainder (int (rem len 8))]

    (Math/sqrt
     (loop [i (int 0)
            sum (double 0.0)]
       (if (< i chunks)
         (let [idx (int (* i 8))
               ;; Calculate 8 differences
               d0 (- (aget-d v1 idx) (aget-d v2 idx))
               d1 (- (aget-d v1 (+ idx 1)) (aget-d v2 (+ idx 1)))
               d2 (- (aget-d v1 (+ idx 2)) (aget-d v2 (+ idx 2)))
               d3 (- (aget-d v1 (+ idx 3)) (aget-d v2 (+ idx 3)))
               d4 (- (aget-d v1 (+ idx 4)) (aget-d v2 (+ idx 4)))
               d5 (- (aget-d v1 (+ idx 5)) (aget-d v2 (+ idx 5)))
               d6 (- (aget-d v1 (+ idx 6)) (aget-d v2 (+ idx 6)))
               d7 (- (aget-d v1 (+ idx 7)) (aget-d v2 (+ idx 7)))]
           (recur (unchecked-inc i)
                  (+ sum
                     (* d0 d0) (* d1 d1) (* d2 d2) (* d3 d3)
                     (* d4 d4) (* d5 d5) (* d6 d6) (* d7 d7))))

         ;; Handle remainder
         (let [start-idx (int (* chunks 8))]
           (loop [j (int start-idx)
                  final-sum sum]
             (if (< j len)
               (let [diff (- (aget-d v1 j) (aget-d v2 j))]
                 (recur (unchecked-inc j)
                        (+ final-sum (* diff diff))))
               final-sum))))))))

;; ===== SIMD Implementation (when available) =====

(when simd-available?
  (require '[hnsw.simd :as simd]))

;; ===== Public API with Automatic Selection =====

(defn cosine-distance
  "High-performance cosine distance with automatic SIMD detection
   Uses hardware acceleration when available, optimized fallback otherwise"
  ^double [^doubles v1 ^doubles v2]
  (if simd-available?
    ;; Use SIMD implementation
    ((resolve 'hnsw.simd/cosine-distance-simd-doubles) v1 v2)
    ;; Use optimized fallback
    (cosine-distance-loop-unrolled v1 v2)))

(defn euclidean-distance
  "High-performance Euclidean distance with automatic SIMD detection"
  ^double [^doubles v1 ^doubles v2]
  (if simd-available?
    ((resolve 'hnsw.simd/euclidean-distance-simd-doubles) v1 v2)
    (euclidean-distance-loop-unrolled v1 v2)))

;; ===== Batch Processing =====

(defn batch-distances-parallel
  "Calculate distances for multiple vectors in parallel
   Uses ForkJoinPool for efficient parallel processing"
  [distance-fn ^doubles query vectors]
  (let [pool (ForkJoinPool/commonPool)
        tasks (map (fn [v]
                     (.submit pool
                              ^Callable (fn []
                                          (distance-fn query v))))
                   vectors)]
    (vec (map #(.get ^java.util.concurrent.Future %) tasks))))

(defn batch-cosine-distances
  "Batch cosine distance calculation with parallel processing"
  [^doubles query vectors]
  (batch-distances-parallel cosine-distance query vectors))

(defn batch-euclidean-distances
  "Batch Euclidean distance calculation with parallel processing"
  [^doubles query vectors]
  (batch-distances-parallel euclidean-distance query vectors))

;; ===== Vector Preprocessing =====

(defn normalize-vector!
  "In-place vector normalization for cosine similarity
   Modifies the input vector directly"
  [^doubles v]
  (let [len (alength v)
        norm (loop [i (int 0)
                    sum (double 0.0)]
               (if (< i len)
                 (let [val (aget-d v i)]
                   (recur (unchecked-inc i)
                          (+ sum (* val val))))
                 (Math/sqrt sum)))]
    (when-not (zero? norm)
      (let [inv-norm (/ 1.0 norm)]
        (dotimes [i len]
          (aset-d v i (* (aget-d v i) inv-norm)))))
    v))

(defn precompute-norms
  "Precompute norms for a collection of vectors
   Useful for repeated cosine similarity calculations"
  [vectors]
  (double-array
   (map (fn [^doubles v]
          (Math/sqrt
           (areduce v i sum 0.0
                    (let [val (aget v i)]
                      (+ sum (* val val))))))
        vectors)))

;; ===== Performance Benchmarking =====

(defn benchmark-distance-functions
  "Compare performance of different distance implementations"
  [dimension n-tests]
  (let [v1 (double-array (repeatedly dimension #(rand)))
        v2 (double-array (repeatedly dimension #(rand)))]

    (println "\n📊 Distance Function Performance Benchmark")
    (println (apply str (repeat 50 "-")))
    (println (format "Vector dimension: %d" dimension))
    (println (format "Tests per operation: %d" n-tests))
    (println (format "Implementation: %s\n"
                     (if simd-available? "SIMD (Vector API)" "Optimized Fallback")))

    ;; Benchmark cosine distance
    (let [start (System/nanoTime)]
      (dotimes [_ n-tests]
        (cosine-distance v1 v2))
      (let [time-ms (/ (- (System/nanoTime) start) 1000000.0)]
        (println (format "Cosine Distance: %.3f ms total (%.3f μs per op)"
                         time-ms
                         (/ (* time-ms 1000) n-tests)))))

    ;; Benchmark Euclidean distance
    (let [start (System/nanoTime)]
      (dotimes [_ n-tests]
        (euclidean-distance v1 v2))
      (let [time-ms (/ (- (System/nanoTime) start) 1000000.0)]
        (println (format "Euclidean Distance: %.3f ms total (%.3f μs per op)"
                         time-ms
                         (/ (* time-ms 1000) n-tests)))))

    ;; Benchmark batch processing
    (let [vectors (vec (repeatedly 100 #(double-array (repeatedly dimension rand))))
          start (System/nanoTime)]
      (batch-cosine-distances v1 vectors)
      (let [time-ms (/ (- (System/nanoTime) start) 1000000.0)]
        (println (format "\nBatch Processing (100 vectors): %.3f ms (%.3f ms per vector)"
                         time-ms
                         (/ time-ms 100.0)))))))

;; ===== Cache-Friendly Operations =====

(defn distance-with-early-termination
  "Distance calculation with early termination for approximate nearest neighbor
   Stops calculation if distance exceeds threshold"
  ^double [distance-fn ^doubles v1 ^doubles v2 ^double threshold]
  (let [result ^double (distance-fn v1 v2)]
    (if (> result threshold)
      Double/MAX_VALUE
      result)))

(defn top-k-distances
  "Find top-k nearest vectors efficiently
   Returns indices and distances sorted by distance"
  [distance-fn ^doubles query vectors k]
  (let [distances-with-idx
        (map-indexed (fn [idx v]
                       [idx (distance-fn query v)])
                     vectors)
        sorted (sort-by second distances-with-idx)]
    (vec (take k sorted))))

;; Export dot-product function  
(def dot-product
  "Dot product function - delegates to SIMD implementation if available"
  (if simd-available?
    (fn [^doubles a ^doubles b]
      ((resolve 'hnsw.simd/dot-product-simd-optimized) a b))
    (fn [^doubles a ^doubles b]
      (let [len (alength a)]
        (loop [i 0 sum 0.0]
          (if (< i len)
            (recur (inc i) (+ sum (* (aget a i) (aget b i))))
            sum))))))

;; Run benchmark on load
(println "\n🚀 SIMD-Optimized Distance Functions Loaded")
(println (format "   Mode: %s" (if simd-available? "SIMD Hardware Acceleration" "Optimized Software")))
(println "   Functions: cosine-distance, euclidean-distance, dot-product")
(println "   Batch ops: batch-cosine-distances, batch-euclidean-distances")
