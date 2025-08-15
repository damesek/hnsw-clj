(ns hnsw.simd
  "SIMD-optimized distance functions using Java Vector API (Java 16+)
   Provides significant speedup for vector operations on modern CPUs"
  (:import [jdk.incubator.vector FloatVector VectorSpecies VectorOperators]))

;; ===== Vector Species Configuration =====
(def ^VectorSpecies SPECIES FloatVector/SPECIES_PREFERRED)

;; Store as primitive long to avoid boxing
(def ^:const ^long SPECIES-LENGTH-LONG (long (.length SPECIES)))

(println (format "🚀 SIMD enabled: %d-bit vectors (%d floats per operation)"
                 (.vectorBitSize SPECIES)
                 SPECIES-LENGTH-LONG))

;; ===== Optimized SIMD Functions with proper type hints =====

(defn dot-product-simd-optimized
  "Highly optimized SIMD dot product with no boxing"
  ^double [^floats a ^floats b]
  (let [len (long (alength a))
        species-len SPECIES-LENGTH-LONG
        upper-bound (long (- len (rem len species-len)))]

    ;; Process main SIMD chunks with primitive loop counters
    (let [main-sum (loop [i (long 0)
                          sum (double 0.0)]
                     (if (< i upper-bound)
                       (let [va (FloatVector/fromArray SPECIES a (int i))
                             vb (FloatVector/fromArray SPECIES b (int i))
                             product (.mul va vb)]
                         (recur (+ i species-len)
                                (+ sum (double (.reduceLanes product VectorOperators/ADD)))))
                       sum))]

      ;; Process remaining elements with primitive types
      (loop [j (long upper-bound)
             final-sum (double main-sum)]
        (if (< j len)
          (recur (inc j)
                 (+ final-sum (* (double (aget a (int j)))
                                 (double (aget b (int j))))))
          final-sum)))))

(defn euclidean-distance-simd-optimized
  "Optimized SIMD Euclidean distance with no boxing"
  ^double [^floats a ^floats b]
  (let [len (long (alength a))
        species-len SPECIES-LENGTH-LONG
        upper-bound (long (- len (rem len species-len)))]

    (Math/sqrt
     (let [main-sum (loop [i (long 0)
                           sum (double 0.0)]
                      (if (< i upper-bound)
                        (let [va (FloatVector/fromArray SPECIES a (int i))
                              vb (FloatVector/fromArray SPECIES b (int i))
                              diff (.sub va vb)
                              squared (.mul diff diff)]
                          (recur (+ i species-len)
                                 (+ sum (double (.reduceLanes squared VectorOperators/ADD)))))
                        sum))]

       (loop [j (long upper-bound)
              final-sum (double main-sum)]
         (if (< j len)
           (let [diff (- (double (aget a (int j)))
                         (double (aget b (int j))))]
             (recur (inc j)
                    (+ final-sum (* diff diff))))
           final-sum))))))

(defn cosine-distance-simd-optimized
  "Optimized SIMD cosine distance with no boxing"
  ^double [^floats a ^floats b]
  (let [len (long (alength a))
        species-len SPECIES-LENGTH-LONG
        upper-bound (long (- len (rem len species-len)))]

    ;; Single pass for all three values with primitive types
    (let [[dot norm-a norm-b]
          (loop [i (long 0)
                 dot (double 0.0)
                 norm-a (double 0.0)
                 norm-b (double 0.0)]
            (if (< i upper-bound)
              (let [va (FloatVector/fromArray SPECIES a (int i))
                    vb (FloatVector/fromArray SPECIES b (int i))
                    prod (.mul va vb)
                    sq-a (.mul va va)
                    sq-b (.mul vb vb)]
                (recur (+ i species-len)
                       (+ dot (double (.reduceLanes prod VectorOperators/ADD)))
                       (+ norm-a (double (.reduceLanes sq-a VectorOperators/ADD)))
                       (+ norm-b (double (.reduceLanes sq-b VectorOperators/ADD)))))

              ;; Process remaining scalar elements
              (loop [j (long i)
                     d dot
                     na norm-a
                     nb norm-b]
                (if (< j len)
                  (let [aj (double (aget a (int j)))
                        bj (double (aget b (int j)))]
                    (recur (inc j)
                           (+ d (* aj bj))
                           (+ na (* aj aj))
                           (+ nb (* bj bj))))
                  [d na nb]))))]

      (let [magnitude (* (Math/sqrt (double norm-a))
                         (Math/sqrt (double norm-b)))]
        (if (zero? magnitude)
          1.0
          (- 1.0 (/ (double dot) magnitude)))))))

;; ===== Batch Operations for Better SIMD Utilization =====

(defn batch-cosine-distances-simd
  "Process multiple vectors at once for better SIMD utilization"
  [^floats query vectors]
  (let [n (count vectors)]
    (if (> n 8)
      (vec (pmap #(cosine-distance-simd-optimized query %) vectors))
      (mapv #(cosine-distance-simd-optimized query %) vectors))))

;; ===== Direct double array support (no conversion) =====

(defn cosine-distance-direct
  "Direct cosine distance for double arrays - no conversion, fully optimized"
  ^double [^doubles a ^doubles b]
  (let [len (long (alength a))]
    (loop [i (long 0)
           dot (double 0.0)
           norm-a (double 0.0)
           norm-b (double 0.0)]
      (if (< i len)
        (let [ai (aget a i)
              bi (aget b i)]
          (recur (inc i)
                 (+ dot (* ai bi))
                 (+ norm-a (* ai ai))
                 (+ norm-b (* bi bi))))
        (let [magnitude (* (Math/sqrt norm-a) (Math/sqrt norm-b))]
          (if (zero? magnitude)
            1.0
            (- 1.0 (/ dot magnitude))))))))

(defn euclidean-distance-direct
  "Direct Euclidean distance for double arrays - fully optimized"
  ^double [^doubles a ^doubles b]
  (let [len (long (alength a))]
    (Math/sqrt
     (loop [i (long 0)
            sum (double 0.0)]
       (if (< i len)
         (let [diff (- (aget a i) (aget b i))]
           (recur (inc i)
                  (+ sum (* diff diff))))
         sum)))))

;; ===== Export main functions =====

(def cosine-distance cosine-distance-simd-optimized)
(def euclidean-distance euclidean-distance-simd-optimized)
(def dot-product dot-product-simd-optimized)

;; Keep original function names for compatibility
(def cosine-distance-simd cosine-distance-simd-optimized)
(def euclidean-distance-simd euclidean-distance-simd-optimized)
(def dot-product-simd dot-product-simd-optimized)

;; Double array wrappers
(defn cosine-distance-simd-doubles
  "Wrapper for double arrays - uses direct optimized implementation"
  ^double [^doubles a ^doubles b]
  (cosine-distance-direct a b))

(defn euclidean-distance-simd-doubles
  "Wrapper for double arrays - uses direct optimized implementation"
  ^double [^doubles a ^doubles b]
  (euclidean-distance-direct a b))
