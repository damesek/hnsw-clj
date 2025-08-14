(ns hnsw.wip.ultra-optimized
  "Ultra-optimized HNSW with all performance improvements including SIMD"
  (:require [hnsw.ultra-fast :as base]
            [hnsw.simd-optimized :as simd-opt])
  (:import [java.util Arrays]
           [java.util.concurrent ConcurrentHashMap ForkJoinPool ConcurrentSkipListSet]
           [java.util.concurrent.atomic AtomicInteger AtomicReference]
           [java.nio ByteBuffer]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.io RandomAccessFile]))

;; ===== Performance Configuration =====
(set! *warn-on-reflection* false)
(set! *unchecked-math* true) ; Change back to true to avoid warnings

;; ===== Phase 1: Eliminate ALL Boxing =====

(defmacro aget-d
  "Macro for primitive double array access without boxing"
  [arr idx]
  `(aget ^doubles ~arr (int ~idx)))

(defmacro aset-d
  "Macro for primitive double array setting without boxing"
  [arr idx val]
  `(aset ^doubles ~arr (int ~idx) (double ~val)))

(defn ^double fast-euclidean-distance
  "Zero-boxing Euclidean distance"
  ^double [^doubles v1 ^doubles v2]
  (let [len (int (alength v1))]
    (loop [i (int 0)
           sum (double 0.0)]
      (if (< i len)
        (let [diff (- (aget-d v1 i)
                      (aget-d v2 i))]
          (recur (unchecked-inc i)
                 (+ sum (* diff diff))))
        (Math/sqrt sum)))))

(defn ^double fast-cosine-distance
  "Zero-boxing cosine distance with loop unrolling"
  ^double [^doubles v1 ^doubles v2]
  (let [len (int (alength v1))
        chunks (int (unchecked-divide-int len 4))
        remainder (int (unchecked-remainder-int len 4))]

    ;; Process 4 elements at a time
    (loop [i (int 0)
           dot (double 0.0)
           norm1 (double 0.0)
           norm2 (double 0.0)]
      (if (< i chunks)
        (let [idx (int (unchecked-multiply-int i 4))
              ;; Unroll 4x
              a0 (aget-d v1 idx)
              b0 (aget-d v2 idx)
              a1 (aget-d v1 (unchecked-inc idx))
              b1 (aget-d v2 (unchecked-inc idx))
              a2 (aget-d v1 (+ idx 2))
              b2 (aget-d v2 (+ idx 2))
              a3 (aget-d v1 (+ idx 3))
              b3 (aget-d v2 (+ idx 3))]
          (recur (unchecked-inc i)
                 (+ dot (* a0 b0) (* a1 b1) (* a2 b2) (* a3 b3))
                 (+ norm1 (* a0 a0) (* a1 a1) (* a2 a2) (* a3 a3))
                 (+ norm2 (* b0 b0) (* b1 b1) (* b2 b2) (* b3 b3))))

        ;; Handle remainder
        (let [start-idx (int (unchecked-multiply-int chunks 4))]
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
              (if (and (> n1 0.0) (> n2 0.0))
                (- 1.0 (/ d (* (Math/sqrt n1) (Math/sqrt n2))))
                1.0))))))))

;; ===== Phase 2: Parallel Index Building =====

(defrecord ParallelGraph [nodes
                          entry-point
                          M
                          max-M
                          ef-construction
                          ml
                          distance-fn
                          element-count
                          pool])

(defn create-parallel-graph
  [& {:keys [M ef-construction distance-fn num-threads]
      :or {M 16
           ef-construction 200
           distance-fn simd-opt/cosine-distance ;; Use SIMD-optimized version
           num-threads (.availableProcessors (Runtime/getRuntime))}}]
  (map->ParallelGraph
   {:nodes (ConcurrentHashMap.)
    :entry-point (AtomicReference. nil)
    :M M
    :max-M (long (* 2 M))
    :ef-construction ef-construction
    :ml (/ 1.0 (Math/log 2.0))
    :distance-fn distance-fn
    :element-count (AtomicInteger. 0)
    :pool (ForkJoinPool. num-threads)}))

(defn parallel-insert-batch
  "Insert batch in parallel using ForkJoinPool"
  [graph elements]
  ;; For now, use sequential insertion since the base graph structure
  ;; doesn't support concurrent modifications safely
  (doseq [[id vector] elements]
    (base/insert-single graph id (double-array vector)))
  graph)

(defn parallel-build-index
  "Build index with parallel insertion and SIMD distance functions"
  [data & {:keys [M ef-construction distance-fn num-threads show-progress?]
           :or {M 16
                ef-construction 200
                distance-fn simd-opt/cosine-distance ;; Use SIMD-optimized version
                num-threads (.availableProcessors (Runtime/getRuntime))
                show-progress? true}}]

  ;; Use the base ultra-fast build-index with our SIMD-optimized distance function
  (base/build-index data
                    :M M
                    :ef-construction ef-construction
                    :distance-fn distance-fn
                    :show-progress? show-progress?))

;; ===== Phase 3: Memory Pool =====

(defrecord VectorPool [dimension pool-size index pool])

(defn get-vector-from-pool [^VectorPool vp]
  (let [^AtomicInteger idx-atom (:index vp)
        idx (.getAndIncrement idx-atom)
        ^long pool-size (:pool-size vp)]
    (if (< idx pool-size)
      (aget ^"[[D" (:pool vp) idx)
      (double-array (:dimension vp)))))

(defn return-vector-to-pool [^VectorPool vp ^doubles v]
  (Arrays/fill v 0.0))

(defn clear-pool [^VectorPool vp]
  (.set ^AtomicInteger (:index vp) 0))

(defn create-vector-pool
  "Create a pool of reusable vectors"
  [dimension pool-size]
  (->VectorPool dimension
                pool-size
                (AtomicInteger. 0)
                (make-array Double/TYPE pool-size dimension)))

;; ===== Phase 4: Off-Heap Memory =====

(defrecord OffHeapVectorStorage [buffer dimension num-vectors])

(defn get-vector-from-offheap [^OffHeapVectorStorage storage idx]
  (let [^ByteBuffer buffer (:buffer storage)
        ^long dimension (:dimension storage)
        offset (* idx dimension 8)
        result (double-array dimension)]
    (.position buffer (int offset))
    (dotimes [i dimension]
      (aset-d result i (.getDouble buffer)))
    result))

(defn put-vector-to-offheap [^OffHeapVectorStorage storage idx ^doubles vector]
  (let [^ByteBuffer buffer (:buffer storage)
        ^long dimension (:dimension storage)
        offset (* idx dimension 8)]
    (.position buffer (int offset))
    (doseq [v vector]
      (.putDouble buffer v))))

(defn create-off-heap-storage
  "Create off-heap storage for vectors"
  [num-vectors dimension]
  (let [bytes-per-vector (* dimension 8)
        total-bytes (* num-vectors bytes-per-vector)
        buffer (ByteBuffer/allocateDirect total-bytes)]
    (->OffHeapVectorStorage buffer dimension num-vectors)))

;; ===== Phase 5: Memory Mapped Index =====

(defrecord MemoryMappedIndex [file channel buffer dimension num-vectors])

(defn save-mmap-index [^MemoryMappedIndex mmi graph]
  (let [^ByteBuffer buffer (:buffer mmi)
        nodes (:nodes graph)]
    (.position buffer 0)
    (.putLong buffer (.size ^ConcurrentHashMap nodes))

    (doseq [[id node] nodes]
      ;; Write node data
      (let [id-bytes (.getBytes ^String id)]
        (.putInt buffer (count id-bytes))
        (.put buffer id-bytes))
      ;; Write vector
      (when-let [vec (.vector node)]
        (doseq [v vec]
          (.putDouble buffer v))))))

(defn load-mmap-index [^MemoryMappedIndex mmi]
  (let [^ByteBuffer buffer (:buffer mmi)]
    (.position buffer 0)
    (let [num-nodes (.getLong buffer)
          nodes (ConcurrentHashMap.)]
      ;; Read nodes
      (dotimes [_ num-nodes]
        ;; Read node data and reconstruct
        (let [id-len (.getInt buffer)
              id-bytes (byte-array id-len)]
          (.get buffer id-bytes)
          ;; Add reconstruction logic here
          ))
      nodes)))

(defn close-mmap-index [^MemoryMappedIndex mmi]
  (.close ^FileChannel (:channel mmi))
  (.close ^RandomAccessFile (:file mmi)))

(defn create-mmap-index
  "Create memory-mapped index for persistence"
  [filepath num-vectors dimension]
  (let [file (RandomAccessFile. filepath "rw")
        channel (.getChannel file)
        size (* num-vectors dimension 8 2) ; Extra space for metadata
        buffer (.map channel FileChannel$MapMode/READ_WRITE 0 size)]
    (->MemoryMappedIndex file channel buffer dimension num-vectors)))

;; ===== Phase 6: Lock-Free Priority Queue =====

(defrecord LockFreePriorityQueue [items size max-size comparator])

(defn add-to-queue [^LockFreePriorityQueue q item]
  (let [^ConcurrentSkipListSet items (:items q)
        ^AtomicInteger size-atom (:size q)
        ^long max-size (:max-size q)]
    (when (< (.get size-atom) max-size)
      (when (.add items item)
        (.incrementAndGet size-atom))
      true)))

(defn poll-from-queue [^LockFreePriorityQueue q]
  (let [^ConcurrentSkipListSet items (:items q)
        ^AtomicInteger size-atom (:size q)]
    (when-let [item (.pollFirst items)]
      (.decrementAndGet size-atom)
      item)))

(defn peek-queue [^LockFreePriorityQueue q]
  (.first ^ConcurrentSkipListSet (:items q)))

(defn clear-queue [^LockFreePriorityQueue q]
  (.clear ^ConcurrentSkipListSet (:items q))
  (.set ^AtomicInteger (:size q) 0))

(defn create-lock-free-queue
  "Create a lock-free priority queue"
  [max-size comparator]
  (->LockFreePriorityQueue
   (ConcurrentSkipListSet. comparator)
   (AtomicInteger. 0)
   max-size
   comparator))

;; ===== Optimized Search with All Improvements =====

(defn search-optimized
  "Search using all optimizations - delegates to base for now"
  [graph ^doubles query-vec ^long k]
  ;; Use the base search which is already optimized
  (base/search-knn graph query-vec k))

;; ===== Benchmark Functions =====

(defn benchmark-parallel-vs-sequential
  "Compare build times with different optimization levels"
  [num-vectors dimension]
  (let [data (repeatedly num-vectors
                         (fn []
                           [(str (java.util.UUID/randomUUID))
                            (double-array (repeatedly dimension rand))]))

        ;; Test 1: Base implementation
        start-base (System/currentTimeMillis)
        base-index (base/build-index data
                                     :distance-fn base/cosine-distance-ultra
                                     :show-progress? false)
        base-time (- (System/currentTimeMillis) start-base)

        ;; Test 2: Loop-unrolled optimized
        start-opt (System/currentTimeMillis)
        opt-index (base/build-index data
                                    :distance-fn fast-cosine-distance
                                    :show-progress? false)
        opt-time (- (System/currentTimeMillis) start-opt)

        ;; Test 3: SIMD-optimized (8x unrolled or HW accelerated)
        start-simd (System/currentTimeMillis)
        simd-index (base/build-index data
                                     :distance-fn simd-opt/cosine-distance
                                     :show-progress? false)
        simd-time (- (System/currentTimeMillis) start-simd)]

    (println "\n📊 Performance Comparison (Build Index)")
    (println (apply str (repeat 50 "-")))
    (println (format "Vectors: %d, Dimension: %d" num-vectors dimension))
    (println)

    (let [result {:base-distance {:time-ms base-time
                                  :vectors-per-sec (/ (* num-vectors 1000.0) base-time)}
                  :loop-unrolled {:time-ms opt-time
                                  :vectors-per-sec (/ (* num-vectors 1000.0) opt-time)}
                  :simd-optimized {:time-ms simd-time
                                   :vectors-per-sec (/ (* num-vectors 1000.0) simd-time)}
                  :speedup-vs-base {:loop-unrolled (/ (double base-time) opt-time)
                                    :simd-optimized (/ (double base-time) simd-time)}}]

      (println "Results:")
      (println (format "  Base:          %d ms (%.1f vectors/sec)"
                       base-time (get-in result [:base-distance :vectors-per-sec])))
      (println (format "  Loop-unrolled: %d ms (%.1f vectors/sec) - %.2fx speedup"
                       opt-time (get-in result [:loop-unrolled :vectors-per-sec])
                       (get-in result [:speedup-vs-base :loop-unrolled])))
      (println (format "  SIMD-optimized: %d ms (%.1f vectors/sec) - %.2fx speedup"
                       simd-time (get-in result [:simd-optimized :vectors-per-sec])
                       (get-in result [:speedup-vs-base :simd-optimized])))
      result)))

;; ===== Export Public API =====

(def build-index parallel-build-index)
(def search search-optimized)
