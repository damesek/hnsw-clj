(ns hnsw.helper.index-io
  "HNSW Index Serialization/Deserialization utilities using EDN format"
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [hnsw.ultra-fast :as ultra])
  (:import [java.util Random]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic AtomicInteger AtomicReference]))

(defn index-data
  [index]
  {:index-type :ultra
   :nodes (into {}
                (map (fn [[key node]]
                       [key {:id (.id node)
                             :vector (vec (.vector node))
                             :level (.level node)
                             :ordinal (.ordinal node)
                             :neighbors (mapv #(vec %) (.neighbors node))}])
                     (.nodes index)))
   :entry-point (.get (.entry-point index))
   :M (.M index)
   :max-M (.max-M index)
   :ef-construction (.ef-construction index)
   :ef-search (.ef-search index)
   :ml (.ml index)
   :normalize-vectors (.normalize-vectors index)
   :compact-precision (.compact-precision index)
   :construction-precision (.construction-precision index)
   :metric (cond
             (or (.normalize-vectors index)
                 (identical? (.distance-fn index) ultra/cosine-distance-ultra)) :cosine
             (identical? (.distance-fn index) ultra/euclidean-distance-ultra) :euclidean
             :else :custom)
   :element-count (.get (.element-count index))})

(defn from-data
  [index-data distance-fn]
  (when (or (:construction-codec index-data) (:construction-codebook index-data))
    (throw (IllegalArgumentException. "This index uses a removed construction codec; rebuild from source vectors")))
  (let [metric (case (:metric index-data)
                 :cosine ultra/cosine-distance-ultra
                 :euclidean ultra/euclidean-distance-ultra
                 :custom (or distance-fn
                             (throw (IllegalArgumentException.
                                     "Loading a custom metric requires its distance function")))
                 nil (or distance-fn ultra/cosine-distance-ultra)
                 (throw (IllegalArgumentException. "Unknown saved distance metric")))
        normalize-vectors (boolean (:normalize-vectors index-data))
        precision (or (:construction-precision index-data) :float64)
        _ (when-not (#{:float32 :float64} precision)
            (throw (IllegalArgumentException. "Invalid saved construction-precision")))
        _ (when (and (= precision :float32) (not normalize-vectors))
            (throw (IllegalArgumentException. "float32 construction requires normalized cosine vectors")))
        nodes (ConcurrentHashMap.)
        next-ordinal (AtomicInteger. 0)]
    (doseq [[key node-data] (:nodes index-data)]
      (let [neighbors (into-array
                       Object
                       (map (fn [items]
                              (doto (ConcurrentHashMap/newKeySet)
                                (.addAll items)))
                            (:neighbors node-data)))
            ordinal (long (or (:ordinal node-data)
                              (.getAndIncrement next-ordinal)))
            vector (double-array (:vector node-data))
            node (ultra/->UltraNode
                  (:id node-data)
                  vector
                  (:level node-data)
                  ordinal
                  neighbors
                  (ultra/construction-vector vector precision))]
        (when (>= ordinal (.get next-ordinal))
          (.set next-ordinal (int (inc ordinal))))
        (.put nodes key node)))
    (let [index (ultra/->UltraGraph
                 nodes
                 (AtomicReference. (:entry-point index-data))
                 (:M index-data)
                 (:max-M index-data)
                 (:ef-construction index-data)
                 (or (:ef-search index-data) 50)
                 (:ml index-data)
                 (if normalize-vectors ultra/cosine-distance-normalized metric)
                 (when normalize-vectors (ultra/create-normalized-distance-engine))
                 (AtomicReference. nil)
                 (AtomicInteger. (.size nodes))
                 next-ordinal
                 (Random. 42)
                 normalize-vectors
                 (or (:compact-precision index-data) :float64)
                 precision
                 (when (= precision :float32) (ultra/create-float-distance-engine))
                 (AtomicReference. nil))]
      (ultra/optimize-search! index))))

(defn write-data!
  "Atomically replace a local EDN file."
  [filepath data]
  (let [destination (.toAbsolutePath (.toPath (io/file filepath)))
        temporary (java.nio.file.Files/createTempFile
                    (.getParent destination) ".hnsw-" ".tmp"
                    (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (binding [*print-length* nil *print-level* nil]
        (spit (.toFile temporary) (pr-str data)))
      (java.nio.file.Files/move
        temporary destination
        (into-array java.nio.file.CopyOption
                    [java.nio.file.StandardCopyOption/ATOMIC_MOVE
                     java.nio.file.StandardCopyOption/REPLACE_EXISTING]))
      (finally (java.nio.file.Files/deleteIfExists temporary)))))

(defn save-index
  "Save a raw HNSW graph. Prefer hnsw.api for application use."
  [index filepath]
  (write-data! filepath (index-data index))
  index)

(defn load-index
  "Read a raw HNSW graph. Legacy custom metrics require their distance function."
  ([filepath] (load-index filepath nil))
  ([filepath distance-fn]
   (let [data (edn/read-string (slurp filepath))]
     (when-not (= :ultra (:index-type data))
       (throw (IllegalArgumentException. "Not an HNSW index")))
     (from-data data distance-fn))))
