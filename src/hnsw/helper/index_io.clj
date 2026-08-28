(ns hnsw.helper.index-io
  "HNSW Index Serialization/Deserialization utilities using EDN format"
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [hnsw.ann.partition.lightning :as lightning]
            [hnsw.ultra-fast :as ultra])
  (:import [hnsw.ann.partition.lightning LightningIndex]
           [java.util Random]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic AtomicInteger AtomicReference]))

(defn- ultra-index-data
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
   :element-count (.get (.element-count index))})

(defn- lightning-index-data
  [^LightningIndex index]
  {:index-type :lightning
   :partitions (mapv (fn [partition]
                       (mapv (fn [[id vector]] [id (vec vector)]) partition))
                     (.partitions index))
   :centroids (mapv vec (.centroids index))
   :vector-norms (into {} (.vector-norms index))
   :element-count (.size ^ConcurrentHashMap (.data-map index))})

(defn save-index
  "Save an UltraGraph or LightningIndex to an EDN file."
  [index filepath]
  (println (format "💾 Saving index to %s..." filepath))
  (let [start (System/currentTimeMillis)
        index-data (if (instance? LightningIndex index)
                     (lightning-index-data index)
                     (ultra-index-data index))]
    (binding [*print-length* nil
              *print-level* nil]
      (spit filepath (pr-str index-data)))
    (let [elapsed (- (System/currentTimeMillis) start)
          file-size (double (/ (.length (io/file filepath)) (* 1024 1024)))]
      (println (format "✅ Index saved in %.2f seconds (%.1f MB)"
                       (double (/ elapsed 1000.0)) file-size))))
  index)

(defn- load-lightning-index
  [index-data]
  (let [partitions (mapv (fn [partition]
                           (mapv (fn [[id vector]]
                                   [id (double-array vector)])
                                 partition))
                         (:partitions index-data))
        centroids (mapv double-array (:centroids index-data))
        data-map (ConcurrentHashMap.)
        vector-norms (ConcurrentHashMap.)]
    (doseq [partition partitions
            [id vector] partition]
      (.put data-map id vector))
    (doseq [[id norm] (:vector-norms index-data)]
      (.put vector-norms id norm))
    (lightning/->LightningIndex partitions
                                centroids
                                data-map
                                ultra/cosine-distance-ultra
                                vector-norms)))

(defn- load-ultra-index
  [index-data distance-fn]
  (let [nodes (ConcurrentHashMap.)
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
            node (ultra/->UltraNode
                  (:id node-data)
                  (double-array (:vector node-data))
                  (:level node-data)
                  ordinal
                  neighbors)]
        (when (>= ordinal (.get next-ordinal))
          (.set next-ordinal (int (inc ordinal))))
        (.put nodes key node)))
    (let [normalize-vectors (boolean (:normalize-vectors index-data))
          internal-distance-fn (if normalize-vectors
                                 ultra/cosine-distance-normalized
                                 distance-fn)
          normalized-distance (when normalize-vectors
                                (ultra/create-normalized-distance-engine))
          index (ultra/->UltraGraph
                 nodes
                 (AtomicReference. (:entry-point index-data))
                 (:M index-data)
                 (:max-M index-data)
                 (:ef-construction index-data)
                 (or (:ef-search index-data) 50)
                 (:ml index-data)
                 internal-distance-fn
                 normalized-distance
                 (AtomicReference. nil)
                 (AtomicInteger. (:element-count index-data))
                 next-ordinal
                 (Random. 42)
                 normalize-vectors)]
      (ultra/optimize-search! index))))

(defn load-index
  "Load an UltraGraph or LightningIndex from an EDN file."
  ([filepath]
   (load-index filepath ultra/cosine-distance-ultra))
  ([filepath distance-fn]
   (if (.exists (io/file filepath))
     (do
       (println (format "📂 Loading index from %s..." filepath))
       (let [start (System/currentTimeMillis)
             index-data (edn/read-string (slurp filepath))
             index (if (= :lightning (:index-type index-data))
                     (load-lightning-index index-data)
                     (load-ultra-index index-data distance-fn))
             elapsed (- (System/currentTimeMillis) start)]
         (println (format "✅ Index loaded in %.2f seconds (%d vectors)"
                          (double (/ elapsed 1000.0))
                          (:element-count index-data)))
         index))
     (do
       (println (format "❌ Index file not found: %s" filepath))
       nil))))

(defn index-exists?
  "Check if an index file exists"
  [filepath]
  (.exists (io/file filepath)))
