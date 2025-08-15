(ns hnsw.helper.index-io
  "HNSW Index Serialization/Deserialization utilities using EDN format"
  (:require [clojure.java.io :as io]
            [clojure.edn :as edn]
            [hnsw.ultra-fast :as ultra])
  (:import [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic AtomicReference AtomicInteger]
           [java.util HashSet]))

(defn save-index
  "Save HNSW index to an EDN file"
  [index filepath]
  (println (format "💾 Saving index to %s..." filepath))
  (let [start (System/currentTimeMillis)
        ;; Convert to serializable format
        nodes-data (into {}
                         (map (fn [[k v]]
                                [k {:id (.id v)
                                    :vector (vec (.vector v))
                                    :level (.level v)
                                    :neighbors (mapv #(vec %) (.neighbors v))}])
                              (.nodes index)))

        index-data {:nodes nodes-data
                    :entry-point (.get (.entry-point index))
                    :M (.M index)
                    :max-M (.max-M index)
                    :ef-construction (.ef-construction index)
                    :ml (.ml index)
                    :element-count (.get (.element-count index))}]

    ;; Write to file
    (spit filepath (pr-str index-data))

    (let [elapsed (- (System/currentTimeMillis) start)
          file-size (double (/ (.length (io/file filepath)) (* 1024 1024)))]
      (println (format "✅ Index saved in %.2f seconds (%.1f MB)"
                       (double (/ elapsed 1000.0)) file-size))))
  index)

(defn load-index
  "Load HNSW index from an EDN file"
  [filepath distance-fn]
  (if (.exists (io/file filepath))
    (do
      (println (format "📂 Loading index from %s..." filepath))
      (let [start (System/currentTimeMillis)
            ;; Read EDN data
            index-data (edn/read-string (slurp filepath))

            ;; Reconstruct nodes
            nodes (ConcurrentHashMap.)
            _ (doseq [[k node-data] (:nodes index-data)]
                (let [neighbors (into-array Object
                                            (map #(HashSet. %) (:neighbors node-data)))
                      node (ultra/->UltraNode
                            (:id node-data)
                            (double-array (:vector node-data))
                            (:level node-data)
                            neighbors)]
                  (.put nodes k node)))

            ;; Reconstruct index using the constructor
            index (ultra/->UltraGraph
                   nodes
                   (AtomicReference. (:entry-point index-data))
                   (:M index-data)
                   (:max-M index-data)
                   (:ef-construction index-data)
                   (:ml index-data)
                   distance-fn
                   (AtomicInteger. (:element-count index-data)))]

        (let [elapsed (- (System/currentTimeMillis) start)]
          (println (format "✅ Index loaded in %.2f seconds (%d vectors)"
                           (double (/ elapsed 1000.0)) (:element-count index-data))))
        index))
    (do
      (println (format "❌ Index file not found: %s" filepath))
      nil)))

(defn index-exists?
  "Check if an index file exists"
  [filepath]
  (.exists (io/file filepath)))
