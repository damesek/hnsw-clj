(ns benchmark.core
  "Comprehensive HNSW implementation benchmarks"
  (:require [hnsw.core :as our-hnsw]
            [hnsw.api :as api]
            [hnsw.ultra-optimized :as ultra]
            [clojure.string :as str])
  (:import [java.nio.file Paths]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ============= Test Data Generation =============

(defn generate-random-vectors
  "Generate random float vectors for testing"
  [num-vectors dimensions]
  (vec (repeatedly num-vectors
                   #(float-array (repeatedly dimensions rand)))))

;; ============= Distance Functions =============

(defn euclidean-distance
  [^floats v1 ^floats v2]
  (let [dim (alength v1)]
    (loop [i 0
           sum 0.0]
      (if (< i dim)
        (let [diff (- (aget v1 i) (aget v2 i))]
          (recur (inc i) (+ sum (* diff diff))))
        (Math/sqrt sum)))))

;; ============= Warmup Function =============

(defn warmup-jvm
  "Warm up the JVM to get more consistent benchmark results"
  [f iterations]
  (dotimes [_ iterations]
    (f)))

;; ============= Our API Implementation =============

(defn benchmark-our-hnsw-api
  [vectors queries k]
  (let [^floats first-vec (first vectors)
        dim (alength first-vec)

        ;; Create index
        start-index-ns (System/nanoTime)
        index (api/create-index
               {:dimensions dim
                :m 16
                :ef-construction 200
                :max-elements (count vectors)
                :distance-fn :euclidean})

        ;; Add vectors
        _ (doseq [[idx v] (map-indexed vector vectors)]
            (api/add! index idx v))
        index-time-ns (- (System/nanoTime) start-index-ns)]))

        ;;