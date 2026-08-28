(ns hnsw.search.vector
  "Java Vector API distance engine for normalized double vectors."
  (:import [hnsw.search VectorDistance]
           [hnsw.ultra_fast NormalizedDistance]))

(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

(deftype VectorNormalizedDistance []
  NormalizedDistance
  (distance [_ left right]
    (VectorDistance/normalizedCosine left right)))

(defn create-distance-engine
  []
  (VectorNormalizedDistance.))
