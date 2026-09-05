(ns hnsw.search.vector
  "Java Vector API distance engines for normalized vectors."
  (:import [hnsw.search VectorDistance]
           [hnsw.ultra_fast NormalizedDistance FloatNormalizedDistance]))

(set! *warn-on-reflection* false)
(set! *unchecked-math* true)

(deftype VectorNormalizedDistance []
  NormalizedDistance
  (distance [_ left right]
    (VectorDistance/normalizedCosine left right)))

(defn create-distance-engine
  []
  (VectorNormalizedDistance.))

(deftype VectorFloatNormalizedDistance []
  FloatNormalizedDistance
  (distanceFloat [_ left right]
    (VectorDistance/normalizedCosineFloat left right)))

(defn create-float-distance-engine
  []
  (VectorFloatNormalizedDistance.))
