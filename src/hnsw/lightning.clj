(ns hnsw.lightning
  "Backward-compatible facade for the relocated Lightning implementation."
  (:require [hnsw.ann.partition.lightning :as implementation]))

(def build-index implementation/build-index)
(def build-lightning-index implementation/build-lightning-index)
(def search-knn implementation/search-knn)
(def search-lightning implementation/search-lightning)
(def index-info implementation/index-info)
