(ns test.simple
  "Simple test to verify the library works"
  (:require [hnsw.api :as hnsw]))

(defn test-basic-operations []
  (println "\n🧪 Testing HNSW Library Basic Operations")
  (println "========================================\n")

  (try
    ;; Create index
    (println "1. Creating index...")
    (def idx (hnsw/index {:dimensions 3
                          :distance-fn :euclidean}))
    (println "   ✅ Index created")

    ;; Add vectors
    (println "\n2. Adding vectors...")
    (hnsw/add! idx "vec1" [1.0 0.0 0.0])
    (hnsw/add! idx "vec2" [0.0 1.0 0.0])
    (hnsw/add! idx "vec3" [0.0 0.0 1.0])
    (hnsw/add! idx "vec4" [1.0 1.0 0.0])
    (hnsw/add! idx "vec5" [0.5 0.5 0.5])
    (println "   ✅ Added 5 vectors")

    ;; Search
    (println "\n3. Searching for nearest neighbors...")
    (let [query [0.9 0.1 0.1]
          results (hnsw/search idx query 3)]
      (println "   Query:" query)
      (println "   Results:")
      (doseq [{:keys [id distance]} results]
        (println (format "     - %s: distance %.3f" id distance))))

    ;; Test with metadata
    (println "\n4. Testing with metadata...")
    (hnsw/add! idx "doc1" [2.0 0.0 0.0] {:title "Document 1" :type "article"})
    (hnsw/add! idx "doc2" [0.0 2.0 0.0] {:title "Document 2" :type "blog"})

    (let [results (hnsw/search idx [1.8 0.2 0.0] 2)]
      (println "   Results with metadata:")
      (doseq [{:keys [id distance metadata]} results]
        (println (format "     - %s: distance %.3f, metadata %s"
                         id distance metadata))))

    ;; Test different distance functions
    (println "\n5. Testing cosine distance...")
    (def idx-cos (hnsw/index {:dimensions 3
                              :distance-fn :cosine}))
    (hnsw/add! idx-cos "a" [1.0 0.0 0.0])
    (hnsw/add! idx-cos "b" [0.707 0.707 0.0])
    (hnsw/add! idx-cos "c" [0.0 1.0 0.0])

    (let [results (hnsw/search idx-cos [1.0 0.1 0.0] 2)]
      (println "   Cosine similarity results:")
      (doseq [{:keys [id distance]} results]
        (println (format "     - %s: distance %.3f" id distance))))

    (println "\n✅ All tests passed!")

    (catch Exception e
      (println "\n❌ Test failed with error:")
      (println (.getMessage e))
      (.printStackTrace e))))

(defn -main [& args]
  (test-basic-operations))
