(ns hnsw.filtered-test
  (:require [clojure.edn :as edn]
            [clojure.test :refer :all]
            [hnsw.api :as api]
            [hnsw.search.filtered :as filtered]
            [hnsw.ultra-fast :as ultra])
  (:import [hnsw.ultra_fast UltraGraph UltraNode]
           [java.nio.file Files]
           [java.util Random]))

(defn- ids [neighbors] (mapv :id neighbors))

(defn- bible-index [entries & options]
  (apply api/index entries :filter-keys #{:book :chapter} options))

(deftest arbitrary-keys-and-multiple-and-conditions
  (let [attributes {:tenant-id 42 "category" "article" :language "hu" :active false
                    :deleted-at nil 17 :numbered [:literal :key] [1 2] nil :nil-key false :false-key}
        entries [["match" [1 0] attributes]
                 ["other-language" [2 0] (assoc attributes :language "en")]
                 ["other-tenant" [3 0] (assoc attributes :tenant-id 43)]
                 ["missing-field" [4 0] (dissoc attributes :deleted-at)]
                 ["no-metadata" [0 0]]]]
    (with-open [index (api/index entries :filter-keys (vec (keys attributes)) :num-threads 1)]
      (is (= (set (keys attributes)) (:filter-keys (api/info index))))
      (is (= ["match" "missing-field"]
             (ids (api/search-filtered index [0 0] 10 {:tenant-id 42 "category" "article" :language "hu"}))))
      (is (= ["match"] (ids (api/search-filtered index [0 0] 10 attributes))))
      (is (= ["match" "other-language" "other-tenant"]
             (ids (api/search-filtered index [0 0] 10 {:deleted-at nil}))))
      (is (= ["match" "other-language" "other-tenant" "missing-field"]
             (ids (api/search-filtered index [0 0] 10 {:active false nil :nil-key false :false-key}))))
      (is (= [] (api/search-filtered index [0 0] 10 {:tenant-id "42"})))
      (is (= [] (api/search-filtered index [0 0] 10 {[:literal :key] [2 1]})))
      (is (= [] (api/search-filtered index [0 0] 10 {"category" :article})))
      (is (= ["other-language"]
             (ids (api/search-filtered index [0 0] 10 {:language "en" :tenant-id 42}))))))
  (let [attributes (into {} (map (fn [position] [position (str position)]) (range 40)))]
    (with-open [index (api/index [["a" [1 0] attributes]
                                 ["b" [0 0] (assoc attributes 39 "different")]]
                                :filter-keys (keys attributes) :num-threads 1)]
      (is (= ["a"] (ids (api/search-filtered index [0 0] 10 attributes)))))))

(deftest configuration-is-opt-in-per-index-and-does-not-affect-ordinary-search
  (let [entries [["a" [1 0] {:tenant 42 :language :hu}]]]
    (with-open [disabled (api/index entries :num-threads 1)
                tenant (api/index entries :filter-keys [:tenant :tenant] :num-threads 1)
                language (api/index entries :filter-keys #{:language} :num-threads 1)]
      (is (= #{} (:filter-keys (api/info disabled))))
      (is (= #{:tenant} (:filter-keys (api/info tenant))))
      (is (= #{:language} (:filter-keys (api/info language))))
      (with-redefs [filtered/build-postings (fn [& _] (throw (AssertionError. "Ordinary queries must not initialize filters")))]
        (is (= (api/search disabled [1 0] 1) (api/search tenant [1 0] 1) (api/search language [1 0] 1)))
        (is (= (api/search-batch disabled [[1 0]] 1) (api/search-batch tenant [[1 0]] 1))))
      (doseq [index [disabled tenant language]]
        (is (nil? (:metadata-filter-cache @(:state index)))))
      (is (thrown? IllegalArgumentException (api/search-filtered disabled [1 0] 1 {:tenant 42})))
      (is (thrown? IllegalArgumentException (api/search-filtered tenant [1 0] 1 {:language :hu})))
      (is (thrown? IllegalArgumentException (api/search-filtered language [1 0] 1 {:tenant 42})))
      (is (= ["a"] (ids (api/search-filtered tenant [1 0] 1 {:tenant 42}))))
      (is (= ["a"] (ids (api/search-filtered language [1 0] 1 {:language :hu}))))))
  (doseq [invalid [nil 1 :tenant "tenant" {:tenant true}]]
    (is (thrown? IllegalArgumentException (api/index [] :filter-keys invalid)))))

(deftest metadata-field-names-have-no-domain-rules
  (with-open [index (api/index [["chapter-only" [0 0] {:chapter "anything"}]
                               ["nil-book" [1 0] {:book nil :chapter -1}]
                               ["blank-book" [2 0] {:book "" :chapter 0}]]
                              :filter-keys #{:book :chapter} :num-threads 1)]
    (is (= ["chapter-only"] (ids (api/search-filtered index [0 0] 10 {:chapter "anything"}))))
    (is (= ["nil-book"] (ids (api/search-filtered index [0 0] 10 {:book nil :chapter -1}))))
    (is (= ["blank-book"] (ids (api/search-filtered index [0 0] 10 {:book "" :chapter 0}))))))

(deftest sparse-postings-match-independent-metadata-intersections
  (let [field-keys [:tenant :category :enabled :nullable :unique :tags]
        metadata (into {} (map (fn [ordinal]
                                [(str ordinal)
                                 (cond-> {:tenant (mod ordinal 3) :category (mod ordinal 5)
                                          :enabled (even? ordinal) :nullable nil
                                          :unique ordinal :tags [(mod ordinal 7)]}
                                   (zero? (mod ordinal 4)) (dissoc :nullable)
                                   (zero? (mod ordinal 11)) (dissoc :tenant))]) (range 200)))
        postings (filtered/build-postings metadata (set field-keys))
        before (into {} (for [[key values] (:fields postings) [value positions] values]
                          [[key value] (vec positions)]))]
    (doseq [anchor (range 0 200 13) mask (range 1 64)]
      (let [criteria (select-keys (get metadata (str anchor))
                                 (keep-indexed #(when (bit-test mask %1) %2) field-keys))]
        (when (seq criteria)
          (let [expected (set (keep (fn [[identifier attributes]]
                                     (when (= criteria (select-keys attributes (keys criteria))) identifier)) metadata))
                actual (filtered/matching-ids postings criteria)]
            (is (= expected (set actual)))
            (is (= (count expected) (count actual)))))))
    (is (= [] (filtered/matching-ids postings {:tenant 9 :nullable nil})))
    (is (= [] (filtered/matching-ids postings {:tenant 1 :unique 0})))
    (is (= before (into {} (for [[key values] (:fields postings) [value positions] values]
                            [[key value] (vec positions)]))))
    (is (= (reduce + (map count (vals metadata)))
           (reduce + (map count (vals before)))))))

(deftest saved-filter-configuration-and-legacy-load-overrides
  (let [path (Files/createTempFile "hnsw-generic-filter" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (with-open [index (api/index [["a" [1 0] {:tenant 42 :language :hu :deleted-at nil}]]
                                  :filter-keys #{:tenant :language :deleted-at} :num-threads 1)]
        (api/search-filtered index [1 0] 1 {:tenant 42 :language :hu :deleted-at nil})
        (api/save-index index (str path)))
      (with-open [loaded (api/load-index (str path))]
        (is (= #{:tenant :language :deleted-at} (:filter-keys (api/info loaded))))
        (is (nil? (:metadata-filter-cache @(:state loaded))))
        (is (= ["a"] (ids (api/search-filtered loaded [1 0] 10 {:tenant 42 :language :hu :deleted-at nil}))))
        (api/clear! loaded)
        (is (= #{:tenant :language :deleted-at} (:filter-keys (api/info loaded))))
        (api/add-vector! loaded "new" [0 0] {:tenant 42 :language :hu :deleted-at nil})
        (is (= ["new"] (ids (api/search-filtered loaded [0 0] 10 {:tenant 42 :deleted-at nil})))))
      (with-open [loaded (api/load-index (str path) :filter-keys [:language])]
        (is (= #{:language} (:filter-keys (api/info loaded))))
        (is (thrown? IllegalArgumentException (api/search-filtered loaded [1 0] 10 {:tenant 42})))
        (is (= ["a"] (ids (api/search-filtered loaded [1 0] 10 {:language :hu})))))
      (doseq [options [[:filter-keys nil] [:unknown true]]]
        (is (thrown? IllegalArgumentException (apply api/load-index (str path) options))))
      (let [saved (edn/read-string (slurp (str path)))]
        (spit (str path) (pr-str (update saved :options dissoc :filter-keys)))
        (with-open [legacy (api/load-index (str path))]
          (is (= #{} (:filter-keys (api/info legacy))))
          (is (= ["a"] (ids (api/search legacy [1 0] 1))))
          (is (thrown? IllegalArgumentException (api/search-filtered legacy [1 0] 10 {:tenant 42}))))
        (with-open [legacy (api/load-index (str path) :filter-keys #{:tenant :language})]
          (is (= ["a"] (ids (api/search-filtered legacy [1 0] 10 {:tenant 42 :language :hu})))))
        (spit (str path) (pr-str (assoc-in saved [:options :filter-keys] :invalid)))
        (is (thrown? IllegalArgumentException (api/load-index (str path)))))
      (finally (Files/deleteIfExists path)))))

(deftest book-and-chapter-are-an-intersection
  (with-open [index (bible-index [["global" [0 0] {:book "Luke" :chapter 3}]
                               ["john-3-b" [2 0] {:book "John" :chapter 3 :text "second"}]
                               ["john-3-a" [1 0] {:book "John" :chapter 3 :text "first"}]
                               ["john-4" [0.5 0] {:book "John" :chapter 4}]
                               ["book-only" [3 0] {:book "John"}]
                               ["no-metadata" [0 0]]
                               ["false-metadata" [0 0] false]
                               ["non-map" [0 0] "John"]
                               ["no-book" [0 0] {:chapter 3}]]
                              :num-threads 1 :ef-search 1)]
    (is (= ["john-3-a" "john-3-b"]
           (ids (api/search-filtered index [0 0] 10 {:book "John" :chapter 3}))))
    (is (= ["john-4" "john-3-a" "john-3-b" "book-only"]
           (ids (api/search-filtered index [0 0] 10 {:book "John"}))))
    (is (= [{:id "john-3-a" :distance 1.0
             :metadata {:book "John" :chapter 3 :text "first"}}]
           (api/search-filtered index [0 0] 1 {:book "John" :chapter 3})))
    (is (= [] (api/search-filtered index [0 0] 10 {:book "john"})))
    (is (= [] (api/search-filtered index [0 0] 10 {:book "John" :chapter 999})))
    (is (= [] (api/search-filtered index [0 0] 10 {:book "Missing"}))))
  (doseq [book [:john 0 42]]
    (with-open [index (bible-index [["a" [1 0] {:book book :chapter 1}]] :num-threads 1)]
      (is (= ["a"] (ids (api/search-filtered index [1 0] 1 {:book book :chapter 1})))))))

(deftest deterministic-ties-and-large-k
  (with-open [index (bible-index (mapv #(vector % [1 0] {:book :john :chapter 1})
                                    ["z" "2" "10" "a"])
                              :metric :cosine :num-threads 1)]
    (is (= ["10" "2"] (ids (api/search-filtered index [1 0] 2 {:book :john}))))
    (is (= ["10" "2" "a" "z"]
           (ids (api/search-filtered index (float-array [1 0]) Integer/MAX_VALUE {:book :john}))))
    (is (every? #(zero? (:distance %)) (api/search-filtered index [1 0] 10 {:book :john})))))

(deftest validates-filter-and-query-before-building-cache
  (with-open [index (bible-index [["a" [1 0] {:book :john :chapter 1}]]
                              :metric :cosine :num-threads 1)]
    (doseq [criteria [nil {} [] {:verse 1} {:book :john :verse 1} {:book :john :predicate identity}]]
      (is (thrown? IllegalArgumentException (api/search-filtered index [1 0] 10 criteria))))
    (doseq [top-k [nil 0 -1 1.5 (inc (long Integer/MAX_VALUE))]]
      (is (thrown? IllegalArgumentException (api/search-filtered index [1 0] top-k {:book :john}))))
    (doseq [query [nil 2 [] [1] [0 0] [Double/NaN 0] [Double/POSITIVE_INFINITY 0]]]
      (is (thrown? IllegalArgumentException (api/search-filtered index query 1 {:book :john}))))
    (is (nil? (:metadata-filter-cache @(:state index))))))

(deftest ordinary-search-and-snapshots-stay-unchanged
  (with-open [index (bible-index [["a" [1 0] {:book :john :chapter 1}]
                               ["b" [0 1] {:book :luke :chapter 1}]]
                              :metric :cosine :compact-precision :float32 :num-threads 1)]
    (let [state @(:state index)
          ^UltraGraph graph (:graph state)
          snapshot (.get (.search-index graph))
          ordinary (api/search index [1 0] 2)
          batch (api/search-batch index [[1 0] [0 1]] 2 :num-threads 2)]
      (is (nil? (:metadata-filter-cache @(:state index))))
      (with-redefs [ultra/search-knn (fn [& _] (throw (AssertionError. "Global ANN must not run")))
                    ultra/optimize-search! (fn [& _] (throw (AssertionError. "Snapshot must not change")))]
        (is (= ["b"] (ids (api/search-filtered index [1 0] 1 {:book :luke :chapter 1})))))
      (is (identical? graph (:graph @(:state index))))
      (is (identical? snapshot (.get (.search-index graph))))
      (is (= state (dissoc @(:state index) :metadata-filter-cache)))
      (is (= ordinary (api/search index [1 0] 2)))
      (is (= batch (api/search-batch index [[1 0] [0 1]] 2 :num-threads 2)))
      (let [cache (:metadata-filter-cache @(:state index))]
        (api/search-filtered index [0 1] 1 {:book :john})
        (is (identical? cache (:metadata-filter-cache @(:state index))))))))

(defn- exact-distance [metric query values]
  (if (= metric :euclidean)
    (Math/sqrt (reduce + (map (fn [left right] (let [delta (- left right)] (* delta delta))) query values)))
    (let [query-norm (Math/sqrt (reduce + (map #(* % %) query)))
          value-norm (Math/sqrt (reduce + (map #(* % %) values)))]
      (- 1.0 (/ (reduce + (map * query values)) (* query-norm value-norm))))))

(deftest exhaustive-subset-matches-independent-ground-truth
  (let [random (Random. 987)
        vector! #(vec (repeatedly 33 (fn [] (- (.nextDouble random) 0.5))))
        entries (mapv (fn [ordinal]
                        [(str ordinal) (vector!)
                         {:book (if (even? ordinal) :john :luke) :chapter (inc (mod ordinal 7))}])
                      (range 120))
        queries (repeatedly 5 vector!)]
    (doseq [metric [:cosine :euclidean]
            precision (if (= metric :cosine) [:float64 :float32] [:float64])]
      (with-open [index (bible-index entries :metric metric :num-threads 1
                                  :construction-precision precision :compact-precision precision)]
        (doseq [query queries
                criteria [{:book :john} {:book :john :chapter 3} {:book :luke :chapter 7}]
                top-k [1 10 200]]
          (let [expected (->> entries
                              (filter (fn [[_ _ metadata]] (= criteria (select-keys metadata (keys criteria)))))
                              (map (fn [[identifier values metadata]]
                                     {:id identifier :distance (exact-distance metric query values) :metadata metadata}))
                              (sort-by (juxt :distance :id))
                              (take top-k))
                actual (api/search-filtered index query top-k criteria)]
            (is (= (ids expected) (ids actual)))
            (is (= (mapv :metadata expected) (mapv :metadata actual)))
            (is (every? #(< (Math/abs (double %)) 1.0e-12)
                        (map - (map :distance expected) (map :distance actual))))))))))

(deftest cache-tracks-mutations-and-live-vectors
  (with-open [index (bible-index [["a" [1 0] {:book :john :chapter 1}]
                               ["b" [2 0] {:book :john :chapter 1}]] :num-threads 1)]
    (let [criteria {:book :john :chapter 1}]
      (is (= ["a" "b"] (ids (api/search-filtered index [0 0] 10 criteria))))
      (let [cache (:metadata-filter-cache @(:state index))]
        (api/update-vector! index "a" [3 0])
        (is (= ["b" "a"] (ids (api/search-filtered index [0 0] 10 criteria))))
        (is (identical? cache (:metadata-filter-cache @(:state index)))))
      (api/add-vector! index "c" [0 0] {:book :john :chapter 1})
      (is (= ["c" "b" "a"] (ids (api/search-filtered index [0 0] 10 criteria))))
      (api/update-vector! index "c" [0 0] {:book :luke :chapter 2})
      (is (= ["b" "a"] (ids (api/search-filtered index [0 0] 10 criteria))))
      (is (= ["c"] (ids (api/search-filtered index [0 0] 10 {:book :luke :chapter 2}))))
      (api/update-vector! index "b" [2 0] {:book :john :chapter 2})
      (is (= ["a"] (ids (api/search-filtered index [0 0] 10 criteria))))
      (let [^UltraGraph graph (:graph @(:state index))
            ordinal (.ordinal ^UltraNode (.get (.nodes graph) "a"))]
        (api/delete-vector! index "a")
        (api/add-vector! index "reused" [0 0] {:book :luke :chapter 1})
        (is (= ordinal (.ordinal ^UltraNode (.get (.nodes graph) "reused"))))
        (is (= [] (api/search-filtered index [0 0] 10 criteria))))
      (api/add-vector! index "a" [4 0] {:book :john :chapter 1})
      (is (= ["a"] (ids (api/search-filtered index [0 0] 10 criteria))))
      (api/update-vector! index "a" [4 0] nil)
      (is (= [] (api/search-filtered index [0 0] 10 criteria)))
      (api/clear! index)
      (is (= [] (api/search-filtered index [0 0] 10 {:book :luke})))
      (api/add-vector! index "new" [0 0] {:book :john :chapter 1})
      (is (= ["new"] (ids (api/search-filtered index [0 0] 10 criteria)))))))

(deftest empty-and-closed-index
  (let [index (bible-index [] :metric :cosine :num-threads 1)]
    (is (= [] (api/search-filtered index [1 0] 10 {:book :john})))
    (api/add-vector! index "a" [1 0] {:book :john :chapter 1})
    (is (= ["a"] (ids (api/search-filtered index [1 0] 10 {:book :john}))))
    (api/close! index)
    (is (thrown? IllegalStateException (api/search-filtered index [1 0] 10 {:book :john})))))

(deftest metadata-persists-but-derived-postings-do-not
  (let [path (Files/createTempFile "hnsw-filtered" ".edn"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (with-open [index (bible-index [["a" [1 0] {:book "John" :chapter 3 :text "árvíz"}]]
                                  :metric :cosine :num-threads 1 :compact-precision :float32)]
        (let [expected (api/search-filtered index [1 0] 10 {:book "John" :chapter 3})]
          (api/save-index index (str path))
          (is (not (contains? (edn/read-string (slurp (str path))) :metadata-filter-cache)))
          (with-open [loaded (api/load-index (str path))]
            (is (nil? (:metadata-filter-cache @(:state loaded))))
            (is (= expected (api/search-filtered loaded [1 0] 10 {:book "John" :chapter 3})))
            (api/delete-vector! loaded "a")
            (is (= [] (api/search-filtered loaded [1 0] 10 {:book "John"}))))))
      (finally (Files/deleteIfExists path)))))

(deftest concurrent-readers-share-cold-cache-and-exclude-writers
  (with-open [index (bible-index [["initial" [1 0] {:book :john :chapter 1}]]
                              :metric :cosine :num-threads 1)]
    (let [builds (atom 0)
          build-postings filtered/build-postings]
      (with-redefs [filtered/build-postings (fn [metadata filter-keys] (swap! builds inc) (build-postings metadata filter-keys))]
        (let [start (promise)
              readers (mapv (fn [_] (future @start (api/search-filtered index [1 0] 10 {:book :john})))
                            (range 8))]
          (deliver start true)
          (doseq [reader readers]
            (is (= ["initial"] (ids (deref reader 30000 [])))))
          (is (= 1 @builds)))))
    (let [readers (mapv (fn [_]
                         (future
                           (every? true?
                             (repeatedly 30
                               #(let [neighbors (api/search-filtered index [1 0] 100 {:book :john :chapter 1})]
                                  (and (seq neighbors)
                                       (every? (fn [neighbor] (= {:book :john :chapter 1} (:metadata neighbor)))
                                               neighbors)))))))
                       (range 4))
          writer (future
                   (dotimes [ordinal 10]
                     (api/add-vector! index (str ordinal) [1 1] {:book :john :chapter 1})
                     (api/update-vector! index (str ordinal) [0 1] {:book :luke :chapter 2})
                     (api/delete-vector! index (str ordinal))))]
      (is (not= ::timeout (deref writer 30000 ::timeout)))
      (doseq [reader readers] (is (true? (deref reader 30000 ::timeout))))
      (is (= ["initial"] (ids (api/search-filtered index [1 0] 10 {:book :john})))))))
