# hnsw-clj

JVM-native HNSW vector search for Clojure: semantic search, recommendations and similarity matching over precomputed embeddings. Parallel indexing, cosine/Euclidean distance, optional SIMD and exact metadata-filtered search; no Python or JNI dependency.

## API

Requires JDK 21 and Clojure CLI. Build with `clojure -M:build`; enable SIMD with `-J--add-modules=jdk.incubator.vector`.

```clojure
(require '[hnsw.api :as hnsw])

(def entries
  [["a" [1.0 0.0] {:tenant 42 :category "article" :language "hu"}]
   ["b" [0.0 1.0] {:tenant 42 :category "article" :language "en"}]])

(def index
  (hnsw/index entries :metric :cosine :num-threads 10
              :construction-precision :float32 :compact-precision :float32
              :filter-keys #{:tenant :category :language}))
```

`index` accepts vectors or `[id vector metadata?]` entries and copies input arrays. IDs become unique strings; vectors must have equal dimensions and finite values (nonzero norm for cosine). Metadata should be immutable EDN. Search returns distance-ordered `{:id … :distance … :metadata …}` maps; metadata is omitted when absent.

| Index option | Default |
| --- | --- |
| `:metric` | `:euclidean`; also `:cosine` |
| `:M` / `:ef-construction` / `:ef-search` | `16` / `200` / `50` |
| `:num-threads` / `:batch-size` / `:seed` | Available processors / `4` / `42` |
| `:construction-precision` / `:compact-precision` | `:float64`; optional `:float32` for cosine |
| `:dimensions` / `:show-progress?` | Inferred / `false` |
| `:filter-keys` | `#{}` (disabled); set or sequence of arbitrary metadata keys |

| API | Purpose | Example |
| --- | --- | --- |
| `search` | Approximate top-k; optional efSearch overrides the index default. | `(hnsw/search index [1.0 0.0] 10 80)` |
| `search-batch` | Parallel queries, preserving input order. | `(hnsw/search-batch index [[1.0 0.0] [0.0 1.0]] 10 :num-threads 10)` |
| `search-filtered` | Exact top-k matching all supplied metadata conditions. | `(hnsw/search-filtered index [1.0 0.0] 10 {:tenant 42 :category "article" :language "hu"})` |
| `add-vector!` | Insert one new ID. | `(hnsw/add-vector! index "c" [0.8 0.2] {:language "hu"})` |
| `add-vectors!` | Insert a validated batch. | `(hnsw/add-vectors! index [["d" [0.7 0.3]] ["e" [0.6 0.4]]])` |
| `update-vector!` | Replace vector and optional metadata; three arguments preserve metadata. | `(hnsw/update-vector! index "c" [0.9 0.1] {:language "en"})` |
| `delete-vector!` | Delete one ID; missing IDs are ignored. | `(hnsw/delete-vector! index "c")` |
| `delete-vectors!` | Delete several IDs with one snapshot refresh. | `(hnsw/delete-vectors! index ["d" "e"])` |
| `size` | Number of live vectors. | `(hnsw/size index)` |
| `info` | Dimensions, graph settings, acceleration and filter keys. | `(hnsw/info index)` |
| `save-index` | Save graph, metadata and configuration as EDN. | `(hnsw/save-index index "index.edn")` |
| `load-index` | Load a trusted file; optionally override filter keys. | `(with-open [loaded (hnsw/load-index "index.edn" :filter-keys #{:language})] (hnsw/size loaded))` |
| `tune` | Offline cosine tuning against a recall target; calibration/validation queries must be disjoint. | `(hnsw/tune entries [[1.0 0.0]] [[0.0 1.0]] :k 1 :target-recall 0.98)` |
| `clear!` | Remove all vectors; retain dimensions and options. | `(hnsw/clear! index)` |
| `close!` | Release references; subsequent operations fail. Also supports `with-open`. | `(hnsw/close! index)` |

Filters use equality with **AND**, on any nonempty subset of configured keys; no two-key limit. Empty filters and unconfigured keys are rejected. Explicit `nil` differs from a missing key. Filtered search scans the matching subset exactly in float64; efSearch does not apply. Its lazy metadata cache rebuilds after metadata changes.

Lower efSearch can reduce ANN recall. Mutations block readers; updates/deletions repair the graph locally and refresh the full search snapshot.

## Benchmark

Hungarian Bible, 768D multilingual MPNet, cosine, top-10, M=16, efConstruction=100, efSearch=80, float32 construction/ANN search, 10 build threads. Apple Silicon (10 CPUs), JDK 21 SIMD, Python hnswlib 0.8.0; five-run medians, API costs included, loading/embedding/warmup excluded.

| Vectors | Implementation | Indexing (s) | Search, 1 thread (ms) | 10-thread QPS | Filtered, 1 thread (ms) | ANN Recall@10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10,000 | Python hnswlib (C++) | 0.678 | 0.336 | 18,660 | 5.739 | 99.09% |
| 10,000 | Clojure (JVM) | 0.684 | 0.151 | 24,352 | 0.046 | 99.07% |
| 31,126 (full Bible) | Python hnswlib (C++) | 2.880 | 0.479 | 12,308 | 21.341 | 98.71% |
| 31,126 (full Bible) | Clojure (JVM) | 3.567 | 0.268 | 14,164 | 0.053 | 98.67% |

ANN uses 497 matched held-out queries; filters use 1,000 aligned queries with varying book AND chapter (~34 eligible verses), warm caches. **Filtering differs: Clojure exact subset scan vs Python hnswlib callback ANN**, not equivalent algorithms. Runtimes were measured separately.

[Commands, methodology and raw results](examples/bible_benchmark/README.md#readme-comparison).
