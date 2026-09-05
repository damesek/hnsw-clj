# Bilingual Bible HNSW benchmark

This example compares Clojure `hnsw-clj` and Python `hnswlib` using identical
HNSW parameters and the same precomputed float32 vectors.

HNSW indexes numeric vectors, not text. For English-to-Hungarian semantic search,
both translations must be embedded into one multilingual vector space. The default
model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.

## README comparison

The concise root README combines these measured results, not synthetic estimates:

- Construction, serial ANN latency, 10-thread batch QPS and ANN recall:
  [managed API comparison](api_overhead_results_2026-09-05.json), optimized Clojure rows
  and matching Python rows, at 10,000 and 31,126 vectors. These use 497 held-out
  queries per size and include each implementation's public API. Clojure build timings
  include validation and compact snapshot creation; filter-cache creation is separate.
- Clojure filtered latency: the `mixed-chapters-exact` row from
  [10k generic-filter results](filtered_generic_10k_results_2026-09-05.json) and
  [full-Bible generic-filter results](filtered_generic_results_2026-09-05.json).
- Python filtered latency: [callback-filter results](filtered_python_results_2026-09-05.json).
  The wrapper resolves/intersects book/chapter postings, calls hnswlib's native filtered
  ANN with a Python membership callback, then creates ID/distance/metadata results.
  It uses one query thread and efSearch=80. This is not a Python exact NumPy scan.

Filtered workloads use the same 1,000 English queries and each aligned Hungarian
verse's book AND chapter: 34.494 / 33.677 eligible verses on average at 10k / full size.
Query SHA-256 was independently verified against the Clojure loader:

| Vectors | Query SHA-256 |
| ---: | --- |
| 10,000 | `df4784c50037522ee881c364328718dd841899fd8b5aa0d7573c448876d990b0` |
| 31,126 | `79a3a5d165139e0ca339b7c882e28a68434dde55ab062add73ffe4ad44acc472` |

Python's filtered recall against exhaustive float64 subset neighbors is 99.970% / 99.950%
(denominator is min(10, eligible count)). Clojure's filtered path is exhaustive float64;
these are different algorithms and precision paths, not an isolated language comparison.
Small selective filters can make callback-filtered HNSW visit much of the graph; the
Clojure subset scan only scores matches. Do not extrapolate this result to broad filters.

Filtered numbers are warmed medians of five rounds: Clojure repeats the 1,000 queries
five times per round, Python once. Posting construction, loading, embeddings, graph
building and exact validation are outside filtered timings. Runtimes were measured
separately, not interleaved. Root README ANN numbers and filtered numbers use their
respective workloads; they are not all measurements from one benchmark invocation.
The fixed John/Psalms filters in the 10k raw report match zero entries; those empty-result
measurements are deliberately excluded from the comparison.

```bash
NUM_VECTORS=10000 bash examples/bible_benchmark/run_filtered_benchmark.sh
NUM_VECTORS=31126 bash examples/bible_benchmark/run_filtered_benchmark.sh
python3 -m venv target/python-benchmark-venv
target/python-benchmark-venv/bin/python -m pip install numpy hnswlib
NUM_VECTORS=10000 bash examples/bible_benchmark/run_filtered_python.sh
NUM_VECTORS=31126 bash examples/bible_benchmark/run_filtered_python.sh
```

`PYTHON` or `HNSW_PYTHON_VENV` selects an existing Python environment. For the unfiltered
columns, use `run_api_overhead.sh` and `run_tuned_python.sh` with the corresponding size;
see the API preprocessing methodology later in this document.

## 1. Prepare shared embeddings

The defaults expect these sibling-repository files:

- `../bibliam.hu/data/bsb-karoli/HunKar.json`
- `../bibliam.hu/data/bsb-karoli/BSB.json`

```bash
./examples/bible_benchmark/run_prepare_embeddings.sh
```

The source files contain 31,126 non-empty Hungarian and 31,086 non-empty English
verses. Canonical book/chapter/verse IDs align 31,027 non-empty pairs. The generated
`hu.f32`, `en.f32`, and `manifest.json` files are ignored by Git.

Useful overrides:

```bash
DEVICE=mps BATCH_SIZE=128 ./examples/bible_benchmark/run_prepare_embeddings.sh
FORCE=1 MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  ./examples/bible_benchmark/run_prepare_embeddings.sh
```

## 2. Run both benchmarks

```bash
./examples/bible_benchmark/run_clojure.sh
./examples/bible_benchmark/run_python.sh
```

The two benchmark commands default to an English-query to Hungarian-index benchmark.
They report parallel index build time, median serial request latency and QPS, median
parallel batch throughput, and exact aligned-verse recall@1, recall@5, and recall@10.

The Clojure launcher compiles the optional Java Vector API kernel and immutable
primitive-array search snapshot before running. It requires a JDK that provides
`jdk.incubator.vector`; without that module the library automatically uses the
portable Clojure search path. This float32 embedding benchmark defaults to
`COMPACT_PRECISION=float32`; use `COMPACT_PRECISION=float64` to measure the exact
double-precision compact path.

The default benchmark uses 1,000 deterministic query verses and five measured runs:

```bash
INDEX_LANGUAGE=hu QUERY_LANGUAGE=en NUM_QUERIES=1000 RUNS=5 THREADS=10 \
  ./examples/bible_benchmark/run_clojure.sh
INDEX_LANGUAGE=hu QUERY_LANGUAGE=en NUM_QUERIES=1000 RUNS=5 THREADS=10 \
  ./examples/bible_benchmark/run_python.sh
```

Use `QUERY_LANGUAGE=hu` for same-language Hungarian search. Use
`NUM_QUERIES=31027` to query every directly aligned non-empty verse.

If `manifest.json` does not exist, both benchmarks run in `synthetic-shape` mode
with 31,173 vectors and 768 dimensions. That mode measures speed only; it does not
measure bilingual semantic quality.

## 3. Interactive text search

Run either implementation with a query argument to print the top 10 matching Bible
verses directly in the terminal:

```bash
./examples/bible_benchmark/run_text_search_clojure.sh \
  "Szeresd felebarátodat, mint magadat"

./examples/bible_benchmark/run_text_search_python.sh \
  "Szeresd felebarátodat, mint magadat"
```

Omit the query to keep the index and MPNet model loaded for repeated searches:

```bash
./examples/bible_benchmark/run_text_search_clojure.sh
./examples/bible_benchmark/run_text_search_python.sh
```

Both commands accept Hungarian or English input, embed it with the multilingual MPNet
model from `manifest.json`, and print rank, Bible reference, cosine similarity,
distance, and verse text. The Clojure command uses Python only as a persistent query
encoder; neighbor search runs in the Clojure/Java HNSW index. The Python command uses
`hnswlib`. Both therefore receive the same normalized query embedding.

Useful overrides:

```bash
INDEX_LANGUAGE=en TOP_K=10 EF_SEARCH=80 \
  ./examples/bible_benchmark/run_text_search_clojure.sh "love your enemies"
```

The first run may install Python dependencies or download the configured model. Run
`run_prepare_embeddings.sh` before searching if `data/manifest.json` is missing.

## Book/chapter filtered API benchmark

```bash
bash examples/bible_benchmark/run_filtered_benchmark.sh > /tmp/hnsw-filtered-results.json
```

This separate benchmark builds one managed index with the manifest items as metadata
and `:filter-keys #{:book :chapter}`. These are application-selected keys, not built-in
library fields; any configured field can be queried alone or together with others.
Defaults: full 31,126-verse Hungarian index, 768D multilingual MPNet, 1,000 aligned
English verse queries, top-10, M=16, efConstruction=100, float32 construction/snapshot,
10 threads, five rounds of 5,000 requests (five repetitions of the query set).
The ordinary unfiltered reference uses efSearch=80; filtered search does not use it.
`RUNS`, `QUERY_REPEATS`, `NUM_QUERIES`, `THREADS`, `INDEX_LANGUAGE`, `QUERY_LANGUAGE`,
`NUM_VECTORS`, `EF_CONSTRUCTION`, and `EF_SEARCH` can be overridden via environment.

Workloads include fixed John 3, John and Psalms filters, plus mixed books/chapters
selected from each query's aligned verse. Fixed-filter trials repeatedly visit the
same subset; mixed-filter trials better exercise changing vector-cache locality.
The script validates result counts and metadata constraints outside the timed regions.

The JSON reports median round-average serial latency, median per-round p50/p95, and
median concurrent throughput. A prestarted shared executor calls the individual managed
API functions with the same scheduling in all workloads; it does not use `search-batch`.
Validation, input copying, read locks and result metadata are included. Console output,
embedding inference, file loading, graph construction and warmup are excluded. Workload
order rotates/reverses between rounds. The first filtered call is reported separately;
warm-JIT cache-rebuild trials drop only the derived postings cache before timing the
next call, not a full add/update/delete operation. They should not be interpreted as
total mutation costs.

**Filtered exact float64 top-10 and global approximate float32 top-10 have different
search scopes and accuracy.** A smaller filtered subset can be faster; a broad exact
scan can be slower than HNSW. This is a same-machine workload comparison, not a claim
that exact search accelerates the unfiltered algorithm or a Python comparison.

### Generic metadata-filter measurements — 2026-09-05

Fresh run of the configurable implementation on the same full-size workload and
JDK/Vector API settings, using `:filter-keys #{:book :chapter}`:

| Workload | Eligible verses | Serial mean, ms | Serial p95, ms | 10-thread QPS |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered HNSW, efSearch=80 | 31,126 | 0.2620 | 0.3580 | 16,391 |
| John 3, exact | 36 | 0.0422 | 0.0440 | 118,568 |
| John, exact | 878 | 0.2344 | 0.2481 | 25,586 |
| Psalms, exact | 2,527 | 0.6877 | 0.8315 | 7,195 |
| Changing chapters, exact | 33.7 average; 4–176 | 0.0527 | 0.0708 | 113,347 |
| Changing books, exact | 1,009 average; 25–2,527 | 0.5342 | 1.0245 | 7,792 |

First filtered call: **50.01 ms**. Warm-JIT postings rebuild: **15.23 ms median**
(13.55–18.74 ms). Configurable filtering is not a speed optimization: the fixed John 3
serial query was about 5.4 microseconds slower than the earlier specialized version in
these samples. Multiple generic conditions now intersect per-field postings rather than
using a hardcoded composite book/chapter list. The measurements were separate runs,
not interleaved comparisons; do not attribute every timing difference to the code change.
The unfiltered search implementation is unchanged.

See [all generic-filter parameters and samples](filtered_generic_results_2026-09-05.json).

### Original book/chapter-specific measurements — 2026-09-05

Historical results from the earlier fixed-field implementation, before configurable
generic postings. The current launcher uses the generic implementation; these numbers
are retained as a baseline, not relabeled as measurements of the generalized version.

JDK 21.0.8, aarch64, 10 available processors, Vector API enabled, default workload
above. These figures include the managed API and verse metadata; they are not directly
interchangeable with earlier metadata-free or `search-batch` throughput results.

| Workload | Eligible verses | Serial mean, ms | Serial p95, ms | 10-thread QPS |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered HNSW, efSearch=80 | 31,126 | 0.2643 | 0.3583 | 16,330 |
| John 3, exact | 36 | 0.0369 | 0.0385 | 154,866 |
| John, exact | 878 | 0.2265 | 0.2381 | 27,101 |
| Psalms, exact | 2,527 | 0.6754 | 0.8248 | 6,775 |
| Changing chapters, exact | 33.7 average; 4–176 | 0.0444 | 0.0595 | 124,091 |
| Changing books, exact | 1,009 average; 25–2,527 | 0.5320 | 1.0371 | 7,802 |

The first filtered call took **36.48 ms**, including lazy postings creation and first-use
execution. With JIT warmed, a forced postings-cache miss took **18.47 ms median**
(14.50–21.00 ms across five samples). This O(N) cost applies again when changed metadata
invalidates the cache, not on every request. It excludes the mutation operation itself.

Chapter-sized subsets perform well here. Do not extrapolate fixed-John throughput to
requests that switch books: varying subsets also change cache locality and this mixed
workload was materially slower. Large exact subsets can be slower than global ANN;
the current endpoint deliberately prefers exhaustive subset rankings over approximate
filtered traversal. These are single-machine, warmed steady-state measurements, not
end-to-end text/embedding latency or a service latency SLA.

All rounds, parameters, cold-call and rebuild samples are retained in
[the raw filtered results](filtered_results_2026-09-05.json).

## Measured results

Apple Silicon, 10 physical CPUs, 5,000 deterministic queries, five measured runs,
`M=16`, `efConstruction=200`, `efSearch=50`, `k=10`:

| Workload | Runtime | Build | Serial latency | 10-thread QPS | Recall@10 |
|----------|---------|-------|----------------|---------------|-----------|
| Hungarian → Hungarian | Clojure float32 | 9.86 s | 0.203 ms | 19,668 | 99.72% |
| Hungarian → Hungarian | Python `hnswlib` | 5.15 s | 0.359 ms | 16,169 | 99.84% |
| English → Hungarian | Clojure float32 | 10.34 s | 0.199 ms | 20,096 | 89.00% |
| English → Hungarian | Python `hnswlib` | 5.17 s | 0.345 ms | 16,023 | 88.82% |

The original Clojure snapshot stored normalized doubles, so ten concurrent searches
streamed roughly twice as much vector data as `hnswlib` and used only two double SIMD
lanes on this 128-bit Vector API runtime. Its public wrapper and raw Java kernel both
measured about 10.6k QPS, ruling out Clojure maps and executor scheduling as the main
bottleneck. The explicit float32 snapshot halves vector traffic, uses four SIMD lanes,
and reaches 22–25% higher 10-thread throughput than `hnswlib` at comparable recall.
The multilingual MPNet vectors for both complete translations took 4 minutes 33
seconds to generate on MPS.

## `efSearch`: latency versus accuracy

Reducing `efSearch` reduces the number of candidates explored and therefore can
reduce recall. It is configurable, but the library does not silently lower it.
Measured on one fixed English → Hungarian Clojure graph:

| `EF_SEARCH` | Serial latency | QPS | Exact verse recall@10 |
|-------------|----------------|-----|-----------------------|
| 10 | 0.123 ms | 8,121 | 82.80% |
| 16 | 0.153 ms | 6,548 | 85.70% |
| 20 | 0.173 ms | 5,779 | 86.95% |
| 30 | 0.231 ms | 4,333 | 88.40% |
| 40 | 0.285 ms | 3,507 | 89.10% |
| 50 | 0.338 ms | 2,955 | 89.80% |
| 80 | 0.492 ms | 2,032 | 90.20% |
| 100 | 0.593 ms | 1,687 | 90.50% |

Keep `EF_SEARCH=50` as the balanced default. For higher recall, try 80–100; for
latency-sensitive approximate retrieval, test 20–40 against your own relevance set.

```bash
EF_SEARCH=80 ./examples/bible_benchmark/run_clojure.sh
EF_SEARCH=80 ./examples/bible_benchmark/run_python.sh
```

The MPNet files are already float32. The benchmark's float32 compact snapshot therefore
matches their source precision, while the mutable HNSW graph remains double precision.
The library default is still `:float64`; float32 is explicit outside this embedding
benchmark because very close double-valued candidates can change order.

## Construction precision benchmark

The optional `:construction-precision :float32` now uses cached float vectors during
cosine graph construction as well. It leaves `M`, `efConstruction`, `efSearch`, and
the double source vectors unchanged. The default construction precision is still
float64. Use the following command to measure the two modes in alternating order:

```bash
NUM_VECTORS=10000 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  ./examples/bible_benchmark/run_construction.sh

NUM_VECTORS=31173 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  ./examples/bible_benchmark/run_construction.sh
```

The launcher requires the shared embeddings and enables the Java Vector API.
Each mode warms up on 2,000 vectors, then builds fresh graphs for each measured
run. It reports graph construction, compact snapshot creation, total time, and
exact ANN recall@10 against exhaustive cosine search. Data loading, embedding
generation, ground-truth computation, and explicit GC are outside build timing.
`BUILD_PRECISIONS=float64` selects only the default path; `THREADS=1` measures
serial construction. The compact search representation is float32 in both modes.

Measured September 5, 2026, Apple Silicon / Java 21.0.8, 768 dimensions,
10 threads, `M=16`, `efConstruction=200`, `efSearch=50`, 1,000 English queries,
five builds per mode (medians):

| Hungarian vectors | Construction precision | Graph construction | Snapshot | Total | ANN recall@10 |
| --- | --- | --- | --- | --- | --- |
| 10,000 | float64 | 1.670 s | 0.070 s | 1.736 s | 98.52% |
| 10,000 | float32 | 1.094 s | 0.074 s | 1.169 s | 98.47% |
| 31,126 | float64 | 9.052 s | 0.237 s | 9.285 s | 97.75% |
| 31,126 | float32 | 6.079 s | 0.259 s | 6.338 s | 97.71% |

Column medians need not sum to the median total. Total build time falls by 32.7%
and 31.7%, respectively. Parallel insertion order varies, so recall varies slightly
between builds; all individual samples and embedding SHA-256 hashes are stored in
[`construction_results_2026-09-05.edn`](construction_results_2026-09-05.edn).

These 10k vectors are a subset of the real Bible embeddings, not the older
synthetic 10k dataset. These figures compare the two Clojure construction modes;
the corresponding Python measurements are below. The earlier aligned-verse hit rate is a
different quality metric and must not be compared directly with this ANN recall.

The cache adds `N × D × 4` bytes while retaining normalized double source vectors:
about 91.2 MiB for all 31,126 verses, plus array headers. This option accelerates
construction at the cost of memory; it does not halve total index memory.
It currently supports the built-in cosine metric only. Scalar fallback works
without the Vector API, but the measured speedup requires the compiled SIMD kernel.

To use it in the existing search benchmark:

```bash
CONSTRUCTION_PRECISION=float32 NUM_QUERIES=5000 THREADS=10 \
  ./examples/bible_benchmark/run_clojure.sh
```

## Matching Python construction benchmark

```bash
NUM_VECTORS=10000 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  bash examples/bible_benchmark/run_construction_python.sh
NUM_VECTORS=31126 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  bash examples/bible_benchmark/run_construction_python.sh
```

The launcher creates a virtual environment and installs NumPy/hnswlib if needed.
`HNSW_PYTHON_VENV` overrides its location. It requires the same prepared embeddings,
uses the same deterministic English query selection, and computes exhaustive
float64 cosine top-10 ground truth outside the measured region. Tiny rounding/tie
differences between NumPy and JVM arithmetic are possible. BLAS is limited to one
thread; hnswlib explicitly uses the requested 10 threads. Each build starts fresh.

Measured Python 3.13.5, hnswlib 0.8.0, NumPy 2.5.2, September 5, 2026:

| Vectors | Build median | ANN recall@10 | Serial query | 10-thread QPS |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.239 s | 98.59% | 0.274 ms | 20,899 |
| 31,126 | 5.601 s | 98.09% | 0.368 ms | 15,616 |

[Individual samples](python_results_2026-09-05.json) retain the configuration and
versions. Python was measured after the earlier Clojure trials, not interleaved.
The Clojure benchmarks intentionally measure the internal HNSW kernel, including
snapshot creation; they exclude the new managed API's copying, validation and locking.
Float32 Clojure total time is 5.7% lower at 10k and 13.2% higher at full size in these
samples. Recall differs slightly; this is not a quality-matched speed guarantee.

## Recall-constrained tuning

```bash
NUM_VECTORS=10000 THREADS=10 BUILD_RUNS=3 QUALITY_QUERIES=1000 \
  bash examples/bible_benchmark/run_tuning.sh
```

This retains `hnsw.api/tune` and the float32/float64 HNSW builder. The default grid
varies `efConstruction=100,200` and `efSearch=50,80,120`. Override
`BUILD_CONFIGS_EDN` to vary `M` or construction precision, for example:

```bash
BUILD_CONFIGS_EDN='[{:M 16 :construction-precision :float32} {:M 24 :construction-precision :float32}]' \
  bash examples/bible_benchmark/run_tuning.sh
```

Identical query vectors are deduplicated in original selection order, then split
50/50 into disjoint calibration and held-out sets. The tuner recommends options only
if all calibration builds and one fresh held-out build meet the recall target.
Defaults remain unchanged; recall on future queries is not guaranteed.

## Fixed tuned configuration: Python vs Clojure

```bash
NUM_VECTORS=10000 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  bash examples/bible_benchmark/run_tuned_python.sh
NUM_VECTORS=10000 THREADS=10 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  bash examples/bible_benchmark/run_tuned_clojure.sh
```

Repeat with `NUM_VECTORS=31126` for the complete nonempty Hungarian corpus.
Both launchers default to **M=16, efConstruction=100, efSearch=80, k=10, seed=42**.
Clojure explicitly uses float32 construction and a float32 compact search snapshot.
No further tuning takes place in this comparison.

Both use the same prepared 768D embeddings and the same deterministic English query
selection. After deduplication, only the second half (held-out queries) is evaluated:
497 queries for each dataset size. Their emitted `query_sha256` hashes must match.
Exhaustive ground truth uses float64 cosine and lexicographic string-ID tie breaking
in both languages; small arithmetic differences are still possible.

Build results are five fresh-build samples and their median; they exclude loading,
embedding inference, ground-truth computation, warmup and explicit GC. Clojure timing
includes public API validation/copying and compact snapshot creation. Python timing
includes index initialization and `add_items`. Both warm up on 2,000 vectors.

Search timing uses the last measured graph, with five repetitions of serial queries
and five repetitions of 10-thread batches. Clojure uses the managed API (including
its per-batch executor overhead); Python uses `hnswlib.knn_query`. Recall is evaluated
for every build. These are equal-parameter, not exactly equal-recall measurements.

### Measured results (September 5, 2026)

Python 3.13.5 / hnswlib 0.8.0 / NumPy 2.5.2; Clojure on Java 21.0.8, macOS ARM64.
Runs were sequential, not interleaved across runtimes. The complete samples,
parameters and matching query hashes are in
[`tuned_results_2026-09-05.json`](tuned_results_2026-09-05.json).

| Vectors | Runtime | Build median | Serial query | 10-thread QPS | Median recall@10 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Python hnswlib | 0.663 s | 0.319 ms | 18,537 | 99.22% |
| 10,000 | Clojure | 0.898 s | 0.184 ms | 15,417 | 99.01% |
| 31,126 | Python hnswlib | 2.856 s | 0.446 ms | 12,710 | 98.71% |
| 31,126 | Clojure | 3.935 s | 0.300 ms | 11,641 | 98.71% |

In these runs, Clojure construction took **1.35× / 1.38×** Python time, while serial
queries took less time. Python had higher batch throughput. The Clojure public batch
API creates an executor per call; the QPS values include that overhead and represent
497-query batches, not a long-running server's sustained throughput. Embedding model
inference is excluded. The older benchmark sections use different parameters/query
sets and must not be mixed into this comparison.

### Why did the earlier Python comparison look better?

At 10k vectors the older construction-only comparison measured **1.169 s Clojure
versus 1.239 s Python**, so Clojure was 5.7% faster in those samples. The tuned
comparison measured **0.898 s versus 0.663 s**: both became faster in absolute time,
but Python improved proportionally more. At full size, the corresponding ratios
changed from 1.13x to 1.38x Clojure/Python time.

These are not a controlled code-regression experiment: efConstruction changed from
200 to 100, and the newer Clojure measurement includes managed API input copying
and validation. Search also changed efSearch (50 to 80), query selection, and batch
size. Timing variance and non-interleaved runtime trials add uncertainty. A proper
regression claim needs the same benchmark, options, data, API boundary and hardware
conditions on both revisions.

## API preprocessing optimization

```bash
NUM_VECTORS=10000 BUILD_RUNS=5 QUALITY_QUERIES=1000 \
  bash examples/bible_benchmark/run_api_overhead.sh
```

The example freezes the old validation, entry preparation and metadata-wrapper
functions in `api_overhead_baseline.clj`. It temporarily rebinds only those helpers
to compare old/new preprocessing **in one JVM with alternating trial order**. Run
it as a standalone benchmark, not inside a service handling other requests.
Construction, snapshot storage, distance arithmetic, M and ef parameters are unchanged.

The new path clones primitive doubles directly, widens primitive floats without
boxed intermediates, and checks finite coordinates and the accumulated norm in one
primitive loop. A transient entry vector and a batch-local ID set replace repeated
persistent collection updates. Existing IDs are checked directly in the metadata map;
single-batch addition no longer copies the full existing ID set. Metadata-free query
results avoid an extra result-vector allocation. Validation and defensive copies
are not skipped.

Preparation timings include all input copies, validation and duplicate checks.
Caller-thread allocation counters cover **preprocessing only**, not graph worker
allocations or peak/live index memory. The incremental-preparation measurement
averages 100 single-item validations against an existing N-ID map; it excludes
actual insertion and snapshot refresh and must not be advertised as insert latency.

Full-build timings include the whole managed constructor, with preparation and
graph-plus-snapshot components recorded separately. Both modes warm up on the full
input for preparation and 2,000 vectors for construction; five fresh builds/mode
are timed by default. Parallel graph construction remains nondeterministic.

Queries use the same 497-query held-out split and SHA-256 convention as the tuned
Python comparison. Old/new API queries run against **one identical graph**, with
result IDs, ordering, distances and metadata checked for equality after every timed
serial and parallel trial. Exact ANN recall is computed outside timing for that
shared graph, not separately for every build sample. Batch QPS still includes
per-call executor creation; executor reuse is not part of this optimization.

Loading, embeddings, warmup, exact ground truth and explicit GC are excluded.
The report retains individual timing/allocation samples and median summaries.

### Measured API results (September 5, 2026)

Java 21.0.8, macOS ARM64, 768D Bible vectors; M=16, efConstruction=100,
efSearch=80, k=10, 10 threads, five samples/mode:

| Initial vectors | Measurement | Old API helpers | Optimized API helpers |
| ---: | --- | ---: | ---: |
| 10,000 | Input preparation | 171.05 ms | 8.67 ms |
| 10,000 | Preparation allocation | 1,545.5 MB | 64.8 MB |
| 10,000 | Full index construction | 0.930 s | 0.684 s |
| 10,000 | Serial query | 0.174 ms | 0.151 ms |
| 10,000 | 10-thread QPS | 17,906 | 24,352 |
| 31,126 | Input preparation | 519.69 ms | 23.30 ms |
| 31,126 | Preparation allocation | 4,811.6 MB | 199.9 MB |
| 31,126 | Full index construction | 4.140 s | 3.567 s |
| 31,126 | Serial query | 0.301 ms | 0.268 ms |
| 31,126 | 10-thread QPS | 11,222 | 14,164 |

MB here means 1,000,000 bytes **allocated during preparation**, not peak resident
memory. Preparation is 19.7x/22.3x faster and allocates 95.8% less. Full build time
falls by 26.5%/13.8%; batch throughput rises by 36.0%/26.2%. Component medians and
separately measured preprocessing microbenchmarks need not sum to total-build medians.
Graph work remains dominant after optimization and parallel timings still vary.

Single-item preparation against existing 10k/31k-ID maps fell from 0.791/2.956 ms
to about 0.0045 ms in both cases. This is only duplicate checking, validation and
copying; actual insertion and the global snapshot refresh still cost extra.

Fresh Python 3.13.5 / hnswlib 0.8.0 / NumPy 2.5.2 measurements use the same tuned
launcher, five builds and the exact same 497 held-out queries (SHA-256 verified):

| Vectors | Optimized Clojure build | Python API build | Clojure/Python time |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.684 s | 0.678 s | 1.01x |
| 31,126 | 3.567 s | 2.880 s | 1.24x |

The 10k difference is less than 1%: treat it as a practical tie, not evidence of a
significant performance difference. Python was run **after** the paired Clojure
experiment, not interleaved across languages. Both public APIs are timed, but their
validation, storage and result representations are not identical.

Old/new Clojure query outputs were identical on the shared graph for every measured
trial. That graph's ANN recall@10 was 99.07%/98.67% at 10k/full size; Python's median
across five fresh builds was 99.09%/98.71%. This is equal-parameter, not strictly
equal-recall benchmarking. No ef parameter, construction precision or distance
kernel was changed by this API optimization.

Raw timing/allocation samples and fresh Python results are saved in
[`api_overhead_results_2026-09-05.json`](api_overhead_results_2026-09-05.json).

## Local deletion/update versus full rebuilding

```bash
NUM_VECTORS=10000 BUILD_RUNS=3 CHURN_CYCLES=3 QUALITY_QUERIES=200 \
  bash examples/bible_benchmark/run_maintenance.sh
```

This benchmark defaults to float32 construction/snapshot, M=16, efConstruction=100,
efSearch=80, k=10, and 10 build threads. It clones one initial topology into local
and legacy-full-rebuild indexes outside timing, alternates their measurement order
between repetitions, then performs three cycles of deleting 1% of the original
corpus and moving one surviving vector. `DELETE_COUNT` overrides batch size.
The local vector update retains its level/ordinal; the legacy path recreates the
whole graph. Only the first local delete includes reverse-edge cache initialization.

JSON output separates mutation time, compact snapshot time, and total API time.
Exact cosine ground truth is recomputed after each mutation outside timing; both
indexes are evaluated against the same surviving/updated corpus and queries.
Embeddings, clone preparation, ground truth and explicit GC are not timed.
The legacy rebuild uses the previous implementation's internal build boundary,
not the public constructor's extra validation/copying overhead.

This measures physical deletion plus graph repair, **not** Python hnswlib's
deletion-mark API. Compact snapshots still refresh globally; see
[maintenance design and limitations](../../docs/LOCAL_MAINTENANCE.md).

### Measured local-maintenance results (September 5, 2026)

Java 21.0.8, macOS ARM64, three repetitions from the same initial topology.
First-cycle median total times, including snapshot refresh and, for the first
local delete, reverse-edge cache initialization:

| Initial vectors | Operation | Previous full rebuild | Local repair | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | Delete 100 vectors | 0.743 s | 0.256 s | 2.9x |
| 10,000 | Update one vector | 0.764 s | 0.075 s | 10.2x |
| 31,126 | Delete 311 vectors | 5.670 s | 1.682 s | 3.4x |
| 31,126 | Update one vector | 6.448 s | 0.449 s | 14.4x |

In the second/third cycles, 10k deletion took 0.181/0.188 s (4.4x/4.0x faster
than rebuilding). At full size it took 1.210/1.613 s (5.1x/3.7x faster).
Full-size single-vector graph mutation itself took only 2–4 ms; the global
snapshot refresh accounts for almost all of the total update latency.

Exact ANN recall@10 uses 198 deduplicated English queries at 10k and 200 at full
size, not the earlier tuner's 497-query held-out split. After three cycles, 300/933
vectors have been removed and one surviving vector has moved three times:

| Initial vectors | Initial recall | Final local recall | Final rebuild median recall |
| ---: | ---: | ---: | ---: |
| 10,000 | 98.59% | 98.48% | 98.64% |
| 31,126 | 98.40% | 98.10% | 98.45% |

The final local/rebuild gap is 0.15/0.35 percentage points. Local repair is faster,
but **does not guarantee unchanged recall** under continued churn. Ground truth
changes with the active corpus, so initial/final recall is not evaluated against
an identical target set. All timings, every cycle's quality, configuration and
methodology are in [`maintenance_results_2026-09-05.json`](maintenance_results_2026-09-05.json).
Timing component medians need not sum to the median total. These measurements
include two live comparison graphs plus a boxed topology clone in memory and must
not be substituted for the independent index-construction benchmark timings.
