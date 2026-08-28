# Bilingual Bible HNSW benchmark

This example compares Clojure `hnsw-clj` and Python `hnswlib` using identical
HNSW parameters and the same precomputed float32 vectors.

HNSW indexes numeric vectors, not text. For English-to-Hungarian semantic search,
both translations must be embedded into one multilingual vector space. The default
model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.

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

Both commands default to an English-query to Hungarian-index benchmark. They report:

- parallel index build time;
- median serial request latency and QPS;
- median parallel batch throughput;
- exact aligned-verse recall@1, recall@5, and recall@10.

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
