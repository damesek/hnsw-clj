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
portable Clojure search path.

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

Apple Silicon, 10 logical CPUs, 2,000 deterministic queries, five measured runs,
`M=16`, `efConstruction=200`, `efSearch=50`, `k=10`:

| Workload | Runtime | Build | Serial latency | 10-thread QPS | Recall@10 |
|----------|---------|-------|----------------|---------------|-----------|
| Hungarian → Hungarian | Clojure | 9.88 s | 0.345 ms | 9,696 | 99.80% |
| Hungarian → Hungarian | Python `hnswlib` | 5.18 s | 0.363 ms | 15,765 | 99.80% |
| English → Hungarian | Clojure | 10.48 s | 0.334 ms | 9,565 | 89.75% |
| English → Hungarian | Python `hnswlib` | 5.16 s | 0.347 ms | 15,722 | 89.55% |

At equal `efSearch`, Clojure serial search is 4–5% faster in these runs. Python's
native float32 batch search remains about 1.6× faster with 10 threads. The multilingual
MPNet vectors for both complete translations took 4 minutes 33 seconds to generate
on MPS.

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

These optimizations do not quantize the MPNet embeddings: the compact Clojure search
snapshot stores normalized doubles and computes the same cosine distances as the
mutable graph. The measured accuracy change in the table comes from `efSearch`, not
from the primitive layout or SIMD kernel.
