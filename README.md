# HNSW-CLJ: Vector Search in Clojure 

### **Status: Pre-alpha (early WIP)**
v0.1.1

Fast HNSW search in Clojure with no third-party dependencies, delivering sub-millisecond results via SIMD- and BLAS-optimized distance core. Achieves 4–16× speedup. The strength of this solution is that it targets the main runtime bottleneck, leverages data-level parallelism, minimizes overhead, and aligns computation with the hardware’s strengths—significantly reducing query latency and increasing throughput.

Backround: About 3–4 years ago I began exploring how to implement HNSW in Clojure without third-party dependencies and with full customizability—because AI keeps changing, and so do the needs. 

<br> 

> **Why accelerating distance computation matters**
> - In HNSW, 70–95% of query time is spent computing distances/inner products.
> - SIMD + Java Vector API: 4–16× fewer instructions, fewer cache misses, lower latency.
> - JBLAS (BLAS kernels): Near-peak CPU throughput with SIMD + multithreading for large batches.
> - Result: Higher QPS, lower P95/P99 latency.

<br> 

## ⚡ Performance Results

### Corrected HNSW construction (August 28, 2026)

The production `hnsw.ultra-fast` builder now follows the insertion flow from the
[original HNSW paper](https://arxiv.org/abs/1603.09320), cross-checked against
[hnswlib](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h) and
[Faiss](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp):

- Level multiplier is `1 / log(M)`, instead of the previous `1 / log(2)`.
- Upper layers use greedy descent before `ef-construction` search on insertion layers.
- New nodes select at most `M` diverse neighbors; `2M` is only the level-0 reciprocal capacity.
- Overflow pruning uses the HNSW diversity heuristic and skips provably dominated reciprocal candidates.
- Cosine indexes normalize stored vectors once and use an equivalent dot-product distance internally.
- The `:seed` option now makes level generation reproducible.
- Batch construction uses one shared graph, lock-free concurrent neighbor reads, per-node
  mutation locks, and atomic entry-point promotion. Level generation remains deterministic;
  parallel edge insertion order is intentionally schedule-dependent.
- Workers claim small dynamic batches to balance expensive insertions. The defaults are all
  available processors and `:batch-size 4`; use `:num-threads 1` for serial construction.

Measured locally with `M=16`, `ef-construction=200`, seed `42`:

| Dataset | Previous builder | Corrected, 1 thread | Corrected, 10 threads | Parallel speedup |
|---------|------------------|---------------------|----------------------|------------------|
| 4,000 × 64 | 9.56 s | 1.53 s | — | — |
| 10,000 × 768 cosine | 69 s documented baseline | 26.84 s | 4.49 s | 5.98× |

On the same 10,000 × 768 benchmark, Python `hnswlib` took 19.65 seconds with one
thread and 3.35 seconds with 10 threads. The Clojure implementation is therefore
1.37× slower serially and 1.34× slower in parallel, while retaining JVM-native data
and a single shared HNSW graph.

```clojure
(ultra/build-index vectors
                   :M 16
                   :ef-construction 200
                   :num-threads 10
                   :batch-size 4)
```

On the deterministic 1,000 × 64 recall check, mean recall@10 is 99.5% and all
per-layer degree limits are preserved. The remaining build hotspot is the vector
distance kernel (81% of sampled CPU time for 768-dimensional vectors).

### Current full-Bible search comparison (August 28, 2026)

The reproducible [bilingual Bible benchmark](examples/bible_benchmark/README.md)
indexes 31,126 non-empty Hungarian verses and queries either Hungarian or English
MPNet-v2 embeddings with identical `M=16`, `efConstruction=200`, `efSearch=50`,
and `k=10` settings:

| Query → index | Runtime | Serial latency | 10-thread QPS | Recall@10 |
|---------------|---------|----------------|---------------|-----------|
| Hungarian → Hungarian | Clojure float32 | 0.203 ms | 19,668 | 99.72% |
| Hungarian → Hungarian | Python `hnswlib` | 0.359 ms | 16,169 | 99.84% |
| English → Hungarian | Clojure float32 | 0.199 ms | 20,096 | 89.00% |
| English → Hungarian | Python `hnswlib` | 0.345 ms | 16,023 | 88.82% |

These rows use 5,000 deterministic queries and five measured runs. At equal
`efSearch=50`, the float32 Clojure search path is 42–44% faster serially and 22–25%
faster at 10-thread throughput. The previous float64 snapshot saturated memory
bandwidth: it occupied twice the vector storage and used two SIMD lanes instead of
four on this 128-bit Vector API runtime. The compact float32 snapshot fixes both
limits while keeping the mutable concurrent graph as the double-precision source of
truth for construction and updates.

Lowering `efSearch` is not a lossless optimization. On the English → Hungarian
workload, one fixed float64 Clojure graph produced this speed/quality curve:

| `efSearch` | Serial latency | Recall@10 |
|------------|----------------|-----------|
| 10 | 0.123 ms | 82.80% |
| 20 | 0.173 ms | 86.95% |
| 30 | 0.231 ms | 88.40% |
| 50 (default) | 0.338 ms | 89.80% |
| 80 | 0.492 ms | 90.20% |
| 100 | 0.593 ms | 90.50% |

The default therefore remains 50. Query-time overrides are supported for workloads
that explicitly prefer latency or recall. The example includes executable Clojure,
Python, and multilingual embedding preparation shell scripts.

The search algorithm still uses the original HNSW greedy upper-layer descent and
bounded `efSearch` base-layer expansion. The default `:float64` snapshot is lossless.
For float32 embedding models, `:compact-precision :float32` halves compact vector
storage and doubles SIMD lane width; it is explicit because arbitrary double inputs
can experience small tie-breaking changes. More aggressive compressed layouts such
as [LVQ](https://arxiv.org/abs/2304.04759) or the
[VSAG](https://www.vldb.org/pvldb/vol18/p5017-cheng.pdf) pipeline should remain
separate opt-in modes with their own recall benchmarks.

Compile and enable the optimized Java 21 Vector API path:

```bash
./scripts/compile_vector_kernel.sh
clojure -J--add-modules=jdk.incubator.vector -M
```

Query-time `efSearch` and ordered parallel batch search are public API options:

```clojure
(def index (hnsw.api/index vectors
                           :metric :cosine
                           :compact-precision :float32))
(hnsw.api/search index query 10 80)
(hnsw.api/search-batch index queries 10 :ef-search 50 :num-threads 10)
```

*I use the Old Hungarian Károli Bible for semantic similarity research, where pairing Hebrew and Greek meanings is a challenging task (it was). In this environment, it becomes immediately clear if something is not working. I tested on 31,173 verses — the full Károli Bible (mpnet-v2, 768-dim embeddings, 492.9 MB index):*

### 🏆 Historical Benchmark Results (August 15, 2025)

#### Multi-threaded Performance (Production)
- **Peak Performance:** **0.186 ms/query** (**5,376 QPS**) with 20 threads
- **Average (5 runs):** **0.195 ms/query** (**5,128 QPS**)
- **Consistency:** <5% variance across runs

#### Detailed Performance Metrics

| Configuration | QPS | Latency (ms) | vs Target | Status |
|--------------|-----|--------------|-----------|--------|
| **Best Run** | **5,376** | **0.186** | +14% | ✅ Exceeded |
| Run 2 | 5,236 | 0.191 | +10% | ✅ Exceeded |
| Run 3 | 4,773 | 0.209 | +1% | ✅ Met |
| **Average** | **5,128** | **0.195** | **+9%** | ✅ **Exceeded** |
| Original Target | 4,719 | 0.212 | - | Baseline |

#### Single-threaded Performance (Baseline)
- **Single query:** 1.093 ms/query (915 QPS)
- **Speedup:** 5.8x with 20 threads parallelization
- **vs Linear search:** 86x faster (single), 500x faster (parallel)

<br> 

### ⚡ SIMD + Java Vector API + JBLAS

| Approach                      | Best for                        | Key benefits                                                                 |
|--------------------------------|---------------------------------|------------------------------------------------------------------------------|
| 🧮 **Java Vector API (SIMD)** | Per-query, low-latency search    | 4–16× fewer instructions, branchless tight loops, cache-friendly arrays, no JNI overhead |
| 📦 **JBLAS (native BLAS)**    | Large batches, high throughput   | Optimized BLAS kernels (OpenBLAS/MKL), GEMM batching, multithreaded + SIMD   |

**Impact:**  
- 4–16× faster distance kernel (CPU/ISA dependent)  
- Lower P95/P99 latency, higher QPS
<br>
<br> 

⚠️ ⚠️  Planned: standalone dep after full functionality and examples. End of August. ⚠️ ⚠️ 

## 🗂️ Available ANN Algorithms in this branch 

### 1. **📐 Graph-based Algorithms** (Graph-based ANN)
Build navigable graph structures, search by hopping from node to node.

| Implementation | Namespace | Description | Build                      | Search | Recall |
|---------------|-----------|-------------|----------------------------|--------|--------|
| **Pure HNSW** | `hnsw.hnsw-search` | Original HNSW algorithm | extreme Slow (300s+) ** ⚠️ | 0.2 ms | 99%+ |
| **Ultra Fast HNSW** | `hnsw.ultra-fast` | Optimized HNSW | Medium                     | 0.3ms  | 98%+ |
| **Ultra Optimized** | `hnsw.ultra-optimized` | Max optimized | Medium                     | 0.2ms  | 98%+ |

** I work on this [WIP]

### 2. **🗃️ Partition-based Algorithms** (Partition-based ANN)
Divide data into partitions, search in parallel.

| Implementation | Namespace | Description | Build | Search | Recall |
|---------------|-----------|-------------|-------|--------|--------|
| **Partitioned HNSW** | `hnsw.partitioned-hnsw` | Multiple HNSW in parallel | 6-7s | 0.4-0.6ms | 91%    |
| **Lightning** | `hnsw.lightning` | Simple partitions | 61ms | 1.5ms | 29% ** |
| **IVF-FLAT** | `hnsw.ivf-flat` | K-means + flat search | 2-5s | 5-10ms | 95%+   |

** lightening parallel ~ like knn++ > 5-7 sec index, 7 - 1x ms very good recall.

### 3. **🔀 Hash-based Algorithms** (Hash-based ANN)
Group similar vectors using hash functions.

| Implementation | Namespace | Description | Build | Search | Recall |
|---------------|-----------|-------------|-------|--------|--------|
| **Hybrid LSH** | `hnsw.hybrid-lsh` | Multi-probe LSH | 0.5-1s | 2-5ms | 45% |

### 4. **🔄 Hybrid Algorithms** (Hybrid ANN)
Combine multiple techniques.

| Implementation | Namespace | Description | Build | Search | Recall |
|---------------|-----------|-------------|-------|--------|--------|
| **IVF-HNSW** | `hnsw.ivf-hnsw` | IVF + HNSW graph | 30-90s | 2-5ms | 85-90% |

### 5. **📉 Dimension Reduction Algorithms** (Dimension Reduction ANN)
First reduce dimensions, then search in reduced space. (my favorite)

| Implementation | Namespace | Description | Build | Search | Recall |
|---------------|-----------|-------------|-------|--------|--------|
| **P-HNSW (PCAF)** | `hnsw.pcaf` | Random Projection + HNSW | ~5s | 1-2ms | 85-90% |
<br> 

### Example Search Session

Automatically pick one verse → compare it with all other verses → return the top 5 most similar items.

```
🔍 Search> love 5

🎯 SEARCHING FOR: "love" (top 5 results)
============================================================
✅ Found 179 verses containing 'love'

📖 First match: [Gen 21:23]
   "Now therefore swear unto me here by God that thou wilt not deal falsely..."

🔗 Top 5 semantically similar verses:
------------------------------------------------------------
⏱️ Search completed in 10.12 ms

 1. [Josh 2:12] (84.0%)
    "Now therefore, I pray you, swear unto me by the LORD..."

 2. [Ps 71:18] (83.4%)
    "Now also when I am old and greyheaded, O God, forsake me not..."

 3. [2Sam 16:19] (81.2%)
    "And again, whom should I serve? should I not serve..."

 4. [1Kings 9:6] (81.0%)
    "But if ye shall at all turn from following me..."

 5. [Gen 27:29] (81.0%)
    "Let people serve thee, and nations bow down to thee..."

🔍 Search> beginning 10

🎯 SEARCHING FOR: "beginning" (top 10 results)
============================================================
✅ Found 8 verses containing 'beginning'

📖 First match: [Gen 1:1]
   "In the beginning God created the heaven and the earth."

⏱️ Search completed in 4.45 ms

 1. [Gen 2:4] (92.2%)
    "These are the generations of the heavens and of the earth..."
 2. [Ps 124:8] (81.2%)
    "Our help is in the name of the LORD, who made heaven and earth."
 3. [Heb 1:10] (80.8%)
    "And, Thou, Lord, in the beginning hast laid the foundation..."
```

## 📊 Features

### ✅ Implemented
- Ultra-optimized distance functions (SIMD via Java Vector API + JBLAS)
- Sub-millisecond search on 31K+ vectors
- Interactive search shell with colored output
- Save/load index to disk
- Multiple distance metrics (Cosine, Euclidean)
- Parallel search with linear scaling
- Performance benchmarking tools

### 🚧 TODO
- [ ] Delete vectors from index
- [ ] Filter functions (by metadata, categories)
- [ ] Update existing vectors
- [ ] Incremental index building
- [ ] REST API server
- [ ] Distributed index support
- [ ] GPU acceleration
- [ ] Memory-mapped indices




## 🛠️ Architecture

### Essential Files (Production)
```
src/hnsw/
├── ultra-optimized.clj   # Main HNSW implementation
├── ultra-fast.clj        # SIMD-optimized distance functions  
└── index-io.clj          # Save/load functionality
```

### Additional Files (Full Implementation)
```
src/hnsw/
├── core.clj              # Examples and demos
├── graph.clj             # Original HNSW algorithm
├── simd.clj              # SIMD support detection
├── simd-optimized.clj    # SIMD implementations
├── filtered.clj          # Filter support (WIP)
├── benchmark.clj         # Benchmarking utilities
└── api.clj               # Public API wrapper
```

### Scripts & Tests
```
scripts/
├── build_index_complete.clj  # Build 31K index
├── interactive_search.clj    # Interactive shell
└── test_*.clj                # Performance tests

final-test-bench/
└── hnsw-clj-our-solution/    # Production-ready demo
    ├── interactive_search_31k.clj
    ├── PERFORMANCE_ANALYSIS*.md
    └── run_interactive_31k.sh
```

## 🚀 Quick Start

```clojure
;; 1. Build index
(require '[hnsw.ultra-optimized :as opt])
(require '[hnsw.index-io :as io])

(def vectors [["id1" (double-array [0.1 0.2 ...])]
              ["id2" (double-array [0.4 0.5 ...])]])

(def index (opt/build-index vectors))
(io/save-index index "my-index.hnsw")

;; 2. Load and search
(def index (io/load-index "my-index.hnsw" opt/fast-cosine-distance))
(def results (opt/search index query-vector 10))
```

## 📈 Historical Benchmarks (before construction fix)

The table below is retained for historical search-performance context. Its build
times predate the corrected HNSW constructor; use the measurements above for current builds.

| Dataset | Vectors | Dims | Build Time | Search (k=10) | QPS | Single Thread |
|---------|---------|------|------------|---------------|-----|---------------|
| Small | 1,000 | 128 | ~7s* | 0.05 ms | 20,000 | (0.1 ms / 10,000 QPS) |
| Medium | 10,000 | 768 | ~69s* | 0.15 ms | 6,667 | (0.5 ms / 2,000 QPS) |
| **Bible** | **31,173** | **768** | **215s** | **0.212 ms** | **4,719** | **(1.199 ms / 834 QPS)** |
| Large | 100,000 | 768 | ~690s* | 0.50 ms | 2,000 | (2.5 ms / 400 QPS) |

*Performance with 20 parallel threads (single thread results in parentheses)*  
**Build times estimated based on Bible dataset (only Bible dataset values are measured)*

## 🔬 Python hnswlib vs Clojure HNSW-CLJ Comparison

### ⚠️ Parameter Differences
**Identical parameters:** M=16, ef_construction=200, ef_search=50, k=10, vectors=31,173, dimensions=768  
**Single difference:** Distance metric - Python (Cosine) vs Clojure (Euclidean)

### 🏆 Validated Performance Results

| Metric | Python hnswlib | Clojure HNSW-CLJ | Winner | Difference |
|--------|---------------|------------------|--------|-----------|
| **Build Time** | 25-30s | 215s | 🐍 Python | 8x faster |
| **Single Thread QPS** | 1,125 | 834 | 🐍 Python | 1.3x faster |
| **Multi Thread QPS (20)** | 1,330 | **4,719** | 🔧 Clojure | **3.5x faster** |
| **Average Latency** | 0.85ms | **0.212ms** | 🔧 Clojure | **4x better** |
| **Parallel Scaling** | 1.3x (GIL) | **5.7x** | 🔧 Clojure | True parallelism |

### 💡 Usage Recommendations

**Use Python hnswlib when:**
- Fast prototyping needed
- Building indices quickly (8x faster)
- Integration with Python ML ecosystem
- Single-threaded applications

**Use Clojure HNSW-CLJ when:**
- Production serving with high QPS requirements
- Real-time applications needing low latency
- High concurrency environments
- Scalable microservices

### 🎯 Optimal Hybrid Approach

```python
# 1. Build index with Python (fast: 25-30 sec)
import hnswlib
index = hnswlib.Index(space='cosine', dim=768)
index.init_index(max_elements=31173, ef_construction=200, M=16)
index.add_items(vectors)
index.save_index('index.bin')
```

```clojure
;; 2. Serve with Clojure (high performance: 4,700+ QPS)
(def index (load-index "index.bin"))
(serve-api index {:port 8080})
```

This combines Python's fast build time with Clojure's excellent serving performance!

*For detailed comparison, see [PARAMETER_COBENCHMARK_RESULTS_ACTUAL.md](BENCHMARK_RESULTS_ACTUAL.md) and [BENCHMARK_RESULTS_PYTHON.md](BENCHMARK_RESULTS_PYTHON.md)*

## 🔧 Requirements

- Clojure 1.11+
- Java 21+ (for Vector API)
- 3GB heap for 31K index

## 📝 License

MIT
