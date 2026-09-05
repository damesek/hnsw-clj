# Slack post: HNSW-CLJ vs Python hnswlib

Attachment: `hnsw_clj_vs_hnswlib_slack.png`

## Copy-ready short version

🚀 *How does HNSW-CLJ compare with Python hnswlib?*

In short: *it is already faster at search, while hnswlib still leads at index construction.*

We measured both implementations on the same full Hungarian Bible index: 31,126 verses, 768-dimensional MPNet vectors, and identical HNSW settings.

• 🔎 *10-thread search:* HNSW-CLJ handles *23.5% more queries per second* on average (19.9k vs 16.1k QPS).
• ⚡ *Single-query search:* HNSW-CLJ has *42.9% lower latency* on average (0.201 ms vs 0.352 ms).
• 🎯 *Accuracy:* effectively equal; the Recall@10 difference is no more than 0.18 percentage points in either workload.
• 🏗️ *Index construction:* Python hnswlib is still approximately *1.96× faster* (5.16 s vs 10.10 s).

One important technical detail: Python `hnswlib` is not pure Python—it calls an optimized native C++ engine. HNSW-CLJ provides a Clojure API, while its fast search path remains entirely inside the JVM using a custom Java Vector API/SIMD kernel, with no external C++ or JNI dependency.

💡 *What does this mean in practice?*

For read-heavy Clojure/JVM services, HNSW-CLJ is already highly competitive: it provides faster search without requiring a separate Python service or native C++ integration. If indexes must be rebuilt from scratch very frequently, hnswlib is currently the stronger choice. The next major optimization target is therefore a float32 index-construction path.

_Test setup: Apple Silicon, 10 physical CPUs, 5,000 deterministic queries, median of 5 measured runs, M=16, efConstruction=200, efSearch=50, k=10._

## One-sentence executive summary

*On the full Bible benchmark, HNSW-CLJ delivers 22–25% higher parallel search throughput with effectively identical accuracy, while index construction remains nearly twice as slow as native C++-backed Python hnswlib.*

## What changed since the previous version?

The latest version introduces reference-aligned HNSW construction, true shared-graph parallel indexing, and an optimized float32 SIMD search path. Compared with the previous version, the 10k × 768 index build dropped from 69 seconds to 4.49 seconds, while 10-thread full-Bible search throughput increased from roughly 10.6k to 20k QPS. These improvements preserve essentially the same recall, halve search-vector memory usage, and keep the complete search engine inside the JVM without external C++ or JNI dependencies.

## Mi változott az előző verzióhoz képest?

A legújabb verzió referenciaimplementációhoz igazított HNSW-indexépítést, valódi közös gráfon dolgozó párhuzamos indexelést és optimalizált float32 SIMD keresési útvonalat vezet be. Az előző verzióhoz képest a 10k × 768-as index építési ideje 69 másodpercről 4,49 másodpercre csökkent, miközben a teljes Biblián mért, 10 szálas keresési teljesítmény körülbelül 10,6k-ról 20k QPS-re nőtt. Mindez lényegében változatlan recall mellett felezi a float32 keresési vektorok memóriaigényét, és a teljes keresőmotort külső C++- vagy JNI-függőség nélkül a JVM-en belül tartja.

## Plain-English explanation

An HNSW index is like a highway network connecting texts by meaning. Index construction builds the roads; search follows the shortcuts to reach the nearest verses. HNSW-CLJ already travels this network faster, while hnswlib can still build the network faster.

## Important caveats

- This benchmark ran on one Apple Silicon machine and one specific dataset of 31,126 vectors with 768 dimensions; it is not a universal ranking.
- English → Hungarian Recall@10 measures retrieval of the aligned verse, so multilingual embedding quality also has a major influence.
- The HNSW-CLJ `float32` optimization currently accelerates the immutable search snapshot; the mutable graph and index-construction path still use double precision.
