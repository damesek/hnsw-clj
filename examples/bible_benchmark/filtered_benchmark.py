#!/usr/bin/env python3
"""Python hnswlib callback filtering on the Clojure benchmark's chapter workload."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from statistics import median

import numpy as np

import benchmark as shared
from construction_benchmark import build


def search_filtered(index, query, criteria, postings, items):
    allowed = postings["book"].get(criteria["book"], frozenset()) & postings["chapter"].get(
        criteria["chapter"], frozenset()
    )
    if not allowed:
        return []
    labels, distances = index.knn_query(
        query, k=min(shared.K, len(allowed)), num_threads=1, filter=allowed.__contains__
    )
    return [
        {"id": items[int(label)]["id"], "distance": float(distance), "metadata": items[int(label)]}
        for label, distance in zip(labels[0], distances[0])
    ]


def main():
    repeats = int(os.environ.get("QUERY_REPEATS", "1"))
    if min(shared.NUM_VECTORS, shared.NUM_QUERIES, shared.RUNS, shared.THREADS, repeats) < 1 or shared.K != 10:
        raise ValueError("Positive counts and K=10 required")
    with (shared.EMBEDDING_DIR / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    vectors, queries, expected, source, dimensions = shared.load_shared_data()
    items = manifest["languages"][shared.INDEX_LANGUAGE]["items"][:len(vectors)]
    started = time.perf_counter()
    postings = {"book": {}, "chapter": {}}
    for label, item in enumerate(items):
        for key in postings:
            postings[key].setdefault(item[key], set()).add(label)
    postings_ms = (time.perf_counter() - started) * 1000.0
    criteria = [{"book": items[int(label)]["book"], "chapter": items[int(label)]["chapter"]}
                for label in expected]
    requests = list(zip(queries, criteria))
    del expected
    index = build(vectors)
    normalized = np.asarray(vectors, dtype=np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    subset_sizes = []
    hits = 0
    total_truth = 0
    for query, selected in requests:
        allowed = sorted(postings["book"][selected["book"]] & postings["chapter"][selected["chapter"]])
        subset_sizes.append(len(allowed))
        prepared = np.asarray(query, dtype=np.float64)
        prepared /= np.linalg.norm(prepared)
        distances = 1.0 - normalized[allowed] @ prepared
        identifiers = np.asarray([items[label]["id"] for label in allowed])
        order = np.lexsort((identifiers, distances))[:shared.K]
        truth = set(identifiers[order].tolist())
        neighbors = search_filtered(index, query, selected, postings, items)
        assert len(neighbors) == min(shared.K, len(allowed))
        assert all(all(neighbor["metadata"][key] == value for key, value in selected.items())
                   for neighbor in neighbors)
        hits += len(truth & {neighbor["id"] for neighbor in neighbors})
        total_truth += len(truth)
    samples = []
    for run in range(1, shared.RUNS + 1):
        started = time.perf_counter()
        checksum = 0
        for _ in range(repeats):
            for query, selected in requests:
                checksum += len(search_filtered(index, query, selected, postings, items))
        milliseconds = (time.perf_counter() - started) * 1000.0
        assert checksum == repeats * sum(min(shared.K, size) for size in subset_sizes)
        sample = {"run": run, "serial_ms": milliseconds / (len(queries) * repeats)}
        samples.append(sample)
        print(f"Run {run}: filtered ms/query={sample['serial_ms']:.6f}", file=sys.stderr, flush=True)
    print(json.dumps({
        "benchmark": "python-hnswlib-mixed-chapters", "python": platform.python_version(),
        "hnswlib": importlib.metadata.version("hnswlib"), "numpy": np.__version__,
        "architecture": platform.machine(), "source": source, "vectors": len(vectors),
        "dimensions": dimensions, "queries": len(queries), "runs": shared.RUNS,
        "query_repeats": repeats, "requests_per_trial": len(queries) * repeats,
        "M": shared.M, "ef_construction": shared.EF_CONSTRUCTION, "ef_search": shared.EF_SEARCH,
        "build_threads": shared.THREADS, "query_threads": 1, "k": shared.K,
        "query_sha256": hashlib.sha256(queries.astype("<f4", copy=False).tobytes()).hexdigest(),
        "eligible": {"min": min(subset_sizes), "max": max(subset_sizes),
                     "mean": sum(subset_sizes) / len(subset_sizes)},
        "median_serial_ms": median(sample["serial_ms"] for sample in samples),
        "filtered_recall": hits / total_truth, "postings_ms": postings_ms, "samples": samples,
        "method": "Same aligned English queries and query-specific book AND chapter as filtered_benchmark.clj. "
                  "Warm single-thread hnswlib callback ANN, efSearch=80 unless overridden. "
                  "Timed wrapper includes postings lookup/intersection, callback, native search and result metadata. "
                  "Loading, embeddings, construction, postings creation and exact subset validation excluded. "
                  "Five per-round average latency samples; not interleaved with Clojure. "
                  "Clojure uses exact float64 subset search, not this filtered ANN algorithm."
    }, indent=2))


if __name__ == "__main__":
    main()
