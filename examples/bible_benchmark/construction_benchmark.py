#!/usr/bin/env python3
"""Repeated hnswlib construction with exhaustive float64 cosine ground truth."""

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from statistics import median

import hnswlib
import numpy as np

import benchmark as shared


def build(vectors):
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), M=shared.M,
                     ef_construction=shared.EF_CONSTRUCTION, random_seed=shared.SEED)
    index.add_items(vectors, np.arange(len(vectors)), num_threads=shared.THREADS)
    index.set_ef(shared.EF_SEARCH)
    return index


def main():
    if not (shared.EMBEDDING_DIR / "manifest.json").is_file():
        raise ValueError("Prepare the shared Bible embeddings first")
    shared.NUM_QUERIES = int(os.environ.get("QUALITY_QUERIES", "1000"))
    build_runs = int(os.environ.get("BUILD_RUNS", "5"))
    if min(shared.NUM_QUERIES, build_runs, shared.THREADS) < 1 or shared.K != 10:
        raise ValueError("Positive sizes and K=10 required")
    vectors, queries, _, source, dimensions = shared.load_shared_data()
    query_split = os.environ.get("QUERY_SPLIT", "all")
    if query_split not in {"all", "validation"}:
        raise ValueError("QUERY_SPLIT must be all or validation")
    selected_queries = len(queries)
    if query_split == "validation":
        _, positions = np.unique(queries, axis=0, return_index=True)
        queries = queries[np.sort(positions)]
        unique_queries = len(queries)
        if unique_queries < 2:
            raise ValueError("At least two distinct queries required for validation")
        queries = queries[unique_queries // 2:]
    else:
        unique_queries = None
    query_hash = hashlib.sha256(queries.astype("<f4", copy=False).tobytes()).hexdigest()
    normalized = np.asarray(vectors, dtype=np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-300)
    normalized_queries = np.asarray(queries, dtype=np.float64)
    normalized_queries /= np.maximum(
        np.linalg.norm(normalized_queries, axis=1, keepdims=True), 1e-300)
    ground_truth = []
    identifiers = np.asarray([str(position) for position in range(len(vectors))])
    for query in normalized_queries:
        distances = 1.0 - normalized @ query
        order = (np.lexsort((identifiers, distances)) if query_split == "validation"
                 else np.argsort(distances, kind="stable"))
        ground_truth.append(set(order[:10].tolist()))
    del normalized, normalized_queries
    warmup = build(vectors[:min(2000, len(vectors))])
    del warmup
    samples = []
    for run in range(1, build_runs + 1):
        gc.collect()
        started = time.perf_counter()
        index = build(vectors)
        seconds = time.perf_counter() - started
        labels, _ = index.knn_query(queries, k=10, num_threads=shared.THREADS)
        recall = sum(len(set(row.tolist()) & truth)
                     for row, truth in zip(labels, ground_truth)) / (10 * len(queries))
        samples.append({"run": run, "total_seconds": seconds, "ann_recall_at_10": recall})
        if run == build_runs:
            for query in queries[:200]:
                index.knn_query(query, k=10, num_threads=1)
            serial = [shared.benchmark_serial(index, queries) for _ in range(5)]
            parallel = [shared.benchmark_parallel(index, queries) for _ in range(5)]
        del index
    print(json.dumps({
        "runtime": "python", "python": platform.python_version(),
        "numpy": np.__version__, "seed": shared.SEED,
        "query_split": query_split, "selected_queries": selected_queries,
        "unique_queries": unique_queries, "query_sha256": query_hash,
        "ground_truth_tie_break": "lexicographic-id" if query_split == "validation" else "numeric-id",
        "hnswlib": importlib.metadata.version("hnswlib"), "source": source,
        "vectors": len(vectors), "dimensions": dimensions, "queries": len(queries),
        "threads": shared.THREADS, "M": shared.M, "ef_construction": shared.EF_CONSTRUCTION,
        "ef_search": shared.EF_SEARCH, "k": 10, "build_runs": build_runs,
        "samples": samples,
        "median_total_seconds": median(row["total_seconds"] for row in samples),
        "median_ann_recall_at_10": median(row["ann_recall_at_10"] for row in samples),
        "serial_latency_ms": median(row["latency_ms"] for row in serial),
        "parallel_qps": median(row["qps"] for row in parallel),
    }, indent=2))


if __name__ == "__main__":
    main()
