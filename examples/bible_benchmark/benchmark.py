#!/usr/bin/env python3
"""Fair hnswlib search benchmark over the shared Bible embedding files."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from pathlib import Path
from statistics import median

import hnswlib
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", EXAMPLE_DIR / "data"))
INDEX_LANGUAGE = os.environ.get("INDEX_LANGUAGE", "hu")
QUERY_LANGUAGE = os.environ.get("QUERY_LANGUAGE", "en")
NUM_VECTORS = int(os.environ.get("NUM_VECTORS", "31173"))
DIMENSIONS = int(os.environ.get("DIMENSIONS", "768"))
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", "1000"))
RUNS = int(os.environ.get("RUNS", "5"))
THREADS = int(os.environ.get("THREADS", str(os.cpu_count() or 1)))
K = int(os.environ.get("K", "10"))
M = int(os.environ.get("M", "16"))
EF_CONSTRUCTION = int(os.environ.get("EF_CONSTRUCTION", "200"))
EF_SEARCH = int(os.environ.get("EF_SEARCH", "50"))
SEED = int(os.environ.get("SEED", "42"))


def load_shared_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    manifest_path = EMBEDDING_DIR / "manifest.json"
    if not manifest_path.exists():
        rng = np.random.default_rng(SEED)
        vectors = rng.random((NUM_VECTORS, DIMENSIONS), dtype=np.float32)
        query_count = min(NUM_QUERIES, NUM_VECTORS)
        indices = (np.arange(query_count, dtype=np.int64) * 7919) % NUM_VECTORS
        return vectors, vectors[indices].copy(), indices, "synthetic-shape", DIMENSIONS

    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    dimensions = int(manifest["dimensions"])
    languages = manifest["languages"]
    target = languages[INDEX_LANGUAGE]
    source = languages[QUERY_LANGUAGE]
    target_count = min(int(target["count"]), NUM_VECTORS)
    target_items = target["items"][:target_count]
    target_vectors = np.memmap(
        EMBEDDING_DIR / target["vectors"],
        dtype="<f4",
        mode="r",
        shape=(int(target["count"]), dimensions),
    )[:target_count]
    source_vectors = np.memmap(
        EMBEDDING_DIR / source["vectors"],
        dtype="<f4",
        mode="r",
        shape=(int(source["count"]), dimensions),
    )
    target_labels = {item["id"]: label for label, item in enumerate(target_items)}
    aligned = [
        (row, target_labels[item["id"]])
        for row, item in enumerate(source["items"])
        if item["id"] in target_labels
    ]
    if not aligned:
        raise RuntimeError("No canonical verse IDs overlap between the selected languages")
    query_count = min(NUM_QUERIES, len(aligned))
    selected = [(index * 7919) % len(aligned) for index in range(query_count)]
    source_rows = np.fromiter((aligned[index][0] for index in selected), dtype=np.int64)
    expected = np.fromiter((aligned[index][1] for index in selected), dtype=np.int64)
    queries = np.asarray(source_vectors[source_rows], dtype=np.float32, order="C")
    source_name = f"{QUERY_LANGUAGE}->{INDEX_LANGUAGE}:{manifest['model']}"
    return target_vectors, queries, expected, source_name, dimensions


def metrics(seconds: float, query_count: int) -> dict[str, float]:
    return {
        "seconds": seconds,
        "latency_ms": seconds * 1000.0 / query_count,
        "qps": query_count / seconds,
    }


def benchmark_serial(index: hnswlib.Index, queries: np.ndarray) -> dict[str, float]:
    started = time.perf_counter()
    checksum = 0
    for query in queries:
        labels, _ = index.knn_query(query, k=K, num_threads=1)
        checksum += int(labels.shape[1])
    result = metrics(time.perf_counter() - started, len(queries))
    result["checksum"] = checksum
    return result


def benchmark_parallel(index: hnswlib.Index, queries: np.ndarray) -> dict[str, float]:
    started = time.perf_counter()
    labels, _ = index.knn_query(queries, k=K, num_threads=THREADS)
    result = metrics(time.perf_counter() - started, len(queries))
    result["checksum"] = int(labels.size)
    return result


def summarize(mode: str, results: list[dict[str, float]]) -> dict[str, float]:
    middle = median(result["seconds"] for result in results)
    summary = metrics(middle, NUM_QUERIES if not results else int(results[0]["checksum"] / K))
    print(
        "RESULT runtime=python "
        f"mode={mode} threads={1 if mode == 'serial' else THREADS} "
        f"seconds={summary['seconds']:.6f} latency_ms={summary['latency_ms']:.6f} "
        f"qps={summary['qps']:.1f}"
    )
    return summary


def print_quality(index: hnswlib.Index, queries: np.ndarray, expected: np.ndarray) -> None:
    labels, _ = index.knn_query(queries, k=max(K, 10), num_threads=THREADS)
    recall_at_1 = np.mean(labels[:, :1] == expected[:, None])
    recall_at_5 = np.mean(np.any(labels[:, :5] == expected[:, None], axis=1))
    recall_at_10 = np.mean(np.any(labels[:, :10] == expected[:, None], axis=1))
    print(
        f"QUALITY exact_verse_recall_at_1={recall_at_1:.4f} "
        f"at_5={recall_at_5:.4f} at_10={recall_at_10:.4f}"
    )


def main() -> None:
    if min(NUM_VECTORS, NUM_QUERIES, RUNS, THREADS, K) < 1:
        raise ValueError("NUM_VECTORS, NUM_QUERIES, RUNS, THREADS and K must be positive")
    vectors, queries, expected, source_name, dimensions = load_shared_data()
    query_count = len(queries)
    print(f"Runtime: Python {platform.python_version()} / hnswlib")
    print(f"Dataset: {source_name}")
    print(
        f"Vectors={len(vectors):,} dimensions={dimensions} queries={query_count:,} "
        f"M={M} efConstruction={EF_CONSTRUCTION} efSearch={EF_SEARCH} k={K}"
    )

    labels = np.arange(len(vectors), dtype=np.int64)
    index = hnswlib.Index(space="cosine", dim=dimensions)
    index.init_index(
        max_elements=len(vectors),
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=SEED,
    )
    started = time.perf_counter()
    index.add_items(vectors, labels, num_threads=THREADS)
    build_seconds = time.perf_counter() - started
    index.set_ef(EF_SEARCH)
    print(f"BUILD runtime=python threads={THREADS} seconds={build_seconds:.6f}")

    warmup = queries[: min(200, query_count)]
    for query in warmup:
        index.knn_query(query, k=K, num_threads=1)
    index.knn_query(warmup, k=K, num_threads=THREADS)

    serial_results = [benchmark_serial(index, queries) for _ in range(RUNS)]
    parallel_results = [benchmark_parallel(index, queries) for _ in range(RUNS)]
    summarize("serial", serial_results)
    summarize("parallel", parallel_results)
    print_quality(index, queries, expected)
    del index, vectors, queries
    gc.collect()


if __name__ == "__main__":
    main()
