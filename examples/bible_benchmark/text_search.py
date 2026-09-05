#!/usr/bin/env python3
"""Interactive text-to-top-10 Bible search using Python hnswlib."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


EXAMPLE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", EXAMPLE_DIR / "data"))
INDEX_LANGUAGE = os.environ.get("INDEX_LANGUAGE", "hu")
NUM_VECTORS = int(os.environ.get("NUM_VECTORS", "31173"))
THREADS = int(os.environ.get("THREADS", str(os.cpu_count() or 1)))
TOP_K = int(os.environ.get("TOP_K", "10"))
M = int(os.environ.get("M", "16"))
EF_CONSTRUCTION = int(os.environ.get("EF_CONSTRUCTION", "200"))
EF_SEARCH = int(os.environ.get("EF_SEARCH", "50"))
SEED = int(os.environ.get("SEED", "42"))
DEVICE = os.environ.get("DEVICE") or None


def load_data() -> tuple[dict, dict, np.ndarray]:
    manifest_path = EMBEDDING_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Embedding manifest not found: {manifest_path}. "
            "Run run_prepare_embeddings.sh first."
        )
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    descriptor = manifest["languages"][INDEX_LANGUAGE]
    count = min(int(descriptor["count"]), NUM_VECTORS)
    dimensions = int(manifest["dimensions"])
    vectors = np.memmap(
        EMBEDDING_DIR / descriptor["vectors"],
        dtype="<f4",
        mode="r",
        shape=(int(descriptor["count"]), dimensions),
    )[:count]
    selected = dict(descriptor)
    selected["items"] = descriptor["items"][:count]
    selected["count"] = count
    return manifest, selected, vectors


def build_index(vectors: np.ndarray, dimensions: int) -> tuple[hnswlib.Index, float]:
    index = hnswlib.Index(space="cosine", dim=dimensions)
    index.init_index(
        max_elements=len(vectors),
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=SEED,
    )
    started = time.perf_counter()
    index.add_items(
        vectors,
        np.arange(len(vectors), dtype=np.int64),
        num_threads=THREADS,
    )
    index.set_ef(max(EF_SEARCH, TOP_K))
    return index, time.perf_counter() - started


def print_results(query: str, labels: np.ndarray, distances: np.ndarray, items: list[dict]) -> None:
    print(f'\nQuery: "{query}"')
    print("=" * 80)
    for rank, (label, distance) in enumerate(zip(labels, distances), start=1):
        item = items[int(label)]
        similarity = 1.0 - float(distance)
        print(
            f"{rank:2d}. {item['book']} {item['chapter']}:{item['verse']} "
            f"| similarity={similarity:.4f} distance={float(distance):.4f}"
        )
        print(f"    {item['text']}")
    print()


def search_once(
    model: SentenceTransformer,
    index: hnswlib.Index,
    items: list[dict],
    query: str,
) -> None:
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    labels, distances = index.knn_query(vector, k=min(TOP_K, len(items)), num_threads=1)
    print_results(query, labels[0], distances[0], items)


def interactive_search(
    model: SentenceTransformer,
    index: hnswlib.Index,
    items: list[dict],
) -> None:
    print("Írj be magyar vagy angol szöveget. Kilépés: :q")
    while True:
        try:
            query = input("Keresés> ").strip()
        except EOFError:
            print()
            return
        if query in {":q", "quit", "exit"}:
            return
        if query:
            search_once(model, index, items, query)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Text to search; omit for interactive mode")
    arguments = parser.parse_args()

    manifest, descriptor, vectors = load_data()
    dimensions = int(manifest["dimensions"])
    print(
        f"Loading {manifest['model']} on {DEVICE or 'auto'} and building "
        f"{descriptor['count']:,}-verse {INDEX_LANGUAGE} index..."
    )
    model = SentenceTransformer(manifest["model"], device=DEVICE)
    index, build_seconds = build_index(vectors, dimensions)
    print(
        f"Python hnswlib index ready in {build_seconds:.2f}s "
        f"(threads={THREADS}, efSearch={EF_SEARCH}, topK={TOP_K})"
    )

    query = " ".join(arguments.query).strip()
    if query:
        search_once(model, index, descriptor["items"], query)
    else:
        interactive_search(model, index, descriptor["items"])


if __name__ == "__main__":
    main()
