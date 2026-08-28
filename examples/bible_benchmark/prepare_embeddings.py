#!/usr/bin/env python3
"""Create shared bilingual MPNet embeddings for the Python and Clojure benchmarks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXAMPLE_DIR.parent.parent
HUNGARIAN_JSON = Path(
    os.environ.get(
        "HUNGARIAN_JSON",
        ROOT_DIR.parent / "bibliam.hu/data/bsb-karoli/HunKar.json",
    )
)
ENGLISH_JSON = Path(
    os.environ.get(
        "ENGLISH_JSON",
        ROOT_DIR.parent / "bibliam.hu/data/bsb-karoli/BSB.json",
    )
)
OUTPUT_DIR = Path(os.environ.get("EMBEDDING_DIR", EXAMPLE_DIR / "data"))
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
DEVICE = os.environ.get("DEVICE") or None
FORCE = os.environ.get("FORCE", "0") == "1"


def canonical_id(book_index: int, chapter: int, verse: int) -> str:
    return f"b{book_index + 1:02d}-c{chapter:03d}-v{verse:03d}"


def load_translation(path: Path) -> tuple[str, list[dict]]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)

    items: list[dict] = []
    for book_index, book in enumerate(payload["books"]):
        for chapter in book["chapters"]:
            chapter_number = int(chapter["chapter"])
            for verse in chapter["verses"]:
                text = verse["text"].strip()
                if not text:
                    continue
                verse_number = int(verse["verse"])
                items.append(
                    {
                        "id": canonical_id(book_index, chapter_number, verse_number),
                        "book_index": book_index,
                        "book": book["name"],
                        "chapter": chapter_number,
                        "verse": verse_number,
                        "text": text,
                    }
                )
    return payload["translation"], items


def encode_language(
    model: SentenceTransformer, language: str, items: list[dict]
) -> tuple[str, int]:
    vector_name = f"{language}.f32"
    vector_path = OUTPUT_DIR / vector_name
    if vector_path.exists() and not FORCE:
        expected_dimensions = model.get_sentence_embedding_dimension()
        expected_size = len(items) * expected_dimensions * np.dtype("<f4").itemsize
        if vector_path.stat().st_size == expected_size:
            print(f"Reusing {vector_path} ({len(items):,} vectors)")
            return vector_name, expected_dimensions

    print(f"Encoding {language}: {len(items):,} verses with {MODEL_NAME}")
    vectors = model.encode(
        [item["text"] for item in items],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype="<f4", order="C")
    temporary_path = vector_path.with_suffix(".tmp")
    vectors.tofile(temporary_path)
    temporary_path.replace(vector_path)
    return vector_name, int(vectors.shape[1])


def main() -> None:
    for path in (HUNGARIAN_JSON, ENGLISH_JSON):
        if not path.exists():
            raise FileNotFoundError(f"Bible JSON not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hu_translation, hu_items = load_translation(HUNGARIAN_JSON)
    en_translation, en_items = load_translation(ENGLISH_JSON)
    common_ids = {item["id"] for item in hu_items} & {item["id"] for item in en_items}

    print(f"Hungarian non-empty verses: {len(hu_items):,}")
    print(f"English non-empty verses:   {len(en_items):,}")
    print(f"Canonical aligned verses:   {len(common_ids):,}")
    print(f"Loading model on device: {DEVICE or 'auto'}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    hu_vectors, hu_dimensions = encode_language(model, "hu", hu_items)
    en_vectors, en_dimensions = encode_language(model, "en", en_items)
    if hu_dimensions != en_dimensions:
        raise RuntimeError("The two embedding matrices have different dimensions")

    manifest = {
        "model": MODEL_NAME,
        "dimensions": hu_dimensions,
        "aligned_nonempty": len(common_ids),
        "languages": {
            "hu": {
                "translation": hu_translation,
                "count": len(hu_items),
                "vectors": hu_vectors,
                "items": hu_items,
            },
            "en": {
                "translation": en_translation,
                "count": len(en_items),
                "vectors": en_vectors,
                "items": en_items,
            },
        },
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    with temporary_manifest.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, separators=(",", ":"))
    temporary_manifest.replace(manifest_path)
    print(f"Wrote shared benchmark data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
