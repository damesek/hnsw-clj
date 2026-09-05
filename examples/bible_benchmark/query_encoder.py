#!/usr/bin/env python3
"""Persistent JSON-lines MPNet query encoder for the Clojure text search example."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


EXAMPLE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", EXAMPLE_DIR / "data"))
DEVICE = os.environ.get("DEVICE") or None


def main() -> None:
    with (EMBEDDING_DIR / "manifest.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    model = SentenceTransformer(manifest["model"], device=DEVICE)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                return
            text = str(request["text"]).strip()
            if not text:
                raise ValueError("text must not be empty")
            vector = model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            response = {"vector": vector.tolist()}
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
