#!/usr/bin/env python3
"""
Generate embeddings for Bible verses using sentence-transformers
"""

import json
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import time

def load_bible_verses(filepath, limit=None):
    """Load Bible verses from TSV file"""
    verses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            parts = line.strip().split('\t')
            if len(parts) == 4:
                verses.append({
                    'id': f"{parts[0]}_{parts[1]}:{parts[2]}",
                    'book': parts[0],
                    'chapter': int(parts[1]),
                    'verse': int(parts[2]),
                    'text': parts[3]
                })
    return verses

def generate_embeddings(verses, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
    """Generate embeddings using sentence-transformers"""
    print(f"🚀 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"📊 Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    texts = [v['text'] for v in verses]
    
    print(f"⏳ Generating embeddings for {len(texts)} verses...")
    start = time.time()
    
    # Generate embeddings in batches
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    elapsed = time.time() - start
    print(f"✅ Generated {len(embeddings)} embeddings in {elapsed:.1f} seconds")
    
    return embeddings

def save_embeddings(verses, embeddings, output_path):
    """Save verses and embeddings to JSON file"""
    data = {
        'metadata': {
            'num_verses': len(verses),
            'embedding_dim': len(embeddings[0]) if len(embeddings) > 0 else 0,
            'model': 'paraphrase-multilingual-mpnet-base-v2'
        },
        'verses': []
    }
    
    for verse, embedding in zip(verses, embeddings):
        data['verses'].append({
            'id': verse['id'],
            'book': verse['book'],
            'chapter': verse['chapter'],
            'verse': verse['verse'],
            'text': verse['text'],
            'embedding': embedding.tolist()
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to {output_path}")

def main():
    # Configuration
    bible_path = "data/KaroliRevid_m.tsv"
    output_path = "data/bible_embeddings.json"
    
    # First test with 100 verses
    limit = 100 if len(sys.argv) <= 1 else int(sys.argv[1])
    
    print("=" * 60)
    print("🔯 Bible Embedding Generator")
    print("=" * 60)
    
    # Load verses
    print(f"\n📖 Loading Bible verses (limit: {limit})...")
    verses = load_bible_verses(bible_path, limit)
    print(f"✅ Loaded {len(verses)} verses")
    
    # Show sample
    if verses:
        print(f"\nFirst verse: {verses[0]['text'][:80]}...")
        print(f"Last verse: {verses[-1]['text'][:80]}...")
    
    # Generate embeddings
    print("\n🧮 Generating embeddings...")
    embeddings = generate_embeddings(verses)
    
    # Save results
    print(f"\n💾 Saving embeddings...")
    save_embeddings(verses, embeddings, output_path)
    
    # Statistics
    print("\n📊 Statistics:")
    print(f"  - Verses processed: {len(verses)}")
    print(f"  - Embedding dimension: {len(embeddings[0])}")
    print(f"  - Output file size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    print("\n✨ Done! You can now load these embeddings in Clojure.")

if __name__ == "__main__":
    main()
