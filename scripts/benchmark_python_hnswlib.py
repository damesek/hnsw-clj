#!/usr/bin/env python3
"""
HNSW-lib Python Benchmark - Comparison with Clojure implementation
Tests single-thread and multi-thread (limited by GIL) performance
"""

import hnswlib
import numpy as np
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import threading
import os

# Configuration matching Clojure benchmark
NUM_VECTORS = 31173
DIMENSIONS = 768
NUM_QUERIES = 100
K = 10
THREAD_COUNTS = [1, 5, 10, 20, 50]

# HNSW parameters (same as Clojure)
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50

def generate_random_vectors(n, d):
    """Generate random vectors matching Clojure benchmark"""
    print(f"📊 Generating {n:,} random vectors ({d} dimensions)...")
    return np.random.rand(n, d).astype('float32')

def build_index(vectors):
    """Build HNSW index"""
    print(f"Building index for {len(vectors):,} vectors...")
    
    # Initialize index
    index = hnswlib.Index(space='cosine', dim=DIMENSIONS)
    index.init_index(max_elements=NUM_VECTORS, ef_construction=EF_CONSTRUCTION, M=M)
    
    # Add vectors with progress
    start = time.time()
    batch_size = 1000
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        labels = np.arange(i, min(i+batch_size, len(vectors)))
        index.add_items(batch, labels)
        if i % 5000 == 0:
            print(f"Progress: {i}/{len(vectors)} ({i*100/len(vectors):.1f}%)")
    
    build_time = (time.time() - start) * 1000  # Convert to ms
    print(f"✅ Index built in {build_time:.1f} ms")
    
    # Set search parameters
    index.set_ef(EF_SEARCH)
    
    return index, build_time

def search_single(args):
    """Single search operation for parallel execution"""
    index, query = args
    labels, distances = index.knn_query(query, k=K)
    return labels, distances

def benchmark_single_thread(index, queries):
    """Benchmark single-threaded search"""
    start = time.time()
    results = []
    for query in queries:
        labels, distances = index.knn_query(query, k=K)
        results.append((labels, distances))
    search_time = (time.time() - start) * 1000
    return search_time, results

def benchmark_multithread(index, queries, num_threads):
    """Benchmark multi-threaded search (limited by GIL)"""
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create args for each query
        args = [(index, query) for query in queries]
        results = list(executor.map(search_single, args))
    
    search_time = (time.time() - start) * 1000
    return search_time, results

def benchmark_multiprocess(index_data, queries, num_processes):
    """Benchmark multi-process search (true parallelism)"""
    # Note: This requires pickleable index, which hnswlib doesn't support well
    # So we'll skip this for now
    pass

def run_benchmark():
    """Run complete benchmark suite"""
    print("\n" + "="*80)
    print("🐍 PYTHON HNSWLIB BENCHMARK")
    print("="*80)
    
    # Generate test data
    vectors = generate_random_vectors(NUM_VECTORS, DIMENSIONS)
    queries = vectors[np.random.choice(NUM_VECTORS, NUM_QUERIES, replace=False)]
    
    print(f"\n📊 Configuration:")
    print(f"  • Vectors: {NUM_VECTORS:,}")
    print(f"  • Dimensions: {DIMENSIONS}")
    print(f"  • Queries: {NUM_QUERIES}")
    print(f"  • k: {K}")
    print(f"  • Thread counts: {THREAD_COUNTS}")
    print(f"  • HNSW params: M={M}, ef_construction={EF_CONSTRUCTION}, ef_search={EF_SEARCH}")
    
    # Build index
    print("\n🔨 Building index...")
    index, build_time = build_index(vectors)
    
    # Warm-up
    print("\n🔥 Warming up...")
    for _ in range(10):
        index.knn_query(queries[0], k=K)
    
    # Run benchmarks
    print("\n📈 Running benchmarks...")
    print("-"*80)
    print(f"{'Threads':<10} {'Build(ms)':<12} {'Search(ms)':<12} {'Latency(ms)':<12} {'QPS':<12}")
    print("-"*80)
    
    results = []
    for num_threads in THREAD_COUNTS:
        if num_threads == 1:
            search_time, _ = benchmark_single_thread(index, queries)
        else:
            # Python's GIL limits true parallelism for CPU-bound tasks
            search_time, _ = benchmark_multithread(index, queries, num_threads)
        
        avg_latency = search_time / NUM_QUERIES
        qps = (NUM_QUERIES * 1000) / search_time
        
        print(f"{num_threads:<10} {build_time:<12.1f} {search_time:<12.1f} {avg_latency:<12.3f} {qps:<12.0f}")
        
        results.append({
            'threads': num_threads,
            'build_ms': build_time,
            'search_ms': search_time,
            'latency_ms': avg_latency,
            'qps': qps
        })
    
    print("-"*80)
    
    # Summary
    single_qps = results[0]['qps']
    best_result = max(results, key=lambda x: x['qps'])
    
    print(f"\n📊 SUMMARY:")
    print(f"  • Single thread QPS: {single_qps:.0f}")
    print(f"  • Best QPS: {best_result['qps']:.0f} ({best_result['threads']} threads)")
    print(f"  • Speedup: {best_result['qps']/single_qps:.1f}x")
    print(f"  • Best latency: {best_result['latency_ms']:.3f} ms")
    
    # GIL limitation note
    print(f"\n⚠️ Note: Python's GIL limits true parallel execution for CPU-bound tasks.")
    print(f"  Multi-threading in Python may not show significant speedup for HNSW search.")
    
    return results

def test_with_multiprocessing():
    """Test with multiprocessing for true parallelism"""
    print("\n" + "="*80)
    print("🔀 MULTIPROCESSING TEST (True Parallelism)")
    print("="*80)
    
    # Generate test data
    vectors = generate_random_vectors(NUM_VECTORS, DIMENSIONS)
    queries = vectors[np.random.choice(NUM_VECTORS, NUM_QUERIES, replace=False)]
    
    # Build index once
    index = hnswlib.Index(space='cosine', dim=DIMENSIONS)
    index.init_index(max_elements=NUM_VECTORS, ef_construction=EF_CONSTRUCTION, M=M)
    index.add_items(vectors)
    index.set_ef(EF_SEARCH)
    
    # Save index to file for multiprocessing
    index_file = '/tmp/hnsw_index.bin'
    index.save_index(index_file)
    
    def search_batch(args):
        """Search function for multiprocessing"""
        queries_batch, index_file = args
        # Each process loads its own index
        idx = hnswlib.Index(space='cosine', dim=DIMENSIONS)
        idx.load_index(index_file)
        idx.set_ef(EF_SEARCH)
        
        results = []
        for q in queries_batch:
            labels, distances = idx.knn_query(q, k=K)
            results.append((labels, distances))
        return results
    
    print("\nTesting with different process counts:")
    print("-"*60)
    print(f"{'Processes':<12} {'Time(ms)':<12} {'QPS':<12} {'Speedup':<12}")
    print("-"*60)
    
    # Single process baseline
    start = time.time()
    _ = [index.knn_query(q, k=K) for q in queries]
    single_time = (time.time() - start) * 1000
    single_qps = (NUM_QUERIES * 1000) / single_time
    print(f"{1:<12} {single_time:<12.1f} {single_qps:<12.0f} {1.0:<12.1f}")
    
    # Test with multiple processes
    for num_proc in [2, 4, 8]:
        if num_proc > cpu_count():
            continue
            
        # Split queries into batches
        batch_size = len(queries) // num_proc
        query_batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        args = [(batch, index_file) for batch in query_batches]
        
        start = time.time()
        with Pool(num_proc) as pool:
            results = pool.map(search_batch, args)
        
        mp_time = (time.time() - start) * 1000
        mp_qps = (NUM_QUERIES * 1000) / mp_time
        speedup = mp_qps / single_qps
        
        print(f"{num_proc:<12} {mp_time:<12.1f} {mp_qps:<12.0f} {speedup:<12.1f}")
    
    print("-"*60)
    
    # Clean up
    if os.path.exists(index_file):
        os.remove(index_file)

if __name__ == "__main__":
    print("🚀 HNSW Python Benchmark - Comparison with Clojure HNSW-CLJ")
    print("="*80)
    
    # Check if hnswlib is installed
    try:
        import hnswlib
        print(f"✅ hnswlib version: {hnswlib.__version__}")
    except ImportError:
        print("❌ hnswlib not installed. Install with: pip install hnswlib")
        exit(1)
    
    # Run main benchmark
    results = run_benchmark()
    
    # Test multiprocessing (true parallelism)
    test_with_multiprocessing()
    
    print("\n" + "="*80)
    print("✅ Benchmark complete!")
    print("="*80)