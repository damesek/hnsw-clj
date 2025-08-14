# HNSW-CLJ Test Suite

## Overview
A comprehensive test suite for the HNSW (Hierarchical Navigable Small World) implementation in Clojure.

## Test Structure

### Core Test Files

1. **`test/data_generator.clj`** - Test data generation utilities
   - Supports multiple data formats: `:vector`, `:double-array`, `:indexed`
   - Various distributions: Gaussian, uniform, clustered
   - Common embedding dimensions from 256 to 3072

2. **`test/simple_test.clj`** - Basic functionality tests
   - Ultra-fast implementation tests
   - Distance function tests
   - Graph implementation tests
   - Quick smoke tests without external dependencies

3. **`test/hnsw/core_test.clj`** - Comprehensive core tests
   - Distance calculations (Euclidean, Cosine)
   - Index creation and search
   - KNN search quality
   - Concurrent operations
   - Performance benchmarks

## Data Formats

The test suite supports three data formats:

### 1. Vector Format (Default)
```clojure
(generate-dataset 100 128 :format :vector)
;; Returns: [[0.1 0.2 ...] [0.3 0.4 ...] ...]
```

### 2. Double Array Format (SIMD Optimized)
```clojure
(generate-dataset 100 128 :format :double-array)
;; Returns: [#double[] #double[] ...]
```

### 3. Indexed Format (Ultra-fast Implementation)
```clojure
(generate-dataset 100 128 :format :indexed)
;; Returns: [["vec_0" #double[]] ["vec_1" #double[]] ...]
```

## Running Tests

### Quick Test
```bash
# Run the ultra-simple test
chmod +x test-ultra-simple.sh
./test-ultra-simple.sh
```

### Comprehensive Test Suite
```bash
# Run all tests
chmod +x run-tests.sh
./run-tests.sh
```

### Individual Test Files
```bash
# Run simple tests
clojure -M:test -m simple-test

# Run core tests
clojure -M:test -m hnsw.core-test

# Run specific test functions
clojure -M:test --focus test-ultra-fast-index
```

### REPL Testing
```clojure
;; Load test namespace
(require '[simple-test :as st])
(require '[clojure.test :as t])

;; Run all tests in namespace
(t/run-tests 'simple-test)

;; Run specific test
(st/test-basic-functionality)
```

## Test Categories

### Quick Tests (`:quick`)
- Basic functionality verification
- Distance calculations
- Small dataset tests (< 100 vectors)
- Run time: < 1 second

### Performance Tests (`:performance`)
- Large dataset tests (1K-10K vectors)
- Build time benchmarks
- Search performance metrics
- Run time: 10-60 seconds

### Integration Tests
- Multi-implementation comparisons
- Data format compatibility
- Concurrent operation testing

## Common Test Patterns

### Testing Ultra-fast Implementation
```clojure
(let [vectors (generate-dataset 100 128 :format :indexed)
      index (ultra/build-index vectors :show-progress? false)
      [_ query-vec] (first vectors)
      results (ultra/search-knn index query-vec 5)]
  (is (= 5 (count results)))
  (is (< (:distance (first results)) 0.01)))
```

### Testing Distance Functions
```clojure
(let [v1 (double-array [1 2 3])
      v2 (double-array [4 5 6])]
  (is (> (simd/euclidean-distance v1 v2) 5.0))
  (is (< (simd/cosine-distance v1 v2) 1.0)))
```

### Testing Graph Implementation
```clojure
(let [g (graph/create-graph)]
  (graph/insert g "v1" [1.0 2.0 3.0])
  (graph/insert g "v2" [4.0 5.0 6.0])
  (let [results (graph/search-knn g [1.0 2.0 3.0] 2)]
    (is (sequential? results))))
```

## Known Issues

1. **Graph implementation** - Currently returns empty results in search
2. **Large dataset tests** - Can be slow without `:show-progress? false`
3. **SIMD availability** - Falls back to software implementation if hardware SIMD not available

## Performance Expectations

- **Ultra-fast build**: ~50ms for 1K vectors, ~500ms for 10K vectors
- **Search latency**: < 1ms for most queries
- **Memory usage**: ~100MB for 10K 768-dim vectors

## Troubleshooting

### ClassCastException with Ultra-fast
Ensure you're using the correct data format:
```clojure
;; Wrong - regular vectors
(ultra/build-index [[1 2 3] [4 5 6]])

;; Correct - indexed format with double arrays
(ultra/build-index [["v0" (double-array [1 2 3])] 
                   ["v1" (double-array [4 5 6])]])
```

### Tests Hanging
Add `:show-progress? false` to build-index calls:
```clojure
(ultra/build-index vectors :show-progress? false)
```

### Distance Function Errors
Ensure using double arrays for SIMD functions:
```clojure
;; Wrong
(simd/euclidean-distance [1 2 3] [4 5 6])

;; Correct
(simd/euclidean-distance (double-array [1 2 3]) 
                        (double-array [4 5 6]))
```
