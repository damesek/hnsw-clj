#!/usr/bin/env bash

# Performance test runner for HNSW-CLJ
echo "========================================="
echo "HNSW-CLJ Performance Test Suite"
echo "========================================="
echo ""
echo "⚠️  This will run performance benchmarks (10-30 seconds)"
echo ""

# Run the performance tests directly by name
clojure -Sdeps '{:paths ["src" "test"]}' -M -e '
(require (quote [clojure.test :as t]))
(require (quote hnsw.core-test))

(println "Running performance tests...")
(println "----------------------------")
(println)

;; Run specific performance tests
(binding [t/*test-out* *out*]
  (let [perf-test-1 (resolve (quote hnsw.core-test/test-build-performance))
        perf-test-2 (resolve (quote hnsw.core-test/test-search-performance))
        results (atom {:pass 0 :fail 0 :error 0})]
    
    ;; Run test-build-performance
    (when perf-test-1
      (println "Running: test-build-performance")
      (println "--------------------------------")
      (try
        (@perf-test-1)
        (swap! results update :pass inc)
        (println "✓ Passed")
        (catch Exception e
          (println (str "✗ Failed: " (.getMessage e)))
          (swap! results update :fail inc))))
    
    (println)
    
    ;; Run test-search-performance
    (when perf-test-2
      (println "Running: test-search-performance")
      (println "---------------------------------")
      (try
        (@perf-test-2)
        (swap! results update :pass inc)
        (println "✓ Passed")
        (catch Exception e
          (println (str "✗ Failed: " (.getMessage e)))
          (swap! results update :fail inc))))
    
    ;; Print summary
    (println)
    (println "============================")
    (println "Performance Test Summary:")
    (println (str "  Tests run: " (+ (:pass @results) (:fail @results))))
    (println (str "  Passed: " (:pass @results)))
    (println (str "  Failed: " (:fail @results)))
    
    (if (zero? (:fail @results))
      (do
        (println)
        (println "✅ All performance tests passed!")
        (System/exit 0))
      (do
        (println)
        (println "❌ Some performance tests failed!")
        (System/exit 1)))))
'

echo ""
echo "Performance benchmarks completed!"
echo ""
echo "Test parameters (optimized for speed):"
echo "• Small: 100 vectors × 128 dimensions"
echo "• Medium: 500 vectors × 128 dimensions"  
echo "• Large: 1,000 vectors × 128 dimensions"
echo "• Queries: 20"
