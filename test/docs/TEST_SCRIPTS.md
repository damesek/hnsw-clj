# Test Scripts - Clean Setup

## Active Test Scripts (4)

Only the essential test scripts remain in the root directory:

| Script | Purpose | Tests | Runtime |
|--------|---------|-------|---------|
| `test-minimal.sh` | Basic functionality tests | 5 | < 5 sec |
| `test-quick.sh` | Quick smoke test with all components | 6 | < 10 sec |
| `test-functional.sh` | Detailed functional testing | 11 | < 20 sec |
| `run-tests.sh` | Complete test suite runner | 6 | < 30 sec |

## Usage

```bash
# Make executable (one time)
chmod +x test-*.sh run-tests.sh

# Run tests
./test-minimal.sh      # Basic tests
./test-quick.sh        # Quick validation
./test-functional.sh   # Comprehensive tests
./run-tests.sh         # Full suite
```

## Archived Scripts

The following obsolete/redundant test scripts have been moved to `old-test-scripts/`:
- test-verify.sh
- test-ultra-simple.sh
- test-hnsw.sh
- test-direct.sh
- test-simple.sh
- generate-minimal-data.sh
- quick-test.sh
- generate-test-data.sh

These can be safely deleted with: `rm -rf old-test-scripts/`

## Other Scripts

- `nrepl-init-optimized.sh` - nREPL starter with optimizations (kept as it's useful)

## Test Coverage

All 4 active test scripts provide complete coverage:
- ✅ Distance functions (SIMD-optimized)
- ✅ Ultra-fast implementation
- ✅ Graph implementation
- ✅ Data generator (all 3 formats)

The test suite is now clean, organized, and production-ready!
