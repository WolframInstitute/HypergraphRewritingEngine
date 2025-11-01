# Hypergraph Benchmark Suite

Comprehensive performance benchmarking infrastructure for the hypergraph rewriting system.

## Quick Start

### Build Benchmarks

```bash
cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release
make benchmark_suite
```

### Run Benchmarks

```bash
# Run all benchmarks (outputs to benchmark_results/)
cmake .. -DCMAKE_BUILD_TYPE=Release && make run_benchmarks

# Run filtered benchmarks (gtest-style patterns with wildcards)
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_FILTER="pattern_matching*" && make run_benchmarks
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_FILTER="canonicalization_2d_sweep" && make run_benchmarks

# Clear filter
cmake .. -UBENCHMARK_FILTER && make run_benchmarks

# Run benchmark suite directly with gtest-style filter
./benchmark_suite --filter='pattern'

# Filter syntax (gtest-style):
# - Wildcards: '*' (any string), '?' (any char)
# - Multiple patterns: separated by ':' (OR)
# - Negative patterns: after '-' (exclude)
# - Examples:
#   "pattern_matching*" - all pattern matching benchmarks
#   "*_2d_sweep" - all 2D parameter sweeps
#   "canonicalization_by_edge_count" - exact match
#   "causal*:transitive*" - multiple patterns (OR)
#   "*overhead*" - substring match
#   "*-*reference*" - all except reference benchmarks
```

### Visualize Results

```bash
# Install Python dependencies (one-time setup)
pip install -r benchmarking/requirements.txt

# Generate plots
python3 benchmarking/plot_benchmarks.py benchmark_results/<commit_hash>/
```

## Output Files

Each benchmark run creates a directory: `benchmark_results/tree-<commit_hash>/`

### CSV Files Generated:

1. **summary.csv** - One row per commit, columns for each benchmark metric
   - Format: `commit_hash,commit_date,timestamp,<benchmark>_<params>_avg_us,...`
   - Use for: Cross-commit comparisons

2. **detailed.csv** - One row per benchmark configuration
   - Format: `benchmark_name,params,samples,min_us,max_us,avg_us,stddev_us,cv_percent`
   - Use for: Per-benchmark analysis

3. **raw_timings.csv** - One row per sample
   - Format: `benchmark_name,params,sample_num,timing_us`
   - Use for: Statistical analysis, outlier detection

4. **samples_convergence.csv** - Convergence tracking
   - Format: `benchmark_name,params,sample_num,cumulative_avg_us,cumulative_stddev_us,cv_percent`
   - Use for: Understanding variance behavior

5. **benchmark_config.txt** - Configuration snapshot
   - Contains: min/max samples, variance threshold, commit info

## Visualization

The `plot_benchmarks.py` script generates:

- **Convergence plots** - How CV changes with sample count
- **1D parameter sweeps** - Performance vs single parameter (with error bars)
- **2D surface plots** - Performance vs two parameters
- **Time-series plots** - Performance across commits (requires multiple runs)

All plots saved to: `benchmark_results/<commit_hash>/plots/`

## Benchmark Categories

### Canonicalization Benchmarks
- Graph canonicalization performance with varying edge counts and symmetry groups
- 2D parameter sweeps

### Pattern Matching Benchmarks
- Pattern matching scalability across thread counts and graph sizes
- 2D thread/size sweeps

### State Management Benchmarks
- State storage and reconstruction overhead

### Event Relationship Benchmarks
- Causal and branchial edge computation overhead

### Evolution Benchmarks
- End-to-end evolution with thread scaling
- Multi-rule evolution

### Wolfram Integration Benchmarks
- End-to-end testing via WolframScript (only runs when WolframScript is available)
- WXF serialization overhead

## Adaptive Sampling

Benchmarks use **variance-driven adaptive sampling**:

1. Start with `BENCHMARK_MIN_SAMPLES=5`
2. Calculate CV (coefficient of variation) = stddev/mean × 100
3. If CV < `BENCHMARK_VARIANCE_THRESHOLD=5%`: STOP
4. Otherwise, add 5 more samples
5. Repeat until CV converges or `BENCHMARK_MAX_SAMPLES=100` reached

### Output Format

```
[ RUN      ] canonicalization_by_edge_count
[   TIMING ] Running 50 calibrated samples for (edges=2, symmetry_groups=1)...
[   TIMING ]   Sample 1/50: 1.20 μs (avg: 1.20 μs, CV: 0.00%)
[   TIMING ]   Sample 2/50: 1.85 μs (avg: 1.53 μs, CV: 21.34%)
...
[       OK ] canonicalization_by_edge_count (edges=10) - 208.94 ± 9.12 μs (n=15, CV=4.3%)

[==========] 14 benchmarks completed
[   SAVED  ] Results written to: benchmark_results/tree-b82299a.../
```

## Adding New Benchmarks

Create a new `.cpp` file in the `benchmarks/` directory:

```cpp
#include "benchmark_framework.hpp"
#include <hypergraph/rewriting.hpp>

using namespace hypergraph;
using namespace benchmark;

BENCHMARK(my_new_benchmark, "Description of what this benchmark measures") {
    for (int param : {10, 20, 50}) {
        BENCHMARK_PARAM("param", param);

        // Setup (not timed)
        auto graph = generate_test_data(param);

        // Code to measure
        BENCHMARK_CODE([&]() {
            my_function(graph);
        }, 10);  // Optional: force specific sample count (omit for adaptive sampling)
    }
}
```

Then add the file to `benchmarks/CMakeLists.txt`:

```cmake
add_executable(benchmark_suite
    canonicalization_benchmarks.cpp
    pattern_matching_benchmarks.cpp
    state_management_benchmarks.cpp
    event_relationship_benchmarks.cpp
    evolution_benchmarks.cpp
    wolfram_integration_benchmark.cpp
    my_new_benchmark.cpp  # Add your file here
)
```

## Project Structure

```
benchmarking/                          # Framework and support files
├── CMakeLists.txt                     # Framework library target
├── benchmark_framework.hpp            # Core infrastructure
├── benchmark_main.cpp                 # Main entry point
├── random_hypergraph_generator.hpp    # Test data generation
├── plot_benchmarks.py                 # Visualization script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file

benchmarks/                            # Benchmark implementations
├── CMakeLists.txt                     # Benchmark suite target
├── canonicalization_benchmarks.cpp
├── pattern_matching_benchmarks.cpp
├── state_management_benchmarks.cpp
├── event_relationship_benchmarks.cpp
├── evolution_benchmarks.cpp
└── wolfram_integration_benchmark.cpp

benchmark_results/                     # Output directory
└── tree-<commit_hash>/
    ├── summary.csv
    ├── detailed.csv
    ├── raw_timings.csv
    ├── samples_convergence.csv
    ├── benchmark_config.txt
    └── plots/
        ├── <benchmark>_1d.png
        └── <benchmark>_2d.png
```

## Troubleshooting

**Issue:** Benchmarks take too long
- **Solution:** Use a filter to run specific benchmarks: `cmake .. -DBENCHMARK_FILTER=benchmark_name`

**Issue:** High variance (CV > 10%)
- **Cause:** System noise, cache effects, turbo boost
- **Solution:** Pin to core, disable turbo, close background apps

**Issue:** Visualization fails
- **Cause:** Missing Python packages
- **Solution:** `pip install -r benchmarking/requirements.txt`

**Issue:** Build fails with "Benchmarks MUST be built in release mode"
- **Solution:** Configure with Release build type: `cmake .. -DCMAKE_BUILD_TYPE=Release`
