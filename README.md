# Hypergraph Rewriting Engine

A high-performance implementation of multiway hypergraph rewriting.

## Features

- **Parallel Pattern Matching**: Task-parallel pipeline with work-stealing scheduler
- **Edge Signature Indexing**: Fast pattern matching via multi-level signature partitioning
- **Incremental Rewriting**: Match reuse and patch based matching around newly added edges
- **Multiway Graph Evolution**: Parallel state evolution and causal/branchial edge creation
- **Lock-free Data Structures**: High-performance concurrent queues and hash maps
- **Canonicalization**: Optimized hypergraph canonicalisation with insertion sort for fast incremental updates
- **Mathematica Integration**: Paclet support

## Development Status

**This code is not ready for production use.**

This repository is uploaded for progress reporting requirements and is not yet released. The implementation is under active development and contains known issues and incomplete features.

## Quick Start

### Build

```bash
mkdir build && cd build
cmake .. && make -j
```

### Run Tests

```bash
# Core functionality tests
./unified_tests
```

### Basic Usage

```cpp
#include <hypergraph/wolfram_evolution.hpp>
#include <hypergraph/rewriting.hpp>

// Create evolution system
hypergraph::WolframEvolution evolution(3, 4, true, false); // 3 steps, 4 threads

// Define rewriting rule: {A,B} -> {{A,B}, {B,C}}
hypergraph::PatternHypergraph lhs, rhs;
lhs.add_edge({hypergraph::PatternVertex::variable(1),
             hypergraph::PatternVertex::variable(2)});
rhs.add_edge({hypergraph::PatternVertex::variable(1),
             hypergraph::PatternVertex::variable(2)});
rhs.add_edge({hypergraph::PatternVertex::variable(2),
             hypergraph::PatternVertex::variable(3)});

evolution.add_rule(hypergraph::RewritingRule(lhs, rhs));

// Run evolution
std::vector<std::vector<hypergraph::GlobalVertexId>> initial = {{1, 2}};
evolution.evolve(initial);

// Get results
const auto& graph = evolution.get_multiway_graph();
std::cout << "States: " << graph.num_states() << std::endl;
std::cout << "Events: " << graph.num_events() << std::endl;
```

## Build Requirements

- C++17 compatible compiler (GCC 8+, Clang 7+)
- CMake 3.15+
- Google Test (automatically downloaded)
- Mathematica 12+ (for paclet build dependencies)