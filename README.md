# Hypergraph Rewriting Engine

A high-performance implementation of multiway hypergraph rewriting.

## Features

- **Parallel Pattern Matching**: Task-parallel pipeline with work-stealing scheduler
- **Edge Signature Indexing**: Fast pattern matching via multi-level signature partitioning
- **Incremental Rewriting**: Match reuse and patch-based matching around newly added edges
- **Multiway Graph Evolution**: Parallel state evolution and causal/branchial edge creation
- **Lock-free Data Structures**: High-performance concurrent queues and hash maps
- **Canonicalization**: Isomorphism-invariant state hashing via Weisfeiler-Leman refinement
- **Mathematica Integration**: Full paclet with LibraryLink bindings

## Development Status

**This code is not ready for production use.**

This repository is uploaded for progress reporting requirements and is not yet released. The implementation is under active development and contains known issues and incomplete features.

## Quick Start

### Build

```bash
mkdir build_linux && cd build_linux
cmake ..
make -j32
```

### Run Tests

```bash
# All tests
./all_tests

# Category-specific tests (faster iteration)
./core_tests         # Fast core functionality
./evolution_tests    # Evolution and pattern matching
./causal_tests       # Causal/branchial graph
./stress_tests       # Determinism and performance
./integration_tests  # Paclet and integration
```

### Basic Usage

```cpp
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>

using namespace hypergraph;

int main() {
    // Create hypergraph and evolution engine
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);  // 4 threads

    // Define rule: {x,y} -> {x,y},{y,z}
    auto rule = make_rule(0)
        .lhs({0, 1})      // Pattern edge {x, y}
        .rhs({0, 1})      // Keep {x, y}
        .rhs({1, 2})      // Add {y, z} with fresh vertex z
        .build();

    engine.add_rule(rule);

    // Initial state and evolution
    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}};
    engine.evolve(initial, 3);  // 3 steps

    // Results
    std::cout << "States: " << hg.num_states() << "\n";
    std::cout << "Events: " << hg.num_events() << "\n";

    // Causal structure
    auto causal = hg.causal_graph().get_causal_edges();
    auto branchial = hg.causal_graph().get_branchial_edges();
}
```

## Core API

### Hypergraph

The central data structure storing all edges, states, and events:

```cpp
Hypergraph hg;

// Create edges and states
EdgeId e1 = hg.create_edge({1, 2, 3});
StateId s = hg.create_state({e1});

// Configure canonicalization
hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
hg.set_event_signature_keys(EVENT_SIG_FULL);
```

### Rules

Rules are defined with pattern variables (0, 1, 2, ...) that bind to vertices:

```cpp
// {x,y},{y,z} -> {x,y},{y,z},{z,x}  (close triangle)
auto rule = make_rule(0)
    .lhs({0, 1})
    .lhs({1, 2})
    .rhs({0, 1})
    .rhs({1, 2})
    .rhs({2, 0})
    .build();
```

### Evolution Engine

```cpp
ParallelEvolutionEngine engine(&hg, num_threads);
engine.add_rule(rule);
engine.evolve(initial_edges, num_steps);

// Optional: explore only from canonical states (deduplication)
engine.set_explore_from_canonical_states_only(true);
```

## Build Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+)
- CMake 3.15+
- Google Test (automatically downloaded)
- Mathematica 12+ (optional, for paclet)

## Supported Platforms

The paclet supports 6 platforms:
- **Linux**: x86-64, ARM64
- **Windows**: x86-64, ARM64
- **macOS**: x86-64 (Intel), ARM64 (Apple Silicon)

### Cross-Compilation

Cross-compilation from Linux is supported for all platforms. The build script automatically detects available toolchains:

- **Linux ARM64**: Requires `gcc-aarch64-linux-gnu`
- **Windows x86-64**: Requires MinGW-w64 (`gcc-mingw-w64-x86-64`)
- **Windows ARM64**: Requires Clang + Windows SDK (auto-detected in WSL2)
- **macOS**: Requires OSXCross

See `paclet_source/README.md` for detailed build instructions.

## Project Structure

```
hypergraph/              # Core hypergraph rewriting library
  include/hypergraph/
    hypergraph.hpp         # Main Hypergraph class
    parallel_evolution.hpp # ParallelEvolutionEngine
    pattern.hpp            # RewriteRule, make_rule()
    pattern_matcher.hpp    # Pattern matching
    causal_graph.hpp       # Causal/branchial edges
    types.hpp              # Core types (VertexId, EdgeId, etc.)
  tests/                   # Hypergraph unit tests

job_system/              # Work-stealing task scheduler
lockfree_deque/          # Lock-free concurrent deque
wxf/                     # Wolfram Exchange Format serialization

visualisation/           # 3D visualization (Vulkan)
  blackhole/               # Physics analysis (geodesics, curvature, etc.)
  scene/                   # Rendering pipeline
  shaders/                 # GLSL/SPIR-V shaders

gpu/                     # GPU compute kernels (experimental)

testing/                 # Test infrastructure (CMake, helpers)
examples/                # Usage examples
benchmarks/              # Performance benchmarks

paclet/                  # Mathematica paclet skeleton
paclet_source/           # LibraryLink FFI implementation
```
