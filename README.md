# Hypergraph Rewriting Engine

A high-performance implementation of multiway hypergraph rewriting with Mathematica integration.

## Status

This project is functional but under active development. No stable release has been made yet and APIs may change between versions.

## Features

- **Multiway Evolution**: Parallel state evolution with causal and branchial graph construction, single synchronisation point (no intra-evolution phase barriers).
- **Parallel Pattern Matching**: SCAN → EXPAND → SINK dataflow pipeline with work-stealing scheduling.
- **Edge Signature Indexing**: Fast candidate generation via multi-level signature partitioning.
- **Incremental Match Forwarding**: Re-use parent-state matches in child states; only find new matches that involve newly produced edges.
- **Canonicalization**: a fast Weisfeiler-Leman (WL) heuristic or exact McKay-style individualisation-refinement (IR) for isomorphism-correct deduplication.
- **Lock-free Data Structures**: concurrent hash map, lock-free list, lock-free deque, thread-safe arena.
- **Mathematica Paclet**: LibraryLink bindings with evolution, canonical/causal/branchial graph extraction, dimension / curvature / geodesic / branchial analyses, and topology / initial-condition generators.

## Installation

### Mathematica Paclet

Install the paclet directly from a release `.paclet` file:

```mathematica
PacletInstall["path/to/WolframInstitute__HypergraphRewriteEngine-0.0.1.paclet"]
Needs["HypergraphRewriting`"]
```

The paclet name is `WolframInstitute/HypergraphRewriteEngine`; its exported context is ``HypergraphRewriting` ``.

### Building from Source

```bash
mkdir build_linux && cd build_linux
cmake .. -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
make -j32 paclet
```

## Usage

### Mathematica

```mathematica
Needs["HypergraphRewriting`"]

(* Rules use symbolic vertices (normalised to numeric internally). *)
rule = {{x, y}, {y, z}} -> {{x, y}, {y, z}, {z, x}};
init = {{1, 2}, {2, 3}, {3, 1}};

(* HGEvolve[rules, initialEdges, steps, property]. Passing a single rule is
   supported; a list of rules is also supported. *)
result = HGEvolve[rule, init, 5, "All"];

(* "All" returns an Association with these keys: *)
result["NumStates"]      (* uint: number of states *)
result["NumEvents"]      (* uint: number of events *)
result["States"]         (* association State -> state edges *)
result["Events"]         (* list of rewriting events *)
result["CausalEdges"]    (* list of (producer, consumer) event pairs *)
result["BranchialEdges"] (* list of event pairs sharing an input state *)

(* Graph properties evaluate directly to Graph objects: *)
HGEvolve[rule, init, 5, "StatesGraph"]
HGEvolve[rule, init, 5, "CausalGraph"]
HGEvolve[rule, init, 5, "BranchialGraph"]
HGEvolve[rule, init, 5, "EvolutionCausalBranchialGraph"]
```

See `paclet_source/README.md` for the full option list (hash strategy, canonicalisation modes, pruning limits, dimension / curvature / geodesic analyses, topology and initial-condition generators).

### C++ API

```cpp
#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/hypergraph.hpp>

using namespace hypergraph;

int main() {
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, 4);  // 4 threads

    // Rule: {x,y},{y,z} -> {x,y},{y,z},{z,x}
    auto rule = make_rule(0)
        .lhs({0, 1}).lhs({1, 2})
        .rhs({0, 1}).rhs({1, 2}).rhs({2, 0})
        .build();

    engine.add_rule(rule);

    std::vector<std::vector<VertexId>> initial = {{1, 2}, {2, 3}, {3, 1}};
    engine.evolve(initial, 5);

    std::cout << "States: " << hg.num_states() << "\n";
    std::cout << "Events: " << hg.num_events() << "\n";
}
```

## Supported Platforms

The paclet includes native libraries for:

| Platform | Architecture |
|----------|--------------|
| Linux | x86-64, ARM64 |
| Windows | x86-64, ARM64 |
| macOS | x86-64 (Intel), ARM64 (Apple Silicon) |

## Build Requirements

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.14+
- Google Test (automatically downloaded)
- Mathematica 13+ (optional, for paclet)

## Cross-Compilation

Build all 6 platforms from Linux:

```bash
./build_all_platforms.sh
```

Required packages (Ubuntu/Debian):

```bash
sudo apt install \
    cmake build-essential \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 \
    clang-22 lld-22
```

| Target | Toolchain |
|--------|-----------|
| Linux ARM64 | `gcc-aarch64-linux-gnu` |
| Windows x86-64 | MinGW-w64 |
| Windows ARM64 | Clang 22 + LLD |
| macOS | [OSXCross](https://github.com/tpoechtrager/osxcross) |

See [CROSS_COMPILATION.md](CROSS_COMPILATION.md) for detailed setup.

## Project Structure

```
hypergraph/     Core CPU engine: evolution, matching, WL/IR canonicalization, storage
gpu/            CUDA port (optional, BUILD_GPU=ON), mirrors the CPU algorithms
job_system/     Work-stealing task scheduler the engine runs on
lockfree_deque/ Lock-free concurrent deque backing the scheduler
common/         Shared primitives (portable intrinsics, shared WL hash core)
wxf/            Wolfram Exchange Format serialization (the WL boundary)

paclet/         Wolfram Language paclet (kernel code, bundled binaries, doc notebooks)
paclet_source/  FFI: run_rewriting_core, the standalone hg_evolve binary, GPU marshaling
reference/      Validation oracle (brute-force ground truth) + golden corpus
tools/          Standalone research/validation probes and profiling harnesses
testing/        Aggregate C++ test target (all_tests)
benchmarks/     Per-area benchmarks;  benchmarking/  is the framework library
visualisation/  Interactive 3D viewer (Vulkan) and physics analysis
```

New here? Users: **[docs/QUICKSTART.md](docs/QUICKSTART.md)**. Developers:
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Testing

```bash
cd build_linux
./all_tests              # All tests
./core_tests             # Core functionality
./evolution_tests        # Evolution and pattern matching
./causal_tests           # Causal/branchial graph
```
