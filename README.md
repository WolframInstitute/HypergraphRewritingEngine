# Hypergraph Rewriting Engine

A high-performance implementation of multiway hypergraph rewriting with Mathematica integration.

## Status

v1.0.0-rc1: feature-complete, gated by CI, model checking and a brute-force oracle. APIs
are stable from this release line onward.

## Features

- **Multiway Evolution**: Parallel state evolution with causal and branchial graph construction, single synchronisation point (no intra-evolution phase barriers, online transitive reduction).
- **Exact Canonicalization**: McKay-style individualisation--refinement (IR) computes an exact canonical hash for every state; deduplication is isomorphism-correct, never heuristic. A content-ordered hash (`Automatic`) and no-dedup (`None`) modes exist for workloads that want them.
- **Determinism Contract**: the state set, event set and causal/branchial relations are a function of the inputs alone — independent of thread count and scheduling.
- **Quotient Exploration**: expand one representative per isomorphism class and reconstruct the raw events, causal edges and branchial pairs exactly.
- **GPU Backend**: CUDA port mirroring the CPU algorithms, with a resident evolution kernel (states re-enter as match tasks without returning to the host) and warp-collective canonicalization sharing the same IR implementation as the CPU.
- **Parallel Pattern Matching**: match-by-hyperedge join with signature-partitioned candidates and work-stealing scheduling.
- **Incremental Match Forwarding**: re-use parent-state matches in child states; only find new matches that involve newly produced edges (selected per rule set by static analysis).
- **Lock-free Data Structures**: concurrent hash map, key set, lock-free list, lock-free deque, thread-safe arena — the shared protocols model-checked under RC11/scoped-RC11 (GenMC, GPUMC) and TLA+ at their recorded bounds, sanitizer-gated in CI.
- **Mathematica Paclet**: LibraryLink bindings with evolution, canonical/causal/branchial graph extraction, dimension / curvature / geodesic / branchial analyses, and topology / initial-condition generators.

## Installation

### Mathematica Paclet

Install from the Wolfram Paclet Repository:

```mathematica
PacletInstall["WolframInstitute/HypergraphRewriteEngine"]
Needs["HypergraphRewriting`"]
```

(Or install a `.paclet` file from the [latest GitHub release](https://github.com/WolframInstitute/HypergraphRewritingEngine/releases) by passing its URL or a local path to `PacletInstall`.)

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
result["BranchialEdges"] (* event pairs that share an input state AND consume a
                            common edge -- two events out of one state that
                            consume disjoint edges are not branchially adjacent *)

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

Build every shipped artifact from Linux (six platform libraries and binaries, plus the
CUDA executables where a toolchain is present):

```bash
./build_all_platforms.sh
```

Required packages (Ubuntu/Debian):

```bash
sudo apt install \
    cmake build-essential \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 \
    clang lld
```

| Target | Toolchain |
|--------|-----------|
| Linux ARM64 | `gcc-aarch64-linux-gnu` |
| Windows x86-64 | MinGW-w64 |
| Windows ARM64 | Clang + LLD |
| macOS | [OSXCross](https://github.com/tpoechtrager/osxcross) |

See [CROSS_COMPILATION.md](CROSS_COMPILATION.md) for detailed setup.

## Project Structure

```
hypergraph/     Core CPU engine: evolution, matching, WL/IR canonicalization, storage
gpu/            CUDA port (optional, BUILD_GPU=ON), mirrors the CPU algorithms
job_system/     Work-stealing task scheduler the engine runs on
lockfree_deque/ Lock-free concurrent deque backing the scheduler
common/         Shared CPU/GPU rules: the IR canonicalization core, join, quotient DP,
                ring/termination/pool protocols, portable intrinsics
wxf/            Wolfram Exchange Format serialization (the WL boundary)

paclet/         Wolfram Language paclet (kernel code, bundled binaries, doc notebooks)
paclet_source/  FFI: run_rewriting_core, the standalone hg_evolve binary, GPU marshaling
reference/      Validation oracle (brute-force ground truth) + golden corpus
verification/   GenMC and GPUMC model-checking harnesses, TLA+ specifications
tools/          Standalone research/validation probes and profiling harnesses
testing/        Aggregate C++ test target (all_tests)
benchmarks/     Per-area benchmarks;  benchmarking/  is the framework library
visualisation/  Viz-event interface (the renderer lives in its own repository)

docs/           Specification, architecture, code map, quickstart
paper/          The technical report (LaTeX; tables generated by tools/dev)
examples/       Small C++ usage examples
cmake/          Cross-compilation toolchain files
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
ctest                    # Registers all_tests (the subset targets are direct-run only)
```

## License

MIT — see [LICENSE.md](LICENSE.md). The paclet declares the same in its
`PacletInfo.wl`.
