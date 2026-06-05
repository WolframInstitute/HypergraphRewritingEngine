# Hypergraph Rewriting Engine

A high-performance implementation of multiway hypergraph rewriting with Mathematica integration.

## Status

This project is functional but under active development. No stable release has been made yet and APIs may change between versions.

## Features

- **Multiway Evolution**: Parallel state evolution with causal and branchial graph construction, single synchronisation point (no intra-evolution phase barriers).
- **Parallel Pattern Matching**: SCAN → EXPAND → SINK dataflow pipeline with work-stealing scheduling.
- **Edge Signature Indexing**: Fast candidate generation via multi-level signature partitioning.
- **Incremental Match Forwarding**: Re-use parent-state matches in child states; only find new matches that involve newly produced edges.
- **Canonicalization**: choice of fast heuristics (WL / UT / incremental UT) or exact McKay-style individualisation-refinement (IR) for isomorphism-correct deduplication.
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

**One command, every platform this machine can target** (recommended). It
auto-installs the cross-toolchains it needs, builds each platform library, and
assembles the distributable archive:

```bash
./build_all_platforms.sh
```

Outputs:

| Artifact | Location |
|----------|----------|
| Per-platform loadable libraries | `paclet/LibraryResources/<Platform>/` |
| Distributable archive | `paclet_archive/WolframInstitute__HypergraphRewriteEngine-<version>.paclet` |

Platforms the current host architecture cannot target (e.g. Windows-ARM64 on
macOS, which needs the Windows SDK) are reported as *skipped*, not failures —
see [Cross-Compilation](#cross-compilation).

**Single platform only** (just the library for the current machine):

```bash
mkdir build && cd build
cmake .. -DBUILD_MATHEMATICA_PACLET=ON
make -j paclet                  # -> paclet/LibraryResources/<Platform>/
make create_paclet_archive      # -> paclet_archive/*.paclet
```

The build auto-detects a Wolfram/Mathematica install
(`/Applications/Wolfram.app`, `Mathematica.app`, or `Wolfram Engine.app`;
their `~/Applications` equivalents; standard Linux/Windows locations). Override
with `MATHEMATICA_HOME=/path/to/Wolfram/Contents` if it lives elsewhere.

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

`./build_all_platforms.sh` builds every target the host can reach from a
declarative toolchain *schema*. Each target lists the toolchain it needs and how
to install it; the script auto-installs missing toolchains, skips targets the
host architecture cannot support, and only fails on a genuine build error
(`--no-install` to opt out of installs, `--help` for all options).

What each host can build:

| Target | Toolchain | macOS host | Linux host |
|--------|-----------|:----------:|:----------:|
| macOS x86-64 / ARM64 | Apple clang (native) / OSXCross | ✅ native | OSXCross |
| Linux x86-64 / ARM64 | GCC cross (`messense` tap / `apt`) | ✅ auto | ✅ native + cross |
| Windows x86-64 | MinGW-w64 (`brew` / `apt`) | ✅ auto | ✅ auto |
| Windows ARM64 | Clang + Windows SDK + MSVC libs | ⏭️ skipped | ⏭️ skipped |

Windows ARM64 needs the Windows SDK and MSVC import libraries, which only exist
on a Windows/WSL host — it is skipped elsewhere rather than aborting the run.

The toolchains are installed automatically, but for reference:

```bash
# macOS (Homebrew)
brew install mingw-w64
brew install messense/macos-cross-toolchains/x86_64-unknown-linux-gnu
brew install messense/macos-cross-toolchains/aarch64-unknown-linux-gnu

# Linux (Ubuntu/Debian)
sudo apt install cmake build-essential \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64
```

See [CROSS_COMPILATION.md](CROSS_COMPILATION.md) for detailed setup.

## Project Structure

```
hypergraph/              Core rewriting library
  include/hypergraph/
    hypergraph.hpp         Hypergraph class
    parallel_evolution.hpp Evolution engine
    pattern.hpp            RewriteRule, make_rule()
    causal_graph.hpp       Causal/branchial edges

job_system/              Work-stealing task scheduler
lockfree_deque/          Lock-free concurrent deque
wxf/                     Wolfram Exchange Format serialization

visualisation/           3D visualization (Vulkan)
  blackhole/               Physics analysis
  scene/                   Rendering pipeline

paclet/                  Mathematica paclet
paclet_source/           LibraryLink FFI
```

## Testing

```bash
cd build_linux
./all_tests              # All tests
./core_tests             # Core functionality
./evolution_tests        # Evolution and pattern matching
./causal_tests           # Causal/branchial graph
```
