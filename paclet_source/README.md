# HypergraphRewriting Paclet

A high-performance Mathematica paclet for hypergraph rewriting and Wolfram Physics model simulation.

## Overview

This paclet provides efficient hypergraph rewriting capabilities with parallel processing support. It includes:

- Fast hypergraph canonicalization
- Pattern matching algorithms
- Multiway rewriting evolution
- Wolfram Physics model simulation
- Parallel processing with lock-free job system
- Causal and branchial graph computation

## Installation

### Prerequisites

- Mathematica 13.0 or later
- C++20-capable compiler (GCC 10+, Clang 12+, MSVC 19.29+)
- CMake 3.14 or later

### Building the Library

This project supports building for multiple platforms: Linux (x86-64, ARM64), Windows (x86-64, ARM64), and macOS (x86-64, ARM64).

#### Build Dependencies

**Required for all builds:**
- CMake 3.14+
- C++20-capable compiler

**Platform-specific cross-compilation toolchains (for multi-platform builds):**

| Platform | Required Toolchain | Installation |
|----------|-------------------|--------------|
| Linux x86-64 | Native GCC/Clang | `sudo apt install build-essential` |
| Linux ARM64 | GCC ARM64 cross-compiler | `sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu` |
| Windows x86-64 | MinGW-w64 | `sudo apt install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64` |
| Windows ARM64 | Clang + Windows SDK* | `sudo apt install clang` |
| macOS x86-64 | OSXCross | See [OSXCross setup](https://github.com/tpoechtrager/osxcross) |
| macOS ARM64 | OSXCross | See [OSXCross setup](https://github.com/tpoechtrager/osxcross) |

**\*Windows ARM64 notes:** When building from WSL2, the build script automatically detects Windows SDK and MSVC libraries from `/mnt/c`. Requires Visual Studio with ARM64 components installed on the Windows host. Outside of WSL2, requires Windows SDK headers and libraries for ARM64.

**The build script gracefully skips platforms** where toolchains are unavailable and provides installation instructions.

#### Multi-Platform Build (Recommended)

Build for all platforms using the provided script:

```bash
# From project root
./build_all_platforms.sh
```

This automatically builds Linux, Windows (via MinGW), and macOS (via OSXCross if available) libraries.

Build for specific platforms:
```bash
./build_all_platforms.sh --linux-only
./build_all_platforms.sh --windows-only
./build_all_platforms.sh --macos-only
```

#### Cross-Compilation

The build system supports cross-compilation from any host OS to all target platforms.

**Linux Native Build:**
```bash
mkdir build_linux && cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**Windows Cross-Compilation (from Linux/WSL):**
```bash
# Install MinGW
sudo apt install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64

# Build
mkdir build_windows && cd build_windows
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**macOS Cross-Compilation (from Linux via OSXCross):**
```bash
# Set up OSXCross (one-time setup)
# See CROSS_COMPILATION.md for detailed instructions

export OSXCROSS_ROOT="$HOME/osxcross"
export PATH="$OSXCROSS_ROOT/target/bin:$PATH"

# Build
mkdir build_macos && cd build_macos
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/macos-cross.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**Linux ARM64 Cross-Compilation (from Linux):**
```bash
# Install ARM64 cross-compiler
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Build
mkdir build_linux_arm64 && cd build_linux_arm64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/linux-cross.cmake \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**Windows ARM64 Cross-Compilation (from Linux):**
```bash
# Install Clang (if not already installed)
sudo apt install clang

# Build
mkdir build_windows_arm64 && cd build_windows_arm64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DWINDOWS_COMPILER=clang \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

For comprehensive cross-compilation documentation, including OSXCross setup and toolchain options, see [CROSS_COMPILATION.md](../CROSS_COMPILATION.md).

#### Platform Libraries

Built libraries are placed in platform-specific directories:
```
paclet/LibraryResources/
├── Linux-x86-64/libHypergraphRewriting.so
├── Linux-ARM64/libHypergraphRewriting.so
├── Windows-x86-64/HypergraphRewriting.dll
├── Windows-ARM64/HypergraphRewriting.dll
├── MacOSX-x86-64/libHypergraphRewriting.dylib
└── MacOSX-ARM64/libHypergraphRewriting.dylib
```

### Installing the Paclet

In Mathematica, evaluate:

```mathematica
(* Install from local directory *)
PacletInstall["/path/to/paclet/directory"]

(* Or create and install a .paclet file *)
pacletFile = CreatePacletArchive["/path/to/paclet/directory"]
PacletInstall[pacletFile]
```

## Usage

### Loading the Paclet

```mathematica
<< HypergraphRewriting`
```

### Basic Evolution

The core function is `HGEvolve`, which performs multiway rewriting evolution:

```mathematica
(* Define rules and initial state *)
rules = {
  {{x, y, z}} -> {{x, y}, {y, z}, {z, x}},
  {{x, y}} -> {{x, z}, {z, y}}
};

initialEdges = {{1, 2, 3}, {3, 4, 5}};

(* Run evolution for 3 steps *)
result = HGEvolve[rules, initialEdges, 3]
```

### Extracting Properties

`HGEvolve` returns different properties based on the `property` argument:

```mathematica
(* Get states *)
states = HGEvolve[rules, initialEdges, 3, "States"]

(* Get events *)
events = HGEvolve[rules, initialEdges, 3, "Events"]

(* Get causal edges *)
causalEdges = HGEvolve[rules, initialEdges, 3, "CausalEdges"]

(* Get branchial edges *)
branchialEdges = HGEvolve[rules, initialEdges, 3, "BranchialEdges"]

(* Get counts *)
numStates = HGEvolve[rules, initialEdges, 3, "NumStates"]
numEvents = HGEvolve[rules, initialEdges, 3, "NumEvents"]
```

### Visualizing Evolution

The paclet provides various graph visualizations:

```mathematica
(* States graph - multiway branching *)
statesGraph = HGEvolve[rules, initialEdges, 3, "StatesGraph"]

(* Causal graph - causal relationships between events *)
causalGraph = HGEvolve[rules, initialEdges, 3, "CausalGraph"]

(* Branchial graph - branchlike relationships between states *)
branchialGraph = HGEvolve[rules, initialEdges, 3, "BranchialGraph"]

(* Combined evolution graph *)
evolutionGraph = HGEvolve[rules, initialEdges, 3, "EvolutionCausalBranchialGraph"]
```

To get graph structure without rendering:
```mathematica
(* Structure variants return Graph objects without vertex styling *)
statesStructure = HGEvolve[rules, initialEdges, 3, "StatesGraphStructure"]
causalStructure = HGEvolve[rules, initialEdges, 3, "CausalGraphStructure"]
```

### Evolution Options

`HGEvolve` accepts the options below. Defaults shown in brackets; the complete
list lives in `paclet/Kernel/HypergraphRewriting.wl` under `Options[HGEvolve]`.

**Hashing and canonicalization**

| Option | Values | Default | Purpose |
|---|---|---|---|
| `"HashStrategy"` | `"WL"`, `"UT"`, `"iUT"` | `"WL"` | Fast heuristic used for state hashing. WL = Weisfeiler-Leman, UT = uniqueness tree, iUT = incremental UT. Heuristics can false-positive on highly symmetric graphs. |
| `"CanonicalizeStates"` | `None`, `Automatic`, `Full` | `None` | `None`: each raw state is its own cell. `Automatic`: no evolution-time dedup; content-ordered ID is computed at output time for display grouping. `Full`: exact isomorphism-based dedup via McKay-style IR canonicalization (no false positives). |
| `"CanonicalizeEvents"` | `None`, `Full`, `Automatic`, or `{keys...}` | `None` | Event dedup by signature. Key list may contain `"InputState"`, `"OutputState"`, `"Step"`, `"Rule"`, `"ConsumedEdges"`, `"ProducedEdges"`. |

**Causal / branchial output**

| Option | Values | Default | Purpose |
|---|---|---|---|
| `"CausalTransitiveReduction"` | `True` / `False` | `True` | Filter redundant causal edges at insertion time (Goranci online TR). |
| `"BranchialStep"` | `All`, `-1`, 1-based index, or `Automatic` | `Automatic` | Which step's branchial graph to return for `"BranchialGraph"` (`-1` = final) and for `"Evolution*Branchial*"` variants. |
| `"EdgeDeduplication"` | `True` / `False` | `True` | One edge per event-pair vs one edge per shared hypergraph edge. |

**Exploration bounds**

| Option | Default | Purpose |
|---|---|---|
| `"MaxSuccessorStatesPerParent"` | `0` (unlimited) | Cap the branching factor. |
| `"MaxStatesPerStep"` | `0` (unlimited) | Cap states created at each generation. |
| `"ExplorationProbability"` | `1.0` | Per-state probability of exploring further. |
| `"ExploreFromCanonicalStatesOnly"` | `False` | Only explore from canonical representatives; requires `"CanonicalizeStates" -> Full`. |

**Uniform-random evolution** (reservoir-sampled match selection)

| Option | Default | Purpose |
|---|---|---|
| `"UniformRandom"` | `False` | Use step-synchronised reservoir sampling over matches. |
| `"MatchesPerStep"` | `0` | Matches to apply per step in uniform-random mode (`0` = all). |

**Genesis / progress / debug**

| Option | Default | Purpose |
|---|---|---|
| `"ShowGenesisEvents"` | `False` | Include synthetic genesis events that "produce" each initial edge. |
| `"ShowProgress"` | `False` | Print per-step progress via WSTP (requires `HAVE_WSTP`). |
| `"DebugFFI"` | `False` | Print FFI data flow before each call. |
| `"AspectRatio"` | `None` | Graph aspect ratio for `*Graph` properties. |

**Analyses** (all `False` by default; enable to compute additional per-state / per-timestep data; see WL source for full sub-options):

- `"DimensionAnalysis"` — per-vertex Hausdorff dimension.
- `"CurvatureAnalysis"` — Ollivier-Ricci / Wolfram-Ricci / dimension-gradient curvature.
- `"GeodesicAnalysis"` — trace geodesic paths.
- `"TopologicalAnalysis"` — K5 / K3,3 minor detection.
- `"EntropyAnalysis"`, `"BranchialAnalysis"`, `"HilbertSpaceAnalysis"`, `"MultispaceAnalysis"`, `"BranchAlignment"`, `"EquilibriumAnalysis"`.

**Initial conditions** (alternative to passing `initialEdges` directly):

Set `"InitialCondition"` to `"Edges"` (default), `"Grid"`, `"Sprinkling"`, `"BrillLindquist"`, `"Poisson"`, or `"Uniform"`, and accompany with the per-condition parameters (`"GridWidth"`, `"GridHeight"`, `"SprinklingDensity"`, `"BrillLindquistMass1"`, `"BrillLindquistSeparation"`, etc.).

**Topology** (wrap the graph on a non-flat surface):

`"Topology" -> "Flat" | "Cylinder" | "Torus" | "Sphere" | "Klein" | "Mobius"` with `"MajorRadius"`, `"MinorRadius"`.

Malformed or unknown options that fail to parse on the FFI side are skipped and recorded in the debug message queue rather than aborting the evolve call. Enable `"DebugFFI"` or drain `getDebugMessages[]` to surface the skips.

### Available Properties

**Data Properties:**
- `"States"` - Association of state IDs to state edges
- `"Events"` - List of rewriting events
- `"CausalEdges"` - Pairs of causally connected event IDs
- `"BranchialEdges"` - Pairs of branchially connected event IDs
- `"NumStates"` - Total number of states
- `"NumEvents"` - Total number of events
- `"NumCausalEdges"` - Number of causal edges
- `"NumBranchialEdges"` - Number of branchial edges
- `"Debug"` - Association with all counts

**Graph Properties (with rendering):**
- `"StatesGraph"` - Multiway states graph
- `"CausalGraph"` - Causal graph of events
- `"BranchialGraph"` - Branchial graph of states
- `"EvolutionGraph"` - Combined evolution visualization
- `"EvolutionCausalGraph"` - Evolution with causal edges
- `"EvolutionBranchialGraph"` - Evolution with branchial edges
- `"EvolutionCausalBranchialGraph"` - Full evolution graph (default)

**Graph Structure Properties (without rendering):**
- `"StatesGraphStructure"`
- `"CausalGraphStructure"`
- `"BranchialGraphStructure"`
- `"EvolutionGraphStructure"`
- `"EvolutionCausalGraphStructure"`
- `"EvolutionBranchialGraphStructure"`
- `"EvolutionCausalBranchialGraphStructure"`

## Examples

### Simple Rewriting

```mathematica
(* Binary splitting rule *)
rules = {{{x, y}} -> {{x, z}, {z, y}}};
initialEdges = {{1, 2}};

(* Evolve and visualize *)
HGEvolve[rules, initialEdges, 3, "StatesGraph"]
```

### Wolfram Physics Models

```mathematica
(* Ternary to binary rule *)
rules = {{{x, y, z}} -> {{x, y}, {y, z}, {z, x}}};
initialEdges = {{1, 2, 3}};

(* Get causal graph *)
HGEvolve[rules, initialEdges, 5, "CausalGraph"]
```

### Multiple Rules

```mathematica
rules = {
  {{x, y, z}} -> {{x, y}, {y, z}, {z, x}},
  {{x, y}} -> {{x, z}, {z, y}}
};
initialEdges = {{1, 2, 3}, {2, 3, 4}};

(* Evolution with both rules *)
HGEvolve[rules, initialEdges, 4, "EvolutionCausalBranchialGraph"]
```

## Performance Notes

### Parallel Processing

- The library uses a custom lock-free job system for parallel rewriting
- Pattern matching is parallelized across multiple threads
- Canonicalization is performed in parallel for state comparison

### Optimization Tips

- Leave `"CanonicalizeStates"` at `None` (the default) when you don't need isomorphism-based deduplication. `Full` mode runs McKay-style IR canonicalization per new state and is only cheap on low-symmetry graphs.
- Use structure properties (e.g. `"StatesGraphStructure"`) to get `Graph` objects without styled vertex rendering.
- For large evolutions, request the count properties first (`"NumStates"`, `"NumEvents"`, `"NumCausalEdges"`, `"NumBranchialEdges"`) before pulling full state / event lists.
- `"MaxStatesPerStep"` and `"MaxSuccessorStatesPerParent"` cap the multiway explosion; combine with `"ExplorationProbability"` for probabilistic thinning.
- `"HashStrategy" -> "iUT"` reuses parent caches for incremental vertex hashing — faster than plain UT on evolutions with many child states.

### Memory Considerations

- States are stored with automatic memory management via smart pointers
- Canonicalization caches are managed internally
- For very large evolutions (>10,000 states), monitor memory usage

## Troubleshooting

### Library Loading Issues

If you encounter library loading errors:

1. **Check library exists**: Verify the library is compiled for your platform in `LibraryResources/$SystemID/`
2. **Rebuild library**: Run `./build_all_platforms.sh` from project root
3. **Manual override**: Set `MATHEMATICA_INSTALL_DIR` if CMake can't find Mathematica

### Cross-Compilation Issues

Common cross-compilation problems:

1. **MinGW not found**: Install with `sudo apt install gcc-mingw-w64-x86-64` (Linux)
2. **OSXCross setup**: See [CROSS_COMPILATION.md](../CROSS_COMPILATION.md) for detailed setup
3. **ARM64 Linux cross-compiler**: Install with `sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`
4. **ARM64 Windows (Clang)**: Install Clang with `sudo apt install clang`; requires Clang with Windows SDK headers
5. **Wrong platform library**: Ensure you're building with the correct toolchain file and `CMAKE_SYSTEM_PROCESSOR`

### Runtime Errors

If `HGEvolve` returns `$Failed`:

1. **Check library loaded**: Look for "Library functions loaded successfully" message when loading paclet
2. **Verify input format**: Rules use `lhs -> rhs` (plain `Rule`, not `RuleDelayed`). Symbolic vertices are normalised to 0-based integers automatically; integer vertices are passed through. Edges are lists of vertices.
3. **Check steps**: Steps must be a positive integer

### Build Issues

Common build problems:

1. **C++20 support**: Use GCC 10+, Clang 12+, or MSVC 2019 16.11+.
2. **CMake version**: Requires CMake 3.14 or later
3. **Mathematica headers**: CMake must find Mathematica installation (set `MATHEMATICA_INSTALL_DIR` if needed)
4. **Thread library**: On Linux, pthread must be available

## Architecture

### Components

- **C++ Core**: High-performance hypergraph rewriting engine
  - Pattern matching with edge signatures
  - Parallel job system with work-stealing deques
  - Canonicalization via nauty-style algorithms
  - Lock-free concurrent hash maps

- **LibraryLink Interface**: Bidirectional WXF serialization
  - Efficient binary protocol (no string parsing)
  - Handles arbitrary nested data structures
  - Zero-copy byte array transfer

- **Wolfram Language Frontend**: Graph visualization and user API
  - Multiple visualization modes
  - Flexible property extraction
  - Integration with WolframPhysics ecosystem

## Contributing

To contribute to this paclet:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and update documentation
5. Submit a pull request

## License

This paclet is part of the Efficient Rewriting project. See the main project license for details.
