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
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
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
  -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
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
  -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
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
  -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
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
  -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
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
| `"HashStrategy"` | `"WL"` | `"WL"` | Fast Weisfeiler-Leman heuristic for state hashing. Can false-positive on highly symmetric graphs; use `CanonicalizeStates -> Full` for exact IR deduplication. |
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

### Complete `Options[HGEvolve]` reference

Every option accepted by `HGEvolve`, with its default and a one-line description. Kept in sync with `paclet/Kernel/HypergraphRewriting.wl`.

**Core evolution**

| Option | Default | Description |
|---|---|---|
| `"HashStrategy"` | `"WL"` | State hash algorithm: `"WL"` (Weisfeiler-Leman). |
| `"CanonicalizeStates"` | `None` | `None` / `Automatic` / `Full`. `Full` uses exact IR canonicalization for dedup. |
| `"CanonicalizeEvents"` | `None` | `None` / `Full` / `Automatic` / `{keys...}`. Keys: `"InputState"`, `"OutputState"`, `"Step"`, `"Rule"`, `"ConsumedEdges"`, `"ProducedEdges"`. |
| `"CausalTransitiveReduction"` | `True` | Online Goranci transitive reduction of causal edges. |
| `"MaxSuccessorStatesPerParent"` | `0` | Cap children per parent; `0` = unlimited. |
| `"MaxStatesPerStep"` | `0` | Cap states created per generation; `0` = unlimited. |
| `"ExplorationProbability"` | `1.0` | Probability of exploring further from each new state. |
| `"ExploreFromCanonicalStatesOnly"` | `False` | Only explore from canonical representatives (requires `"CanonicalizeStates" -> Full`). |
| `"EdgeDeduplication"` | `True` | One edge per event pair vs one edge per shared hypergraph edge. |
| `"BranchialStep"` | `Automatic` | Which step's branchial graph to return. `All`, `-1` (= final), or 1-based step. |

**Uniform-random evolution**

| Option | Default | Description |
|---|---|---|
| `"UniformRandom"` | `False` | Use step-synchronised reservoir sampling over matches. |
| `"MatchesPerStep"` | `0` | Matches to apply per step in uniform-random mode; `0` = all. |

**Output shape / progress / debug**

| Option | Default | Description |
|---|---|---|
| `"ShowProgress"` | `False` | Print per-step progress to the frontend via WSTP. |
| `"ShowGenesisEvents"` | `False` | Create synthetic genesis events for initial states. |
| `"AspectRatio"` | `None` | Graph aspect ratio for `*Graph` properties. |
| `"DebugFFI"` | `False` | Log FFI request/response metadata before each call. |
| `"IncludeStateContents"` | `False` | Include full edge lists in returned state records. |
| `"IncludeEventContents"` | `False` | Include full consumed/produced edge lists in returned event records. |
| `"RandomSeed"` | `Automatic` | Seed for reproducible runs. |

**Dimension analysis** (per-vertex Hausdorff dimension)

| Option | Default | Description |
|---|---|---|
| `"DimensionAnalysis"` | `False` | Enable per-vertex Hausdorff dimension computation. |
| `"DimensionFormula"` | `"LinearRegression"` | `"LinearRegression"` or `"DiscreteDerivative"`. |
| `"DimensionRadius"` | `{1, 5}` | `{minR, maxR}` radii for dimension estimation. |
| `"DimensionColorBy"` | `"Mean"` | `"Mean"`, `"Variance"`, `"Min"`, or `"Max"`. |
| `"DimensionPalette"` | `"TemperatureMap"` | `ColorData` palette for dimension rendering. |
| `"DimensionRange"` | `Automatic` | `{min, max}` or `Automatic` for color-scale range. |
| `"DimensionPerVertex"` | `False` | Include per-vertex dimension data in each state record. |
| `"DimensionTimestepAggregation"` | `False` | Include `PerTimestep` aggregation section. |

**Geodesic analysis** (trace test particles)

| Option | Default | Description |
|---|---|---|
| `"GeodesicAnalysis"` | `False` | Enable geodesic path tracing. |
| `"GeodesicSources"` | `Automatic` | Source vertex list; `Automatic` = auto-select near high-dim regions. |
| `"GeodesicMaxSteps"` | `50` | Maximum path length. |
| `"GeodesicBundleWidth"` | `5` | Number of paths in each bundle. |
| `"GeodesicFollowGradient"` | `False` | Follow dimension gradient vs random walk. |
| `"GeodesicDimensionPercentile"` | `0.9` | Percentile threshold for auto-selecting sources near high-dim regions. |

**Topological / particle analysis** (Robertson-Seymour defect detection)

| Option | Default | Description |
|---|---|---|
| `"TopologicalAnalysis"` | `False` | Detect K5 / K3,3 minors (non-planarity). |
| `"TopologicalCharge"` | `False` | Compute per-vertex topological charge. |
| `"DetectK5Minors"` | `True` | Check for K5 minors. |
| `"DetectK33Minors"` | `True` | Check for K3,3 bipartite minors. |
| `"DetectDimensionSpikes"` | `True` | Flag vertices whose local dimension exceeds the spike threshold. |
| `"DetectHighDegree"` | `True` | Flag high-degree vertices. |
| `"DimensionSpikeThreshold"` | `1.5` | Multiplier above mean to flag a spike. |
| `"DegreePercentile"` | `0.95` | Percentile for high-degree flagging (top 5% by default). |
| `"ChargeRadius"` | `3.0` | Radius for local topological-charge computation. |
| `"ChargePerVertex"` | `False` | Include per-vertex charge data in each state record. |
| `"ChargeTimestepAggregation"` | `False` | Include `PerTimestep` aggregation section for charge. |

**Curvature analysis** (Ollivier-Ricci, Wolfram-Ricci, dimension-gradient)

| Option | Default | Description |
|---|---|---|
| `"CurvatureAnalysis"` | `False` | Enable per-vertex curvature computation. |
| `"CurvatureMethod"` | `"All"` | `"OllivierRicci"`, `"WolframRicci"`, `"DimensionGradient"`, `"Both"`, or `"All"`. |
| `"CurvaturePerVertex"` | `False` | Include per-vertex curvature data in each state record. |
| `"CurvatureTimestepAggregation"` | `False` | Include `PerTimestep` aggregation section. |

**Branch alignment** (curvature shape-space PCA)

| Option | Default | Description |
|---|---|---|
| `"BranchAlignment"` | `False` | Compute PCA alignment across branches (requires `"CurvatureAnalysis" -> True`). |
| `"BranchAlignmentMethod"` | `"WolframRicci"` | Which curvature to align on: `"WolframRicci"` or `"OllivierRicci"`. |

**Entropy analysis**

| Option | Default | Description |
|---|---|---|
| `"EntropyAnalysis"` | `False` | Enable graph entropy / information-measure computation. |
| `"EntropyTimestepAggregation"` | `False` | Include `PerTimestep` aggregation section. |

**Hilbert space analysis** (state bitvector inner products)

| Option | Default | Description |
|---|---|---|
| `"HilbertSpaceAnalysis"` | `False` | Enable Hilbert-space analysis. |
| `"HilbertStep"` | `-1` | Step to analyse; `-1` = final. |
| `"HilbertScope"` | `"Global"` | `"Global"`, `"PerTimestep"`, or `"Both"`. |

**Branchial analysis** (distribution sharpness and branch entropy)

| Option | Default | Description |
|---|---|---|
| `"BranchialAnalysis"` | `False` | Enable branchial analysis. |
| `"BranchialScope"` | `"Global"` | `"Global"`, `"PerTimestep"`, or `"Both"`. |
| `"BranchialPerVertex"` | `False` | Include per-vertex sharpness data. |

**Multispace analysis** (vertex/edge probabilities across branches)

| Option | Default | Description |
|---|---|---|
| `"MultispaceAnalysis"` | `False` | Enable multispace analysis. |
| `"MultispaceScope"` | `"Global"` | `"Global"`, `"PerTimestep"`, or `"Both"`. |

**Initial conditions**

| Option | Default | Description |
|---|---|---|
| `"InitialCondition"` | `"Edges"` | `"Edges"`, `"Grid"`, `"Sprinkling"`, `"BrillLindquist"`, `"Poisson"`, or `"Uniform"`. |

**Topology** (wrap graph on a non-flat surface)

| Option | Default | Description |
|---|---|---|
| `"Topology"` | `"Flat"` | `"Flat"`, `"Cylinder"`, `"Torus"`, `"Sphere"`, `"Klein"`, or `"Mobius"`. |
| `"MajorRadius"` | `10.0` | Major radius for curved topologies. |
| `"MinorRadius"` | `3.0` | Minor radius for torus. |

**Grid initial condition**

| Option | Default | Description |
|---|---|---|
| `"GridWidth"` | `10` | Grid width. |
| `"GridHeight"` | `10` | Grid height. |
| `"GridHoles"` | `{}` | List of `{x, y, radius}` for circular holes in the grid. |

**Sprinkling / Minkowski initial condition**

| Option | Default | Description |
|---|---|---|
| `"SprinklingDensity"` | `500` | Number of spacetime points to sprinkle. |
| `"SprinklingSpatialDim"` | `2` | Spatial dimensionality: 1, 2, or 3. |
| `"SprinklingTimeExtent"` | `10.0` | Time dimension extent. |
| `"SprinklingSpatialExtent"` | `10.0` | Spatial dimension extent. |
| `"SprinklingLightconeAngle"` | `1.0` | Speed of light (c = 1 default). |
| `"SprinklingAlexandrovCutoff"` | `5.0` | Max proper-time separation between connected events. |
| `"SprinklingTransitivityReduction"` | `True` | Remove redundant causal edges. |
| `"SprinklingMaxEdgesPerVertex"` | `50` | Connectivity cap. |

**Brill-Lindquist initial condition** (two-black-hole approximation)

| Option | Default | Description |
|---|---|---|
| `"BrillLindquistMass1"` | `3.0` | Mass of first black hole. |
| `"BrillLindquistMass2"` | `3.0` | Mass of second black hole. |
| `"BrillLindquistSeparation"` | `10.0` | Distance between black holes. |
| `"BrillLindquistBoxX"` | `{-15.0, 15.0}` | X domain. |
| `"BrillLindquistBoxY"` | `{-15.0, 15.0}` | Y domain. |

**Sampling**

| Option | Default | Description |
|---|---|---|
| `"EdgeThreshold"` | `Automatic` | Max vertex separation for edge creation. |
| `"PoissonMinDistance"` | `1.0` | Minimum separation for Poisson-disk sampling. |

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

## Exported Functions Besides HGEvolve

The paclet exports a range of companion functions: initial-condition generators, analysis routines, and plot helpers. Every signature and return-shape below is taken from `paclet/Kernel/HypergraphRewriting.wl`. In Mathematica, `?HG*` produces the same summaries from the attached `::usage` strings.

### Initial condition generators

All return an `Association` containing at least `"Edges"` (a list of pair-edges) and `"VertexCoordinates"` (an `Association` from `VertexId` to `{x, y}` or `{x, y, z}`). Functions that produce embedded 3D surfaces also return `"VertexCoordinates3D"`. Initial-condition results are first-class inputs to `HGEvolve` (via `HGEvolve[rules, icResult, steps, …]`) and can be converted to a `Graph` via `HGToGraph`.

| Function | Signature | Returns (in addition to `"Edges"` + `"VertexCoordinates"`) |
|---|---|---|
| `HGGrid` | `HGGrid[width, height]` | — |
| `HGGridWithHoles` | `HGGridWithHoles[width, height, holes]` — `holes` is a list of `{centerX, centerY, radius}` | — |
| `HGCylinder` | `HGCylinder[resolution, height]` — wraps horizontally (theta direction), open vertically | `"VertexCoordinates3D"` |
| `HGTorus` | `HGTorus[resolution]` — both theta and phi wrap | `"VertexCoordinates3D"` |
| `HGSphere` | `HGSphere[resolution]` — UV-sampled | `"VertexCoordinates3D"` |
| `HGKleinBottle` | `HGKleinBottle[resolution, height]` — theta wraps with z-flip (non-orientable) | — |
| `HGMobiusStrip` | `HGMobiusStrip[resolution, width]` — theta wraps with z-flip, finite z | `"VertexCoordinates3D"` |
| `HGMinkowskiSprinkling` | `HGMinkowskiSprinkling[n, opts]` — causal set by Poisson sprinkling of `n` points, connected by causal structure | `"SpacetimePoints"`, `"DimensionEstimate"` |
| `HGBrillLindquist` | `HGBrillLindquist[n, {mass1, mass2}, separation, opts]` — discrete spacetime around two black holes; vertex density ∝ conformal factor ψ⁴ | `"HorizonCenters"` |
| `HGPoissonDisk` | `HGPoissonDisk[n, minDistance, opts]` — blue-noise distribution with a minimum vertex separation | — |
| `HGUniformRandom` | `HGUniformRandom[n, opts]` — uniform point cloud | — |

`HGMinkowskiSprinkling` options: `"SpatialDim"`, `"TimeExtent"`, `"SpatialExtent"`, `"LightconeAngle"`, `"AlexandrovCutoff"`, `"TransitivityReduction"`, `"MaxEdgesPerVertex"` (defaults match the `"Sprinkling*"` options of `HGEvolve`).

### Conversion

| Function | Signature | Description |
|---|---|---|
| `HGToGraph` | `HGToGraph[icResult]`, `HGToGraph[edges]`, `HGToGraph[edges, coords]` | Convert an initial-condition `Association`, a raw edge list, or an edge list with coordinates into a Mathematica `Graph`. |

### Analysis

| Function | Signature | Description |
|---|---|---|
| `HGHausdorffAnalysis` | `HGHausdorffAnalysis[edges, opts]` | Compute local Hausdorff dimension per vertex on an unevolved edge list. Same options as `HGEvolve`'s `Dimension*` family. |
| `HGBranchAlignment` | `HGBranchAlignment[edges, curvature]`, `HGBranchAlignment[evolutionResult, stateId]` | Curvature-weighted PCA embedding of a single state (used by the `Plot1D/2D/3D` helpers below). |
| `HGAlignAllBranches` | `HGAlignAllBranches[evolutionResult, step]` | Align every branch at a given step into a shared PCA frame. |
| `HGBranchAlignmentBatch` | `HGBranchAlignmentBatch[evolutionResult]` or `HGBranchAlignmentBatch[evolutionResult, curvatureMethod]` | PCA alignment for **every** state in an evolution, keyed by state id. Returns `<\|"PerState" -> …, "PerTimestep" -> …, "Global" -> …\|>`. `curvatureMethod` defaults to `"WolframRicci"`; `"OllivierRicci"` is the other supported choice. Requires the evolution to have been run with `"CurvatureAnalysis" -> True`. |

### Plots

Each plot takes the matching analysis result (or the whole evolution `Association` with the relevant analysis flags enabled) and produces a `Graphics` object.

| Function | Signature | Description |
|---|---|---|
| `HGStateDimensionPlot` | `HGStateDimensionPlot[edges, opts]` | Single hypergraph with vertices coloured by local dimension. |
| `HGTimestepUnionPlot` | `HGTimestepUnionPlot[evolutionResult, step, opts]` | Union of all states at `step`, dimension-coloured. |
| `HGDimensionFilmstrip` | `HGDimensionFilmstrip[evolutionResult, opts]` | Grid of per-step `HGTimestepUnionPlot`s. |
| `HGGeodesicPlot` | `HGGeodesicPlot[evolutionResult, stateId, opts]` | Geodesic paths overlaid on a state, dimension-coloured. Requires `"GeodesicAnalysis" -> True`. |
| `HGGeodesicFilmstrip` | `HGGeodesicFilmstrip[evolutionResult, opts]` | List of lists of `HGGeodesicPlot`s, one sub-list per timestep. |
| `HGLensingPlot` | `HGLensingPlot[evolutionResult, stateId, opts]` | Gravitational-lensing deflection-angle vs impact-parameter plot, with GR prediction overlaid (uses Brill-Lindquist horizon centers). |
| `HGBranchAlignmentPlot1D` | `HGBranchAlignmentPlot1D[alignment, opts]` — `alignment` is a single state's result or the per-state association | Curvature vs vertex rank, ordered along PC1. |
| `HGBranchAlignmentPlot2D` | `HGBranchAlignmentPlot2D[alignment, opts]` | PC1 vs PC2, coloured by curvature. |
| `HGBranchAlignmentPlot3D` | `HGBranchAlignmentPlot3D[alignment, opts]` | PC1 vs PC2 vs PC3, coloured by curvature. |

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
