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
- C++ compiler with C++17 support
- CMake 3.14 or later

### Building the Library

This project supports building for multiple platforms: Linux, Windows, and macOS.

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

For comprehensive cross-compilation documentation, including OSXCross setup and toolchain options, see [CROSS_COMPILATION.md](../CROSS_COMPILATION.md).

#### Platform Libraries

Built libraries are placed in platform-specific directories:
```
paclet/LibraryResources/
├── Linux-x86-64/libHypergraphRewriting.so
├── Windows-x86-64/HypergraphRewriting.dll
└── MacOSX-x86-64/libHypergraphRewriting.dylib
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
  {{x_, y_, z_}} :> {{x, y}, {y, z}, {z, x}},
  {{x_, y_}} :> {{x, z}, {z, y}}
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

`HGEvolve` supports various options:

```mathematica
HGEvolve[rules, initialEdges, steps, property,
  "CanonicalizeStates" -> True,         (* Canonicalize states for isomorphism detection *)
  "CanonicalizeEvents" -> False,        (* Canonicalize events *)
  "CausalTransitiveReduction" -> True,  (* Remove transitive causal edges *)
  "EarlyTermination" -> False,          (* Stop when a seen state is encountered *)
  "PatchBasedMatching" -> False,        (* Use patch-based pattern matching *)
  "FullCapture" -> True,                (* Capture all states including duplicates *)
  "AspectRatio" -> 1/2                  (* Graph aspect ratio *)
]
```

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
rules = {{{x_, y_}} :> {{x, z}, {z, y}}};
initialEdges = {{1, 2}};

(* Evolve and visualize *)
HGEvolve[rules, initialEdges, 3, "StatesGraph"]
```

### Wolfram Physics Models

```mathematica
(* Ternary to binary rule *)
rules = {{{x_, y_, z_}} :> {{x, y}, {y, z}, {z, x}}};
initialEdges = {{1, 2, 3}};

(* Get causal graph *)
HGEvolve[rules, initialEdges, 5, "CausalGraph"]
```

### Multiple Rules

```mathematica
rules = {
  {{x_, y_, z_}} :> {{x, y}, {y, z}, {z, x}},
  {{x_, y_}} :> {{x, z}, {z, y}}
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

- Enable `"EarlyTermination"` to stop evolution when cycles are detected
- Disable `"CanonicalizeStates"` if isomorphism detection is not needed
- Use structure properties (e.g., `"StatesGraphStructure"`) to avoid rendering overhead
- For large evolutions, extract counts first (`"NumStates"`, `"NumEvents"`) before full data

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
3. **Wrong platform library**: Ensure you're building with the correct toolchain file

### Runtime Errors

If `HGEvolve` returns `$Failed`:

1. **Check library loaded**: Look for "Library functions loaded successfully" message when loading paclet
2. **Verify input format**: Rules must be in `lhs :> rhs` format, edges must be lists of integers
3. **Check steps**: Steps must be a positive integer

### Build Issues

Common build problems:

1. **C++17 support**: Use GCC 7+, Clang 5+, or MSVC 2017+
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
