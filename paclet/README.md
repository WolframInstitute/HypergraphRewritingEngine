# HypergraphRewriting Paclet

A high-performance Mathematica paclet for hypergraph rewriting and Wolfram Physics model simulation.

## Overview

This paclet provides efficient hypergraph rewriting capabilities with parallel processing support. It includes:

- Fast hypergraph canonicalization 
- Pattern matching algorithms
- Multiway rewriting evolution
- Wolfram Physics model simulation
- Parallel processing support

## Installation

### Prerequisites

- Mathematica 13.0 or later
- C++ compiler with C++17 support
- CMake 3.12 or later

### Building the Library

#### Option 1: Build from Project Root (Recommended)

1. Build the entire project including the paclet:
```bash
mkdir build && cd build
cmake ..
make -j4
```

2. Build just the paclet library:
```bash
make paclet
```

#### Option 2: Build Paclet Separately

1. Navigate to the paclet directory:
```bash
cd paclet/LibraryResources
```

2. Create a build directory and compile:
```bash
mkdir build
cd build
cmake ..
make -j4
```

The paclet library will be automatically placed in the correct platform directory (`LibraryResources/Linux-x86-64/`, etc.).

### Installing the Paclet

In Mathematica, evaluate:

```mathematica
(* Install from local directory *)
PacletInstall["/path/to/your/paclet/directory"]

(* Or create and install a .paclet file *)
pacletFile = CreatePacletArchive["/path/to/your/paclet/directory"]
PacletInstall[pacletFile]
```

## Usage

### Loading the Paclet

```mathematica
<< HypergraphRewriting`
```

### Basic Operations

#### Creating Hypergraphs

```mathematica
(* Create a simple hypergraph *)
hg = HGCreate[{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}]
```

#### Canonicalization

```mathematica
(* Get canonical form *)
canonical = HGCanonical[hg]
```

#### Pattern Matching

```mathematica
(* Find pattern matches *)
pattern = {{1, 2}, {2, 3}}
matches = HGPatternMatch[hg, pattern]
```

#### Rewriting Rules

```mathematica
(* Define and apply rewriting rules *)
rule = RewritingRule[{{1, 2, 3}}, {{1, 2}, {2, 3}, {3, 1}}]
newHG = HGApplyRule[hg, rule]
```

#### Multiway Evolution

```mathematica
(* Evolve using multiple rules *)
rules = {
    RewritingRule[{{1, 2, 3}}, {{1, 2}, {2, 3}, {3, 1}}],
    RewritingRule[{{1, 2}}, {{1, 3, 2}}]
}
evolution = HGMultiwayEvolution[hg, rules, 3]
```

#### Wolfram Models

```mathematica
(* Create a Wolfram Physics model *)
initial = {{1, 2, 3}, {3, 4, 5}}
rules = {
    RewritingRule[{{x_, y_, z_}}, {{x, y}, {y, z}, {z, x}}]
}
model = HGWolframModel[initial, rules, 5]
```

### Performance Options

#### Parallel Processing

```mathematica
(* Enable parallel processing *)
HGSetParallel[True]

(* Disable parallel processing *)
HGSetParallel[False]
```

#### Statistics and Cache Management

```mathematica
(* Get performance statistics *)
stats = HGGetStats[]

(* Clear internal caches *)
HGClearCache[]
```

## API Reference

### Core Functions

- `HGCreate[edges]` - Create hypergraph from edge list
- `HGCanonical[hg]` - Get canonical form
- `HGPatternMatch[hg, pattern]` - Find pattern matches
- `HGApplyRule[hg, rule]` - Apply single rewriting rule
- `HGApplyRules[hg, rules, steps]` - Apply multiple rules for specified steps
- `HGMultiwayEvolution[hg, rules, steps]` - Multiway evolution
- `HGWolframModel[init, rules, steps]` - Create Wolfram model

### Utility Functions

- `HGSetParallel[enabled]` - Enable/disable parallel processing
- `HGGetStats[]` - Get performance statistics
- `HGClearCache[]` - Clear internal caches

### Data Types

- `HypergraphObject[...]` - Represents a hypergraph
- `RewritingRule[lhs, rhs]` - Represents a rewriting rule
- `MultiwayState[...]` - Represents a multiway evolution state
- `WolframModel[...]` - Represents a Wolfram physics model

## Examples

See the included example notebook `HypergraphRewritingExamples.nb` for detailed usage examples and tutorials.

## Performance Notes

- The library uses efficient C++ implementations for core algorithms
- Pattern matching utilizes edge signatures for fast lookups
- Parallel processing can significantly speed up multiway evolution
- Canonicalization results are cached for improved performance

## Troubleshooting

### Library Loading Issues

If you encounter library loading errors:

1. Ensure the C++ library is compiled for your platform
2. Check that all dependencies are installed
3. Verify Mathematica can find the library in the correct platform directory

### Compilation Issues

Common compilation problems:

1. **Mathematica headers not found**: Ensure Mathematica is properly installed and CMake can find it
2. **C++17 support**: Use a modern compiler (GCC 7+, Clang 5+, MSVC 2017+)
3. **Thread library**: On Linux, ensure pthread is available

### Memory Usage

For large hypergraphs or long evolution runs:

- Monitor memory usage with `HGGetStats[]`
- Clear caches periodically with `HGClearCache[]`
- Consider limiting the number of evolution steps

## Contributing

To contribute to this paclet:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This paclet is part of the Efficient Rewriting project. See the main project license for details.