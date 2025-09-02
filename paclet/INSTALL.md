# HypergraphRewriting Paclet Installation Guide

## Quick Start

1. **Build from project root (recommended):**
   ```bash
   mkdir build && cd build
   cmake ..
   make paclet  # Build just the paclet
   # Or: make -j4  # Build entire project including paclet
   ```

2. **Install the paclet in Mathematica:**
   ```mathematica
   PacletInstall["/path/to/efficient_rewriting_final/paclet"]
   << HypergraphRewriting`
   ```

3. **Test the installation:**
   ```mathematica
   hg = HGCreate[{{1,2,3}, {2,3,4}}]
   HGCanonical[hg]
   ```

## Detailed Installation

### Prerequisites

- **Mathematica 13.0+** with LibraryLink support
- **C++ Compiler** with C++17 support:
  - Linux: GCC 7+ or Clang 5+
  - macOS: Xcode Command Line Tools
  - Windows: Visual Studio 2017+ or MinGW-w64
- **CMake 3.12+**
- **pthreads** (usually available on Unix systems)

### Step 1: Build the Native Library

#### Option A: Build from Project Root (Recommended)

Navigate to the project root:
```bash
cd /path/to/efficient_rewriting_final
```

Create and enter build directory:
```bash
mkdir build
cd build
```

Configure with CMake:
```bash
# Basic configuration
cmake ..

# Or specify Mathematica location if not auto-detected
cmake -DMathematica_INSTALL_DIR="/Applications/Mathematica.app/Contents" ..

# For release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# To disable paclet building
cmake -DBUILD_MATHEMATICA_PACLET=OFF ..

# For WSL2 with Windows Wolfram installation
cmake -DMATHEMATICA_INSTALL_DIR="/mnt/c/Program Files/Wolfram Research/Wolfram/14.1" ..
```

Build the library:
```bash
make paclet  # Build just the paclet
# Or: make -j4  # Build entire project
```

#### Option B: Build Paclet Separately

Navigate to the LibraryResources directory:
```bash
cd paclet/LibraryResources
```

Create and enter build directory:
```bash
mkdir build
cd build
```

Configure and build:
```bash
cmake ..
make -j4
```

The library will be automatically placed in the correct platform directory.

### Step 2: Install the Paclet

#### Option A: Install from Directory
```mathematica
(* Replace with your actual path *)
pacletDir = "/path/to/efficient_rewriting_final/paclet";
PacletInstall[pacletDir]
```

#### Option B: Create and Install .paclet File
```mathematica
pacletDir = "/path/to/efficient_rewriting_final/paclet";
pacletFile = CreatePacletArchive[pacletDir];
PacletInstall[pacletFile]
```

#### Option C: Development Installation
For development, use temporary loading:
```mathematica
PacletDirectoryAdd["/path/to/efficient_rewriting_final/paclet"]
<< HypergraphRewriting`
```

### Step 3: Load and Test

Load the paclet:
```mathematica
<< HypergraphRewriting`
```

Basic functionality test:
```mathematica
(* Create a simple hypergraph *)
hg = HGCreate[{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}]

(* Test canonicalization *)
canonical = HGCanonical[hg]

(* Test pattern matching *)
pattern = {{1, 2}}
matches = HGPatternMatch[hg, pattern]

(* Test rewriting *)
rule = RewritingRule[{{1, 2, 3}}, {{1, 2}, {2, 3}, {3, 1}}]
result = HGApplyRule[hg, rule]

Print["Installation successful!"]
```

## Platform-Specific Notes

### Linux
- Install development packages: `sudo apt-get install build-essential cmake`
- Ensure pthread is available (usually default)
- May need to set `LD_LIBRARY_PATH` if Mathematica libraries not found

### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- CMake available via Homebrew: `brew install cmake`
- May need to accept Xcode license: `sudo xcodebuild -license accept`

### Windows
- Use Visual Studio Developer Command Prompt
- May need to specify generator: `cmake -G "Visual Studio 16 2019" ..`
- Ensure Mathematica installation directory is accessible

## Troubleshooting

### CMake Can't Find Mathematica
```bash
# Specify Mathematica installation explicitly
cmake -DMathematica_INSTALL_DIR="/usr/local/Wolfram/Mathematica/13.0" ..
```

### Library Loading Errors
1. Check library exists in correct platform directory:
   ```
   paclet/LibraryResources/Linux-x86-64/libHypergraphRewriting.so
   ```

2. Verify library dependencies:
   ```bash
   ldd libHypergraphRewriting.so  # Linux
   otool -L libHypergraphRewriting.dylib  # macOS
   ```

3. Check Mathematica can load the library:
   ```mathematica
   FindLibrary["HypergraphRewriting"]
   ```

### Compilation Errors
1. **C++17 not supported:** Upgrade compiler or add flags:
   ```bash
   cmake -DCMAKE_CXX_FLAGS="-std=c++17" ..
   ```

2. **Header not found:** Install Mathematica development files or specify path:
   ```bash
   cmake -DMathematica_INCLUDE_DIRS="/path/to/mathematica/includes" ..
   ```

3. **Linking errors:** Ensure all dependencies are available and linkable

### Runtime Issues
1. **Function not found:** Rebuild library and reinstall paclet
2. **Memory errors:** Check for proper resource cleanup in usage
3. **Performance issues:** Enable parallel processing with `HGSetParallel[True]`

## Performance Optimization

### Compiler Optimizations
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
```

### Parallel Processing
```mathematica
(* Enable parallel processing *)
HGSetParallel[True]

(* Monitor performance *)
stats = HGGetStats[]

(* Clear caches if memory usage is high *)
HGClearCache[]
```

### Memory Management
- Use `HGClearCache[]` periodically for long runs
- Monitor memory usage with system tools
- Consider limiting evolution steps for large hypergraphs

## Verification

Run the complete test suite:
```mathematica
(* Load example notebook *)
NotebookOpen["Documentation/HypergraphRewritingExamples.nb"]

(* Or run basic verification *)
TestID[HGCreate[{{1,2,3}}], HypergraphObject[_]]
TestID[HGCanonical[HGCreate[{{1,2,3}}]], {{0,1,2}}]
```

## Uninstallation

To remove the paclet:
```mathematica
PacletUninstall["HypergraphRewriting"]
```

## Getting Help

1. Check the documentation: `?HypergraphRewriting`
2. View examples: Open `Documentation/HypergraphRewritingExamples.nb`
3. Check performance: `HGGetStats[]`
4. Report issues: Include system info, Mathematica version, and error messages