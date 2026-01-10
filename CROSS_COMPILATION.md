# Cross-Compilation Guide

This project supports building for Linux, Windows, and macOS from any host platform using flexible cross-compilation toolchains.

## Dependencies

### All Platforms (Ubuntu/Debian)

Install all cross-compilation dependencies at once:

```bash
# All dependencies for all 6 target platforms
sudo apt install \
    cmake build-essential \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 \
    clang-22 lld-22 \
    llvm-dev libxml2-dev uuid-dev libssl-dev \
    libbz2-dev zlib1g-dev
```

**Note:** `clang-22` requires adding the LLVM apt repository first (see Windows ARM64 Setup below).

### Per-Platform Dependencies

| Target Platform | Required Packages |
|-----------------|-------------------|
| Linux x86-64 | `build-essential cmake` |
| Linux ARM64 | `gcc-aarch64-linux-gnu g++-aarch64-linux-gnu` |
| Windows x86-64 | `gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64` |
| Windows ARM64 | `clang-22 lld-22` + Windows SDK with ARM64 libs (see below) |
| macOS (both) | OSXCross + `clang llvm-dev libxml2-dev uuid-dev libssl-dev libbz2-dev zlib1g-dev` |

### Windows ARM64 Setup (WSL2)

Windows ARM64 cross-compilation requires clang 22+, the Windows SDK, and MSVC ARM64 libraries.

**1. Install clang-22 from LLVM repository:**
```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main"
sudo apt update
sudo apt install clang-22 lld-22

# Set as default
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100
sudo update-alternatives --install /usr/bin/lld-link lld-link /usr/bin/lld-link-22 100
```

**2. Install Visual Studio ARM64 build tools (on Windows):**
- Open Visual Studio Installer → Modify → Individual Components
- Search "ARM64" and select "MSVC v143 - VS 2022 C++ ARM64 build tools (Latest)"

Or via command line:
```powershell
vs_installer.exe modify --installPath "C:\Program Files\Microsoft Visual Studio\2022\Community" --add Microsoft.VisualStudio.Component.VC.Tools.ARM64
```

The toolchain automatically finds Windows SDK and MSVC at `/mnt/c/Program Files/...`

### OSXCross Setup

For macOS cross-compilation, OSXCross must be installed at `~/osxcross` (or set `OSXCROSS_ROOT`):

```bash
cd ~
git clone https://github.com/tpoechtrager/osxcross
cd osxcross/tarballs
wget https://github.com/joseluisq/macosx-sdks/releases/download/12.3/MacOSX12.3.sdk.tar.xz
cd ..
./build.sh
```

See the [OSXCross section](#from-linux-using-osxcross) below for detailed instructions.

## Quick Start

### Build All Platforms

```bash
./build_all_platforms.sh
```

This will build for all platforms where toolchains are available.

### Build Specific Platform

```bash
./build_all_platforms.sh --linux-only
./build_all_platforms.sh --windows-only
./build_all_platforms.sh --macos-only
```

## Platform-Specific Setup

### Linux Builds

**Native Linux:**
```bash
mkdir build_linux && cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**From Windows/WSL:**
```bash
# WSL can build Linux binaries natively
mkdir build_linux && cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**From macOS (cross-compilation):**
```bash
# Install Linux cross-compiler or use Docker
# Docker approach recommended:
docker run --rm -v $(pwd):/src -w /src ubuntu:22.04 bash -c "
  apt update && apt install -y cmake g++ make &&
  mkdir -p build_linux && cd build_linux &&
  cmake .. -DBUILD_MATHEMATICA_PACLET=ON &&
  make -j paclet
"
```

### Windows Builds

**From Linux/WSL using MinGW:**
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

**From macOS using MinGW:**
```bash
# Install MinGW via Homebrew
brew install mingw-w64

# Build
mkdir build_windows && cd build_windows
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON
make -j$(sysctl -n hw.ncpu) paclet
```

**Using Clang instead of MinGW:**
```bash
mkdir build_windows && cd build_windows
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
  -DWINDOWS_COMPILER=clang \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON
make -j$(nproc) paclet
```

**Native Windows (MSVC or MinGW):**
```powershell
mkdir build_windows
cd build_windows
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MATHEMATICA_PACLET=ON
cmake --build . --config Release --target paclet
```

### macOS Builds

**Native macOS:**
```bash
mkdir build_macos && cd build_macos
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MATHEMATICA_PACLET=ON
make -j$(sysctl -n hw.ncpu) paclet
```

**From Linux using OSXCross:**

OSXCross allows cross-compiling to macOS from Linux. One-time setup required (~30-60 minutes).

**Step 1: Install Dependencies**
```bash
# Ubuntu/Debian
sudo apt install clang llvm-dev libxml2-dev uuid-dev libssl-dev \
  bash patch make tar xz-utils bzip2 gzip sed cpio libbz2-dev zlib1g-dev

# Arch Linux
sudo pacman -S clang llvm libxml2 libbz2 zlib

# Fedora/RHEL
sudo dnf install clang llvm-devel libxml2-devel libuuid-devel openssl-devel \
  bzip2-devel zlib-devel
```

**Step 2: Clone OSXCross**
```bash
cd ~  # or wherever you want to install
git clone https://github.com/tpoechtrager/osxcross
cd osxcross
```

**Step 3: Obtain macOS SDK**

You need an official macOS SDK from Xcode. Several options:

**Option A: Pre-packaged SDKs (easiest)**
```bash
cd tarballs

# Download pre-packaged SDK (macOS 12.3, works for most cases)
wget https://github.com/joseluisq/macosx-sdks/releases/download/12.3/MacOSX12.3.sdk.tar.xz

# Or macOS 11.3 (for older compatibility)
# wget https://github.com/joseluisq/macosx-sdks/releases/download/11.3/MacOSX11.3.sdk.tar.xz
```

**Option B: Extract from Xcode (if you have access to a Mac)**
```bash
# On macOS with Xcode installed:
./tools/gen_sdk_package_pbzx.sh  # Creates SDK tarball

# Copy the resulting .tar.* file to Linux machine at:
# osxcross/tarballs/MacOSXxx.x.sdk.tar.xz
```

**Option C: Download from Apple (requires Apple Developer account)**
- Download Xcode from https://developer.apple.com/download/
- Extract SDK using OSXCross tools

**Step 4: Build OSXCross**
```bash
# Verify SDK is in tarballs/
ls -lh tarballs/

# Build the cross-compiler toolchain (takes 30-60 minutes)
./build.sh

# Optional: Build additional tools
./build_compiler_rt.sh  # For sanitizers
```

**Step 5: Set Environment**
```bash
# Add to ~/.bashrc or ~/.zshrc for permanent setup
export OSXCROSS_ROOT="$HOME/osxcross"
export PATH="$OSXCROSS_ROOT/target/bin:$PATH"

# Or for current session only
export OSXCROSS_ROOT=$(pwd)
export PATH="$OSXCROSS_ROOT/target/bin:$PATH"
```

**Step 6: Verify Installation**
```bash
# Check compilers are available
x86_64-apple-darwin21-clang --version
x86_64-apple-darwin21-clang++ --version

# Should see "clang version" output
```

**Step 7: Build Project for macOS**
```bash
cd /path/to/hypergraph_project
mkdir build_macos && cd build_macos

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/macos-cross.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MATHEMATICA_PACLET=ON

make -j$(nproc) paclet
```

**Troubleshooting OSXCross:**
- **SDK version mismatch**: Use SDK version matching your target macOS
- **Missing tools**: Ensure all dependencies installed
- **Linker errors**: Try `./build_compiler_rt.sh` for additional libraries
- **PATH issues**: Verify `OSXCROSS_ROOT` points to correct directory

**SDK Version Selection:**
- macOS 10.15+: Use MacOSX11.3+ SDK
- macOS 12+: Use MacOSX12.3+ SDK
- Older macOS: Use matching SDK version

The toolchain will automatically find the SDK if `OSXCROSS_ROOT` is set correctly.

**From Windows:**
OSXCross doesn't officially support Windows. Options:
1. Use WSL (see Linux instructions above)
2. Use Docker with OSXCross image
3. Build on actual macOS hardware

## Compiler Selection

### Override Compiler via Environment Variables

```bash
# Use specific compiler
export CC=clang
export CXX=clang++
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake
```

### Windows Compiler Options

```bash
# MinGW (default)
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake -DWINDOWS_COMPILER=mingw

# Clang
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake -DWINDOWS_COMPILER=clang

# MSVC/clang-cl
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake -DWINDOWS_COMPILER=msvc
```

## Architecture Support

### x86_64 (default)

```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake
```

### ARM64 (aarch64)

```bash
# Set architecture
export TARGET_ARCH=aarch64
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64
```

**Note:** ARM64 support requires appropriate cross-compilers:
- Windows ARM64: `aarch64-w64-mingw32-gcc`
- Linux ARM64: `aarch64-linux-gnu-gcc`
- macOS ARM64 (Apple Silicon): Native or OSXCross with ARM SDK

## Toolchain Files

The project includes three toolchain files:

1. **cmake/toolchains/windows-cross.cmake** - Cross-compile to Windows
2. **cmake/toolchains/linux-cross.cmake** - Cross-compile to Linux
3. **cmake/toolchains/macos-cross.cmake** - Cross-compile to macOS

Each toolchain:
- Auto-detects available compilers (GCC, Clang, MSVC)
- Respects `CC`/`CXX` environment variables
- Supports multiple architectures (x86_64, ARM64)
- Provides clear error messages with installation instructions

## Mathematica/Wolfram Integration

The build system automatically finds Mathematica/Wolfram installations:

**Manual Override:**
```bash
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
  -DMATHEMATICA_INSTALL_DIR="/path/to/Mathematica"
```

**WSL Users:**
```bash
# Automatically detects Windows Mathematica
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
  -DMATHEMATICA_INSTALL_DIR="/mnt/c/Program Files/Wolfram Research/Mathematica/13.3"
```

## Output Locations

Built libraries are placed in:
```
paclet/LibraryResources/
├── Linux-x86-64/libHypergraphRewriting.so
├── Windows-x86-64/HypergraphRewriting.dll
└── MacOSX-x86-64/libHypergraphRewriting.dylib
```

ARM64 builds go in corresponding directories:
```
paclet/LibraryResources/
├── Linux-ARM64/
├── Windows-ARM64/
└── MacOSX-ARM64/
```

## Troubleshooting

### MinGW Not Found

```bash
# Ubuntu/Debian
sudo apt install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64

# Arch Linux
sudo pacman -S mingw-w64-gcc

# macOS
brew install mingw-w64
```

### OSXCross Setup Issues

See https://github.com/tpoechtrager/osxcross for detailed setup instructions.

Common issues:
- Missing Xcode Command Line Tools
- SDK version mismatch
- PATH not set correctly

### Library Not Loading

**Windows DLLs missing dependencies:**
- Use `-static-libgcc -static-libstdc++` (automatic in toolchain)
- Check for missing runtime DLLs with `ldd` or Dependency Walker

**macOS dylib loading issues:**
- Verify SDK version matches deployment target
- Check code signing if running on recent macOS

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Build All Platforms

on: [push]

jobs:
  build-matrix:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            platform: linux
          - os: ubuntu-latest
            platform: windows
          - os: macos-latest
            platform: macos

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          if [ "${{ matrix.platform }}" = "windows" ]; then
            sudo apt install gcc-mingw-w64-x86-64
          fi

      - name: Build
        run: ./build_all_platforms.sh --${{ matrix.platform }}-only
```

## Development Tips

### Faster Iteration

Build only what you need:
```bash
# Build just the library, skip tests
make -j$(nproc) HypergraphRewriting

# Build specific target
make -j$(nproc) hypergraph
```

### Parallel Builds

```bash
# Use all CPU cores
make -j$(nproc)

# Limit parallel jobs
make -j4
```

### Clean Rebuild

```bash
# Clean build directory
rm -rf build_*
./build_all_platforms.sh
```

## Platform-Agnostic Code

This project is designed to be platform-agnostic:
- No platform-specific code in core libraries
- All platform differences handled via CMake
- Standard C++17, no OS-specific APIs
- Cross-platform threading (C++11 threads)

Adding new platforms requires only:
1. Create toolchain file in `cmake/toolchains/`
2. Update `paclet/LibraryResources/CMakeLists.txt` platform detection
3. Test build and runtime
