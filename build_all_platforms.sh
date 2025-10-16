#!/bin/bash
# Build hypergraph rewriting library for all platforms
# Creates a complete multi-platform paclet

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Hypergraph Rewriting Library for All Platforms ===${NC}\n"

# Configuration
BUILD_LINUX=true
BUILD_WINDOWS=true
BUILD_MACOS=true
BUILD_JOBS=${BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Parse arguments
for arg in "$@"; do
    case $arg in
        --linux-only)
            BUILD_LINUX=true
            BUILD_WINDOWS=false
            BUILD_MACOS=false
            ;;
        --windows-only)
            BUILD_LINUX=false
            BUILD_WINDOWS=true
            BUILD_MACOS=false
            ;;
        --macos-only)
            BUILD_LINUX=false
            BUILD_WINDOWS=false
            BUILD_MACOS=true
            ;;
        --no-linux)
            BUILD_LINUX=false
            ;;
        --no-windows)
            BUILD_WINDOWS=false
            ;;
        --no-macos)
            BUILD_MACOS=false
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build for all platforms (default) or specific platforms:"
            echo "  --linux-only      Build only Linux"
            echo "  --windows-only    Build only Windows"
            echo "  --macos-only      Build only macOS"
            echo "  --no-linux        Skip Linux build"
            echo "  --no-windows      Skip Windows build"
            echo "  --no-macos        Skip macOS build"
            echo ""
            echo "Environment variables:"
            echo "  BUILD_JOBS        Number of parallel jobs (default: auto-detect)"
            echo "  OSXCROSS_ROOT     Path to OSXCross (for macOS builds)"
            exit 0
            ;;
    esac
done

# Check prerequisites
check_prerequisite() {
    local name=$1
    local command=$2
    local install_hint=$3

    if ! command -v "$command" &> /dev/null; then
        echo -e "${YELLOW}Warning: $name not found${NC}"
        echo -e "  Install: $install_hint"
        return 1
    fi
    return 0
}

echo -e "${GREEN}Checking prerequisites...${NC}"
check_prerequisite "CMake" "cmake" "sudo apt install cmake (Linux) or brew install cmake (macOS)"

# Linux build
if $BUILD_LINUX; then
    echo -e "\n${GREEN}=== Building for Linux (x86_64) ===${NC}"
    mkdir -p build_linux
    cd build_linux

    if [ ! -f CMakeCache.txt ]; then
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_MATHEMATICA_PACLET=ON \
            || { echo -e "${RED}Linux CMake configuration failed${NC}"; exit 1; }
    fi

    make -j"$BUILD_JOBS" paclet \
        || { echo -e "${RED}Linux build failed${NC}"; exit 1; }

    echo -e "${GREEN}✓ Linux build complete${NC}"
    cd "$PROJECT_ROOT"
fi

# Windows build
if $BUILD_WINDOWS; then
    echo -e "\n${GREEN}=== Building for Windows (x86_64) ===${NC}"

    if ! check_prerequisite "MinGW" "x86_64-w64-mingw32-gcc" "sudo apt install gcc-mingw-w64-x86-64 (Linux)"; then
        echo -e "${RED}Skipping Windows build - MinGW not available${NC}"
        BUILD_WINDOWS=false
    else
        mkdir -p build_windows
        cd build_windows

        if [ ! -f CMakeCache.txt ]; then
            cmake .. \
                -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/windows-cross.cmake \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_MATHEMATICA_PACLET=ON \
                || { echo -e "${RED}Windows CMake configuration failed${NC}"; exit 1; }
        fi

        make -j"$BUILD_JOBS" paclet \
            || { echo -e "${RED}Windows build failed${NC}"; exit 1; }

        echo -e "${GREEN}✓ Windows build complete${NC}"
        cd "$PROJECT_ROOT"
    fi
fi

# macOS build (x86_64)
if $BUILD_MACOS; then
    echo -e "\n${GREEN}=== Building for macOS (x86_64) ===${NC}"

    # Check for OSXCross
    OSXCROSS_ROOT=${OSXCROSS_ROOT:-/opt/osxcross}
    if [ ! -d "$OSXCROSS_ROOT" ]; then
        echo -e "${YELLOW}Warning: OSXCross not found at $OSXCROSS_ROOT${NC}"
        echo -e "${YELLOW}Set OSXCROSS_ROOT environment variable or install OSXCross${NC}"
        echo -e "${YELLOW}See: https://github.com/tpoechtrager/osxcross${NC}"
        echo -e "${RED}Skipping macOS build${NC}"
        BUILD_MACOS=false
    else
        export OSXCROSS_ROOT
        export PATH="$OSXCROSS_ROOT/target/bin:$PATH"
        mkdir -p build_macos
        cd build_macos

        if [ ! -f CMakeCache.txt ]; then
            cmake .. \
                -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/macos-cross.cmake \
                -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_MATHEMATICA_PACLET=ON \
                || { echo -e "${RED}macOS x86_64 CMake configuration failed${NC}"; exit 1; }
        fi

        make -j"$BUILD_JOBS" paclet \
            || { echo -e "${RED}macOS x86_64 build failed${NC}"; exit 1; }

        echo -e "${GREEN}✓ macOS x86_64 build complete${NC}"
        cd "$PROJECT_ROOT"
    fi
fi

# macOS build (ARM64)
if $BUILD_MACOS; then
    echo -e "\n${GREEN}=== Building for macOS (ARM64) ===${NC}"

    # Check for OSXCross (already validated above)
    if [ -d "$OSXCROSS_ROOT" ]; then
        export OSXCROSS_ROOT
        export PATH="$OSXCROSS_ROOT/target/bin:$PATH"
        mkdir -p build_macos_arm64
        cd build_macos_arm64

        if [ ! -f CMakeCache.txt ]; then
            cmake .. \
                -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/macos-cross.cmake \
                -DCMAKE_SYSTEM_PROCESSOR=arm64 \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_MATHEMATICA_PACLET=ON \
                || { echo -e "${RED}macOS ARM64 CMake configuration failed${NC}"; exit 1; }
        fi

        make -j"$BUILD_JOBS" paclet \
            || { echo -e "${RED}macOS ARM64 build failed${NC}"; exit 1; }

        echo -e "${GREEN}✓ macOS ARM64 build complete${NC}"
        cd "$PROJECT_ROOT"
    fi
fi

# Summary
echo -e "\n${GREEN}=== Build Summary ===${NC}"
[ -f paclet/LibraryResources/Linux-x86-64/libHypergraphRewriting.so ] && echo -e "${GREEN}✓ Linux-x86-64${NC}"
[ -f paclet/LibraryResources/Windows-x86-64/HypergraphRewriting.dll ] && echo -e "${GREEN}✓ Windows-x86-64${NC}"
[ -f paclet/LibraryResources/MacOSX-x86-64/libHypergraphRewriting.dylib ] && echo -e "${GREEN}✓ MacOSX-x86-64${NC}"
[ -f paclet/LibraryResources/MacOSX-ARM64/libHypergraphRewriting.dylib ] && echo -e "${GREEN}✓ MacOSX-ARM64${NC}"

echo -e "\n${GREEN}Paclet libraries located in: paclet/LibraryResources/${NC}"
echo -e "\nTo create paclet archive: cd build_<platform> && make paclet_archive"
