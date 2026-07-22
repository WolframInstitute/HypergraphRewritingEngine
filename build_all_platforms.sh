#!/bin/bash
# Build the multi-platform Wolfram paclet library.
#
# Host-aware and fault-tolerant: it builds every target the current host can reach, SKIPS
# the ones it cannot (missing cross-toolchain, or a target this host simply cannot produce)
# with a clear reason, and never lets one target's failure abort the others. The run only
# fails (non-zero exit) if a target we actually ATTEMPTED errors out.
#
# Linux host cross-compiles all six via native gcc / aarch64-linux-gnu / mingw-w64 / clang /
# OSXCross. macOS host builds the macOS slices natively and the rest via cross-toolchains
# where available.
#
#   Usage: ./build_all_platforms.sh [FILTER]
#     FILTER  optional substring/regex; only targets whose name matches are built
#             (e.g. "Windows", "MacOSX", "Linux-x86-64"). Legacy --linux-only /
#             --windows-only / --macos-only are accepted as aliases.
#
#   Env: BUILD_JOBS (default: nproc), OSXCROSS_ROOT (default: ~/osxcross)
#
#   Toolchain deps (Debian/Ubuntu host):
#     sudo apt install cmake build-essential \
#          gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
#          gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 clang lld
#     macOS targets additionally need OSXCross: https://github.com/tpoechtrager/osxcross

set -uo pipefail   # deliberately NOT -e: per-target failures are handled inline.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
HOST_OS="$(uname -s)"                          # Linux | Darwin
OSXCROSS_ROOT="${OSXCROSS_ROOT:-$HOME/osxcross}"
LR="paclet/LibraryResources"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# Filter (positional, or legacy --*-only aliases)
FILTER=""
case "${1:-}" in
    --help|-h)
        sed -n '2,26p' "$0"; exit 0 ;;
    --linux-only)   FILTER="Linux"  ;;
    --windows-only) FILTER="Windows";;
    --macos-only)   FILTER="MacOSX" ;;
    "")             FILTER=""        ;;
    *)              FILTER="$1"      ;;
esac

BUILT=(); SKIPPED=(); FAILED=()
have() { command -v "$1" >/dev/null 2>&1; }
selected() { [[ -z "$FILTER" || "$1" =~ $FILTER ]]; }
skip() { echo -e "${YELLOW}skip   $1 — $2${NC}"; SKIPPED+=("$1 ($2)"); }

# build_target NAME BUILD_DIR OUTPUT_LIB [extra cmake args...]
build_target() {
    local name="$1" dir="$2" out="$3"; shift 3
    echo -e "\n${GREEN}=== $name ===${NC}"
    mkdir -p "$dir"
    rm -f "$out"   # so a failed build can't pass the existence check on a stale library
    if ! cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON "$@"; then
        echo -e "${RED}$name: CMake configuration failed${NC}"; FAILED+=("$name"); return 1
    fi
    if ! cmake --build "$dir" --target paclet -j"$BUILD_JOBS"; then
        echo -e "${RED}$name: build failed${NC}"; FAILED+=("$name"); return 1
    fi
    if [[ -f "$out" ]]; then
        echo -e "${GREEN}$name: OK${NC}"; BUILT+=("$name")
    else
        echo -e "${RED}$name: build reported success but $out is missing${NC}"; FAILED+=("$name"); return 1
    fi
}

echo -e "${GREEN}=== Building the Hypergraph Rewriting paclet (host: $HOST_OS, jobs: $BUILD_JOBS) ===${NC}"
[[ -n "$FILTER" ]] && echo -e "${YELLOW}filter: /$FILTER/${NC}"

# ---- Linux x86-64 ----
if selected "Linux-x86-64"; then
    if [[ "$HOST_OS" == "Linux" ]]; then
        build_target "Linux-x86-64" build_linux "$LR/Linux-x86-64/libHypergraphRewriting.so"
    else
        skip "Linux-x86-64" "host is $HOST_OS; no native x86-64 Linux compiler"
    fi
fi

# ---- Linux ARM64 ----
if selected "Linux-ARM64"; then
    if have aarch64-linux-gnu-gcc; then
        build_target "Linux-ARM64" build_linux_arm64 "$LR/Linux-ARM64/libHypergraphRewriting.so" \
            -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64
    else
        skip "Linux-ARM64" "aarch64-linux-gnu-gcc not found (apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu)"
    fi
fi

# ---- Windows x86-64 ----
if selected "Windows-x86-64"; then
    if have x86_64-w64-mingw32-gcc; then
        build_target "Windows-x86-64" build_windows "$LR/Windows-x86-64/HypergraphRewriting.dll" \
            -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-cross.cmake
    else
        skip "Windows-x86-64" "mingw not found (apt install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64)"
    fi
    # The Windows CUDA engine (hg_evolve_gpu.exe) can't be cross-compiled with mingw — on Windows
    # nvcc requires MSVC as its host compiler. Build it natively (best-effort) via the Windows
    # toolchain when it is present; the mingw DLL/CPU exe above are the required Windows artifacts.
    if [[ -e "/mnt/c/Program Files/CMake/bin/cmake.exe" \
          && -d "/mnt/c/Program Files/Microsoft Visual Studio/2022" \
          && -n "$(ls -d '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/'v*.* 2>/dev/null)" ]]; then
        if ./build_windows_gpu.sh; then
            BUILT+=("Windows-x86-64/hg_evolve_gpu.exe")
        else
            FAILED+=("Windows-x86-64/hg_evolve_gpu.exe")
        fi
    else
        skip "Windows-x86-64/hg_evolve_gpu.exe" "native Windows MSVC+CUDA toolchain not found (VS2022 + CUDA Toolkit + CMake)"
    fi
fi

# ---- Windows ARM64 ----
if selected "Windows-ARM64"; then
    if have clang; then
        build_target "Windows-ARM64" build_windows_arm64 "$LR/Windows-ARM64/HypergraphRewriting.dll" \
            -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-cross.cmake \
            -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DWINDOWS_COMPILER=clang
    else
        skip "Windows-ARM64" "clang not found (apt install clang lld)"
    fi
fi

# ---- macOS x86-64 + ARM64 ----
macos_native() {  # native slices on a macOS host
    selected "MacOSX-x86-64" && build_target "MacOSX-x86-64" build_macos \
        "$LR/MacOSX-x86-64/libHypergraphRewriting.dylib" -DCMAKE_OSX_ARCHITECTURES=x86_64
    selected "MacOSX-ARM64" && build_target "MacOSX-ARM64" build_macos_arm64 \
        "$LR/MacOSX-ARM64/libHypergraphRewriting.dylib" -DCMAKE_OSX_ARCHITECTURES=arm64
}
macos_cross() {   # OSXCross from a non-macOS host
    export OSXCROSS_ROOT PATH="$OSXCROSS_ROOT/target/bin:$PATH"
    selected "MacOSX-x86-64" && build_target "MacOSX-x86-64" build_macos \
        "$LR/MacOSX-x86-64/libHypergraphRewriting.dylib" \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64
    selected "MacOSX-ARM64" && build_target "MacOSX-ARM64" build_macos_arm64 \
        "$LR/MacOSX-ARM64/libHypergraphRewriting.dylib" \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=arm64
}
if selected "MacOSX-x86-64" || selected "MacOSX-ARM64"; then
    if [[ "$HOST_OS" == "Darwin" ]]; then
        macos_native
    elif [[ -d "$OSXCROSS_ROOT/target/bin" ]]; then
        macos_cross
    else
        selected "MacOSX-x86-64" && skip "MacOSX-x86-64" "not a macOS host and OSXCross not at $OSXCROSS_ROOT"
        selected "MacOSX-ARM64"  && skip "MacOSX-ARM64"  "not a macOS host and OSXCross not at $OSXCROSS_ROOT"
    fi
fi

# ---- Summary ----
echo -e "\n${GREEN}=== Summary ===${NC}"
for t in "${BUILT[@]:-}";   do [[ -n "$t" ]] && echo -e "${GREEN}  ✓ built    $t${NC}"; done
for t in "${SKIPPED[@]:-}"; do [[ -n "$t" ]] && echo -e "${YELLOW}  - skipped  $t${NC}"; done
for t in "${FAILED[@]:-}";  do [[ -n "$t" ]] && echo -e "${RED}  ✗ FAILED   $t${NC}"; done
echo -e "\nLibraries in: $LR/"

if (( ${#FAILED[@]} > 0 )); then
    echo -e "\n${RED}${#FAILED[@]} attempted target(s) failed — do not create the paclet archive.${NC}"
    exit 1
fi
echo -e "\n${GREEN}✓ ${#BUILT[@]} target(s) built, ${#SKIPPED[@]} skipped, 0 failed.${NC}"
echo -e "To bundle the paclet: cd build_<host-platform> && make create_paclet_archive"
