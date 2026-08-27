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
#   Usage: ./build_all_platforms.sh [clean] [FILTER]
#     clean   wipe each selected target's build dir before configuring, forcing a fresh
#             CMake configure. REQUIRED after a toolchain change: CMake reads a toolchain's
#             *_LINKER_FLAGS_INIT (and other *_INIT vars) only on the first configure, so an
#             incremental build silently keeps the old flags. `clean` guarantees they apply.
#     FILTER  optional substring/regex; only targets whose name matches are built
#             (e.g. "Windows", "MacOSX", "Linux-x86-64"). Legacy --linux-only /
#             --windows-only / --macos-only are accepted as aliases.
#     clean and FILTER are order-independent, e.g. `./build_all_platforms.sh clean Windows`.
#
#   Env: BUILD_JOBS (default: nproc), OSXCROSS_ROOT (default: ~/osxcross)
#        HG_REQUIRE_GPU=1  the Windows CUDA exe becomes a REQUIRED target: an absent toolchain
#                          or a failed build is FAILED, not SKIPPED. Set this for a release. A
#                          skip leaves the previous exe in the platform directory to be
#                          archived, so without this a broken CUDA config ships a stale binary
#                          and says only "skipped".
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

# Args (order-independent): optional `clean` flag plus an optional FILTER substring;
# legacy --*-only aliases still map to a filter.
CLEAN=0
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --help|-h)      sed -n '2,31p' "$0"; exit 0 ;;
        --clean|clean)  CLEAN=1 ;;
        --linux-only)   FILTER="Linux"  ;;
        --windows-only) FILTER="Windows";;
        --macos-only)   FILTER="MacOSX" ;;
        "")             ;;
        *)              FILTER="$arg"   ;;
    esac
done

BUILT=(); SKIPPED=(); FAILED=()
have() { command -v "$1" >/dev/null 2>&1; }
selected() { [[ -z "$FILTER" || "$1" =~ $FILTER ]]; }
skip() { echo -e "${YELLOW}skip   $1 — $2${NC}"; SKIPPED+=("$1 ($2)"); }

# build_target NAME BUILD_DIR OUTPUT_LIB [extra cmake args...]
build_target() {
    local name="$1" dir="$2" out="$3"; shift 3
    local platdir; platdir="$(dirname "$out")"   # LibraryResources/<platform>
    echo -e "\n${GREEN}=== $name ===${NC}"
    # `clean`: wipe the build dir so CMake re-reads the toolchain from scratch (its *_INIT
    # linker flags are honoured only on a first configure).
    (( CLEAN )) && rm -rf "$dir"
    mkdir -p "$dir"
    # Clear the artifacts we verify so a failed build can't pass on stale files.
    rm -f "$out" "$platdir"/hg_evolve "$platdir"/hg_evolve.exe
    if ! cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON "$@"; then
        echo -e "${RED}$name: CMake configuration failed${NC}"; FAILED+=("$name"); return 1
    fi
    # Build BOTH the LibraryLink library (paclet) AND the standalone evolution process (hg_evolve).
    # The process binary is the primary evolution path on every platform — HGEvolve runs the engine
    # in it so a crash/abort kills the process, not the notebook (the in-engine abort mechanism was
    # removed in favour of this). Shipping only the library would silently fall back to running the
    # engine in-kernel, defeating the isolation. hg_evolve cross-compiles like any executable.
    if ! cmake --build "$dir" --target paclet hg_evolve -j"$BUILD_JOBS"; then
        echo -e "${RED}$name: build failed${NC}"; FAILED+=("$name"); return 1
    fi
    if [[ ! -f "$out" ]]; then
        echo -e "${RED}$name: build reported success but the library $out is missing${NC}"; FAILED+=("$name"); return 1
    fi
    if ! { [ -e "$platdir/hg_evolve" ] || [ -e "$platdir/hg_evolve.exe" ]; }; then
        echo -e "${RED}$name: the hg_evolve process binary was not produced in $platdir${NC}"; FAILED+=("$name"); return 1
    fi
    echo -e "${GREEN}$name: OK (library + hg_evolve process)${NC}"; BUILT+=("$name")
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

# ---- Linux x86-64 CUDA engine ----
# The GPU engine is a SEPARATE binary from the CPU one and is built from its own directory, so
# the CPU leg above stays a pure host build. It is the same deliverable the Windows leg below
# produces -- the paclet's TargetDevice -> "GPU" runs whichever hg_evolve_gpu it finds for the
# running platform -- and it is required on a release for the same reason: SKIPPING does not
# remove a previous binary from the platform directory, so on a release the difference between
# skipping and failing is the difference between shipping a stale binary and knowing you cannot.
if selected "Linux-x86-64"; then
    if [[ "$HOST_OS" == "Linux" ]] && have nvcc; then
        echo -e "\n${GREEN}=== Linux-x86-64/hg_evolve_gpu ===${NC}"
        gpu_out="$LR/Linux-x86-64/hg_evolve_gpu"
        rm -f "$gpu_out"          # so a failed build cannot pass on the previous file
        (( CLEAN )) && rm -rf build_linux_gpu
        if cmake -S . -B build_linux_gpu -DCMAKE_BUILD_TYPE=Release \
                 -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON -DBUILD_GPU=ON \
                 -DCMAKE_CUDA_ARCHITECTURES=89 \
           && cmake --build build_linux_gpu --target hg_evolve_gpu -j"$BUILD_JOBS" \
           && [[ -f "$gpu_out" ]]; then
            echo -e "${GREEN}Linux-x86-64/hg_evolve_gpu: OK${NC}"
            BUILT+=("Linux-x86-64/hg_evolve_gpu")
        elif [[ "${HG_REQUIRE_GPU:-0}" == "1" ]]; then
            echo -e "${RED}Linux-x86-64/hg_evolve_gpu: build failed and HG_REQUIRE_GPU=1${NC}"
            FAILED+=("Linux-x86-64/hg_evolve_gpu")
        else
            skip "Linux-x86-64/hg_evolve_gpu" "optional GPU build did not complete; shipping CPU-only Linux"
        fi
    elif [[ "${HG_REQUIRE_GPU:-0}" == "1" ]]; then
        echo -e "${RED}Linux-x86-64/hg_evolve_gpu: nvcc not found and HG_REQUIRE_GPU=1${NC}"
        FAILED+=("Linux-x86-64/hg_evolve_gpu (toolchain absent)")
    else
        skip "Linux-x86-64/hg_evolve_gpu" "nvcc not found (CUDA Toolkit)"
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
    # THE WINDOWS CPU ARTIFACTS ARE BUILT NATIVELY WHERE MSVC EXISTS, and cross-compiled only
    # where it does not. mingw-w64 corrupts the heap at WORKER-THREAD exit: the engine runs every
    # evolution on worker threads and keeps thread_local scratch, so a mingw-built binary faults
    # at teardown on a threaded run, after the answer has been written to stdout -- which is why
    # it went unnoticed.
    #
    # THE CLAIM IS CHECKED, not asserted here: verification/mingw/ holds the smallest program that
    # shows it, with no engine code in it, run by ctest as `mingw_tls_teardown` with each cell
    # declaring CLEAN or CORRUPT. The identical source under MSVC 14.42 is clean 5/5 where mingw
    # is 116 (STATUS_HEAP_CORRUPTION) 10/10, so the variable is the toolchain. Note what the cells
    # also establish: a thread_local with a non-trivial destructor is NOT on its own enough -- the
    # baseline cell is exactly that and is clean -- and the manifestation is heap-layout sensitive
    # enough that the output filename alone flips it. A mingw configuration that looks clean has
    # not been shown to be clean.
    #
    # A Linux host with no Visual Studio still gets a cross-built pair, because a binary with a
    # teardown fault beats no binary at all -- but it is the fallback, not the shipping route.
    if [[ -e "/mnt/c/Program Files/CMake/bin/cmake.exe" \
          && -d "/mnt/c/Program Files/Microsoft Visual Studio/2022" ]]; then
        if ./build_windows_msvc.sh cpu $([[ "$CLEAN" == "1" ]] && echo clean); then
            BUILT+=("Windows-x86-64 (native MSVC)")
        else
            echo -e "${RED}Windows-x86-64: native MSVC build failed${NC}"
            FAILED+=("Windows-x86-64 (native MSVC)")
        fi
    elif have x86_64-w64-mingw32-gcc; then
        echo -e "${YELLOW}Windows-x86-64: no Visual Studio found; cross-compiling with mingw-w64.${NC}"
        echo    "  mingw-w64 corrupts the heap at worker-thread exit; see verification/mingw/,"
        echo    "  which reproduces it with no engine code and is run by ctest. The identical"
        echo    "  source is clean under MSVC, so the variable is the toolchain."
        echo    "  The engine keeps thread_local scratch and runs on worker threads, so the binary"
        echo    "  this produces faults at teardown on any threaded run. The answer it writes is"
        echo    "  correct and complete; the process exit code is not. Install Visual Studio 2022"
        echo    "  with the C++ workload to build the shipping artifact instead."
        build_target "Windows-x86-64" build_windows "$LR/Windows-x86-64/HypergraphRewriting.dll" \
            -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-cross.cmake
    else
        skip "Windows-x86-64" "no Visual Studio and no mingw"
    fi
    # The Windows CUDA engine (hg_evolve_gpu.exe) can't be cross-compiled with mingw — on Windows
    # nvcc requires MSVC as its host compiler. Build it natively (best-effort) via the Windows
    # toolchain when it is present; the mingw DLL/CPU exe above are the required Windows artifacts.
    if [[ -e "/mnt/c/Program Files/CMake/bin/cmake.exe" \
          && -d "/mnt/c/Program Files/Microsoft Visual Studio/2022" \
          && -n "$(ls -d '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/'v*.* 2>/dev/null)" ]]; then
        # Best-effort by default: the six platform libraries are the required artifacts, and an
        # optional GPU build must not block them, so a failure routes to SKIPPED.
        #
        # HG_REQUIRE_GPU=1 is what a RELEASE run sets. Skipping does not remove the previous
        # exe from the platform directory -- it leaves it there to be archived -- so on a
        # release the difference between SKIPPED and FAILED is the difference between shipping
        # a stale binary and knowing you cannot ship.
        # `clean` propagates: the caller asking for a fresh configure means the GPU leg's cache
        # is as suspect as every other target's.
        if ./build_windows_msvc.sh gpu $([[ "$CLEAN" == "1" ]] && echo clean); then
            BUILT+=("Windows-x86-64/hg_evolve_gpu.exe")
        elif [[ "${HG_REQUIRE_GPU:-0}" == "1" ]]; then
            echo -e "${RED}Windows-x86-64/hg_evolve_gpu.exe: build failed and HG_REQUIRE_GPU=1${NC}"
            FAILED+=("Windows-x86-64/hg_evolve_gpu.exe")
        else
            skip "Windows-x86-64/hg_evolve_gpu.exe" "optional GPU build did not complete (see log above); shipping CPU-only Windows"
        fi
    elif [[ "${HG_REQUIRE_GPU:-0}" == "1" ]]; then
        echo -e "${RED}Windows-x86-64/hg_evolve_gpu.exe: no native MSVC+CUDA toolchain and HG_REQUIRE_GPU=1${NC}"
        FAILED+=("Windows-x86-64/hg_evolve_gpu.exe (toolchain absent)")
    else
        skip "Windows-x86-64/hg_evolve_gpu.exe" "native Windows MSVC+CUDA toolchain not found (VS2022 + CUDA Toolkit + CMake)"
    fi
fi

# ---- Windows ARM64 ----
if selected "Windows-ARM64"; then
    if have clang; then
        # MultiThreaded (/MT) folds the C/C++ runtime in statically: clang targets the MSVC ABI
        # here, CMake honours CMAKE_MSVC_RUNTIME_LIBRARY (CMP0091 NEW), and the static ARM64 CRT
        # libs (libcmt/libcpmt/libvcruntime/libucrt) ship with VS+SDK. Result: the ARM64 DLL and
        # hg_evolve.exe import only KERNEL32/WS2_32 -- no VC++ redistributable dependency.
        build_target "Windows-ARM64" build_windows_arm64 "$LR/Windows-ARM64/HypergraphRewriting.dll" \
            -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-cross.cmake \
            -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DWINDOWS_COMPILER=clang \
            -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
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
    # THE DEPLOYMENT TARGET IS PASSED EVERY TIME, not left to the toolchain's default. The
    # toolchain sets it only `if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)`, so a build directory
    # configured once at some other floor keeps that floor forever. A release run that silently
    # builds against a floor nobody chose is the hazard; passing it on the command line makes
    # the toolchain's answer win on every configure.
    #
    # 14.4 IS THE FLOOR BECAUSE park.hpp SELECTS ON THE HEADER, NOT ON THE TARGET. With a 14.4
    # SDK installed, `__has_include(<os/os_sync_wait_on_address.h>)` succeeds and HG_PARK_OS_SYNC
    # is chosen -- so the objects call os_sync_wait_on_address, which does not exist before macOS
    # 14.4. A lower floor here does not widen support; it produces a dylib whose load command
    # advertises a system the code cannot run on. The engine takes no mutex fallback, so the
    # choice is 14.4 or no macOS build, and this is the former stated once.
    local osx_min="${HG_MACOS_DEPLOYMENT_TARGET:-14.4}"
    selected "MacOSX-x86-64" && build_target "MacOSX-x86-64" build_macos \
        "$LR/MacOSX-x86-64/libHypergraphRewriting.dylib" \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$osx_min"
    selected "MacOSX-ARM64" && build_target "MacOSX-ARM64" build_macos_arm64 \
        "$LR/MacOSX-ARM64/libHypergraphRewriting.dylib" \
        -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$osx_min"
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

# ---- What is actually in the paclet now ----
#
# A per-target BUILT line says a build ran; it says nothing about the other files sitting in the
# platform directories, which are what a release ships. Every artifact carries the commit it was
# built from (paclet_source/build_stamp.hpp), so one scan reports the whole shipped set against
# HEAD -- including the ones this run skipped, which is precisely where a stale file hides.
#
# Advisory here and REQUIRED at sign-off: a filtered or single-platform run is expected to leave
# other platforms behind, so failing this script on it would make the filter useless.
echo -e "\n${GREEN}=== Shipped artifacts vs HEAD ===${NC}"
python3 tools/dev/artifact_stamp_check.py || \
    echo -e "${YELLOW}(advisory here; release sign-off requires this to be clean)${NC}"

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
