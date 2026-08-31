#!/bin/bash
# Build the Windows engine binaries with the NATIVE MSVC toolchain and install them into the
# paclet. Takes `cpu`, `gpu`, or neither (meaning both).
#
# The other Windows binaries (HypergraphRewriting.dll, hg_evolve.exe) are cross-compiled from WSL
# with mingw-w64 by build_all_platforms.sh, but CUDA cannot be built that way: on Windows nvcc
# requires MSVC (cl.exe) as its host compiler. So this drives a NATIVE Windows build from WSL via the
# Windows cmake.exe + the Visual Studio generator + the CUDA Toolkit, then copies the resulting
# hg_evolve_gpu.exe into paclet/LibraryResources/Windows-x86-64/.
#
# The Visual Studio generator compiles .cu through the CUDA<->Visual Studio MSBuild integration
# ("CUDA <ver>.props"). The CUDA installer copies those props into Visual Studio's BuildCustomizations
# folder ONLY for VS versions it recognizes, so a newer VS 2022 ends up with the Toolkit but no props,
# and `-T cuda=<version>` (which looks only in that VS folder) reports "CUDA not found". We therefore
# point the toolset at the Toolkit's OWN copy of the props (shipped in the Toolkit itself) via
# `-T cuda=<toolkit path>` — no manual copy into Visual Studio and no admin required.
#
# Requirements on the Windows side (auto-detected under /mnt/c): Visual Studio 2022 with the C++
# toolset (any edition), an NVIDIA CUDA Toolkit, and CMake.
#
# THE CPU ARTIFACTS ARE BUILT HERE RATHER THAN CROSS-COMPILED, and that is not a preference.
# mingw-w64 corrupts the heap when it destroys a thread_local object with a non-trivial
# destructor at WORKER-THREAD exit. Twenty-five lines reproduce it with no engine code -- four
# thread_local std::vectors touched from joined std::threads -- at 20/20 runs under mingw and
# 0/20 under MSVC on the same machine, and 0/20 under mingw when the same objects are used only
# on the main thread. The engine runs every evolution on worker threads and keeps thread_local
# scratch, so a mingw-built binary hits it on any threaded run; it went unnoticed because the
# corruption is detected at teardown, after the answer has been written to stdout.
#
#   ./build_windows_msvc.sh                   # both, shippable
#   ./build_windows_msvc.sh cpu               # DLL + hg_evolve.exe only (no CUDA needed)
#   ./build_windows_msvc.sh gpu clean         # GPU only, fresh configure (toolset change)
#   HG_GPU_ARCHS=89 ./build_windows_msvc.sh   # single arch (faster; e.g. just Ada/RTX 40xx)
set -euo pipefail

CLEAN=0
DO_CPU=0
DO_GPU=0
for arg in "$@"; do
    case "$arg" in
        --clean|clean) CLEAN=1 ;;
        cpu)           DO_CPU=1 ;;
        gpu)           DO_GPU=1 ;;
        "")            ;;
        *)             echo "error: unknown argument '$arg' (expected cpu, gpu or clean)"; exit 2 ;;
    esac
done
# Naming neither means both, so the release path stays one call.
if [[ "$DO_CPU" == "0" && "$DO_GPU" == "0" ]]; then DO_CPU=1; DO_GPU=1; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# THE SHIPPING ARCHITECTURE SET IS DEFINED ONCE, IN gpu/CMakeLists.txt (HG_GPU_ARCHS). This
# script used to carry its own copy of the same literal, which is one rule written twice: the
# two agree until one is edited, and the failure that produces is a release whose Windows and
# Linux GPU binaries carry SASS for different cards. Nothing is passed unless the caller asked
# for something, and then only HG_GPU_ARCHS -- CMAKE_CUDA_ARCHITECTURES is deliberately NOT
# forwarded, because gpu/CMakeLists.txt derives it from HG_GPU_ARCHS only when the caller has
# not set it, so setting both here would make the default branch unreachable.
GPU_ARCH_ARGS=()
[[ -n "${HG_GPU_ARCHS:-}" ]] && GPU_ARCH_ARGS=(-DHG_GPU_ARCHS="$HG_GPU_ARCHS")
# One build directory PER MODE. The two configures differ in BUILD_GPU and in whether a CUDA
# toolset is pinned, and CMake cannot switch a cache between them without a wipe -- sharing one
# directory would make every alternating call a full native rebuild.
BUILD_WIN_GPU='C:/Temp/hg_gpu_build'
BUILD_WSL_GPU='/mnt/c/Temp/hg_gpu_build'  # same dir, WSL view (Windows-local disk, no 9P)
BUILD_WIN_CPU='C:/Temp/hg_cpu_build'
BUILD_WSL_CPU='/mnt/c/Temp/hg_cpu_build'
DEST="paclet/LibraryResources/Windows-x86-64"

# --- locate the Windows toolchain under /mnt/c ---
# THE COMMIT IS PASSED IN, not discovered. This configure runs under Windows cmake.exe against a
# \\wsl.localhost path, where the CMakeLists' own `git rev-parse` finds no usable repository and
# the stamp comes out `commit=unknown` -- which artifact_stamp_check correctly reports as unknown
# provenance, so the one binary this script exists to produce was the one artifact that could not
# say what it was built from. Resolved on the Linux side, where git works, and handed over.
HG_BUILD_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

WINCMAKE="$(ls '/mnt/c/Program Files/CMake/bin/cmake.exe' 2>/dev/null | head -1 || true)"
[[ -n "$WINCMAKE" ]] || { echo "error: Windows cmake.exe not found (install CMake on Windows)"; exit 1; }

[[ -d "/mnt/c/Program Files/Microsoft Visual Studio/2022" ]] || \
    { echo "error: Visual Studio 2022 not found (need the 'Desktop development with C++' workload)"; exit 1; }
GEN="Visual Studio 17 2022"

if [[ "$DO_GPU" == "1" ]]; then
    # Highest installed CUDA Toolkit. Use its FULL path as the toolset so the VS generator uses the
    # Toolkit's bundled integration props, not the (possibly missing) copy in Visual Studio.
    CUDA_DIR_WSL="$(ls -d '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/'v*.* 2>/dev/null | sort -V | tail -1 || true)"
    [[ -n "$CUDA_DIR_WSL" ]] || { echo "error: no CUDA Toolkit under /mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"; exit 1; }
    CUDA_VER="$(basename "$CUDA_DIR_WSL" | sed 's/^v//')"
    CUDA_DIR_WIN="$(wslpath -w "$CUDA_DIR_WSL")"
    # The props the generator needs live here; if the Toolkit was installed without them, say so plainly.
    [[ -f "$CUDA_DIR_WSL/extras/visual_studio_integration/MSBuildExtensions/CUDA $CUDA_VER.props" ]] || {
        echo "error: the CUDA $CUDA_VER Toolkit is missing its Visual Studio integration props at"
        echo "         $CUDA_DIR_WSL/extras/visual_studio_integration/MSBuildExtensions/"
        echo "       Re-run the CUDA installer and include 'Visual Studio Integration' (the props ship with it)."
        exit 1
    }
    echo "==> toolchain: $GEN, CUDA $CUDA_VER (toolset via Toolkit path), archs [${HG_GPU_ARCHS:-gpu/CMakeLists.txt default}]"
else
    echo "==> toolchain: $GEN (CPU only, no CUDA needed)"
fi


SRC="$(wslpath -w "$ROOT")"
# WIPE ONLY WHEN THE CACHE IS ACTUALLY INCOMPATIBLE. A cache from a different generator makes
# CMake abort with "generator does not match", which is what this guards; a cache from the SAME
# generator is reusable, and reusing it is the difference between a re-link and a multi-hour
# native CUDA rebuild. That matters because the release re-stamps every artifact at the final
# commit, and a leg that can only ever build from scratch cannot be re-stamped affordably.
# `clean` forces the wipe, for a toolset change CMake cannot detect on its own.
# `if` rather than `[[ ... ]] && ...`: under `set -e` an AND-list whose left side is false is the
# whole statement and its non-zero status ends the script, so the no-cache case would exit here.
# ONE BODY FOR BOTH LEGS. They differ in three arguments and in which targets they build; the
# cache guard, the Windows-local working directory and the install are the same question twice.
configure_and_build() {
    local mode="$1" build_win="$2" build_wsl="$3"; shift 3
    local cached_gen=""
    if [[ -f "$build_wsl/CMakeCache.txt" ]]; then
        cached_gen="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$build_wsl/CMakeCache.txt")"
    fi
    # WIPE ONLY WHEN THE CACHE IS ACTUALLY INCOMPATIBLE. A cache from a different generator makes
    # CMake abort with "generator does not match", which is what this guards; a cache from the
    # SAME generator is reusable, and reusing it is the difference between a re-link and a
    # multi-hour native rebuild. That matters because the release re-stamps every artifact at the
    # final commit, and a leg that can only build from scratch cannot be re-stamped affordably.
    # `clean` forces the wipe, for a toolset change CMake cannot detect on its own.
    if [[ "$CLEAN" == "1" || "$cached_gen" != "$GEN" ]]; then
        [[ -n "$cached_gen" ]] && echo "==> $mode: wiping build dir ('$cached_gen' != '$GEN')"
        rm -rf "$build_wsl"
    else
        echo "==> $mode: reusing build dir (generator matches); pass 'clean' to force a configure"
    fi
    mkdir -p "$build_wsl"
    echo "==> $mode: configuring (native MSVC)"
    # HG_ENGINE_STATS defaults ON in CMakeLists.txt for test and development builds; the shipped
    # binaries carry no diagnostic counters, the same as every leg of build_all_platforms.sh.
    run_wincmake -S "$SRC" -B "$build_win" -G "$GEN" -A x64 "$@" \
        -DHG_ENGINE_STATS=OFF \
        -DHG_BUILD_COMMIT="$HG_BUILD_COMMIT" \
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
        -DBUILD_VISUALIZATION=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF
}

# CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded folds the C/C++ runtime statically (/MT) into every
# target, so the shipped binaries carry no VC++ redistributable dependency. Combined with the
# static CUDA runtime on the GPU leg, the only remaining runtime dependency there is the NVIDIA
# driver. Needs CMake policy CMP0091 (NEW) -- default in the CMake versions we use.
#
# RUN cmake.exe FROM A WINDOWS-LOCAL DIRECTORY. This script cds to the repository, which on WSL
# is a \\wsl.localhost UNC path; a Windows process inheriting that as its working directory
# fails with "Invalid argument" before it reads an argument. -S and -B are absolute Windows
# paths, so the working directory decides nothing here except whether cmake starts.
run_wincmake() { ( cd /mnt/c/Temp && "$WINCMAKE" "$@" ); }

install_artifact() {
    local built="$1" name="$2"
    [[ -f "$built" ]] || { echo "error: build did not produce $built"; exit 1; }
    mkdir -p "$DEST"
    cp -f "$built" "$DEST/$name"
    echo "==> installed $DEST/$name ($(du -h "$DEST/$name" | cut -f1))"
}

if [[ "$DO_CPU" == "1" ]]; then
    configure_and_build cpu "$BUILD_WIN_CPU" "$BUILD_WSL_CPU" -DBUILD_GPU=OFF
    echo "==> cpu: building the paclet library and the one-shot binary"
    run_wincmake --build "$BUILD_WIN_CPU" --config Release --target HypergraphRewriting --parallel
    run_wincmake --build "$BUILD_WIN_CPU" --config Release --target hg_evolve --parallel
    # WHERE THE ARTIFACTS LAND IS THE GENERATOR'S CHOICE, not ours: the project pins its output
    # directory, so a multi-config generator writes to the build root rather than a Release/
    # subdirectory. Both are searched instead of assumed, because assuming cost one build.
    for name in hg_evolve.exe HypergraphRewriting.dll; do
        found=""
        for cand in "$BUILD_WSL_CPU/$name" \
                    "$BUILD_WSL_CPU/Release/$name" \
                    "$BUILD_WSL_CPU/paclet_source/Release/$name"; do
            [[ -f "$cand" ]] && { found="$cand"; break; }
        done
        [[ -n "$found" ]] || { echo "error: build did not produce $name anywhere under $BUILD_WSL_CPU"; exit 1; }
        install_artifact "$found" "$name"
    done
fi

if [[ "$DO_GPU" == "1" ]]; then
    # nvcc's front end parses the host branches of every HG_HD function too, and it does not know
    # MSVC's compiler intrinsics unless their declarations are in scope: the `_MSC_VER` branch of
    # hgcommon::ir_ctz64 names _BitScanForward64, which <intrin.h> declares and which cl.exe
    # accepts undeclared. Pre-including <intrin.h> into every CUDA translation unit gives the
    # front end the declaration; the device branch (__CUDA_ARCH__) is unaffected.
    configure_and_build gpu "$BUILD_WIN_GPU" "$BUILD_WSL_GPU" \
        -T "cuda=$CUDA_DIR_WIN" -DBUILD_GPU=ON "${GPU_ARCH_ARGS[@]+"${GPU_ARCH_ARGS[@]}"}" \
        -DCMAKE_CUDA_FLAGS="-include intrin.h"
    # The generator only emits the CUDA targets if CMake found the compiler. Without this the
    # build below dies with a confusing MSB1009.
    [[ -f "$BUILD_WSL_GPU/paclet_source/hg_evolve_gpu.vcxproj" ]] || {
        echo "error: CMake did not find a CUDA compiler, so no GPU project was generated." >&2
        echo "       Check that nvcc runs: '$CUDA_DIR_WSL/bin/nvcc.exe' --version" >&2
        exit 1
    }
    echo "==> gpu: building hg_evolve_gpu (CUDA kernels + host, then device-link)"
    run_wincmake --build "$BUILD_WIN_GPU" --config Release --target hg_evolve_gpu --parallel
    install_artifact "$BUILD_WSL_GPU/hg_evolve_gpu.exe" hg_evolve_gpu.exe
fi

echo "==> done."
