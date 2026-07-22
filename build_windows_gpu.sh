#!/bin/bash
# Build the Windows CUDA engine binary (hg_evolve_gpu.exe) and install it into the paclet.
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
#   ./build_windows_gpu.sh                   # broad arch set (Turing..Hopper), shippable
#   HG_GPU_ARCHS=89 ./build_windows_gpu.sh   # single arch (faster; e.g. just Ada/RTX 40xx)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCHS="${HG_GPU_ARCHS:-75;80;86;89;90}"   # CC numbers; broad = runs on many GPUs
BUILD_WIN='C:/Temp/hg_gpu_build'
BUILD_WSL='/mnt/c/Temp/hg_gpu_build'      # same dir, WSL view (Windows-local disk, no 9P)
DEST="paclet/LibraryResources/Windows-x86-64"

# --- locate the Windows toolchain under /mnt/c ---
WINCMAKE="$(ls '/mnt/c/Program Files/CMake/bin/cmake.exe' 2>/dev/null | head -1 || true)"
[[ -n "$WINCMAKE" ]] || { echo "error: Windows cmake.exe not found (install CMake on Windows)"; exit 1; }

[[ -d "/mnt/c/Program Files/Microsoft Visual Studio/2022" ]] || \
    { echo "error: Visual Studio 2022 not found (need the 'Desktop development with C++' workload)"; exit 1; }
GEN="Visual Studio 17 2022"

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
echo "==> toolchain: $GEN, CUDA $CUDA_VER (toolset via Toolkit path), archs [$ARCHS]"

SRC="$(wslpath -w "$ROOT")"
# Fresh build dir: a cache from a different toolset/generator makes CMake abort with a
# "generator/toolset does not match" error.
rm -rf "$BUILD_WSL"
mkdir -p "$BUILD_WSL"

echo "==> configuring (native MSVC + nvcc)"
# CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded folds the C/C++ runtime statically (/MT) into every
# target (hypergraph, wxf, hg_gpu, hg_evolve_gpu, and the nvcc host objects), so hg_evolve_gpu.exe
# has no VC++ redistributable dependency. Combined with the static CUDA runtime (CUDA_RUNTIME_LIBRARY
# Static), the only remaining runtime dependency is the NVIDIA driver (nvcuda.dll), which is present
# wherever a usable GPU is. Needs CMake policy CMP0091 (NEW) -- default in the CMake versions we use.
"$WINCMAKE" -S "$SRC" -B "$BUILD_WIN" -G "$GEN" -A x64 -T "cuda=$CUDA_DIR_WIN" \
    -DBUILD_GPU=ON -DHG_GPU_ARCHS="$ARCHS" -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
    -DBUILD_VISUALIZATION=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF

# The generator only emits the CUDA targets if CMake found the compiler. If it didn't, the build
# below would die with a confusing MSB1009; catch it here.
[[ -f "$BUILD_WSL/paclet_source/hg_evolve_gpu.vcxproj" ]] || {
    echo "error: CMake did not find a CUDA compiler, so no GPU project was generated." >&2
    echo "       Check that nvcc runs: '$CUDA_DIR_WSL/bin/nvcc.exe' --version" >&2
    exit 1
}

echo "==> building hg_evolve_gpu (compiles the CUDA kernels + host, then device-links)"
"$WINCMAKE" --build "$BUILD_WIN" --config Release --target hg_evolve_gpu --parallel

EXE="$BUILD_WSL/hg_evolve_gpu.exe"
[[ -f "$EXE" ]] || { echo "error: build did not produce $EXE"; exit 1; }
mkdir -p "$DEST"
cp -f "$EXE" "$DEST/hg_evolve_gpu.exe"
echo "==> installed $DEST/hg_evolve_gpu.exe ($(du -h "$DEST/hg_evolve_gpu.exe" | cut -f1))"
echo "==> done. HGEvolve[..., \"TargetDevice\" -> \"GPU\"] will now run on the device under a Windows kernel."
