#!/bin/bash
# Build the Windows CUDA engine binary (hg_evolve_gpu.exe) and install it into the paclet.
#
# The other Windows binaries (HypergraphRewriting.dll, hg_evolve.exe) are cross-compiled from
# WSL with mingw-w64 by build_all_platforms.sh, but CUDA cannot be built that way: on Windows
# nvcc requires MSVC (cl.exe) as its host compiler. So this drives a NATIVE Windows build from
# WSL via the Windows cmake.exe + the Visual Studio generator + the CUDA Toolkit, then copies the
# resulting hg_evolve_gpu.exe into paclet/LibraryResources/Windows-x86-64/.
#
# Requirements on the Windows side (auto-detected under /mnt/c): Visual Studio 2022 with the C++
# toolset, an NVIDIA CUDA Toolkit (with its Visual Studio integration), and CMake.
#
#   ./build_windows_gpu.sh            # broad arch set (Turing..Hopper), shippable
#   HG_GPU_ARCHS=89 ./build_windows_gpu.sh   # single arch (faster; e.g. just Ada/RTX 40xx)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARCHS="${HG_GPU_ARCHS:-75;80;86;89;90}"          # CC numbers; broad = runs on many GPUs
BUILD_WIN='C:/Temp/hg_gpu_build'
BUILD_WSL='/mnt/c/Temp/hg_gpu_build'             # same dir, WSL view (Windows-local disk, no 9P)
DEST="paclet/LibraryResources/Windows-x86-64"

# --- locate the Windows toolchain under /mnt/c ---
WINCMAKE="$(ls '/mnt/c/Program Files/CMake/bin/cmake.exe' 2>/dev/null | head -1 || true)"
[[ -n "$WINCMAKE" ]] || { echo "error: Windows cmake.exe not found (install CMake on Windows)"; exit 1; }

# Visual Studio generator: prefer 2022.
VS2022="/mnt/c/Program Files/Microsoft Visual Studio/2022"
[[ -d "$VS2022" ]] || { echo "error: Visual Studio 2022 not found under $VS2022"; exit 1; }
GEN="Visual Studio 17 2022"

# Highest installed CUDA Toolkit version (the VS generator selects it as the toolset).
CUDA_VER="$(ls -d '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/'v*.* 2>/dev/null \
    | sed 's#.*/v##' | sort -V | tail -1 || true)"
[[ -n "$CUDA_VER" ]] || { echo "error: no CUDA Toolkit under /mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"; exit 1; }
echo "==> toolchain: $GEN, CUDA $CUDA_VER, archs [$ARCHS]"

SRC="$(wslpath -w "$ROOT")"
mkdir -p "$BUILD_WSL"

echo "==> configuring (native MSVC + nvcc)"
"$WINCMAKE" -S "$SRC" -B "$BUILD_WIN" -G "$GEN" -A x64 -T "cuda=$CUDA_VER" \
    -DBUILD_GPU=ON -DHG_GPU_ARCHS="$ARCHS" -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
    -DBUILD_VISUALIZATION=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF

echo "==> building hg_evolve_gpu (compiles the CUDA kernels + host, then device-links)"
"$WINCMAKE" --build "$BUILD_WIN" --config Release --target hg_evolve_gpu --parallel

EXE="$BUILD_WSL/hg_evolve_gpu.exe"
[[ -f "$EXE" ]] || { echo "error: build did not produce $EXE"; exit 1; }
mkdir -p "$DEST"
cp -f "$EXE" "$DEST/hg_evolve_gpu.exe"
echo "==> installed $DEST/hg_evolve_gpu.exe ($(du -h "$DEST/hg_evolve_gpu.exe" | cut -f1))"
echo "==> done. HGEvolve[..., \"TargetDevice\" -> \"GPU\"] will now run on the device under a Windows kernel."
