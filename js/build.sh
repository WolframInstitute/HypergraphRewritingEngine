#!/bin/bash
# Builds the two WebAssembly modules of the hgevolve JavaScript package and copies them into
# js/dist: hgevolve.mjs (no threads) and hgevolve-threads.mjs (Web Workers). Needs an activated
# Emscripten SDK (emcmake on PATH). Usage: js/build.sh [build root, default build_js]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build_js}"
JOBS="${BUILD_JOBS:-4}"
command -v emcmake > /dev/null || { echo "emcmake not found: source <emsdk>/emsdk_env.sh" >&2; exit 2; }
mkdir -p "$ROOT/js/dist"
# The MIT licence goes with every copy of the package.
cp "$ROOT/LICENSE.md" "$ROOT/js/LICENSE.md"
for variant in single threads; do
    flags="-fwasm-exceptions"
    [ "$variant" = threads ] && flags="$flags -pthread"
    emcmake cmake -S "$ROOT" -B "$OUT/$variant" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="$flags" \
        -DBUILD_WOLFRAM_LANGUAGE_PACLET=OFF \
        -DBUILD_GPU=OFF \
        -DBUILD_VISUALIZATION=OFF \
        -DBUILD_TESTS=OFF
    cmake --build "$OUT/$variant" --target hgevolve_js -j"$JOBS"
    name=hgevolve
    [ "$variant" = threads ] && name=hgevolve-threads
    cp "$OUT/$variant/$name.mjs" "$OUT/$variant/$name.wasm" "$ROOT/js/dist/"
done
ls -la "$ROOT/js/dist"
