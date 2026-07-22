#!/bin/bash
# One command to build the release .paclet: platform libraries -> documentation -> archive.
#
#   ./build_paclet.sh              full build; examples are evaluated during doc generation
#                                  (needs a cloud-connected Wolfram to render WolframModelPlot etc.)
#   ./build_paclet.sh structure    input-only docs (for a machine that can't evaluate inline):
#                                   stops after docs so you can Evaluate Notebook in the front
#                                   end, then finish with the create_paclet_archive line it prints
#   ./build_paclet.sh nodocs       assemble using the COMMITTED notebooks as-is (do not regenerate);
#                                   for the machine that pulls docs rendered + committed elsewhere
#
#   Add `clean` (any position, e.g. `./build_paclet.sh clean nodocs`) to force a fresh
#   platform build — required after a toolchain change, which an incremental build ignores.
#
# Output: paclet_archive/WolframInstitute__HypergraphRewriteEngine-<version>.paclet
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Args (order-independent): `clean` forwards a clean rebuild; structure|nodocs pick the doc mode.
MODE=""
CLEAN=""
for a in "$@"; do
    case "$a" in
        clean|--clean)    CLEAN="clean" ;;
        structure|nodocs) MODE="$a" ;;
        "")               ;;
        *) echo "error: unknown argument '$a' (expected: clean, structure, nodocs)"; exit 2 ;;
    esac
done

echo "==> [1/3] platform libraries"
./build_all_platforms.sh ${CLEAN:+"$CLEAN"}

if [[ "$MODE" == "nodocs" ]]; then
    echo "==> [2/3] documentation: using the committed notebooks as-is (not regenerating)"
    ls paclet/Documentation/English/ReferencePages/Symbols/*.nb >/dev/null 2>&1 || {
        echo "error: no committed notebooks under paclet/Documentation/English/ — pull them first, or run without 'nodocs'."; exit 1; }
else
    echo "==> [2/3] documentation"
    ./build_docs.sh ${MODE:+"$MODE"}
fi

# create_paclet_archive lives in the host's native build directory (configured with the paclet
# enabled by build_all_platforms.sh).
case "$(uname -s)" in
    Linux)  BUILDDIR=build_linux ;;
    Darwin) BUILDDIR=build_macos ;;
    *)      BUILDDIR="$(ls -d build_linux build_macos 2>/dev/null | head -1)" ;;
esac
[[ -n "${BUILDDIR:-}" && -d "$BUILDDIR" ]] || { echo "error: native build dir not found (did build_all_platforms.sh run?)"; exit 1; }

if [[ "$MODE" == "structure" ]]; then
    echo "==> input-only docs written. Evaluate them in the front end (Evaluate Notebook), then:"
    echo "       cmake --build $BUILDDIR --target create_paclet_archive"
    exit 0
fi

echo "==> [3/3] paclet archive"
cmake --build "$BUILDDIR" --target create_paclet_archive
echo "==> done:"
ls -la paclet_archive/*.paclet
