#!/bin/bash
# Build the paclet documentation notebooks from the markdown sources in
# paclet/Documentation/Source/ into paclet/Documentation/English/.
#
# Turnkey: vendors MarkdownToNotebook (git submodule), finds wolframscript (native, or the
# Windows install from WSL), and runs the converter. No manual checkout, no paths to pass.
#
#   ./build_docs.sh              generate + evaluate examples (renders the engine's output)
#   ./build_docs.sh structure    input-only cells; evaluate later in the front end
#                                (use this if this machine can't reach the Wolfram resource
#                                 system to evaluate inline)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1. Ensure the MarkdownToNotebook submodule is present.
if [[ ! -f tools/MarkdownToNotebook/MarkdownToNotebook.wl ]]; then
    echo "==> fetching MarkdownToNotebook submodule"
    # NOT --recursive: MarkdownToNotebook's own example submodules are SSH-only (sw1sh/*)
    # and we don't need them; the converter itself is all that's required.
    git submodule update --init tools/MarkdownToNotebook
fi

# 2. Find wolframscript: native on PATH, else the Windows install via WSL.
run_ws() {  # run_ws <wl-script> <args...>
    local script="$1"; shift
    if command -v wolframscript >/dev/null 2>&1; then
        wolframscript -file "$script" "$@"
    else
        local exe
        exe="$(ls /mnt/c/Program\ Files/Wolfram*/*/*/wolframscript.exe 2>/dev/null | sort -V | tail -1 || true)"
        [[ -n "$exe" ]] || { echo "error: wolframscript not found (install Wolfram Engine or Mathematica)"; exit 1; }
        # Windows kernel needs Windows paths.
        local wargs=(); local a
        for a in "$@"; do wargs+=("$a"); done
        "$exe" -file "$(wslpath -w "$script")" "${wargs[@]}"
    fi
}

echo "==> building documentation notebooks (${1:-full evaluation})"
run_ws "$ROOT/tools/build_docs.wls" ${1:+"$1"}
echo "==> done — notebooks in paclet/Documentation/English/"
