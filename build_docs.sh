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

MODE="${1:-}"                       # "structure" or empty
ENGLISH="paclet/Documentation/English"
SCRIPT="$ROOT/tools/build_docs.wls"

# 1. Ensure the MarkdownToNotebook submodule is present.
if [[ ! -f tools/MarkdownToNotebook/MarkdownToNotebook.wl ]]; then
    echo "==> fetching MarkdownToNotebook submodule"
    # NOT --recursive: MarkdownToNotebook's own example submodules are SSH-only (sw1sh/*)
    # and we don't need them; the converter itself is all that's required.
    git submodule update --init tools/MarkdownToNotebook
fi

# 2. Resolve wolframscript: native on PATH, else the Windows install driven from WSL.
if command -v wolframscript >/dev/null 2>&1; then
    WS_KIND=native
    WS_EXE=wolframscript
else
    WS_EXE="$(ls /mnt/c/Program\ Files/Wolfram*/*/*/wolframscript.exe 2>/dev/null | sort -V | tail -1 || true)"
    [[ -n "$WS_EXE" ]] || { echo "error: wolframscript not found (install Wolfram Engine or Mathematica)"; exit 1; }
    WS_KIND=windows
fi

# 3. Build the notebooks.
if [[ "$WS_KIND" == windows ]]; then
    # The Windows Wolfram kernel cannot reliably Export onto the WSL 9P share
    # (\\wsl.localhost\...): large notebooks (HGEvolve, the guide) fail with Export::noopen —
    # the same file-locking limitation create_paclet_archive sidesteps by staging through
    # /mnt/c/Temp. So have the kernel write to a Windows-local directory, then copy the
    # notebooks into place with a Linux cp (which owns the ext4 filesystem and never trips
    # the 9P lock, even if a notebook is open in the front end).
    STAGE_WSL=/mnt/c/Temp/hg_docs_stage
    STAGE_WIN='C:\Temp\hg_docs_stage'
    rm -rf "$STAGE_WSL"
    mkdir -p "$STAGE_WSL"
    echo "==> building documentation notebooks (${MODE:-full evaluation}) [staged via $STAGE_WIN]"
    "$WS_EXE" -file "$(wslpath -w "$SCRIPT")" ${MODE:+"$MODE"} "out=$STAGE_WIN"
    echo "==> copying notebooks into $ENGLISH/"
    mkdir -p "$ENGLISH"
    cp -rf "$STAGE_WSL"/. "$ENGLISH"/
    rm -rf "$STAGE_WSL"
else
    echo "==> building documentation notebooks (${MODE:-full evaluation})"
    "$WS_EXE" -file "$SCRIPT" ${MODE:+"$MODE"}
fi
echo "==> done — notebooks in $ENGLISH/"
