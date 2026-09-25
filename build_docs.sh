#!/bin/bash
# Build the paclet documentation notebooks from the markdown sources in docs/en/ into
# paclet/Documentation/English/ (tools/build_docs.wls describes the mapping).
#
# Turnkey: vendors MarkdownToNotebook (git submodule), finds wolframscript (native, or the
# Windows install from WSL), and runs the converter. No manual checkout, no paths to pass.
#
#   ./build_docs.sh                 generate and evaluate examples (renders the engine's output)
#   ./build_docs.sh structure       input-only cells, into docs/en/.generated/
#   ./build_docs.sh only=<regex>    only the sources whose file name matches <regex>
#
# After placing the notebooks it deletes every notebook under the destination that no source
# maps to, so a renamed or removed page leaves no notebook behind.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARGS=("$@")                          # any of: structure, only=<regex>
MODE="full evaluation"
ENGLISH="paclet/Documentation/English"
for a in "${ARGS[@]}"; do
    if [[ "$a" == structure ]]; then MODE=structure; ENGLISH="docs/en/.generated"; fi
done
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
    echo "==> building documentation notebooks ($MODE) [staged via $STAGE_WIN]"
    # wolframscript can exit non-zero on the benign exit-time license-release message ("The product
    # exited because of a license error") even after a clean DONE. Do not let that abort placement:
    # gate on build_docs.wls's .build_ok sentinel, which is written only after every notebook wrote
    # and verified.
    # dest= lets the converter read the incremental-build manifest and the already-placed
    # notebooks from the real English/ (as a Windows path), so unchanged docs are skipped.
    set +e
    mkdir -p "$ENGLISH"
    "$WS_EXE" -file "$(wslpath -w "$SCRIPT")" ${ARGS[@]+"${ARGS[@]}"} \
        "out=$STAGE_WIN" "dest=$(wslpath -w "$ROOT/$ENGLISH")"
    ws_rc=$?
    set -e
    if [[ ! -f "$STAGE_WSL/.build_ok" ]]; then
        echo "error: documentation generation did not complete (wolframscript rc=$ws_rc); $ENGLISH/ left unchanged." >&2
        exit 1
    fi
    rm -f "$STAGE_WSL/.build_ok"

    # Place each notebook into English/ WITHOUT a truncating overwrite. A notebook open in the
    # front end holds a 9P lock: `cp -f` onto it triggers its --force behaviour (remove a dest it
    # cannot open, then fail to recreate under the lock) and DESTROYS the file. So copy the stage
    # onto ext4 first (fresh files, no lock), then move each notebook into English/ with a
    # same-filesystem rename — rename replaces the directory entry without opening the target, so
    # an open-in-front-end notebook is updated cleanly (the front end keeps the old inode).
    echo "==> placing notebooks into $ENGLISH/"
    EXT4_STAGE="paclet/Documentation/.english_stage"
    rm -rf "$EXT4_STAGE"
    cp -rf "$STAGE_WSL"/. "$EXT4_STAGE"/
    # Move every staged file into English/ (the rebuilt notebooks plus the incremental manifest);
    # cached docs wrote nothing to the stage, so their existing notebooks are left untouched. Only
    # notebooks are counted for the message.
    placed=0
    failed_place=()
    while IFS= read -r -d '' src; do
        rel="${src#"$EXT4_STAGE"/}"
        dst="$ENGLISH/$rel"
        mkdir -p "$(dirname "$dst")"
        if mv -f "$src" "$dst" 2>/dev/null; then
            [[ "$rel" == *.nb ]] && placed=$((placed + 1))
        else
            failed_place+=("$rel")
        fi
    done < <(find "$EXT4_STAGE" -type f -print0)
    rm -rf "$EXT4_STAGE" "$STAGE_WSL"
    echo "==> placed $placed rebuilt notebook(s) (unchanged docs left in place)"
    if ((${#failed_place[@]})); then
        echo "error: could not place ${#failed_place[@]} notebook(s) into $ENGLISH/:" >&2
        printf '   %s\n' "${failed_place[@]}" >&2
        exit 1
    fi
else
    echo "==> building documentation notebooks ($MODE)"
    # Same license-exit guard as the staged path (native writes straight into English/).
    set +e
    "$WS_EXE" -file "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"}
    ws_rc=$?
    set -e
    if [[ ! -f "$ENGLISH/.build_ok" ]]; then
        echo "error: documentation generation did not complete (wolframscript rc=$ws_rc)." >&2
        exit 1
    fi
    rm -f "$ENGLISH/.build_ok"
fi

# 4. Delete the notebooks no source maps to (.doc_pages lists the ones that do).
if [[ -f "$ENGLISH/.doc_pages" ]]; then
    while IFS= read -r -d '' nb; do
        rel="${nb#"$ENGLISH"/}"
        if ! tr '\\' '/' < "$ENGLISH/.doc_pages" | tr -d '\r' | grep -qxF "$rel"; then
            rm -f "$nb"
            echo "==> removed $nb (no source maps to it)"
        fi
    done < <(find "$ENGLISH/Guides" "$ENGLISH/Tutorials" "$ENGLISH/ReferencePages/Symbols" \
                  -name '*.nb' -print0 2>/dev/null)
fi
echo "==> done — notebooks in $ENGLISH/"
