#!/usr/bin/env bash
# Build and run the mingw-w64 thread_local teardown cells, checking each against its declared
# verdict.
#
# WHY THIS EXISTS. The Windows x86-64 artifacts are built natively with MSVC and the mingw
# cross-build is a warned fallback, because mingw-w64 corrupts the heap when a worker thread
# exits. That decision lived in a comment in build_all_platforms.sh, which nothing checked. This
# runs the reproducer instead, so the claim is a command anyone can repeat and a toolchain that
# fixes the defect is noticed rather than assumed.
#
# EACH CELL DECLARES ITS VERDICT, in the table below: CLEAN (every run exits 0) or CORRUPT (at
# least one run exits 116, WSL's truncation of STATUS_HEAP_CORRUPTION). The one-worker CLEAN cells
# are the calibration -- they are one knob away from the corrupting one, so a build that reports
# CORRUPT for all of them is not reproducing this defect, it is reporting something else.
#
# THE CORRUPTION IS INTERMITTENT, so a CORRUPT verdict asks for one hit in HG_TLS_REPS runs.
# Measured 2026-10-06 (WSL kernel 6.18.40.1, x86_64-w64-mingw32-g++ 13-posix), 20 runs per cell
# at one worker: two_tls_static_alloc 3/20 at 16 blocks x 8 rounds, 2/20 at 32 x 4, every other
# cell 0/20. With two or more workers every cell corrupts,
# the single-thread_local baseline included (8 blocks, 2 workers, 8 rounds: baseline 5/20,
# two_tls_static_alloc 9/20; 16 x 4 x 16: baseline 4/10, the rest 9-10/10): baseline_4w pins that.
#
# THE CORRUPT CELL IS A PINNED REPRODUCER, the same pattern verification/genmc/run.sh uses for
# its violation harnesses: it passes exactly when the defect is still reachable. When a future
# mingw-w64 fixes it, this fails and says so, and that is the signal to revisit whether the
# cross-build still needs its warning.
#
# A CLEAN CELL MEANS CLEAN AT THIS LAYOUT, NOT "THIS KNOB IS NOT NEEDED". The manifestation is
# heap-layout sensitive: the corrupting cell built as t.exe corrupts 3 of 3 and the identical
# source and flags built as two_tls_static_alloc.exe is clean 3 of 3. Every cell is therefore
# built to the SAME binary name in its own directory, which is what makes them comparable; with
# the name held constant the knob map reproduces exactly. It still cannot distinguish "this knob
# is necessary" from "this knob shifted the layout enough to hide it", and the same caution
# applies to any engine configuration that appears clean under mingw.
#
# SKIPS RATHER THAN FAILS (exit 2, registered as SKIP_RETURN_CODE by CMakeLists) when the mingw
# cross-compiler is absent, or when this host cannot execute a Windows binary. Neither is
# vendored and neither exists on a Linux CI runner.
#
# Usage:
#   verification/mingw/run.sh            every cell
#   verification/mingw/run.sh <cell>     one cell, by name
#
# Environment:
#   MINGW_CXX   the cross compiler   (default: x86_64-w64-mingw32-g++)
#   HG_TLS_REPS runs per cell   (default: 40)

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/tls_teardown.cpp"

MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"
REPS="${HG_TLS_REPS:-40}"

# WSL's truncation of STATUS_HEAP_CORRUPTION (0xC0000374 & 0xFF).
readonly kCorrupt=116

if ! command -v "$MINGW_CXX" >/dev/null 2>&1; then
    echo "run.sh: '$MINGW_CXX' not found; skipping (set MINGW_CXX to override)"
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# name|flags|blocks workers rounds|verdict
# 16 blocks, one worker, eight rounds for the knob map. Sequential rounds matter -- one worker
# exiting once is clean at every size measured, including 256 MB.
CELLS=(
  "baseline|                                                          |16 1 8|CLEAN"
  "two_tls|-DTWO_TLS                                                  |16 1 8|CLEAN"
  "two_tls_alloc|-DTWO_TLS -DGUARD_ALLOCATES                          |16 1 8|CLEAN"
  "two_tls_static|-DTWO_TLS -DGUARD_TOUCHES_STATIC                    |16 1 8|CLEAN"
  "two_tls_static_alloc|-DTWO_TLS -DGUARD_TOUCHES_STATIC -DGUARD_ALLOCATES|16 1 8|CORRUPT"
  "split_fn|-DTWO_TLS -DSPLIT_FN -DGUARD_TOUCHES_STATIC -DGUARD_ALLOCATES|16 1 8|CLEAN"
  "three_tls|-DTHREE_TLS                                              |16 1 8|CLEAN"
  "baseline_4w|                                                       |16 4 16|CORRUPT"
)

run_cell() {
    local name="$1" flags="$2" verdict="$4"
    local -a args
    read -r -a args <<< "$3"
    # ONE FIXED BINARY NAME, each cell in its own directory. The output filename is itself a
    # layout axis: the same source and flags built as t.exe corrupts 3/3 and as
    # two_tls_static_alloc.exe is clean 3/3. Holding the name constant is what makes the cells
    # comparable to each other.
    local dir="$WORK/$name"
    mkdir -p "$dir"
    local exe="$dir/t.exe"

    # shellcheck disable=SC2086
    if ! "$MINGW_CXX" -std=c++17 -O2 -static $flags -o "$exe" "$SRC" 2>"$dir/cc.err"; then
        echo "--- $name: COMPILE FAILED"
        tail -20 "$dir/cc.err"
        return 3
    fi

    local codes=() rc
    for _ in $(seq 1 "$REPS"); do
        ( cd "$dir" && ./t.exe "${args[@]}" >/dev/null 2>&1 )
        rc=$?
        codes+=("$rc")
        # A host that cannot run a Windows binary at all reports 126/127, not a program status.
        if [ "$rc" -eq 126 ] || [ "$rc" -eq 127 ]; then
            echo "run.sh: cannot execute a Windows binary here (exit $rc); skipping"
            exit 2
        fi
    done

    local want_zero=0
    [ "$verdict" = "CLEAN" ] && want_zero=1

    local hits=0 other=0
    for rc in "${codes[@]}"; do
        if [ "$rc" -eq "$kCorrupt" ]; then hits=$((hits + 1))
        elif [ "$rc" -ne 0 ]; then other=$((other + 1)); fi
    done
    local ok=0
    if [ "$other" -eq 0 ]; then
        if [ "$want_zero" -eq 1 ]; then [ "$hits" -eq 0 ] && ok=1
        else [ "$hits" -gt 0 ] && ok=1; fi
    fi

    if [ "$ok" -eq 1 ]; then
        echo "--- $name: $verdict as declared ($hits of ${#codes[@]} runs corrupted)"
        return 0
    fi
    if [ "$verdict" = "CORRUPT" ]; then
        echo "--- $name: declared CORRUPT, got exits ${codes[*]}."
        echo "    The defect is no longer reachable in this configuration. If mingw-w64 fixed it,"
        echo "    change this cell's verdict and revisit the warning in build_all_platforms.sh."
    else
        echo "--- $name: declared CLEAN, got exits ${codes[*]} -- a cell one knob away from the"
        echo "    reproducer is now corrupting too, so the condition mapped in tls_teardown.cpp"
        echo "    is wider than recorded."
    fi
    return 1
}

fail=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r name flags cargs verdict <<< "$cell"
    if [ $# -ge 1 ] && [ "$1" != "$name" ]; then continue; fi
    echo "=== $name ==="
    run_cell "$name" "$flags" "$cargs" "$verdict" || fail=1
done
exit $fail
