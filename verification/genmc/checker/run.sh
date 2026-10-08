#!/usr/bin/env bash
# The reproducers for the checker fork (branch hg-fixes-0.19 of github.com/richardassar/genmc,
# also as genmc-0.19.0-fixes.patch here), each a program of a few lines. Run after building the
# checker:
#
#   verification/genmc/checker/run.sh
#
# A reproducer whose first line is `// Expect: <text>` must print <text>; every other one must
# report "No errors were detected". A line `// Args: <flags>` in the first three passes the flags
# to the checker, and a line `// Env: NAME=value` in the first three sets that variable for it.
# Measured on v0.17.0 without the fixes: it aborts on the allocation and promotion reproducers
# with an internal check or reports a non-allocated access, does not finish dependence_dag_paths
# within the ten minutes this script gives it, reports no race on the cas_fail_* reproducers,
# fails memmove_overlap_tail's assertion, reports no error on copy_longer_than_unroll, prints no
# assertion message or error site, and aborts on opaque_memcpy_struct, runtime_length_memset and
# thread_atexit_records_nothing. Measured on the fork with HG_GENMC_NO_STUTTER=1: it does not
# finish failed_cas_acquire or weak_cas_push_stutter. uninit_heap_read_reported pins the report's
# wording, and the weak_cas_spurious_* pair pins that a spurious failure of a weak CAS is explored
# wherever it is not a repeat of the attempt before it.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GENMC="${GENMC:-$HOME/genmc-019/build/bin/genmc}"
fail=0
for src in "$HERE"/*.cpp; do
    name="$(basename "$src" .cpp)"
    expect="$(sed -n '1s|^// Expect: ||p' "$src")"
    [ -n "$expect" ] || expect='No errors were detected'
    read -r -a args <<<"$(sed -n '1,3s|^// Args: ||p' "$src")"
    read -r -a envs <<<"$(sed -n '1,3s|^// Env: ||p' "$src")"
    out="$(env "${envs[@]}" timeout 600 "$GENMC" --disable-estimation "${args[@]}" -- -std=c++17 "$src" 2>&1)"
    if grep -qF "$expect" <<<"$out"; then
        echo "ok    $name ($expect; $(grep -oE 'explored: [0-9]+' <<<"$out"))"
    else
        echo "FAIL  $name (expected: $expect)"; grep -E 'Error|INTERNAL|error|No errors' <<<"$out" | head -3; fail=1
    fi
done
exit $fail
