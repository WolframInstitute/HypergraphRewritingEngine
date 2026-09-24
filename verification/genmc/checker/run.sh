#!/usr/bin/env bash
# The reproducers for genmc-0.17.0-fixes.patch, each a program of a few lines that the unpatched
# checker gets wrong and the patched one gets right. Run after building the checker:
#
#   verification/genmc/checker/run.sh
#
# A reproducer whose first line is `// Expect: <text>` must print <text>; every other one must
# report "No errors were detected". A checker without the patch aborts on the allocation and
# promotion reproducers with an internal check or reports a non-allocated access, does not finish
# dependence_dag_paths within the ten minutes this script gives it, reports no race on the
# cas_fail_* reproducers and fails memmove_overlap_tail's assertion.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GENMC="${GENMC:-$HOME/genmc/build/bin/genmc}"
fail=0
for src in "$HERE"/*.cpp; do
    name="$(basename "$src" .cpp)"
    expect="$(sed -n '1s|^// Expect: ||p' "$src")"
    [ -n "$expect" ] || expect='No errors were detected'
    out="$(timeout 600 "$GENMC" --disable-estimation -- -std=c++17 "$src" 2>&1)"
    if grep -qF "$expect" <<<"$out"; then
        echo "ok    $name ($expect; $(grep -oE 'explored: [0-9]+' <<<"$out"))"
    else
        echo "FAIL  $name (expected: $expect)"; grep -E 'Error|INTERNAL|error|No errors' <<<"$out" | head -3; fail=1
    fi
done
exit $fail
