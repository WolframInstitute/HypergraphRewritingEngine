#!/usr/bin/env bash
# The reproducers for genmc-0.17.0-fixes.patch, each a program of a few lines that the unpatched
# checker gets wrong and the patched one verifies. Run after building the checker:
#
#   verification/genmc/checker/run.sh
#
# Every one must report "No errors were detected"; a checker without the patch aborts on the
# first three with an internal check or reports a non-allocated access, and does not finish
# dependence_dag_paths within the ten minutes this script gives it.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GENMC="${GENMC:-$HOME/genmc/build/bin/genmc}"
fail=0
for src in "$HERE"/*.cpp; do
    name="$(basename "$src" .cpp)"
    out="$(timeout 600 "$GENMC" --disable-estimation -- -std=c++17 "$src" 2>&1)"
    if grep -q 'No errors were detected' <<<"$out"; then
        echo "ok    $name ($(grep -oE 'explored: [0-9]+' <<<"$out"))"
    else
        echo "FAIL  $name"; grep -E 'Error|INTERNAL|error' <<<"$out" | head -3; fail=1
    fi
done
exit $fail
