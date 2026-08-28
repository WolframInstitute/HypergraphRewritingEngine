#!/bin/bash
# Determinism gate over the generated corpus: the output must not depend on worker count.
#
# WHY THIS EXISTS SEPARATELY FROM test_determinism_fuzzing. That suite repeats a run at one
# worker count and asserts the counts agree across repeats, on eight hand-written rules whose
# largest case is a few hundred states. It cannot reach a divergence that needs contention at
# scale to appear, and the corpus contains workloads reaching hundreds of thousands of raw states.
# This runs the SAME workloads the scaling sweep runs, and compares one worker against many --
# one worker is the ground truth, because with a single worker there is no interleaving to depend
# on.
#
# The canonical state count and the event count are the invariants: the state set and event set
# are a function of the inputs alone. Raw state ids and raw counts are NOT compared -- which raw
# representative stands for a class is allowed to depend on arrival order.
#
# Input: "<workload> <depth>" per line on stdin. Output: one row per workload, PASS or MISMATCH.
# Exit 1 if any workload mismatched, so it can gate.
set -u
BIN=${1:?bench binary}
THREADS=${2:-8}
REPEATS=${3:-1}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-900}

fail=0
# THE QUIET GATE, AND IT IS NOT OPTIONAL. A timing run on a machine with other work on it produces
# a number indistinguishable from a clean one, and nothing downstream can separate them later. An
# orphaned poller left by an earlier session ran `pgrep` across the whole process table every ten
# seconds through an entire evening of measurements on the benchmark box, and was found only when
# the box was handed to someone else.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/preflight_quiet.sh" --wait "${PREFLIGHT_WAIT:-120}" || {
    echo "$(basename "$0"): refusing to measure on a machine that is not quiet" >&2
    exit 1
}

printf '%-22s %-6s %-26s %s\n' workload depth "canonical (1t vs ${THREADS}t)" verdict
while read -r w d; do
    [ -z "${w:-}" ] && continue
    case "$w" in \#*) continue;; esac

    # Truncated runs are excluded, not compared. Past a container ceiling the engine returns
    # valid partial work and which states got in is the arrival race, so a difference there is
    # the ceiling talking, not the determinism contract.
    b_out=$(timeout "$PER_RUN_TIMEOUT" "$BIN" "$d" 1 1 "$w" 2>&1)
    if printf '%s\n' "$b_out" | grep -q 'capacity limit reached'; then
        printf '%-22s %-6s %-26s %s\n' "$w" "$d" "-" "SKIP(truncated)"
        continue
    fi
    base=$(printf '%s\n' "$b_out" \
           | grep -oE 'canonical=[0-9]+ raw=[0-9]+' | head -1 | grep -oE 'canonical=[0-9]+' | cut -d= -f2)
    if [ -z "$base" ]; then
        printf '%-22s %-6s %-26s %s\n' "$w" "$d" "-" "TIMEOUT_OR_FAIL(1t)"
        fail=1
        continue
    fi

    seen=""
    r=0
    while [ "$r" -lt "$REPEATS" ]; do
        v=$(timeout "$PER_RUN_TIMEOUT" "$BIN" "$d" 1 "$THREADS" "$w" 2>/dev/null \
            | grep -oE 'canonical=[0-9]+ raw=[0-9]+' | head -1 | grep -oE 'canonical=[0-9]+' | cut -d= -f2)
        [ -z "$v" ] && v="FAIL"
        seen="$seen $v"
        r=$((r + 1))
    done

    bad=0
    for v in $seen; do [ "$v" = "$base" ] || bad=1; done
    if [ "$bad" = "1" ]; then
        printf '%-22s %-6s %-26s %s\n' "$w" "$d" "$base vs$seen" "MISMATCH"
        fail=1
    else
        printf '%-22s %-6s %-26s %s\n' "$w" "$d" "$base" PASS
    fi
done
exit $fail
