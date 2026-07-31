#!/usr/bin/env bash
# Instruction and cache cost of one commit, appended to a CSV keyed by that commit.
#
# WHY THIS EXISTS. #43 recorded the non-Full modes as "16% above the pre-sweep baseline, and the
# rest is unexplained". Re-measured months later it was 24.4%, and the reason it went stale is
# that measuring it was a manual act nobody repeated. Worse, the split could not be recovered
# afterwards: once twenty commits have landed, a single total says the cost moved and cannot say
# which landing moved it. Attribution has to be collected AS the commits land, or not at all.
#
# WHY CALLGRIND AND NOT WALL CLOCK. Wall clock on this box drifts more than 10% run to run, which
# is larger than most of the effects worth attributing. Callgrind counts instructions
# deterministically and cachegrind simulates the cache, so both are immune to host load and to
# what else happened to be running. Single-threaded, because per-function attribution is only
# clean when the threads are not interleaved -- the point here is WHICH CODE costs, not throughput.
#
# WHAT IT DOES NOT DO. It does not check out other commits or bisect. It measures the tree as it
# stands and records the commit that tree is at, so it can be run at each green checkpoint and the
# series accumulates. Sweeping history means checking out and re-running it, which is expensive
# and is the caller's decision rather than this script's.
#
# Usage:  tools/cost_sweep.sh [output.csv]
#         default output: benchmark_results/cost_sweep.csv

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT="${1:-benchmark_results/cost_sweep.csv}"
BUILD="build_linux"
mkdir -p "$(dirname "$OUT")"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "cost_sweep: valgrind not found; the whole point is a deterministic instrument" >&2
    exit 1
fi

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY=""
git diff --quiet 2>/dev/null || DIRTY="+dirty"
SUBJECT="$(git log -1 --format=%s 2>/dev/null | tr ',' ' ' || echo unknown)"

# A tree with uncommitted changes is not attributable to the commit it claims, so it is marked
# rather than silently recorded as that commit.
if [ -n "$DIRTY" ]; then
    echo "cost_sweep: working tree is dirty; rows will be marked ${COMMIT}${DIRTY}" >&2
fi

./tools/safe_build.sh "$BUILD" profile_evolve 4 >/dev/null 2>&1 || {
    echo "cost_sweep: profile_evolve failed to build" >&2; exit 1; }

if [ ! -s "$OUT" ]; then
    echo "commit,subject,workload,mode,instructions,d1_misses,lld_misses,dram_bytes" > "$OUT"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for mode in full auto none; do
    # Instructions, from callgrind.
    valgrind --tool=callgrind --callgrind-out-file="$TMP/cg.$mode" \
        "./$BUILD/profile_evolve" 5 "$mode" 1 >/dev/null 2>"$TMP/cg.$mode.err"
    ir=$(grep -oE "refs:[ ]*[0-9,]+" "$TMP/cg.$mode.err" | tr -cd '0-9')

    # Cache misses, from cachegrind. LLd misses times the line size is the DRAM-traffic axis:
    # an instruction count alone cannot distinguish work from waiting for memory.
    # --cache-sim=yes is REQUIRED: cachegrind 3.22 reports only instruction refs without it, and
    # the miss columns come back silently zero rather than absent, which reads like a workload
    # that never touches memory.
    valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file="$TMP/ch.$mode" \
        "./$BUILD/profile_evolve" 5 "$mode" 1 >/dev/null 2>"$TMP/ch.$mode.err"
    d1=$(grep -oE "D1  misses:[ ]*[0-9,]+" "$TMP/ch.$mode.err" | head -1 | tr -cd '0-9')
    lld=$(grep -oE "LLd misses:[ ]*[0-9,]+" "$TMP/ch.$mode.err" | head -1 | tr -cd '0-9')
    dram=$(( ${lld:-0} * 64 ))

    echo "${COMMIT}${DIRTY},${SUBJECT},wolfram-5step,${mode},${ir:-0},${d1:-0},${lld:-0},${dram}" >> "$OUT"
    printf "%-12s %-5s  instructions=%-12s D1=%-10s LLd=%-8s DRAM=%s B\n" \
        "${COMMIT}${DIRTY}" "$mode" "${ir:-0}" "${d1:-0}" "${lld:-0}" "$dram"
done

echo "cost_sweep: appended 3 rows to $OUT"
