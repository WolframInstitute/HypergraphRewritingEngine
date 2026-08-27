#!/bin/bash
# Thread-scaling sweep over the generated corpus, one row per workload.
#
# THE DEPTH IS PER WORKLOAD, and that is the point. The corpus spans four orders of magnitude in
# work at a fixed depth, so a single depth measures the large workloads properly and measures the
# engine's per-call floor on the small ones -- and a floor-dominated row reports a speedup that is
# a statement about process startup, not about the engine. Each workload is therefore run at the
# shallowest depth that reaches TARGET raw states, estimated from its own growth rate.
#
# Input: the depth plan on stdin, one "<workload> <depth>" per line (tools/dev/corpus_depth_plan.py
# derives it from a corpusgrow run). Output: one row per workload, speedup at each thread count.
#
# Usage: corpus_scale_sweep.sh <bin> <threads-csv> <iters> < plan.txt
set -u
BIN=${1:?bench binary}
THREADS=${2:-1,2,4,8,16}
ITERS=${3:-5}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-900}

printf '%-22s %-6s %s\n' workload depth "speedup per thread count ($THREADS)"
while read -r w d; do
    [ -z "${w:-}" ] && continue
    # A TRUNCATED RUN IS NOT A MEASUREMENT. Past a container ceiling the engine returns valid
    # partial work with a warning, and WHICH states got in is decided by the arrival race -- so
    # the counts, and any ratio taken from them, vary between runs and between thread counts for
    # a reason that has nothing to do with the engine's concurrency.
    #
    # The depth plan estimates work from a growth rate and cannot predict the ceiling, which the
    # match storage reaches at a state count that depends on the shape. So the sweep BACKS OFF:
    # a truncated depth is retried one shallower, down to one, and the depth it settled on is
    # printed in the row. A workload that truncates even at depth one is reported as such, never
    # dropped, so the summary cannot read as coverage it does not have.
    out=""
    rc=0
    while [ "$d" -ge 1 ]; do
        out=$(timeout "$PER_RUN_TIMEOUT" "$BIN" "$d" "$ITERS" "$THREADS" "$w" 2>/dev/null)
        rc=$?
        [ $rc -ne 0 ] && break
        printf '%s\n' "$out" | grep -q 'capacity limit reached' || break
        d=$((d - 1))
    done
    if [ $rc -ne 0 ]; then
        printf '%-22s %-6s TIMEOUT_OR_FAIL rc=%d\n' "$w" "$d" "$rc"
        continue
    fi
    if printf '%s\n' "$out" | grep -q 'capacity limit reached'; then
        printf '%-22s %-6s %s\n' "$w" "$d" "TRUNCATED even at depth 1 (capacity ceiling)"
        continue
    fi
    sp=$(printf '%s\n' "$out" | grep -oE 'speedup=[0-9.]+' | cut -d= -f2 | tr '\n' ' ')
    raw=$(printf '%s\n' "$out" | grep -oE 'raw=[0-9]+' | head -1 | cut -d= -f2)
    printf '%-22s %-6s %s raw=%s\n' "$w" "$d" "$sp" "${raw:-?}"
done
