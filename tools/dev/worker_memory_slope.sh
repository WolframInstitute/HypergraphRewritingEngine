#!/usr/bin/env bash
# Per-worker memory slope on a SMALL thread sweep, capped, so the question can be asked on a
# shared box.
#
# WHY THIS EXISTS. tools/dev/scaling_sweep.py's memory section is the instrument for this, and it
# sweeps 1/8/16/24/32 threads -- which on the shape workload reaches 7.5 GB resident and is not
# runnable on a 19 GB desktop shared with other work. The question it answers first, though, is
# only whether resident set grows PER WORKER and how steeply, and that is visible at 1/2/4. This
# runs the same binary with the same arguments as thread_memory_cost() so the numbers are
# comparable to the table, over a thread list that fits, under a hard address-space cap.
#
# The cap is the point. A run that dies on an allocation failure is a result; a run that pages a
# shared machine is an outage for everything else on it. Each measurement runs ALONE and under
# `ulimit -v`, so a regression that wants more than the cap fails loudly instead of reaching swap.
#
# Usage:
#   tools/dev/worker_memory_slope.sh [threads-csv] [rule] [edges] [steps]
# Defaults are scaling_sweep.py's first shape workload (growth:1:9) at 1,2,4 threads.
#
# Environment:
#   HG_MEM_CAP_MB   per-process address-space cap in MB (default: 45% of MemAvailable)

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build_linux/sampling_cost_smoke"

THREADS="${1:-1,2,4}"
RULE="${2:-growth}"
EDGES="${3:-1}"
STEPS="${4:-9}"

if [ ! -x "$BIN" ]; then
    echo "worker_memory_slope.sh: $BIN not built" >&2
    exit 2
fi

if [ -z "${HG_MEM_CAP_MB:-}" ]; then
    avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    HG_MEM_CAP_MB=$(( avail_kb * 45 / 100 / 1024 ))
    [ "$HG_MEM_CAP_MB" -lt 512 ] && HG_MEM_CAP_MB=512
fi
echo "cap ${HG_MEM_CAP_MB} MB per process; workload ${RULE} ${EDGES} ${STEPS}; one at a time"
echo

printf '%8s %10s %10s %10s %12s %12s\n' threads wall_s user_s sys_s peak_rss_MB minor_faults
prev_rss=""
for t in ${THREADS//,/ }; do
    out="$(mktemp)"
    ( ulimit -v $((HG_MEM_CAP_MB * 1024))
      exec /usr/bin/time -v "$BIN" off "$RULE" "$EDGES" "$STEPS" "$t" 4 full ) \
        >/dev/null 2>"$out"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  ${t}t: FAILED (exit $rc) -- the cap held, which is the point of having one"
        tail -3 "$out"
        rm -f "$out"
        continue
    fi
    wall="$(awk -F': ' '/Elapsed \(wall clock\)/ {print $2}' "$out")"
    user="$(awk -F': ' '/User time/ {print $2}' "$out")"
    sys="$(awk -F': ' '/System time/ {print $2}' "$out")"
    rss_kb="$(awk -F': ' '/Maximum resident set size/ {print $2}' "$out")"
    faults="$(awk -F': ' '/Minor \(reclaiming a frame\) page faults/ {print $2}' "$out")"
    rss_mb=$(( rss_kb / 1024 ))
    printf '%8s %10s %10s %10s %12s %12s\n' "$t" "$wall" "$user" "$sys" "$rss_mb" "$faults"
    if [ -n "$prev_rss" ]; then
        echo "         (+$(( rss_mb - prev_rss )) MB since the previous thread count)"
    fi
    prev_rss="$rss_mb"
    rm -f "$out"
done
