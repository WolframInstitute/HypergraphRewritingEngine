#!/usr/bin/env bash
# Where a thread sweep's cycles go when instructions stay flat: per-thread-count perf stat
# over the demand-fill taxonomy (AMD Zen: ls_dmnd_fills_from_sys.*), so a falling IPC is
# attributed to the level that serves the fills -- local L2, local CCX, another CCX's cache
# (near_cache: the cross-L3-domain term), or DRAM -- rather than guessed at.
#
#   tools/dev/perf_fill_sweep.sh <bench-binary> <depth> <workload> [threads...]
#
# One line of `perf stat -x,` output per event per thread count, written to
# perf_fill_<workload>_<T>.txt beside a combined log. The bench binary must be a release
# build (stats compile-out); the binary's own stamp says which (hgcommon/build_stamp.hpp).
# The near_cache event needs an AMD part; on other vendors substitute the equivalent
# cross-domain fill event and say so in the provenance of whatever consumes the numbers.
set -uo pipefail
BIN="${1:?bench binary}"
DEPTH="${2:?depth}"
WORKLOAD="${3:?workload}"
shift 3
THREADS=("${@:-1}")
[ ${#THREADS[@]} -eq 0 ] && THREADS=(1 8 16 32)

EVENTS=instructions,cycles,ls_dmnd_fills_from_sys.all,ls_dmnd_fills_from_sys.local_l2,ls_dmnd_fills_from_sys.local_ccx,ls_dmnd_fills_from_sys.near_cache,ls_dmnd_fills_from_sys.far_cache,ls_dmnd_fills_from_sys.dram_io_near,ls_dmnd_fills_from_sys.dram_io_far

OUTDIR="${PERF_FILL_OUT:-/tmp}"
LOG="$OUTDIR/perf_fill_${WORKLOAD}.log"
echo "start $(date -u +%FT%TZ) bin=$BIN depth=$DEPTH workload=$WORKLOAD" > "$LOG"
"$BIN" --build-info >> "$LOG" 2>&1 || true
for T in "${THREADS[@]}"; do
  echo "=== threads=$T" >> "$LOG"
  perf stat -e "$EVENTS" -x, -o "$OUTDIR/perf_fill_${WORKLOAD}_${T}.txt" \
    "$BIN" "$DEPTH" 1 "$T" "$WORKLOAD" >> "$LOG" 2>&1
  cat "$OUTDIR/perf_fill_${WORKLOAD}_${T}.txt" >> "$LOG"
done
echo "end $(date -u +%FT%TZ)" >> "$LOG"
echo "$LOG"
