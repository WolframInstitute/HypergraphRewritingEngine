#!/usr/bin/env bash
# Run a verification command (GenMC harness, TLC model check) under the same
# process-group watchdog as safe_build: a hard ceiling on group RSS, a floor on
# machine-available memory, and a wall-clock cap. Model checkers grow their
# exploration state without bound; ulimit -v is per-process and gets ignored or
# warned away by the JVM and by genmc's allocator, so the guard watches the SUM
# and kills ONLY this group -- never a machine-wide pattern kill.
#
# An enumeration that cannot finish inside these budgets does not belong on this
# 19 GB shared box: shrink the harness bound or move the argument to TLA+ (state
# -bounded, seconds) instead of raising the ceiling.
#
# Usage: tools/safe_verify.sh <ceiling-gb> <wall-seconds> <cmd> [args...]
#   e.g. tools/safe_verify.sh 6 900 verification/genmc/run.sh concurrent_map_agreement

set -uo pipefail
CEILING_GB=${1:?usage: safe_verify.sh <ceiling-gb> <wall-seconds> <cmd> [args...]}
WALL_S=${2:?usage: safe_verify.sh <ceiling-gb> <wall-seconds> <cmd> [args...]}
shift 2
FLOOR_GB=4

avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
echo "safe_verify: ceiling ${CEILING_GB}GB group RSS, floor ${FLOOR_GB}GB avail, wall ${WALL_S}s: $*"

setsid "$@" &
PGID=$!
start=$(date +%s)
rc=""
while :; do
    if ! kill -0 "$PGID" 2>/dev/null; then wait "$PGID"; rc=$?; break; fi
    now=$(date +%s)
    rss_kb=$(ps -o rss= -g "$PGID" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    rss_gb=$(( rss_kb / 1048576 ))
    tripped=""
    if (( rss_gb >= CEILING_GB )); then tripped="group RSS ${rss_gb}GB >= ${CEILING_GB}GB"; fi
    if (( $(avail_gb) <= FLOOR_GB )); then tripped="machine available <= ${FLOOR_GB}GB"; fi
    if (( now - start >= WALL_S )); then tripped="wall clock ${WALL_S}s exceeded"; fi
    if [ -n "$tripped" ]; then
        echo "safe_verify: KILLING the run -- $tripped" >&2
        kill -TERM -- -"$PGID" 2>/dev/null; sleep 2; kill -KILL -- -"$PGID" 2>/dev/null
        wait "$PGID" 2>/dev/null
        rc=124
        break
    fi
    sleep 2
done
echo "safe_verify: done rc=$rc, available now $(avail_gb)GB"
exit "${rc:-1}"
