#!/usr/bin/env bash
# Build with a hard ceiling on what the build may consume.
#
# This machine is shared with the user's tmux session, their work, and other Claude sessions.
# A runaway build does not merely take a long time -- if it exhausts memory and swap, the whole
# environment goes down with it. That has already nearly happened once, at -j24.
#
# ulimit is not sufficient. `ulimit -v` is PER PROCESS, and the hazard here is the TOTAL across
# a tree of compilers: twenty cc1plus at a gigabyte each are individually well-behaved and
# collectively fatal. So this runs the build in its own process group and watches the sum.
#
# Two independent triggers, because either alone can be fooled:
#   RSS CEILING    total resident set of the build's process group. Catches this build.
#   AVAILABLE FLOOR MemAvailable across the machine. Catches this build being the last straw
#                  when something else is already heavy -- which is the case that actually
#                  threatens the user's session.
#
# On a trip the build's process group is killed, and ONLY that group: never a machine-wide
# pattern kill, which would assassinate the other sessions' compilers too.
#
# Usage: tools/safe_build.sh <build-dir> [target] [jobs]
#   jobs defaults to 2 for a directory configured with CUDA, else 4.

set -uo pipefail

BUILD_DIR=${1:?usage: safe_build.sh <build-dir> [target] [jobs]}
TARGET=${2:-}
CEILING_GB=${SAFE_BUILD_CEILING_GB:-8}
FLOOR_GB=${SAFE_BUILD_FLOOR_GB:-4}

# CUDA gets ONE job. Measured, not guessed: a single nvcc on gpu/src/match.cu peaks near 5 GB
# on its own -- deep template instantiation plus device lambdas -- so -j2 alone breaches an
# 8 GB ceiling. Host TUs are far cheaper and tolerate 4.
if [[ -z "${3:-}" ]]; then
    if grep -qs "CMAKE_CUDA_COMPILER:" "$BUILD_DIR/CMakeCache.txt"; then JOBS=1; else JOBS=4; fi
else
    JOBS=$3
fi

avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

start_avail=$(avail_gb)
echo "safe_build: dir=$BUILD_DIR target=${TARGET:-<all>} -j$JOBS"
echo "safe_build: ceiling ${CEILING_GB}GB group RSS, floor ${FLOOR_GB}GB available (now ${start_avail}GB)"

if (( start_avail < FLOOR_GB )); then
    echo "safe_build: REFUSING to start -- only ${start_avail}GB available, floor is ${FLOOR_GB}GB" >&2
    exit 2
fi

# Own process group, so the watchdog can kill the build and every compiler it spawned without
# touching anything else on the machine.
if [[ -n "$TARGET" ]]; then
    setsid cmake --build "$BUILD_DIR" --target "$TARGET" -j"$JOBS" &
else
    setsid cmake --build "$BUILD_DIR" -j"$JOBS" &
fi
BUILD_PID=$!
PGID=$(ps -o pgid= -p "$BUILD_PID" 2>/dev/null | tr -d ' ')

peak=0
tripped=""
while kill -0 "$BUILD_PID" 2>/dev/null; do
    # Sum RSS over the build's process group only.
    rss_kb=$(ps -eo pgid=,rss= --no-headers 2>/dev/null \
             | awk -v g="$PGID" '$1==g {s+=$2} END {print s+0}')
    rss_gb=$(( rss_kb / 1048576 ))
    (( rss_kb > peak )) && peak=$rss_kb

    if (( rss_gb >= CEILING_GB )); then
        tripped="group RSS ${rss_gb}GB reached the ${CEILING_GB}GB ceiling"
        break
    fi
    a=$(avail_gb)
    if (( a < FLOOR_GB )); then
        tripped="only ${a}GB available, below the ${FLOOR_GB}GB floor"
        break
    fi
    sleep 1
done

if [[ -n "$tripped" ]]; then
    echo "safe_build: KILLING the build -- $tripped" >&2
    [[ -n "$PGID" ]] && kill -TERM -"$PGID" 2>/dev/null
    sleep 2
    [[ -n "$PGID" ]] && kill -KILL -"$PGID" 2>/dev/null
    wait "$BUILD_PID" 2>/dev/null
    echo "safe_build: lower the job count and retry" >&2
    exit 3
fi

wait "$BUILD_PID"; rc=$?
echo "safe_build: done rc=$rc, peak group RSS $(( peak / 1048576 ))GB, available now $(avail_gb)GB"
exit $rc
