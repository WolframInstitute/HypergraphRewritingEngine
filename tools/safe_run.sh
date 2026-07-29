#!/usr/bin/env bash
# Run a binary with a hard ceiling on what it may consume AND how long it may take.
#
# safe_build.sh guards COMPILATION. This guards EXECUTION, which is a different hazard and a
# worse one for GPU work: a persistent kernel whose termination detector never fires does not
# merely take a long time, it occupies the device indefinitely. On this machine the GPU also
# drives the display, so a wedged kernel is not a background nuisance -- it takes the desktop
# with it, and the other sessions on the box with that.
#
# Three independent triggers, because each catches what the others cannot:
#   WALL CLOCK      a run that has not finished by now is not going to. This is the one that
#                   catches a spinning kernel, which consumes no extra memory at all and so
#                   trips neither of the other two.
#   RSS CEILING     total resident set of the run's process group. Catches this run.
#   AVAILABLE FLOOR MemAvailable across the machine. Catches this run being the last straw when
#                   something else is already heavy -- the case that actually threatens the
#                   user's session.
#
# On a trip the run's process group is killed, and ONLY that group: never a machine-wide
# pattern kill, which would take out the other sessions' work too. SIGTERM first so a CUDA
# process can tear its context down (which is what stops the kernel on the device), then
# SIGKILL if it will not go.
#
# Usage: tools/safe_run.sh <seconds> <command> [args...]
#   SAFE_RUN_CEILING_GB   group RSS ceiling, default 8
#   SAFE_RUN_FLOOR_GB     MemAvailable floor, default 3

set -uo pipefail

TIMEOUT=${1:?usage: safe_run.sh <seconds> <command> [args...]}
shift
CEILING_GB=${SAFE_RUN_CEILING_GB:-8}
FLOOR_GB=${SAFE_RUN_FLOOR_GB:-3}

avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
group_rss_gb() {
    local pgid=$1
    ps -e -o pgid=,rss= 2>/dev/null | awk -v g="$pgid" '$1==g {s+=$2} END {printf "%d", s/1048576}'
}

start_avail=$(avail_gb)
echo "safe_run: timeout ${TIMEOUT}s, ceiling ${CEILING_GB}GB group RSS, floor ${FLOOR_GB}GB available (now ${start_avail}GB)"
echo "safe_run: $*"

if (( start_avail < FLOOR_GB )); then
    echo "safe_run: REFUSING to start -- only ${start_avail}GB available, floor is ${FLOOR_GB}GB" >&2
    exit 2
fi

# Own process group, so the watchdog can kill this run and everything it spawned without
# touching anything else on the machine.
setsid "$@" &
RUN_PID=$!
PGID=$(ps -o pgid= -p "$RUN_PID" 2>/dev/null | tr -d ' ')
if [[ -z "$PGID" ]]; then
    wait "$RUN_PID"; exit $?
fi

peak=0
elapsed=0
tripped=""
while kill -0 "$RUN_PID" 2>/dev/null; do
    sleep 1
    elapsed=$((elapsed + 1))
    rss=$(group_rss_gb "$PGID")
    (( rss > peak )) && peak=$rss
    avail=$(avail_gb)

    if (( elapsed >= TIMEOUT )); then
        tripped="wall clock: still running after ${TIMEOUT}s"
    elif (( rss > CEILING_GB )); then
        tripped="group RSS ${rss}GB exceeded the ${CEILING_GB}GB ceiling"
    elif (( avail < FLOOR_GB )); then
        tripped="only ${avail}GB available, below the ${FLOOR_GB}GB floor"
    fi

    if [[ -n "$tripped" ]]; then
        echo "safe_run: KILLING process group $PGID -- $tripped" >&2
        # SIGTERM first: a CUDA process needs to tear down its context, and that is what
        # actually stops work already running on the device. SIGKILL cannot do it politely.
        kill -TERM -"$PGID" 2>/dev/null
        for _ in $(seq 1 10); do
            kill -0 "$RUN_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -"$PGID" 2>/dev/null
        wait "$RUN_PID" 2>/dev/null
        echo "safe_run: killed after ${elapsed}s, peak group RSS ${peak}GB" >&2
        exit 124
    fi
done

wait "$RUN_PID"
rc=$?
echo "safe_run: done rc=$rc after ${elapsed}s, peak group RSS ${peak}GB, available now $(avail_gb)GB"
exit $rc
