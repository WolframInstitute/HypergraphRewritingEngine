#!/usr/bin/env bash
# Decide whether this machine is quiet enough for a timing measurement, and say what is on it.
#
# ONE IMPLEMENTATION, TWO CALLERS, TWO MACHINES. rich_sweep.sh gates every timed point on a
# rented box; paper_tables.py gates the Wolfram comparison on this desktop. The desktop is the
# harder case and the reason this exists: it runs the developer's editor, other agents and other
# projects, so "the machine is idle" is never true by default there. A wall time taken on it
# without checking is a measurement of whatever else was running.
#
# WHAT IT CHECKS
#   * no process from a watch list is running -- compilers, model checkers, JVMs, profilers, and
#     this project's own measurement and test binaries;
#   * the 1-minute load average is below QUIET_LOAD.
#
# THE 1-MINUTE AVERAGE IS THE RIGHT ONE and it is checked rather than the instantaneous count
# because it DECAYS: a machine that finished a saturating build ten seconds ago still has warm
# queues and a cold cache, and a run started there reports that, not the engine.
#
# PROCESS NAMES ARE COMPARED TRUNCATED TO 15 CHARACTERS, which is what the kernel stores in comm
# and therefore all pgrep -x can ever match. A pattern longer than that matches NOTHING and pgrep
# exits non-zero while printing a warning -- read as "count zero", it turns this gate into a
# rubber stamp. Both sampling_cost_smoke (20) and bench_cpu_evolve (16) are over the limit, and
# that defect shipped in the first version of this check.
#
# pgrep -x, never -f: an -f pattern matches this script's own command line and the shell that
# launched it, so the gate would report itself as contention and wait for ever.
#
# Usage:
#   tools/dev/quiet_gate.sh check          exit 0 if quiet, 1 if not; prints one status line
#   tools/dev/quiet_gate.sh wait [secs]    block until quiet, or until secs elapse (default 600)
#   . tools/dev/quiet_gate.sh              source it, then call quiet_status / quiet_wait
#
# Environment:
#   QUIET_LOAD    1-minute load average that counts as quiet (default 1.5)
#   QUIET_EXTRA   extra process names to watch, space separated

QUIET_LOAD=${QUIET_LOAD:-1.5}
QUIET_NAMES=${QUIET_NAMES:-"genmc cc1plus cicc nvcc ptxas java valgrind ncu nsys \
sampling_cost_smoke bench_cpu_evolve bench_gpu_evolve all_tests hg_gpu_tests \
gpu_differential_tests wolfram WolframKernel wolframscript"}

# Sets QUIET_BUSY to the names found running (empty when none) and QUIET_LOAD1 to the load.
quiet_status() {
    QUIET_BUSY=""
    local n short count
    for n in $QUIET_NAMES ${QUIET_EXTRA:-}; do
        short=$(printf '%.15s' "$n")
        # pgrep -c PRINTS 0 AND EXITS NON-ZERO when nothing matches, so `|| echo 0` appends a
        # second zero and the test sees "0\n0", which is not an integer. Take pgrep's output and
        # substitute a default only when it produced none.
        count=$(pgrep -c -x "$short" 2>/dev/null)
        [ -z "$count" ] && count=0
        if [ "$count" -gt 0 ]; then
            QUIET_BUSY="$QUIET_BUSY $n($count)"
        fi
    done
    QUIET_LOAD1=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
    [ -z "$QUIET_BUSY" ] && awk "BEGIN{exit !($QUIET_LOAD1 < $QUIET_LOAD)}"
}

# Block until quiet. Returns 0 if it became quiet, 1 if it timed out (caller decides what to do;
# rich_sweep marks the point contended rather than dropping it).
quiet_wait() {
    local budget=${1:-600} waited=0
    while :; do
        if quiet_status; then
            [ "$waited" -gt 0 ] && echo "quiet after ${waited}s (load $QUIET_LOAD1)" >&2
            return 0
        fi
        if [ "$waited" -ge "$budget" ]; then
            echo "NOT QUIET after ${waited}s (load $QUIET_LOAD1, busy:$QUIET_BUSY)" >&2
            return 1
        fi
        [ "$waited" = 0 ] \
            && echo "waiting for a quiet machine (load $QUIET_LOAD1, busy:$QUIET_BUSY)" >&2
        sleep 5
        waited=$(( waited + 5 ))
    done
}

# Only act when RUN, not when sourced.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-check}" in
        check)
            if quiet_status; then
                echo "QUIET load=$QUIET_LOAD1"
                exit 0
            fi
            echo "BUSY load=$QUIET_LOAD1 threshold=$QUIET_LOAD running:${QUIET_BUSY:-none}"
            exit 1
            ;;
        wait)
            quiet_wait "${2:-600}"
            exit $?
            ;;
        *)
            echo "usage: $0 check|wait [secs]" >&2
            exit 2
            ;;
    esac
fi
