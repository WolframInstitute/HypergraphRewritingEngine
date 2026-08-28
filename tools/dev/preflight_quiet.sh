#!/usr/bin/env bash
# A HARD GATE IN FRONT OF EVERY MEASUREMENT. Exits non-zero when the machine is not quiet, and
# chains the measurement behind itself so a dirty machine cannot produce a number at all.
#
#   tools/dev/preflight_quiet.sh -- ./tools/dev/corpus_scale_sweep.sh ...
#
# WHY IT IS A GATE AND NOT A WARNING. A timing run on a machine with someone else's work on it
# produces a number that looks exactly like a clean one, and nothing downstream can tell them
# apart afterwards. The only place the distinction survives is here, before the run starts.
#
# WHAT IT IS FOR, measured rather than supposed: a wait-loop left behind by an earlier session --
# `until ! pgrep -f ...; do sleep 10; done` -- ran `pgrep` across the whole process table every ten
# seconds on the benchmark box for an entire evening of measurements, and was found only when the
# box was handed to someone else. A poller is invisible in `top` and it is not invisible in a
# median.
#
# SELF-EXCLUSION IS THE HARD PART. `pgrep -f <pattern>` matches the command line running it, and
# this script's own command line contains every pattern it searches for. Bracketing the pattern is
# not enough once a chained command also names a real path. So this excludes BY PID: itself, every
# ancestor up to init, and every descendant.
set -u

MAX_LOAD=${MAX_LOAD:-1.0}
WAIT_S=0
STRICT_USERS=${STRICT_USERS:-1}

usage() {
    cat <<'USAGE'
usage: preflight_quiet.sh [--max-load N] [--wait SECONDS] [--allow-users] [-- CMD ...]

  --max-load N     1-minute load average must be below N (default 1.0, or $MAX_LOAD)
  --wait SECONDS   poll until quiet, up to SECONDS, instead of failing immediately.
                   Load average is a DECAYING mean, so a machine that has just finished
                   a build reads busy for minutes after it is idle. This waits that out.
  --allow-users    do not fail when another user is logged in
  -- CMD ...       run CMD only if every check passes; its exit status becomes ours

Exit: 0 quiet (and CMD succeeded, when given); 1 not quiet; 2 usage.
USAGE
}

CMD=()
while [ $# -gt 0 ]; do
    case "$1" in
        --max-load) MAX_LOAD="${2:?}"; shift 2 ;;
        --wait)     WAIT_S="${2:?}"; shift 2 ;;
        --allow-users) STRICT_USERS=0; shift ;;
        -h|--help)  usage; exit 2 ;;
        --)         shift; CMD=("$@"); break ;;
        *)          echo "preflight: unknown argument '$1'" >&2; usage; exit 2 ;;
    esac
done

# Every PID in this script's own LINEAGE -- ancestors as well as descendants.
#
# ANCESTORS ARE THE HALF THAT IS EASY TO MISS, and leaving them out makes the gate reject every
# legitimate run rather than every dirty one. When a sweep chains this in front of itself the
# process tree is `timeout -> sweep -> preflight`, so the sweep is this script's PARENT and the
# timeout its GRANDPARENT; both carry the sweep's name on their command line and both match the
# patterns below. Caught by running the gate from inside a sweep, which reported the sweep itself.
own_pids() {
    local out="$$" frontier="$$" next child p
    # up: every ancestor to init
    p=$$
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
        [ -n "$p" ] && out="$out $p"
    done
    # down: every descendant
    for _ in 1 2 3 4 5; do
        next=""
        for p in $frontier; do
            for child in $(pgrep -P "$p" 2>/dev/null); do
                out="$out $child"; next="$next $child"
            done
        done
        [ -z "$next" ] && break
        frontier="$next"
    done
    echo "$out"
}

# Work this project starts, and the leftovers it starts them from. Grouped so a report names the
# kind of thing found rather than only a pattern.
BENCH_RE='all_tests|bench_cpu_evolve|bench_gpu_evolve|sampling_cost_smoke|profile_evolve|hg_evolve|ir_vs_wl|benchmark_suite'
SWEEP_RE='corpus_scale_sweep|corpus_determinism_sweep|corpus_depth_plan|rich_sweep|rich_plots|paper_tables|run_sweep|run_rich|hunt\.sh'
BUILD_RE='cmake|ninja|cc1plus|nvcc|cicc|ptxas|ld\.lld|collect2'
VERIFY_RE='genmc|tlc|tla2tools|valgrind|callgrind'
# A poller has no binary of its own: it is a shell holding a sleep loop, which is what makes it
# survive a session and stay invisible.
POLLER_RE='until +!|while +true|while +:|pgrep -f|sleep [0-9]+; *done'

offenders() {
    local mine; mine=" $(own_pids) "
    ps -eo pid,user,etimes,args --no-headers 2>/dev/null | while read -r pid user etimes args; do
        case "$mine" in *" $pid "*) continue ;; esac
        case "$args" in *preflight_quiet*) continue ;; esac
        local kind=""
        case "$args" in
            *[Cc]laude*|*clangd*|*node_modules*) continue ;;
        esac
        if   printf '%s' "$args" | grep -qE "$BENCH_RE";  then kind="benchmark"
        elif printf '%s' "$args" | grep -qE "$SWEEP_RE";  then kind="sweep"
        elif printf '%s' "$args" | grep -qE "$BUILD_RE";  then kind="build"
        elif printf '%s' "$args" | grep -qE "$VERIFY_RE"; then kind="verification"
        elif printf '%s' "$args" | grep -qE "$POLLER_RE"; then kind="poller/orphan"
        fi
        [ -n "$kind" ] && printf '%-14s pid=%-7s user=%-10s age=%-7ss %s\n' \
            "$kind" "$pid" "$user" "$etimes" "$(printf '%.100s' "$args")"
    done
}

load1() { awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0; }

other_users() {
    who 2>/dev/null | awk '{print $1}' | sort -u | grep -vx "$(id -un)" | tr '\n' ' '
}

check_once() {
    local bad=0
    FOUND="$(offenders)"
    if [ -n "$FOUND" ]; then
        echo "preflight: FAIL -- work already running:" >&2
        printf '%s\n' "$FOUND" >&2
        bad=1
    fi
    local l; l="$(load1)"
    if awk -v a="$l" -v b="$MAX_LOAD" 'BEGIN{exit !(a >= b)}'; then
        echo "preflight: FAIL -- 1-minute load $l is not below $MAX_LOAD" >&2
        bad=1
    fi
    if [ "$STRICT_USERS" = "1" ]; then
        local u; u="$(other_users)"
        if [ -n "$u" ]; then
            echo "preflight: FAIL -- other users logged in: $u" >&2
            bad=1
        fi
    fi
    return $bad
}

deadline=$(( $(date +%s) + WAIT_S ))
while :; do
    if check_once; then break; fi
    [ "$(date +%s)" -ge "$deadline" ] && {
        echo "preflight: machine not quiet; refusing to measure" >&2
        exit 1
    }
    sleep 5
done

echo "preflight: quiet (load $(load1), no foreign work, no orphans)"
[ ${#CMD[@]} -eq 0 ] && exit 0
exec "${CMD[@]}"
