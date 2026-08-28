#!/usr/bin/env bash
# Hunt the determinism family under OVERSUBSCRIPTION, and keep every firing.
#
#   flake_hunt.sh start [instances] [filter] [cpus]
#                                       launch, detached, survives logout
#   flake_hunt.sh status                how many rounds, how many firings
#   flake_hunt.sh stop                  stop it -- REQUIRED before any timing runs
#
# WHY A NARROW CPU SET, and not just many threads. The engine PINS its workers, but only when
# ensure_default_cpu_order finds at least two CPUs spanning more than one cache domain. A CI
# runner has 2-4 cores in one domain, so it bails and every worker is placed by the OS; this box
# has 32 cores over 8 L3 instances, so every worker is pinned. Those are different regimes, and
# the firings all come from the unpinned one. Confining the process to two CPUs reproduces it:
# hardware_concurrency then reports 2, the domain check finds one domain and declines to pin, and
# 32 requested workers land unpinned on two cores -- which is what CI does.
#
# taskset therefore works here, though it does NOT work against a run that pins: there the pin
# overrides the mask.
#
# WHY OVERSUBSCRIPTION. Every CI firing of CausalDeterminism.NonQuotientFullyDeterministic came
# from a runner with 2-4 cores answering a request for up to 32 threads. A 32-core box running
# one instance barely preempts, and 4,320 local runs in that regime produced nothing. Several
# instances at once puts many runnable threads on each core, which is the condition the firings
# share -- and the engine pins its workers, so taskset cannot reproduce it (the pin overrides the
# mask).
#
# WHY IT MUST BE STOPPED EXPLICITLY. A loop left running is exactly the orphan that silently
# skews a later benchmark. `stop` is not optional housekeeping; run it before any measurement,
# and tools/dev/preflight_quiet.sh will refuse the box until you have.
set -uo pipefail

ROOT="${HG_HUNT_ROOT:-$HOME/hg_session/src}"
OUT="${HG_HUNT_OUT:-$HOME/flake_hunt}"
BIN="$ROOT/build_linux/all_tests"
TAG=hg_flake_hunt

# ROUNDS PER HOUR IS THE WHOLE POINT, so the default is the test that fires most rather than the
# whole family. Of twenty firings in a week of CI, thirteen were
# NonQuotientFullyDeterministic; it also runs in about six seconds against the family's forty-odd,
# and under this much contention the difference decides whether a rare event is reached at all.
# The family is still reachable -- pass a filter as the third argument.
FILTER="${3:-CausalDeterminism.NonQuotientFullyDeterministic}"

case "${1:-status}" in
start)
    instances="${2:-12}"
    # Each instance gets its OWN pair of CPUs, so it sees a two-core machine exactly as a CI
    # runner does, while as many run at once as the box has pairs. Sharing one pair between all
    # of them would be a different regime again -- 32 workers per instance times twelve on two
    # cores is not what CI does.
    ncpu="$(nproc)"
    cpus="${4:-auto}"
    [ -x "$BIN" ] || { echo "no test binary at $BIN" >&2; exit 2; }
    mkdir -p "$OUT"
    for i in $(seq 1 "$instances"); do
        if [ "$cpus" = auto ]; then
            a=$(( (2 * (i - 1)) % ncpu )); b=$(( (2 * (i - 1) + 1) % ncpu ))
            runner="taskset -c $a,$b"
        else
            runner="taskset -c $cpus"
        fi
        setsid nohup bash -c '
            out="'"$OUT"'"; bin="'"$BIN"'"; filter="'"$FILTER"'"; i="'"$i"'"; runner="'"$runner"'"
            round=0
            while [ ! -f "$out/STOP" ]; do
                round=$((round + 1))
                log="$out/run_${i}_${round}.log"
                if ! $runner "$bin" --gtest_filter="$filter" > "$log" 2>&1; then
                    # A KILL IS NOT A FIRING. `stop` pkills the binaries, so every run in flight
                    # exits non-zero with nothing flushed -- and recording those as firings
                    # manufactures a reproduction that did not happen. Measured: one stop produced
                    # sixteen empty FIRING logs. A real firing names the failing test.
                    if [ -f "$out/STOP" ] || ! grep -q '\[  FAILED  \]' "$log"; then
                        rm -f "$log"
                    else
                        # A FIRING IS THE WHOLE POINT: keep it entire, and keep going.
                        mv "$log" "$out/FIRING_${i}_${round}_$(date -u +%Y%m%dT%H%M%SZ).log"
                    fi
                else
                    rm -f "$log"
                fi
                echo "$round" > "$out/rounds_$i"
            done
        ' >/dev/null 2>&1 &
        disown
    done
    echo "$TAG: $instances instances started; output in $OUT"
    ;;
status)
    printf 'rounds: '
    cat "$OUT"/rounds_* 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0
    printf 'firings: %s\n' "$(ls "$OUT"/FIRING_* 2>/dev/null | wc -l)"
    ls "$OUT"/FIRING_* 2>/dev/null | tail -5
    ;;
stop)
    # The flag goes down BEFORE the kill, so a run cut short by it is discarded rather than
    # recorded. See the loop above.
    mkdir -p "$OUT"
    touch "$OUT/STOP"
    # Bracket the pattern so this command's own line does not match it, and kill the test
    # binaries directly rather than the shells, which exit on their own once the round ends.
    pkill -f '[a]ll_tests --gtest_filter' 2>/dev/null
    sleep 2
    pkill -f '[a]ll_tests --gtest_filter' 2>/dev/null
    echo "stopped; remove $OUT/STOP before starting again"
    ;;
*)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
