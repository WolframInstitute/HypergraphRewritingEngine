#!/usr/bin/env bash
# Hunt the determinism family under OVERSUBSCRIPTION, and keep every firing.
#
#   flake_hunt.sh start [instances]     launch, detached, survives logout
#   flake_hunt.sh status                how many rounds, how many firings
#   flake_hunt.sh stop                  stop it -- REQUIRED before any timing runs
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
    [ -x "$BIN" ] || { echo "no test binary at $BIN" >&2; exit 2; }
    mkdir -p "$OUT"
    for i in $(seq 1 "$instances"); do
        setsid nohup bash -c '
            out="'"$OUT"'"; bin="'"$BIN"'"; filter="'"$FILTER"'"; i="'"$i"'"
            round=0
            while [ ! -f "$out/STOP" ]; do
                round=$((round + 1))
                log="$out/run_${i}_${round}.log"
                if ! "$bin" --gtest_filter="$filter" > "$log" 2>&1; then
                    # A FIRING IS THE WHOLE POINT: keep it entire, and keep going.
                    mv "$log" "$out/FIRING_${i}_${round}_$(date -u +%Y%m%dT%H%M%SZ).log"
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
