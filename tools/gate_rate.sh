#!/usr/bin/env bash
# Failure RATE of a gate, over N independent runs.
#
# WHY THIS EXISTS. A gate answers pass/fail, and for a deterministic gate that is the whole
# answer. For a gate guarding a race it is not: "passed" and "did not fire this time" produce
# identical output, so a single green run is compatible with a defect that fires one time in
# thirty. Acting on one sample per arm is how an innocent change got convicted earlier in this
# work -- the change was reverted, the rate was unchanged, and the real cause was elsewhere.
#
# The rate is the measurement. A fix moves it to 0/N for an N large enough to have seen the
# old rate many times over; anything less is a smaller sample, not a repair.
#
# WHY SEPARATE PROCESSES rather than --gtest_repeat. Repeat runs the iterations inside one
# process, so the allocator, the page cache and every lazily-initialised singleton are warm
# after the first. A race that depends on first-touch page faults or on cold-start thread
# scheduling can be invisible under repeat and visible across processes. Independent processes
# are also what the real observation came from: a full-suite run, once.
#
# Usage:
#   tools/gate_rate.sh <gtest-filter> [N] [--binary PATH] [--load K] [--max-rate R]
#
#   <gtest-filter>  passed through as --gtest_filter
#   N               iterations (default 30)
#   --binary PATH   test binary (default build_linux/all_tests)
#   --load K        run K spinning threads alongside, to reproduce suite-like scheduling
#   --max-rate R    exit 1 if the observed failure rate exceeds R (a float, e.g. 0.0)
#                   omitted: always exit 0, because this is an instrument and not a gate
#
# Examples:
#   tools/gate_rate.sh 'CausalDeterminism.*' 100
#   tools/gate_rate.sh 'MatchCompleteness.*' 200 --max-rate 0.0
#   tools/gate_rate.sh '*Quotient*' 50 --load 4

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FILTER="${1:-}"
if [ -z "$FILTER" ]; then
    sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
shift

N=30
BINARY="build_linux/all_tests"
LOAD=0
MAX_RATE=""

# A leading bare number is the iteration count; everything else is a flag.
case "${1:-}" in ''|-*) ;; *) N="$1"; shift ;; esac

while [ $# -gt 0 ]; do
    case "$1" in
        --binary)   BINARY="$2"; shift 2 ;;
        --load)     LOAD="$2";   shift 2 ;;
        --max-rate) MAX_RATE="$2"; shift 2 ;;
        *) echo "gate_rate: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

if [ ! -x "$BINARY" ]; then
    echo "gate_rate: '$BINARY' is not executable; build it first" >&2
    exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; [ -n "${LOAD_PIDS:-}" ] && kill $LOAD_PIDS 2>/dev/null' EXIT

# Background load. The one real observation of the quotient failure came from inside a full
# suite run and never reproduced standalone, so scheduling pressure is part of the experiment
# rather than a nuisance to be eliminated.
LOAD_PIDS=""
if [ "$LOAD" -gt 0 ] 2>/dev/null; then
    for _ in $(seq 1 "$LOAD"); do
        ( while :; do :; done ) &
        LOAD_PIDS="$LOAD_PIDS $!"
    done
    echo "gate_rate: $LOAD background spinners running"
fi

echo "gate_rate: '$FILTER' x $N  ($BINARY)"

RUNS_FAILED=0
for i in $(seq 1 "$N"); do
    "./$BINARY" --gtest_filter="$FILTER" >"$TMP/run.$i" 2>&1
    rc=$?
    # A crash produces no [  FAILED  ] line at all, so the exit code is what catches it.
    # Recording it under a synthetic name keeps it out of the per-test tally while still
    # counting against the run total -- a gate that segfaults has not passed.
    if [ $rc -ne 0 ]; then
        RUNS_FAILED=$((RUNS_FAILED + 1))
        if grep -q "^\[  FAILED  \]" "$TMP/run.$i"; then
            grep "^\[  FAILED  \]" "$TMP/run.$i" | sed 's/^\[  FAILED  \] //' \
                | sed 's/ (.*//' | grep -v '^$' | sort -u >> "$TMP/failures"
        else
            echo "<crash-or-no-verdict rc=$rc>" >> "$TMP/failures"
        fi
        cp "$TMP/run.$i" "$TMP/keep.$i"
    fi
    printf "\r  %d/%d runs, %d failed" "$i" "$N" "$RUNS_FAILED"
done
printf "\n"

RATE=$(awk -v f="$RUNS_FAILED" -v n="$N" 'BEGIN{printf "%.4f", (n?f/n:0)}')
echo
echo "gate_rate: $RUNS_FAILED/$N runs failed  (rate $RATE)"

if [ -s "$TMP/failures" ]; then
    echo "gate_rate: per-test failure counts"
    sort "$TMP/failures" | uniq -c | sort -rn | while read -r c name; do
        r=$(awk -v c="$c" -v n="$N" 'BEGIN{printf "%.4f", c/n}')
        printf "  %4d/%-5d  %-8s  %s\n" "$c" "$N" "$r" "$name"
    done
    echo
    # Output of the first failing run, because a rate without an instance is not actionable and
    # the reproducing run is exactly the thing that is hard to obtain again.
    FIRST="$(ls "$TMP"/keep.* 2>/dev/null | head -1)"
    if [ -n "$FIRST" ]; then
        echo "gate_rate: first failing run ($FIRST):"
        grep -E "^\[|Failure|Expected|Actual|Which is|  [0-9]+/" "$FIRST" | head -40
    fi
fi

if [ -n "$MAX_RATE" ]; then
    if awk -v r="$RATE" -v m="$MAX_RATE" 'BEGIN{exit !(r>m)}'; then
        echo "gate_rate: rate $RATE exceeds --max-rate $MAX_RATE" >&2
        exit 1
    fi
    echo "gate_rate: rate $RATE within --max-rate $MAX_RATE"
fi
exit 0
