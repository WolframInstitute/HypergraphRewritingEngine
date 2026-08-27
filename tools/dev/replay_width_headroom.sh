#!/usr/bin/env bash
# WHAT A SHARED INSTANCE WORKLIST WOULD BE WORTH, measured rather than argued.
#
# The per-instance replay is driven one POINT at a time -- a point being (canonical class, depth)
# -- so its parallel width is the class count. On a workload that collapses to a handful of
# classes the width is a handful whatever the worker count, while the WORK is proportional to the
# instance count. Measured with tools/bench_cpu_evolve's own counters: cycle4 is 7 classes over
# 68,185 instances and multirule is 7 over 146,599, against wpp's 2,677 over 15,967.
#
# THE WIDTH RATIO IS NOT A SPEEDUP. It says only that the decomposition is not the binding
# constraint; the core count is. What decides the wall-clock win is the replay's SHARE of
# runtime, and that is what this measures:
#
#   full     the shipped configuration -- record set on, so the replay runs
#   RAW=0    record set off, which is the switch that stops the replay
#
# at one and at eight PINNED threads. The replay's share is (full - raw0)/full at one thread;
# how much of the serial residue it accounts for is the difference between the two scalings.
# A workload whose scaling improves when the replay is switched off has the replay as its serial
# part, and that is the fraction a shared worklist could return.
#
# wpp is the control: its class count already exceeds any worker count, so it has no width to
# gain and a shared worklist could only cost it -- 6.0 instances per class is a list too short to
# divide. If wpp's numbers move here, the effect is not the one this is looking for.
#
# QUIET IS A PRECONDITION, NOT A PREFERENCE. These are wall-clock scaling numbers on a
# heterogeneous part; a contended run inflates the eight-thread arm far more than the
# single-thread arm and manufactures exactly the result this looks for.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BENCH=${BENCH:-build_linux/bench_cpu_evolve}
CPUS=${CPUS:-0,2,4,6,8,10,12,14}      # performance_cpus() on this 14900K
STEPS=${STEPS:-6}
ITERS=${ITERS:-5}
LIMIT=${LIMIT:-2.0}

[ -x "$BENCH" ] || { echo "no $BENCH" >&2; exit 1; }

for i in $(seq 1 120); do
    L=$(awk '{print $1}' /proc/loadavg)
    if awk -v l="$L" -v k="$LIMIT" 'BEGIN{exit !(l < k)}'; then
        echo "quiet at load $L"; break
    fi
    [ "$i" = 120 ] && { echo "NEVER QUIET (load $L) -- refusing to measure" >&2; exit 2; }
    sleep 30
done

ms() {  # ms <env-assignment-or-empty> <workload> <threads>
    local env="$1" w="$2" t="$3"
    env $env "$BENCH" "$STEPS" "$ITERS" "$t" "$w" "$CPUS" 2>/dev/null \
        | grep -oE 'median_ms=[0-9.]+' | head -1 | cut -d= -f2
}

printf "%-11s %10s %10s %8s | %10s %10s %8s | %9s\n" \
       workload full_1t full_8t full_sp raw0_1t raw0_8t raw0_sp replay_share
for w in "$@"; do
    f1=$(ms "" "$w" 1);  f8=$(ms "" "$w" 8)
    r1=$(ms "HG_BENCH_RAW=0" "$w" 1); r8=$(ms "HG_BENCH_RAW=0" "$w" 8)
    [ -z "$f1" ] && { echo "$w: no result"; continue; }
    awk -v w="$w" -v f1="$f1" -v f8="$f8" -v r1="$r1" -v r8="$r8" 'BEGIN{
        printf "%-11s %10.2f %10.2f %7.2fx | %10.2f %10.2f %7.2fx | %8.1f%%\n",
               w, f1, f8, f1/f8, r1, r8, r1/r8, 100.0*(f1-r1)/f1 }'
done
