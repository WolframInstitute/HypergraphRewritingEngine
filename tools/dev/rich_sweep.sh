#!/usr/bin/env bash
# Collect the LHS-shape data set: how far each rule shape evolves, how big the relations it
# builds are, and how its parallel efficiency behaves.
#
# WHY IT IS TWO PHASES WITH DIFFERENT CONCURRENCY. The two things being collected have opposite
# requirements and mixing them silently corrupts one of them.
#
#   DEPTH phase -- states, events, causal edges, branchial edges at each depth. These counts are
#   DETERMINISTIC and thread-independent, so the machine can be saturated: several runs at once,
#   each on a share of the cores. Wall time from this phase is NOT a measurement and is not used
#   as one; it exists only to decide when to stop going deeper.
#
#   SCALING phase -- wall time against thread count. A timing run shares nothing: one job at a
#   time. The box is dedicated, so the interference this guards against is THIS SWEEP's own
#   concurrency: four saturating jobs would each report the other three. Serial by construction,
#   and the long pole.
#
# HOW DEEP. Each shape is pushed one depth at a time until a run exceeds DEPTH_BUDGET_S, then
# stopped. That is what makes "as deep as this shape goes" a measured boundary per shape rather
# than one depth guessed for all of them -- the shapes differ by orders of magnitude in cost per
# depth, so a single fixed depth would either truncate the cheap ones or never finish the dear.
#
# Every run emits one RICH key=value line; nothing here parses or aggregates, so a partial run is
# still a usable data set and re-running appends rather than replaces.
#
# Usage:  tools/dev/rich_sweep.sh <build-dir> <out-dir> [depth|scaling|all]

set -uo pipefail

LAST_QUIET_LOAD=""
BUILD="${1:?build dir}"
OUT="${2:?out dir}"
WHICH="${3:-all}"
BIN="$BUILD/sampling_cost_smoke"

[ -x "$BIN" ] || { echo "no $BIN" >&2; exit 1; }
mkdir -p "$OUT"

# RECORD THE MACHINE THAT MEASURED, next to the data it measured. The figures are drawn later,
# often on a different computer from the one that produced the rows, and paper_tables stamps the
# machine it is RUNNING on. Without this file the fragments claim they were measured wherever
# they were drawn, which is a false statement about provenance in a paper whose whole table
# convention is that a table carries its machine with it.
mkdir -p "$OUT"
{ command -v lscpu >/dev/null 2>&1 && lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}'; } \
    > "$OUT/measured_on.txt" 2>/dev/null || true
[ -s "$OUT/measured_on.txt" ] || uname -srm > "$OUT/measured_on.txt"

NPROC=$(nproc)
DEPTH_BUDGET_S=${DEPTH_BUDGET_S:-240}     # stop deepening a shape once one run costs this much
SCALE_BUDGET_S=${SCALE_BUDGET_S:-900}     # a single scaling point may not exceed this
CONC=${CONC:-4}                           # concurrent jobs in the depth phase
THREADS_PER=$(( NPROC / CONC )); [ "$THREADS_PER" -lt 1 ] && THREADS_PER=1

# rule:init_edges. The seed must be at least as large as the pattern or nothing matches, and the
# star shapes get a SMALL seed on purpose: a hub of degree d offers C(d,n) matches for an n-edge
# star, so a seed sized like the chains' would spend the whole budget at depth 2.
#
# The DEPTH phase takes every shape -- it is cheap per shape and the counts are the point. The
# SCALING phase takes a subset, because it is serial and seven thread points per shape is the
# long pole: one representative of each axis (size 1-4, hub, tree, ring, disconnected, arity).
SHAPES=${SHAPES:-"chain1a2:8 chain2a2:8 chain3a2:8 chain4a2:8 \
star2a2:4 star3a2:4 star4a2:4 tree3a2:8 tree4a2:8 cycle3a2:3 cycle4a2:4 \
disc2a2:4 disc3a2:3 chain2a3:8 chain3a3:8 chain2a4:8 mixed2:8 mixed3:8 mixed4:8 \
growth:1 pair:4 triple:5 quad:6 disc:4"}
SCALE_SHAPES=${SCALE_SHAPES:-"chain1a2:8 chain2a2:8 chain3a2:8 chain4a2:8 \
star3a2:4 tree4a2:8 cycle4a2:4 disc2a2:4 chain3a3:8 mixed3:8"}

# THE MEMORY CAP LIVES IN THE BINARY; THIS PASSES IT THE NUMBER. sampling_cost_smoke calls
# setrlimit(RLIMIT_AS) on itself before doing anything, so a run is bounded whoever starts it --
# which matters, because the batch that once drove a 19 GB shared workstation into swap invoked
# that binary directly and would have bypassed a cap enforced only here.
#
# What this script contributes is the one thing the binary cannot know: its own CONCURRENCY. A
# per-process share of free memory is not a bound when the depth phase runs CONC of them at once,
# so the figure passed down is divided by CONC. Setting the value rather than applying a second
# `ulimit` keeps one mechanism: two would drift, and the lower would silently win.
_avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 4096)
MEM_CAP_MB=${MEM_CAP_MB:-$(( _avail_mb * 6 / 10 / CONC ))}
[ "$MEM_CAP_MB" -lt 512 ] && MEM_CAP_MB=512
export HG_MEM_CAP_MB="$MEM_CAP_MB"
capped() {                      # capped <timeout-seconds> <command...>
    local t="$1"; shift
    timeout "$t" "$@"
}

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# A TIMING RUN STARTS ONLY ON A QUIET MACHINE, CHECKED, NOT ASSUMED. This is not hypothetical:
# a verification campaign left one process alive into the first scaling shape of this sweep's own
# first run, and those timings had to be thrown away. The check runs before EVERY timed point.
#
# The check itself lives in quiet_gate.sh and is shared with the Wolfram comparison in
# paper_tables.py, which needs the same guarantee on a desktop that is never idle by default.
# One implementation: a second copy would drift, and this one already carried a defect worth not
# repeating twice (process names truncate at 15 characters, so a longer pgrep -x pattern matches
# nothing and the gate becomes a rubber stamp).
. "$(dirname "${BASH_SOURCE[0]}")/quiet_gate.sh"

wait_quiet() {                 # wait_quiet <what-for>
    local what="$1"
    if quiet_wait 600; then
        LAST_QUIET_LOAD="$QUIET_LOAD1"
        return 0
    fi
    say "  NOT QUIET -- $what measured anyway, marked"
    LAST_QUIET_LOAD="$QUIET_LOAD1!contended"
    return 1
}

# ---------------------------------------------------------------- depth phase
depth_one() {          # depth_one <rule> <init_edges> <max_depth_file>
    local rule="$1" init="$2" marker="$3"
    local d=1 secs
    while [ "$d" -le 14 ]; do
        local log="$OUT/depth_${rule}_d${d}.log"
        local t0=$(date +%s)
        capped $(( DEPTH_BUDGET_S * 3 )) "$BIN" off "$rule" "$init" "$d" "$THREADS_PER" 4 full \
            > "$log" 2>&1
        local rc=$?
        secs=$(( $(date +%s) - t0 ))
        if [ $rc -ne 0 ]; then
            say "$rule depth $d: exit $rc after ${secs}s -- stopping this shape"
            echo "$rule stopped_at_depth $d reason exit$rc" >> "$marker"
            return
        fi
        grep -h "^RICH" "$log" >> "$OUT/rich_depth.txt"
        say "$rule depth $d: ${secs}s  $(grep -oE 'states=[0-9]+ .*branchial_edges=[0-9]+' "$log" | head -1)"
        if [ "$secs" -ge "$DEPTH_BUDGET_S" ]; then
            echo "$rule stopped_at_depth $d reason budget${secs}s" >> "$marker"
            return
        fi
        d=$(( d + 1 ))
    done
    echo "$rule stopped_at_depth 14 reason depth_cap" >> "$marker"
}

if [ "$WHICH" = depth ] || [ "$WHICH" = all ]; then
    say "DEPTH phase: $CONC concurrent, $THREADS_PER threads each, budget ${DEPTH_BUDGET_S}s/run"
    : > "$OUT/depth_markers.txt"
    for s in $SHAPES; do
        depth_one "${s%%:*}" "${s##*:}" "$OUT/depth_markers.txt" &
        while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
    done
    wait
    say "DEPTH phase done: $(grep -c . "$OUT/rich_depth.txt" 2>/dev/null || echo 0) rows"
fi

# -------------------------------------------------------------- scaling phase
# The depth used for each shape is the DEEPEST one whose serial cost is inside SCALE_BUDGET_S.
# Read from the depth phase's own rows, so the two phases cannot disagree about what was run.
if [ "$WHICH" = scaling ] || [ "$WHICH" = all ]; then
    say "SCALING phase: one job at a time, quiet-gated"
    for s in $SCALE_SHAPES; do
        rule="${s%%:*}"; init="${s##*:}"
        best_d=""
        for d in $(seq 14 -1 1); do
            row=$(grep -h "rule=$rule .*steps=$d " "$OUT/rich_depth.txt" 2>/dev/null | head -1)
            [ -z "$row" ] && continue
            ms=$(echo "$row" | grep -oE 'ms=[0-9.]+' | cut -d= -f2)
            # The depth-phase run used THREADS_PER threads; a serial run is slower, so allow
            # headroom rather than pretending that time is the serial time.
            if awk "BEGIN{exit !($ms/1000.0 * $THREADS_PER < $SCALE_BUDGET_S)}"; then
                best_d="$d"; break
            fi
        done
        [ -z "$best_d" ] && { say "$rule: no depth fits the scaling budget, skipped"; continue; }
        say "$rule: scaling at depth $best_d"
        for th in 1 2 4 8 16 24 32; do
            [ "$th" -gt "$NPROC" ] && continue
            for rep in 1 2 3; do
                wait_quiet "$rule t=$th rep=$rep"
                echo "QUIET rule=$rule threads=$th rep=$rep load=$LAST_QUIET_LOAD" \
                    >> "$OUT/quiet_log.txt"
                capped "$SCALE_BUDGET_S" "$BIN" off "$rule" "$init" "$best_d" "$th" 4 full \
                    > "$OUT/scale_${rule}_d${best_d}_t${th}_r${rep}.log" 2>&1 || break
                grep -h "^RICH" "$OUT/scale_${rule}_d${best_d}_t${th}_r${rep}.log" \
                    >> "$OUT/rich_scaling.txt"
                # One repeat is enough once a point runs long: scheduling noise is milliseconds.
                ms=$(grep -oE 'ms=[0-9.]+' "$OUT/scale_${rule}_d${best_d}_t${th}_r${rep}.log" \
                     | cut -d= -f2 | head -1)
                awk "BEGIN{exit !($ms > 10000)}" && break
            done
            say "  $rule t=$th done"
        done
    done
    say "SCALING phase done: $(grep -c . "$OUT/rich_scaling.txt" 2>/dev/null || echo 0) rows"
fi

say "ALL DONE"
