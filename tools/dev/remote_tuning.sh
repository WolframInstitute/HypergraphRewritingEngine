#!/usr/bin/env bash
# PHASE 2 of the rented-box session: where many-core efficiency goes, and whether the cause
# the paper names is the cause. Run after remote_session.sh, which leaves the clone and both
# builds in place; this adds no build of its own unless asked for the arena arm.
#
#   bash tools/dev/remote_tuning.sh [workload] [depth]     # defaults: wpp 7
#
# THE CLAIM UNDER TEST (paper, sec:evaluation): efficiency falls beyond sixteen workers
# because of MEMORY ACQUISITION rather than lock contention -- system time grows with the
# per-worker resident set and minor-fault count while user time grows far less. That was
# measured to 24 threads on a heterogeneous laptop-class part. A 64-core homogeneous box
# either reproduces it with a steeper slope or refutes it, and the difference decides
# whether the lever is allocation (arena block size, huge pages, NUMA placement) or the
# scheduler.
#
# THE ARMS, cheapest first. Each is an environment change over the SAME binary, so a
# difference between arms cannot be a code difference:
#   base    as shipped
#   thp     transparent huge pages forced on; the arena's 1 MB blocks are the unit that
#           faults, so if fault COUNT is the cost, 2 MB pages cut it by construction
#   numa    numactl --interleave=all, if the box has more than one NUMA node; separates
#           "faults cost" from "remote memory costs"
# A fourth arm (arena block size) needs a rebuild and is run by hand once these three say
# whether allocation is implicated at all.
set -uo pipefail

WORKLOAD="${1:-wpp}"
DEPTH="${2:-7}"
ITERS="${3:-3}"
ROOT="$HOME/hg_session"
SRC="$ROOT/src"
OUT="$ROOT/tuning_$(date -u +%Y%m%dT%H%M%SZ).tsv"
cd "$SRC" || { echo "no clone at $SRC — run remote_session.sh first"; exit 1; }

if [ "$(id -u)" = 0 ]; then SUDO=""; else SUDO="sudo"; fi

mapfile -t FIRST_THREADS < <(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}')
NPHYS=${#FIRST_THREADS[@]}
NNODES=$(lscpu | awk '/^NUMA node\(s\)/ {print $3}')
say() { echo "==> $*"; }
say "$NPHYS physical cores, $NNODES NUMA node(s), workload=$WORKLOAD depth=$DEPTH"

# Thread counts: powers of two to the core count, then the core count itself. The pin set
# for N threads is the FIRST N physical cores, so a small count never straddles more of the
# machine than it needs and the curve is not measuring placement drift.
COUNTS=(); n=1
while [ "$n" -lt "$NPHYS" ]; do COUNTS+=("$n"); n=$((n*2)); done
COUNTS+=("$NPHYS")

printf 'arm\tthreads\twall_s\tuser_s\tsys_s\tminor_faults\tmax_rss_kb\tmedian_ms\n' > "$OUT"

run_one() {           # arm, threads, prefix-command...
  local arm="$1" th="$2"; shift 2
  local set; set=$(IFS=,; echo "${FIRST_THREADS[*]:0:$th}")
  local tf="$ROOT/.t.$$" bf="$ROOT/.b.$$"
  /usr/bin/time -f "%e\t%U\t%S\t%R\t%M" -o "$tf" \
    "$@" ./build_linux/bench_cpu_evolve "$DEPTH" "$ITERS" "$th" "$WORKLOAD" "$set" > "$bf" 2>&1
  local med; med=$(grep -o 'median_ms=[0-9.]*' "$bf" | tail -1 | cut -d= -f2)
  printf '%s\t%s\t%s\t%s\n' "$arm" "$th" "$(cat "$tf")" "${med:-NA}" >> "$OUT"
  printf '  %-5s %3s threads  %s  median_ms=%s\n' "$arm" "$th" "$(cat "$tf" | tr '\t' ' ')" "${med:-NA}"
  rm -f "$tf" "$bf"
}

say "arm: base"
$SUDO sh -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled' 2>/dev/null || true
for th in "${COUNTS[@]}"; do run_one base "$th"; done

say "arm: thp (transparent huge pages always)"
if $SUDO sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled' 2>/dev/null; then
  for th in "${COUNTS[@]}"; do run_one thp "$th"; done
  $SUDO sh -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled' 2>/dev/null || true
else
  say "  skipped: cannot write transparent_hugepage/enabled on this box"
fi

if [ "${NNODES:-1}" -gt 1 ] && command -v numactl >/dev/null; then
  say "arm: numa (interleave=all, $NNODES nodes)"
  for th in "${COUNTS[@]}"; do run_one numa "$th" numactl --interleave=all; done
else
  say "arm: numa skipped ($NNODES node(s), numactl $(command -v numactl >/dev/null && echo present || echo absent))"
fi

say "wrote $OUT"
echo
echo "READ IT LIKE THIS: within an arm, if sys_s and minor_faults climb with threads while"
echo "user_s stays near flat, the paper's cause holds and allocation is the lever. If thp"
echo "flattens sys_s, the cost is fault COUNT and the arena block size is the next knob. If"
echo "numa flattens it instead, the cost is remote memory and placement is. If neither moves"
echo "and wall time still stops improving, the cause is not memory and the paper's sentence"
echo "needs rewriting before v1.0.0 ships it."
