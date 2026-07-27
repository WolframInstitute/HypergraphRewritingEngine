#!/usr/bin/env bash
# A/B timing harness for engine-level changes.
#
# Rebuilds libhypergraph, links the standalone tools that exercise the hot paths,
# and reports the MINIMUM wall time over N repetitions (min rejects host-load noise;
# the mean does not). Every reported number carries its (tool, config, commit, date).
#
# Usage: tools/perf_ab.sh <label> [reps]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
LABEL=${1:-run}
REPS=${2:-5}
OUT=${PERF_AB_OUT:-/tmp/claude-1000/-home-fly-my-projects-efficient-rewriting-final/a60043bf-6f3b-4908-8552-e3b0f6cc7105/scratchpad}
mkdir -p "$OUT"

cmake --build build --target hypergraph -j32 >/dev/null

CXXFLAGS="-O2 -std=c++17 -I$ROOT/hypergraph/include -I$ROOT/common/include -I$ROOT/job_system/include -I$ROOT/lockfree_deque/include"
for t in profile_evolve; do
  g++ $CXXFLAGS "tools/$t.cpp" -o "$OUT/$t" "$ROOT/build/libhypergraph.a" -pthread
done
g++ -O2 -std=c++17 -I"$ROOT/hypergraph/include" -I"$ROOT/common/include" \
    tools/ir_malloc_bench.cpp hypergraph/src/ir_canonicalization.cpp \
    -o "$OUT/ir_malloc_bench" -pthread

COMMIT=$(git rev-parse --short HEAD)
echo "== $LABEL  (commit $COMMIT, $(date -u +%Y-%m-%dT%H:%MZ), min of $REPS) =="

bench() {  # name, then command
  local name=$1; shift
  local best=999999
  for _ in $(seq "$REPS"); do
    local t0 t1 ms
    t0=$(date +%s%N); "$@" >/dev/null 2>&1; t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    (( ms < best )) && best=$ms
  done
  printf '  %-34s %6d ms\n' "$name" "$best"
}

bench "profile_evolve 5 full (1 thread)" "$OUT/profile_evolve" 5 full
bench "profile_evolve 6 full (1 thread)" "$OUT/profile_evolve" 6 full
bench "profile_evolve 6 none (1 thread)" "$OUT/profile_evolve" 6 none
bench "profile_evolve 6 auto (1 thread)" "$OUT/profile_evolve" 6 auto
bench "profile_evolve 6 full (8 threads)" "$OUT/profile_evolve" 6 full 8
bench "profile_evolve 7 full (8 threads)" "$OUT/profile_evolve" 7 full 8
"$OUT/ir_malloc_bench"
