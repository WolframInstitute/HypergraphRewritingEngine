#!/usr/bin/env bash
# Performance-regression gate: run build_linux/benchmark_suite for the current tree and fail when
# any metric worsens against a baseline run by more than a tolerance.
#
# THE MACHINE MUST BE QUIET. The run is refused unless tools/dev/quiet_gate.sh reports quiet: a
# wall time taken beside a compiler or another measurement is a number about that other process.
# benchmark_suite is added to the watch list so two gates cannot overlap.
#
# NO CPU LIST, NO AFFINITY. The suite is launched with the engine's default placement:
# ensure_default_cpu_order (job_system/include/job_system/job_system.hpp) seats workers on the
# performance cores, cache-domain-major, and a caller that names CPUs overrides it. This gate
# names none, so the measurement is of the placement users get.
#
# WHERE THE RESULTS GO. Each invocation writes a fresh directory
# benchmark_results/gate/<utc-stamp>-<short-hash>/, and the suite creates commit-<hash>/ (or
# tree-<hash>/ when changes are staged) under it, so a gate run never merges into the
# per-benchmark files of an earlier run of the same commit and never becomes the default baseline
# by accident. The baseline is the newest benchmark_results/commit-*/summary.csv unless
# PERF_BASELINE or --baseline names a directory.
#
# Usage:
#   tools/dev/perf_gate.sh [--baseline DIR] [--filter PATTERN] [--tolerance F] [--stat S] [--plots]
#
#   --baseline DIR   directory holding the baseline summary.csv (env PERF_BASELINE; default: the
#                    newest benchmark_results/commit-*)
#   --filter PAT     benchmark_suite --filter= pattern; metrics the filter leaves out are reported
#                    as `missing` by perf_compare.py and do not fail the gate
#   --tolerance F    fractional worsening that fails a metric (default 0.10)
#   --stat S         gated statistic: avg_us (default), min_us, max_us
#   --plots          regenerate plots for the new run with benchmarking/plot_benchmarks.py
#
# Exit status: 0 no regression; 1 regression; 2 usage; 3 machine not quiet; 4 missing
# prerequisite (binary or baseline); 5 benchmark_suite failed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SUITE=build_linux/benchmark_suite
COMPARE=tools/dev/perf_compare.py
QUIET=tools/dev/quiet_gate.sh
PLOT=benchmarking/plot_benchmarks.py
OUT_ROOT=benchmark_results/gate

BASELINE=${PERF_BASELINE:-}
FILTER=""
TOLERANCE=0.10
STAT=avg_us
PLOTS=0

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//' >&2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --baseline) BASELINE=${2:?--baseline needs a directory}; shift 2 ;;
        --baseline=*) BASELINE=${1#*=}; shift ;;
        --filter) FILTER=${2:?--filter needs a pattern}; shift 2 ;;
        --filter=*) FILTER=${1#*=}; shift ;;
        --tolerance) TOLERANCE=${2:?--tolerance needs a number}; shift 2 ;;
        --tolerance=*) TOLERANCE=${1#*=}; shift ;;
        --stat) STAT=${2:?--stat needs a name}; shift 2 ;;
        --stat=*) STAT=${1#*=}; shift ;;
        --plots) PLOTS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "perf_gate: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# Quiet precondition. quiet_gate.sh prints one status line and exits 1 when the machine is busy.
status=$(QUIET_EXTRA="${QUIET_EXTRA:-} benchmark_suite" "$QUIET" check) || {
    echo "perf_gate: refusing to run on a busy machine: $status" >&2
    exit 3
}
echo "perf_gate: $status"

if [ ! -x "$SUITE" ]; then
    echo "perf_gate: $SUITE is missing; build it with: cmake --build build_linux --target benchmark_suite -j4" >&2
    exit 4
fi

# Baseline: the newest commit-*/summary.csv by modification time, unless one was named.
if [ -z "$BASELINE" ]; then
    BASELINE=$(find benchmark_results -mindepth 2 -maxdepth 2 -path 'benchmark_results/commit-*/summary.csv' \
                    -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)
    if [ -z "$BASELINE" ]; then
        echo "perf_gate: no benchmark_results/commit-*/summary.csv to use as a baseline; run $SUITE once, or pass --baseline" >&2
        exit 4
    fi
fi
if [ ! -f "$BASELINE/summary.csv" ]; then
    echo "perf_gate: baseline $BASELINE has no summary.csv" >&2
    exit 4
fi
echo "perf_gate: baseline $BASELINE"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "perf_gate: the working tree has uncommitted changes; the suite labels the run by HEAD (or by the staged tree) and measures the binary as built" >&2
fi

RUN_ROOT="$OUT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
if [ -e "$RUN_ROOT" ]; then
    echo "perf_gate: $RUN_ROOT already exists" >&2
    exit 4
fi
mkdir -p "$RUN_ROOT"
echo "perf_gate: results under $RUN_ROOT"

# The suite resolves its git commands against the working directory, which is $ROOT here. Output
# goes to the terminal and to a log beside the results; pipefail carries the suite's exit status.
suite_args=(--output="$ROOT/$RUN_ROOT")
[ -n "$FILTER" ] && suite_args+=(--filter="$FILTER")
if ! "$SUITE" "${suite_args[@]}" 2>&1 | tee "$RUN_ROOT/benchmark_suite.log"; then
    echo "perf_gate: $SUITE failed; see $RUN_ROOT/benchmark_suite.log" >&2
    exit 5
fi

HEAD_DIR=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d \( -name 'commit-*' -o -name 'tree-*' \))
if [ "$(printf '%s\n' "$HEAD_DIR" | grep -c .)" -ne 1 ] || [ ! -f "$HEAD_DIR/summary.csv" ]; then
    echo "perf_gate: expected exactly one commit-*/ or tree-*/ directory with a summary.csv under $RUN_ROOT" >&2
    exit 5
fi
echo "perf_gate: head results $HEAD_DIR"

if [ "$PLOTS" = 1 ]; then
    python3 "$PLOT" "$HEAD_DIR"
fi

compare_status=0
python3 "$COMPARE" "$BASELINE/summary.csv" "$HEAD_DIR/summary.csv" --tolerance "$TOLERANCE" --stat "$STAT" \
    | tee "$RUN_ROOT/perf_compare.txt" || compare_status=${PIPESTATUS[0]}
echo "perf_gate: report $RUN_ROOT/perf_compare.txt"
exit "$compare_status"
