#!/usr/bin/env bash
# Run GPUMC harnesses against the device sources.
#
#   verification/gpumc/run.sh <harness> [-D... ] [-- gpumc args]   one run, GPUMC's own output
#   verification/gpumc/run.sh cells [harness[:cell]...]             cells checked (default: all)
#   verification/gpumc/run.sh list                                   every cell, not run
#   verification/gpumc/run.sh stop                                   kill every run.sh container
#
# GPUMC is a stateless model checker for scoped-RC11, the GPU memory model: threads are grouped
# into CTAs and an access carries a scope. It is a fork of GenMC 0.9 on LLVM 14 and runs from a
# container: the CAV 2025 artifact (figshare 28789703, GPL-3.0+), loaded once with `docker load`,
# with checker/gpumc-fixes.patch applied by checker/build.sh as gpumc_docker:hg-fixes.
# HG_GPUMC_IMAGE overrides the image.
#
# DEVICE SOURCES. A harness that runs engine device code includes device/hg_gpumc.hpp first and
# then the engine header or .cu itself; device/ is on the include path ahead of the toolkit, so
# <cuda_runtime.h>, <cuda/atomic> and <cooperative_groups.h> are the model's. The container
# compiles a copy of gpu/ with every <<<grid, block>>> launch configuration deleted (a launch
# becomes a call), and nothing else changed. A harness thread calls a __global__ or __device__
# function directly, as one CUDA thread of the block it entered. device/genmc_libc.hpp is included
# first: it completes GPUMC's C library headers for libstdc++, whose header directories (copied
# into the image) come ahead of GPUMC's, and compiles the host code of an included .cu without
# exceptions. FP contraction is off: the interpreter has no fused multiply-add. A harness that
# includes gpu/src/persistent.cu passes --disable-function-inliner: GPUMC's inliner did not
# finish on k_persistent_evolve in 10 minutes.
#
# CELLS. Each harness lists its cells in comment lines of the form
#
#     // GPUMC-CELL: <name> <PASS|FAIL|BLOCKED> [-DNAME ...] [-- gpumc args]
#
# A PASS cell must finish with "No errors were detected", no race and at least one complete
# execution. GPUMC reports a data race (two conflicting accesses, one of them non-atomic, unordered
# by happens-before) at the end of the run and still exits 0, so run.sh reads "Races Detected" as
# a failure. A FAIL cell is a calibration: the checker must report an error or a race. A BLOCKED
# cell is a calibration whose broken shape lets no execution complete and reports no error.
# `cells` prints one line per cell with the verdict, the executions explored and the wall time,
# and exits 1 when any cell's verdict differs from its expectation. HG_GPUMC_TIMEOUT (seconds,
# default 0 = none) bounds each run; a cell that times out is reported TIMEOUT and fails. Lines
# tagged GPUMC-CELL-LONG are cells that run for hours; HG_GPUMC_LONG=1 includes them.
#
# The tree is mounted read-only; scratch is a tmpfs the container owns.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMAGE="${HG_GPUMC_IMAGE:-gpumc_docker:hg-fixes}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    cat >&2 <<EOF
run.sh: no GPUMC image '$IMAGE'.
Obtain the CAV 2025 artifact (figshare 28789703), then:
    unzip -j <artifact>.zip 'cav_25_artifact_314/gpumc_docker.tar' && docker load -i gpumc_docker.tar
    verification/gpumc/checker/build.sh
EOF
    exit 2
fi

CFLAGS="-std=c++17 -fno-exceptions -fdeclspec -ffp-contract=off -Wno-dangling-else"
CFLAGS+=" -I /opt/hg-stdcxx -I /opt/hg-stdcxx-target"
CFLAGS+=" -include /src/verification/gpumc/device/genmc_libc.hpp -I /src/verification/gpumc/device"
CFLAGS+=" -I /tmp/gpu/include -I /tmp/gpu/src -I /src/common/include"

# run_raw <harness path relative to ROOT> <defines> <gpumc args>: GPUMC's output on stdout.
# Containers are named hg-gpumc-<pid>-<n>. Killing run.sh (or the docker client under it) leaves
# GPUMC running in its container; HG_GPUMC_TIMEOUT runs inside the container, and `run.sh stop`
# kills every container run.sh started.
run_raw() {
    local src="$1" defs="$2" gargs="$3"
    docker run --rm --name "hg-gpumc-$$-$RANDOM" -v "$ROOT:/src:ro" --entrypoint /bin/sh "$IMAGE" -c "
        cp -r /src/gpu /tmp/gpu &&
        find /tmp/gpu \( -name '*.cu' -o -name '*.hpp' \) -exec sed -i -E 's/<<<[^;]*>>>//g' {} + &&
        timeout ${HG_GPUMC_TIMEOUT:-0} genmc $gargs -- $CFLAGS $defs /src/$src
    "
}

harness_path() {
    local name="$1"
    if [ -f "$HERE/$name.cpp" ]; then echo "verification/gpumc/$name.cpp"; return 0; fi
    if [ -f "$HERE/checker/$name.cpp" ]; then echo "verification/gpumc/checker/$name.cpp"; return 0; fi
    echo "run.sh: no such harness '$name'" >&2
    return 2
}

# cells_of <harness path>: "name|expect|defines|gpumc args" per cell.
cells_of() {
    local tag='GPUMC-CELL'
    [ -n "${HG_GPUMC_LONG:-}" ] && tag='GPUMC-CELL\(-LONG\)\{0,1\}'
    sed -n "s|^// $tag: *||p" "$ROOT/$1" | while read -r name expect rest; do
        local defs="${rest%%--*}" gargs=""
        [[ "$rest" == *--* ]] && gargs="${rest#*--}"
        echo "$name|$expect|$(echo $defs)|$(echo $gargs)"
    done
}

run_cells() {
    local fail=0 names=("$@")
    if [ ${#names[@]} -eq 0 ]; then
        for f in "$HERE"/*.cpp "$HERE"/checker/*.cpp; do names+=("$(basename "$f" .cpp)"); done
    fi
    printf '%-40s %-28s %-6s %-8s %14s %9s\n' harness cell expect verdict executions seconds
    for spec in "${names[@]}"; do
        # <harness> or <harness>:<cell>
        local n="${spec%%:*}" only=""
        [[ "$spec" == *:* ]] && only="${spec#*:}"
        local path; path="$(harness_path "$n")" || { fail=1; continue; }
        local cells; cells="$(cells_of "$path")"
        if [ -z "$cells" ]; then
            printf '%-40s %-28s %s\n' "$n" "-" "no GPUMC-CELL lines"
            fail=1
            continue
        fi
        while IFS='|' read -r cname expect defs gargs; do
            [ -n "$only" ] && [ "$cname" != "$only" ] && continue
            local t0 out rc verdict execs
            t0=$(date +%s.%N)
            out="$(run_raw "$path" "$defs" "$gargs" 2>&1)"
            rc=$?
            execs="$(echo "$out" | sed -n 's/^Number of complete executions explored: //p' | tail -1)"
            if [ $rc -eq 124 ]; then
                verdict=TIMEOUT
            elif grep -q "^Races Detected" <<< "$out"; then
                verdict=FAIL
            elif [ $rc -eq 0 ] && grep -q "No errors were detected" <<< "$out" && [ "${execs:-0}" -gt 0 ]; then
                verdict=PASS
            elif [ $rc -eq 0 ] && grep -q "No errors were detected" <<< "$out"; then
                verdict=BLOCKED
            elif grep -qE "Error detected|Assertion violation|Race detected|race" <<< "$out"; then
                verdict=FAIL
            else
                verdict="ERROR($rc)"
            fi
            printf '%-40s %-28s %-6s %-8s %14s %9.1f\n' "$n" "$cname" "$expect" "$verdict" \
                "${execs:--}" "$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{print b-a}')"
            if [ "$verdict" != "$expect" ]; then
                fail=1
                echo "$out" | grep -E "Error|Assertion|race|Race|rror:" | head -8 | sed 's/^/    /'
            fi
        done <<< "$cells"
    done
    return $fail
}

case "${1:-}" in
    "")
        sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
        exit 2 ;;
    stop)
        ids="$(docker ps -q --filter name=hg-gpumc-)"
        [ -n "$ids" ] && docker kill $ids
        exit 0 ;;
    list)
        for f in "$HERE"/*.cpp "$HERE"/checker/*.cpp; do
            p="${f#$ROOT/}"
            cells_of "$p" | sed "s|^|$(basename "$f" .cpp) |"
        done ;;
    cells|all)
        shift
        run_cells "$@"
        exit $? ;;
    *)
        path="$(harness_path "$1")" || exit 2
        shift
        defs=() gargs=()
        while [ $# -gt 0 ] && [ "$1" != "--" ]; do defs+=("$1"); shift; done
        [ "${1:-}" = "--" ] && shift
        gargs=("$@")
        run_raw "$path" "${defs[*]:-}" "${gargs[*]:-}"
        exit $? ;;
esac
