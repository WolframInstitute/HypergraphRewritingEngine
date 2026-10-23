#!/usr/bin/env bash
# The release gate: the checks that take hours and run once per release candidate, not per edit.
#
#   tools/dev/release_gate.sh sanitizers [tool...]   compute-sanitizer over the three GPU suites
#   tools/dev/release_gate.sh gpumc                  every GPUMC cell, the long cells included
#   tools/dev/release_gate.sh genmc                  every GenMC harness and calibration
#   tools/dev/release_gate.sh tla                    every TLC configuration
#   tools/dev/release_gate.sh all                    the four, in that order
#
# Exit status is 0 when every leg passed and 1 otherwise; each leg prints PASS or FAIL with its
# log under $HG_RELEASE_LOGS (default release_gate_logs/).
#
# SANITIZERS. The tools are memcheck (out-of-bounds and misaligned device accesses), initcheck
# (reads of uninitialised device memory), synccheck (illegal __syncthreads and __syncwarp use) and
# racecheck (shared-memory hazards only: the engine's global-memory protocols are GPUMC's, see
# verification/gpumc/COVERAGE.md). Each runs over hg_gpu_tests, gpu_differential_tests and
# gpu_ffi_tests (or the suites $HG_GPU_SUITES names) from $HG_GPU_BUILD (default build_gpu) with
# --error-exitcode, so a report fails the leg. compute-sanitizer needs /usr/lib/wsl/lib on PATH
# under WSL ("Unable to find injection library" otherwise); this script adds it.
#
# Address space: ulimit -v 96000000. initcheck reserves host address space for the device memory
# it tracks, and under 48000000 hg::gpu::evolve's host allocations throw std::bad_alloc.
#
# Tests the sanitizers cannot run, left out of every sanitizer leg:
#   CapacityOverflow.AnEngineThatDoesNotFitStopsAndReports, CapacityOverflow.TheEstimateCovers
#   TheAllocation: both size device allocations from cudaMemGetInfo, which counts the
#   sanitizer's own device memory (under racecheck the allocation sized from the query fails;
#   under initcheck the measured engine exceeds its estimate by 38 MB).
#   GpuBinaryGate.NoUsableDeviceIsAStructuredRefusal: its worker hides the device with
#   CUDA_VISIBLE_DEVICES; started under the sanitizer, the worker answers the job it must refuse.
#   gpu_ffi_tests' kernels run in worker processes, so the sanitizer follows child processes.
# API errors are not reported: CapacityOverflow.AFailedEngineConstructionReturnsItsDeviceMemory
# provokes cudaMalloc failures and checks them.
#
# GPUMC needs docker and the gpumc_docker:hg-fixes image (verification/gpumc/checker/build.sh);
# GenMC needs the checker verification/genmc/run.sh names; TLC needs a JVM and tla2tools.jar.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
GPU_BUILD="${HG_GPU_BUILD:-build_gpu}"
LOGS="${HG_RELEASE_LOGS:-release_gate_logs}"
CS="${COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}"
read -r -a SUITES <<< "${HG_GPU_SUITES:-hg_gpu_tests gpu_differential_tests gpu_ffi_tests}"
UNSANITIZABLE="CapacityOverflow.AnEngineThatDoesNotFitStopsAndReports"
UNSANITIZABLE+=":CapacityOverflow.TheEstimateCoversTheAllocation"
UNSANITIZABLE+=":GpuBinaryGate.NoUsableDeviceIsAStructuredRefusal"
export PATH="$PATH:/usr/lib/wsl/lib"
mkdir -p "$LOGS"
fail=0

leg() {   # leg <name> <command...>: run, log, print the verdict
    local name="$1"; shift
    local log="$LOGS/$name.log"
    if "$@" > "$log" 2>&1; then
        printf 'PASS  %s\n' "$name"
    else
        printf 'FAIL  %s  (exit %s, %s)\n' "$name" "$?" "$log"
        fail=1
    fi
}

sanitizers() {
    local tools=("$@")
    [ ${#tools[@]} -eq 0 ] && tools=(memcheck initcheck synccheck racecheck)
    [ -x "$CS" ] || { echo "FAIL  sanitizers: $CS not found"; fail=1; return; }
    for t in "${tools[@]}"; do
        for s in "${SUITES[@]}"; do
            [ -x "$GPU_BUILD/$s" ] || { echo "FAIL  $t-$s: $GPU_BUILD/$s not built"; fail=1; continue; }
            leg "$t-$s" bash -c "ulimit -v 96000000; exec \"$CS\" --tool $t \
                --report-api-errors no --target-processes all --error-exitcode 9 \
                \"$GPU_BUILD/$s\" --gtest_filter=-$UNSANITIZABLE"
        done
    done
}

case "${1:-}" in
    sanitizers) shift; sanitizers "$@" ;;
    gpumc) leg gpumc env HG_GPUMC_LONG=1 verification/gpumc/run.sh cells ;;
    genmc) leg genmc verification/genmc/run.sh all ;;
    tla)   leg tla verification/tla/run.sh ;;
    all)
        sanitizers
        leg gpumc env HG_GPUMC_LONG=1 verification/gpumc/run.sh cells
        leg genmc verification/genmc/run.sh all
        leg tla verification/tla/run.sh
        ;;
    *) sed -n '2,9p' "$0"; exit 2 ;;
esac
exit $fail
