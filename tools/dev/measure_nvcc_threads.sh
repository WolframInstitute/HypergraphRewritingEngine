#!/usr/bin/env bash
# Peak compiler memory and wall clock for one device translation unit, under a hard cap.
#
# WHY THIS EXISTS. gpu/src/persistent.cu is what makes the device build unschedulable
# on a shared box. Measured 2026-08-05 with tools/safe_build.sh: it drove MemAvailable
# from 15.5 GB to 3 GB before the guard killed it, so roughly 12.5 GB for a single -j1
# compile, and it takes upwards of ten minutes. The other five device TUs build without
# trouble, and it is not the largest of them -- preprocessed it is 122,434 lines against
# evolve.cu's 137,699, which compiles fine. So the cost is in what the compiler must do
# with this TU, not in how much source it reads.
#
# WHAT CONSUMES IT: cicc, the CUDA frontend, which is LLVM-based. Both runs below died
# with `LLVM ERROR: out of memory` while cicc held the memory; ptxas is not the culprit.
# An earlier reading blamed ptxas because "ptxas died due to signal 15" appeared in a
# killed build -- that was the process holding the signal when the group was killed, not
# the process holding the memory.
#
# REFUTED HERE, so it is not retried: --threads. The build passes
#     --generate-code=arch=compute_89,code=[compute_89,sm_89]     (two code objects)
#     --threads=0                                                 (use all CPUs)
# and the guess was that nvcc overlaps the two code-generation passes, each with its own
# working set, so -j1 bounds make but not nvcc's own fan-out. Measured, capped at 10 GB
# of address space:
#     --threads=1   peak RSS 8,490,308 KB   9:11 elapsed   LLVM ERROR: out of memory
#     --threads=0   peak RSS 8,773,304 KB   9:44 elapsed   LLVM ERROR: out of memory
# 283 MB apart, about 3%. --threads is not the multiplier. It parallelises per-ARCHITECTURE
# code generation, and cicc runs once per VIRTUAL architecture, of which there is one here.
# Do not spend the window on that flag again.
#
# WHAT REMAINS TO TEST: the optimisation level for this TU alone (-O2/-O1 reduces what
# LLVM must do, at a perf cost that needs its own measurement), and moving the replay's
# recursive device-function cycle out of the kernel's TU.
#
# SAFETY. `ulimit -v` is per process and is inherited by the cicc/ptxas children, so a run
# that wants more dies as a clean allocation failure instead of paging the machine into
# swap. There is no memory watchdog on this box; the cap is the only guard. Note the cap
# bounds ADDRESS SPACE while `time -v` reports RESIDENT set, so a run can die at the cap
# with a reported peak below it -- both figures above did.
#
# Usage: tools/dev/measure_nvcc_threads.sh [virtual-cap-KB] [extra nvcc flags ...]
#   tools/dev/measure_nvcc_threads.sh 14680064 -O2
#   tools/dev/measure_nvcc_threads.sh 10485760 --threads=1

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build_gpu"
CAP=${1:-10485760}          # address space per process, KB
shift || true
EXTRA=("$@")
SRC="$ROOT/gpu/src/persistent.cu"
NVCC=/usr/local/cuda-13.0/bin/nvcc
RSP="$BUILD/gpu/CMakeFiles/hg_gpu.dir/includes_CUDA.rsp"

if [ ! -f "$RSP" ]; then
    echo "no include response file at $RSP -- configure build_gpu first" >&2
    exit 1
fi

cd "$BUILD/gpu" || exit 1
out=$(mktemp /tmp/persistent_measure_XXXX.o)
trap 'rm -f "$out"' EXIT

echo "=== $(basename "$SRC")  cap $((CAP / 1024)) MB address space  extra: ${EXTRA[*]:-none} ==="
nice -n 15 bash -c "ulimit -v $CAP; exec /usr/bin/time -v $NVCC \
    -forward-unknown-to-host-compiler \
    --options-file '$RSP' \
    -O3 -DNDEBUG -std=c++17 \
    '--generate-code=arch=compute_89,code=[compute_89,sm_89]' \
    --expt-relaxed-constexpr --Werror=cross-execution-space-call \
    ${EXTRA[*]} \
    -x cu -rdc=true -c '$SRC' -o '$out'" 2>&1 \
    | grep -E 'Maximum resident set size|Elapsed \(wall clock\)|Exit status|out of memory|error' \
    | sed 's/^/    /'
