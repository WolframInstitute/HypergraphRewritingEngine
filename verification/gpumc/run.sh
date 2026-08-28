#!/usr/bin/env bash
# Run a GPUMC harness against the REAL shared device sources.
#
#   verification/gpumc/run.sh <harness-name-without-.cpp> [extra gpumc args...]
#   verification/gpumc/run.sh all
#
# WHAT GPUMC IS, AND WHY IT IS NOT GenMC. GPUMC is a stateless model checker for scoped-RC11, the
# GPU memory model: threads are organised into CTAs and an access carries a SCOPE, so
# synchronisation between two threads depends on how close they are. RC11 has no scopes, so GenMC
# would check a program the device does not run. GPUMC extends the same GenMC/TruSt line, which is
# why a harness looks like the CPU ones.
#
# HOW A SCOPE IS WRITTEN. Not as an argument to the atomic -- as an annotation call placed
# immediately before the access it qualifies:
#
#     __VERIFIER_memory_scope_device();
#     __atomic_fetch_add(&x, 1, __ATOMIC_RELEASE);
#
# and the thread hierarchy is declared per thread with __VERIFIER_thread_local_id / _group_id /
# _global_id / _kernel_id.
#
# THE TOOL RUNS IN A CONTAINER, and is not built here. It is a fork of GenMC 0.9 supporting LLVM
# up to 15; this tree builds against LLVM 18, and installing an LLVM 14 beside it on a shared
# machine to run one checker is not a trade worth making. The image is the CAV 2025 artifact
# (figshare 28789703, GPL-3.0+), loaded once with `docker load`. Set HG_GPUMC_IMAGE to override.
#
# A 32-BIT COMPARE-EXCHANGE ALWAYS REPORTS FAILURE in this build, while reading exactly the
# expected value. The trace shows the CAS read and no CAS write, so no execution completes and
# the harness's own assertion fires as though the protocol were broken. The same exchange on a
# 64-bit word, under the same memory orders, succeeds. Widen the harness's word before suspecting
# the code; verification/gpumc/hash_insert_elects_one.cpp records where this was measured.
#
# ONE STEP, unlike the GenMC runner. This driver compiles the input itself and handles C++ -- the
# harness drives a shared template, so that matters -- and takes compiler flags after `--`.
# The documented --input-from-bitcode-file path is NOT used: it segfaults in this build on
# bitcode from either clang or clang++, including a five-line program, while the same source
# compiled directly by the driver verifies. Measured both ways before choosing.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMAGE="${HG_GPUMC_IMAGE:-gpumc_docker:latest}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    cat >&2 <<EOF
run.sh: no GPUMC image '$IMAGE'.
Obtain the CAV 2025 artifact (figshare 28789703), then:
    unzip -j <artifact>.zip 'cav_25_artifact_314/gpumc_docker.tar' && docker load -i gpumc_docker.tar
EOF
    exit 2
fi

run_one() {
    local name="$1"; shift
    local src="$HERE/$name.cpp"
    [ -f "$src" ] || { echo "run.sh: no such harness '$src'" >&2; return 2; }
    echo "=== $name ==="

    # The tree is mounted READ-ONLY: a checker must not be able to edit the sources it checks.
    # Scratch goes to a tmpfs the container owns.
    docker run --rm -v "$ROOT:/src:ro" --entrypoint /bin/sh "$IMAGE" -c "
        genmc $* -- -std=c++17 -I /src/common/include ${HG_HARNESS_DEFINES:-} \
            /src/verification/gpumc/$name.cpp
    "
    local rc=$?
    echo "--- $name: gpumc exit $rc"
    return $rc
}

if [ "${1:-}" = "all" ]; then
    fail=0
    for src in "$HERE"/*.cpp; do
        run_one "$(basename "$src" .cpp)" || fail=1
        echo
    done
    exit $fail
fi

[ $# -ge 1 ] || { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
run_one "$@"
