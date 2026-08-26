#!/usr/bin/env bash
# Run a GenMC harness against the REAL engine headers.
#
# GenMC enumerates the executions of the RC11 memory model for a bounded program. A harness here
# is a small main() that includes the engine's own header and calls its own functions -- not a
# hand-written model of them. A model drifts from the code the moment the code changes, and a
# re-implementation proves a property of the re-implementation. These harnesses break when the
# header breaks, which is the entire point.
#
# SIZE THE HARNESS BEFORE RUNNING IT. `run.sh <name> --mode=estimate` samples executions and
# prints the total-executions and time-to-completion estimates in about a tenth of a second.
# A harness whose estimate does not fit the budget in tools/safe_verify.sh is REDUCED -- fewer
# threads, fewer operations, the same window -- and the reduction is calibrated by breaking the
# property and checking the smaller harness still reports it. Estimation is also a bug-finder in
# its own right: it explores real executions, so a violation it samples is a genuine
# counterexample, delivered in seconds rather than after an enumeration that may never finish.
#
# WHAT A CLEAN RUN MEANS. GenMC is sound and complete for the BOUNDED program it is given:
# exhaustive over the interleavings and reads-from choices of RC11, for that thread count, that
# operation count and those inputs. It is NOT a proof for unbounded thread counts. State the bound
# with the result; "verified" without the bound is a claim the tool never made.
#
# WHY TWO STEPS RATHER THAN `genmc -- file.cpp`.
# Driving the compilation itself, GenMC puts its own runtime-include/c directory first, which
# replaces stdlib.h with a model declaring only what the checker interprets. libstdc++'s <string>
# then fails to find std::strtoul and friends, so any C++ translation unit reaching <string> --
# which concurrent_map.hpp does, for its precondition messages -- cannot be compiled that way.
#
# So: compile to LLVM IR with clang, taking SYSTEM headers for the C and C++ libraries and
# GenMC's headers for exactly the four the checker must interpret (pthread.h, assert.h and the two
# they include). Then hand the IR to GenMC. The interpreter recognises threading and assertions
# through those declarations -- pthread_t is __VERIFIER_thread_t and assert routes to the
# checker's own trap -- while everything else is the real standard library the engine compiles
# against normally.
#
# Usage:
#   verification/genmc/run.sh <harness-name-without-.cpp> [extra genmc args...]
#   verification/genmc/run.sh all
#
# Environment:
#   GENMC          path to the genmc binary        (default: ~/genmc/build/bin/genmc)
#   GENMC_INCLUDE  path to its runtime-include/c   (default: derived from GENMC)
#   CLANGXX        clang++ to emit the IR          (default: /usr/lib/llvm-18/bin/clang++)
#   OPT            matching llvm opt               (default: /usr/lib/llvm-18/bin/opt)
#   HG_HARNESS_DEFINES  extra -D flags for the harness compile. A harness carrying a CALIBRATION
#                  arm -- the defect reinstated behind an ifdef -- is run through it with this,
#                  so the calibration is a command anyone can repeat rather than a claim in a
#                  comment. Word-split on purpose; pass several as one string.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

GENMC="${GENMC:-$HOME/genmc/build/bin/genmc}"
CLANGXX="${CLANGXX:-/usr/lib/llvm-18/bin/clang++}"
OPT="${OPT:-/usr/lib/llvm-18/bin/opt}"

if [ ! -x "$GENMC" ]; then
    cat >&2 <<EOF
run.sh: genmc not found at '$GENMC'. See verification/genmc/README.md for the build, or set GENMC.
EOF
    exit 2
fi
if [ ! -x "$OPT" ]; then
    echo "run.sh: opt not found at '$OPT'; set OPT" >&2
    exit 2
fi
if [ ! -x "$CLANGXX" ]; then
    echo "run.sh: clang++ not found at '$CLANGXX'; set CLANGXX" >&2
    exit 2
fi

# runtime-include/c lives in the genmc SOURCE tree, not next to the built binary.
if [ -z "${GENMC_INCLUDE:-}" ]; then
    for cand in "$(dirname "$GENMC")/../../lli/runtime-include/c" \
                "$(dirname "$GENMC")/../include/genmc/c" \
                "$HOME/genmc/lli/runtime-include/c"; do
        [ -f "$cand/pthread.h" ] && { GENMC_INCLUDE="$(cd "$cand" && pwd)"; break; }
    done
fi
if [ -z "${GENMC_INCLUDE:-}" ] || [ ! -f "$GENMC_INCLUDE/pthread.h" ]; then
    echo "run.sh: could not find genmc's runtime-include/c; set GENMC_INCLUDE" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Exactly the headers the checker must interpret. Everything else resolves to the system
# library, so the harness compiles against the same declarations the engine does.
mkdir -p "$WORK/shim/bits"
for h in pthread.h assert.h genmc.h genmc_internal.h; do
    [ -f "$GENMC_INCLUDE/$h" ] && cp "$GENMC_INCLUDE/$h" "$WORK/shim/$h"
done
# glibc's <stdlib.h> reaches bits/pthreadtypes.h, which defines pthread_t and the mutex/barrier
# unions that GenMC's pthread.h has already defined as its own __VERIFIER_ types. Both definitions
# are visible in the same translation unit and clang rejects the redefinition. An empty file at
# that path lets GenMC's definitions stand; nothing in glibc's headers uses those types itself.
: > "$WORK/shim/bits/pthreadtypes.h"

INCLUDES=(
    -I"$WORK/shim"
    -I"$ROOT/hypergraph/include"
    -I"$ROOT/common/include"
    -I"$ROOT/job_system/include"
    -I"$ROOT/lockfree_deque/include"
)

run_one() {
    local name="$1"; shift
    local src="$HERE/$name.cpp"
    [ -f "$src" ] || { echo "run.sh: no such harness '$src'" >&2; return 2; }

    echo "=== $name ==="

    # Compile at -O0, then optimise with a chosen pass list. Neither half is arbitrary.
    #
    # -O0 alone gives the checker an event for every access to every local, which is a state
    # space orders of magnitude larger than the shared accesses actually under test. The locals
    # have to be promoted to registers.
    #
    # -O1 and above cannot be used to do it: the loop-idiom pass turns the entry-array
    # initialisation into one memset spanning several entries, and the checker's promotion of
    # memory intrinsics requires the destination's pointee to be at least as large as the copy,
    # so it fails an internal check. -Os and -Oz instead emit llvm.umax, which the interpreter
    # does not implement.
    #
    # So: -O0 for the code shape, -Xclang -disable-O0-optnone so the functions are not marked
    # optnone (which would make every subsequent pass a no-op), and a pass list that promotes and
    # inlines but never runs loop-idiom. instcombine and simplifycfg preserve atomic operations,
    # which is what the checker reads the program from.
    if ! "$CLANGXX" -std=c++17 -O0 -Xclang -disable-O0-optnone -S -emit-llvm \
            "${INCLUDES[@]}" -DHG_VERIFICATION=1 ${HG_HARNESS_DEFINES:-} \
            -o "$WORK/$name.raw.ll" "$src" 2>"$WORK/$name.cc.err"; then
        echo "--- $name: COMPILE FAILED"
        tail -30 "$WORK/$name.cc.err"
        return 3
    fi
    if ! "$OPT" -passes='always-inline,inline,sroa,early-cse,instcombine,simplifycfg,adce,globaldce,strip-dead-prototypes' \
            -S "$WORK/$name.raw.ll" -o "$WORK/$name.ll" 2>"$WORK/$name.opt.err"; then
        echo "--- $name: OPT FAILED"
        tail -20 "$WORK/$name.opt.err"
        return 3
    fi

    # Harness-specific GenMC flags travel with the harness in a `// GENMC-ARGS:` line, so the
    # bound a harness needs is stated next to the property it bounds.
    local extra
    extra="$(sed -n 's|^// GENMC-ARGS: *||p' "$src" | head -1)"

    # A `// GENMC-EXPECT: violation` harness is a PINNED REPRODUCER of a known-reachable defect:
    # it passes exactly when the checker still finds the violation, so the suite notices if the
    # window silently moves or is closed without the marker being flipped.
    local expect
    expect="$(sed -n 's|^// GENMC-EXPECT: *||p' "$src" | head -1)"

    # shellcheck disable=SC2086
    "$GENMC" $extra "$@" "$WORK/$name.ll"
    local rc=$?
    if [ "$expect" = "violation" ]; then
        if [ $rc -eq 42 ]; then
            echo "--- $name: EXPECTED violation still reachable (pinned reproducer) -> pass"
            return 0
        fi
        echo "--- $name: expected a violation and got exit $rc -- the window moved or closed;"
        echo "    if it was fixed on purpose, flip the GENMC-EXPECT marker in the harness"
        return 1
    fi
    echo "--- $name: genmc exit $rc"
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

[ $# -ge 1 ] || { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
name="$1"; shift
run_one "$name" "$@"
