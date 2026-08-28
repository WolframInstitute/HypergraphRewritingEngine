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
#   HG_GENMC_DEBUG_INFO set to compile with -g, so the checker's --print-error-trace names source
#                  lines. Off by default: debug metadata more than doubles the pruned module
#                  (111,538 -> 274,828 lines on the evolve() harness) and the transformation
#                  phase's memory with it.
#   HG_HARNESS_DEFINES  extra -D flags for the harness compile. A harness carrying a CALIBRATION
#                  arm -- the defect reinstated behind an ifdef -- is run through it with this,
#                  so the calibration is a command anyone can repeat rather than a claim in a
#                  comment. Word-split on purpose; pass several as one string.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

GENMC="${GENMC:-$HOME/genmc/build/bin/genmc}"
CLANGXX="${CLANGXX:-/usr/lib/llvm-18/bin/clang++}"
LLVM_LINK="${LLVM_LINK:-$(dirname "$CLANGXX")/llvm-link}"
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

# GENMC_WORK names a directory to build in and KEEPS it, so the module the checker was handed can
# be read after the run. Diagnosing a composed harness means looking at the pruned .ll.
if [ -n "${GENMC_WORK:-}" ]; then
    WORK="$GENMC_WORK"; mkdir -p "$WORK"
else
    WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fi

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
    # A PATH runs a harness that is not in this directory, which is how one under development is
    # exercised without `all` picking it up before it is ready.
    local src
    case "$name" in
        */*|*.cpp) src="$name"; name="$(basename "$name" .cpp)" ;;
        *)         src="$HERE/$name.cpp" ;;
    esac
    [ -f "$src" ] || { echo "run.sh: no such harness '$src'" >&2; return 2; }

    echo "=== $name ==="

    # Compile-time defines travel with the harness in a `// GENMC-DEFINES:` line, scoped to this
    # invocation by shadowing HG_HARNESS_DEFINES locally. They reach the engine translation
    # units as well as the harness and are part of the IR cache key, so a harness can bound a
    # structure the engine instantiates -- a map's working capacity, say -- and get its own
    # engine build rather than a stale one or a global rebuild.
    local file_defines
    file_defines="$(sed -n 's|^// GENMC-DEFINES: *||p' "$src" | head -1)"
    local HG_HARNESS_DEFINES="${HG_HARNESS_DEFINES:-} $file_defines"

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
    # A harness that declares `// GENMC-LINK: engine` is checked against the COMPOSED engine
    # rather than one header: every engine translation unit is compiled to IR and linked in, so
    # the checker runs the real evolve(), the real job system and the real structures together.
    # Without it a harness sees only what its own includes define, and a defect that lives in the
    # composition of the structures is invisible to it -- which is what every harness here was
    # before.
    # The marker's VALUE picks the source set. `engine` is everything; `job_system` is the job
    # system alone, which is 543 lines against the engine's tens of thousands -- so a harness about
    # the job system is checkable where one reaching evolve() is not.
    local link_engine link_srcs=()
    link_engine="$(sed -n 's|^// GENMC-LINK: *||p' "$src" | head -1)"
    case "$link_engine" in
        "")         ;;
        engine)     link_srcs=("$ROOT"/hypergraph/src/*.cpp "$ROOT"/job_system/src/*.cpp
                               "$HERE"/genmc_support.cpp) ;;
        job_system) link_srcs=("$ROOT"/job_system/src/*.cpp "$HERE"/genmc_support.cpp) ;;
        *) echo "--- $name: unknown GENMC-LINK target '$link_engine'" >&2; return 2 ;;
    esac

    # Two declarations the composed build needs and a single-header harness does not.
    #
    # The pthread shim: GenMC's pthread.h covers what a harness written against one structure
    # touches and leaves the condition-variable, once, key and affinity families commented out.
    # libstdc++ does not know that -- bits/gthr-default.h aliases the whole pthread surface as
    # soon as a translation unit reaches <memory>, which anything holding a unique_ptr does.
    #
    # HG_PARK_VERIFICATION: on Linux the engine parks in syscall(SYS_futex), which the checker
    # cannot see, so a parked worker would vanish from the exploration. The verification backend
    # spins on the same word, which the contract already permits (park may return spuriously).
    local extra_cc=()
    if [ -n "$link_engine" ]; then
        extra_cc=(-include "$HERE/genmc_pthread_shim.h" -DHG_PARK_VERIFICATION=1)
    fi

    if ! "$CLANGXX" -std=c++17 -O0 ${HG_GENMC_DEBUG_INFO:+-g} -Xclang -disable-O0-optnone -S -emit-llvm \
            "${INCLUDES[@]}" "${extra_cc[@]}" -DHG_VERIFICATION=1 -DHG_ENGINE_STATS=0 ${HG_HARNESS_DEFINES:-} \
            -o "$WORK/$name.raw.ll" "$src" 2>"$WORK/$name.cc.err"; then
        echo "--- $name: COMPILE FAILED"
        tail -30 "$WORK/$name.cc.err"
        return 3
    fi

    local to_opt="$WORK/$name.raw.ll"
    local opt_extra=()
    # NO instcombine. It folds a small constant-aggregate memcpy into ONE integer store of the
    # whole aggregate -- `uint32_t raws[2] = {10, 12}` became `store i64 51539607562` -- and
    # the program then reads the halves as i32. The checker resolves a read by the write of the
    # same width, so a 32-bit read of a 64-bit store finds nothing and the checker stops with
    # an internal check (resolveAccessValue) instead of a verdict. Measured on the composition
    # harness for the causal in-edge order; the same module verifies with instcombine removed.
    local passes='always-inline,inline,sroa,early-cse,simplifycfg,adce,globaldce,strip-dead-prototypes'
    if [ -n "$link_engine" ]; then
        # The engine's IR is CACHED, keyed on the inputs that produce it: every source and header
        # it is built from, plus the flags. Any change to any of them gives a different key and a
        # cold directory, so the cache cannot serve a stale module -- and an unchanged tree makes
        # `run.sh all` compile the engine once for the whole suite rather than once per harness.
        local key
        key="$( { ls -lL --time-style=+%s "$ROOT"/hypergraph/src/*.cpp "$ROOT"/job_system/src/*.cpp \
                     "$ROOT"/hypergraph/include/hypergraph/*.hpp "$ROOT"/common/include/hgcommon/*.hpp \
                     "$ROOT"/job_system/include/job_system/*.hpp "$ROOT"/lockfree_deque/include/*/*.hpp \
                     "$HERE"/genmc_pthread_shim.h "$HERE"/genmc_support.cpp 2>/dev/null
                 echo "${HG_HARNESS_DEFINES:-}" "$link_engine" "${HG_GENMC_DEBUG_INFO:-}" "stats0"; "$CLANGXX" --version | head -1
               } | md5sum | cut -c1-16 )"
        local cache="${GENMC_IR_CACHE:-$ROOT/.genmc_ir_cache}/$key"
        mkdir -p "$cache"

        local units=()
        for u in "${link_srcs[@]}"; do
            local un; un="$(basename "$u" .cpp)"
            if [ ! -s "$cache/$un.ll" ]; then
                if ! "$CLANGXX" -std=c++17 -O0 ${HG_GENMC_DEBUG_INFO:+-g} -Xclang -disable-O0-optnone -S -emit-llvm \
                        "${INCLUDES[@]}" "${extra_cc[@]}" -DHG_VERIFICATION=1 -DHG_ENGINE_STATS=0 ${HG_HARNESS_DEFINES:-} \
                        -o "$cache/$un.ll.tmp" "$u" 2>"$WORK/engine_$un.cc.err"; then
                    echo "--- $name: ENGINE TU $un FAILED TO COMPILE"
                    tail -20 "$WORK/engine_$un.cc.err"
                    return 3
                fi
                mv "$cache/$un.ll.tmp" "$cache/$un.ll"
            fi
            units+=("$cache/$un.ll")
        done
        if ! "$LLVM_LINK" -S "$WORK/$name.raw.ll" "${units[@]}" -o "$WORK/$name.linked.ll" \
                2>"$WORK/$name.link.err"; then
            echo "--- $name: LINK FAILED"
            tail -20 "$WORK/$name.link.err"
            return 3
        fi
        to_opt="$WORK/$name.linked.ll"

        # LIBSTDC++'S TYPEINFO OBJECTS ARE DEFINED HERE, ZERO-FILLED. A class derived from a
        # standard exception carries a typeinfo whose initializer points at the base's --
        # hg::common::CapacityExhausted's at _ZTISt12length_error -- and that base typeinfo is
        # declared in the module and defined only in the C++ runtime. The interpreter assigns no
        # address to an undefined global, so materialising the derived initializer dereferences
        # null in ExecutionEngine::InitializeMemory and the checker segfaults before the first
        # thread runs. MEASURED on the composed engine: every attempt to construct the evolution
        # engine died there.
        #
        # A typeinfo object is read by throw, catch and dynamic_cast and by nothing else. The
        # engine throws only on a programmer error and never catches or downcasts, so on every
        # path the checker explores no thread reads these bytes; a zero-filled definition
        # changes no shared memory and no control flow. The declarations' own types are kept.
        # THE SAME HOLDS FOR EVERY OTHER UNDEFINED DATA SYMBOL, for a reason one line above the
        # typeinfo one in the interpreter: collectStaticAddresses (lli/Runtime/Interpreter.cpp)
        # walks every global variable of the module, declarations included, and re-initialises
        # each from its initializer -- which a declaration does not have. @stderr (the
        # placement diagnostic's fprintf, never enabled by a harness) and std::nothrow (a tag
        # object, never read) are the two the engine declares.
        local ti_ll="$WORK/$name.typeinfo.ll"
        grep -oE '^@[A-Za-z0-9_.$"]+ = external (unnamed_addr )?(global|constant) [^,]+' \
            "$WORK/$name.linked.ll" \
          | sed -E 's/^(@[^ ]+) = external (unnamed_addr )?(global|constant) (.*)$/\1 = \2\3 \4 zeroinitializer/' \
          > "$ti_ll"
        # A definition of a named struct type has to travel with a declaration that uses it.
        if [ -s "$ti_ll" ]; then
            local ty_ll="$WORK/$name.typeinfo.types.ll"
            : > "$ty_ll"
            for t in $(grep -oE '%"[^"]+"' "$ti_ll" | sort -u); do
                grep -F "$t = type " "$WORK/$name.linked.ll" >> "$ty_ll"
            done
            cat "$ti_ll" >> "$ty_ll"; mv "$ty_ll" "$ti_ll"
        fi
        if [ -s "$ti_ll" ]; then
            if ! "$LLVM_LINK" -S "$WORK/$name.linked.ll" "$ti_ll" -o "$WORK/$name.linked2.ll" \
                    2>>"$WORK/$name.link.err"; then
                echo "--- $name: TYPEINFO LINK FAILED"; tail -20 "$WORK/$name.link.err"; return 3
            fi
            to_opt="$WORK/$name.linked2.ll"
        fi

        # INTERNALIZE BEFORE globaldce, or the prune does nothing. Linking the engine brings in
        # every exported symbol, and globaldce cannot prove any of them dead while they are
        # externally visible -- so the checker is handed the whole engine whether or not main
        # reaches it. MEASURED: the same probe is 154 lines and verifies in 0s when linked alone,
        # and 130,815 lines and does not finish in 90s when linked with the engine. Internalizing
        # everything but main takes that back to 140 lines and 0s.
        passes="internalize,globaldce,$passes"
        opt_extra=(-internalize-public-api-list=main)

        # lower-constant-intrinsics because __builtin_constant_p survives the link as
        # llvm.is.constant, which GenMC's code generator does not implement. It goes AFTER inline
        # -- naming a function pass first makes opt parse the whole list as a function pipeline
        # and reject the module passes in it.
        passes="${passes/inline,/inline,lower-constant-intrinsics,}"
    fi

    if ! "$OPT" ${opt_extra[@]+"${opt_extra[@]}"} -passes="$passes" \
            -S "$to_opt" -o "$WORK/$name.ll" 2>"$WORK/$name.opt.err"; then
        echo "--- $name: OPT FAILED"
        tail -20 "$WORK/$name.opt.err"
        return 3
    fi


    # What the checker was actually handed. For a composed harness this is the number that
    # decides whether it finishes: the module is the whole engine minus whatever main cannot
    # reach, and a harness that reaches more of it costs more before a single execution runs.
    if [ -n "$link_engine" ]; then
        echo "    composed: $(wc -l < "$to_opt") lines linked, $(wc -l < "$WORK/$name.ll") after prune"
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

    # A COMPOSED HARNESS RUNS UNDER A MEMORY CAP. The checker's analysis of a module the size of
    # the engine is measured at 18 GB of resident memory on evolve(), which on a shared 19 GB box
    # is a machine in swap. The cap is half of what is available at launch (HG_GENMC_MEM_MB
    # overrides it), and a checker that needs more fails inside its own process instead of
    # taking the box down: the same self-capping every numerical run and CUDA build here uses.
    local mem_cap_kb=""
    if [ -n "$link_engine" ]; then
        local avail_kb; avail_kb="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
        mem_cap_kb="${HG_GENMC_MEM_MB:+$((HG_GENMC_MEM_MB * 1024))}"
        mem_cap_kb="${mem_cap_kb:-$((avail_kb / 2))}"
        echo "    memory cap: $((mem_cap_kb / 1024)) MB (address space)"
    fi
    # shellcheck disable=SC2086
    ( [ -n "$mem_cap_kb" ] && ulimit -v "$mem_cap_kb"; exec "$GENMC" $extra "$@" "$WORK/$name.ll" )
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
        # genmc_support.cpp is LINKED INTO composed harnesses, not run as one -- it supplies the
        # definitions the interpreter lacks and has no main. Running it reports an error about
        # this directory's layout rather than about the engine.
        [ "$(basename "$src")" = "genmc_support.cpp" ] && continue
        run_one "$(basename "$src" .cpp)" || fail=1
        echo
    done
    exit $fail
fi

[ $# -ge 1 ] || { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
name="$1"; shift
run_one "$name" "$@"
