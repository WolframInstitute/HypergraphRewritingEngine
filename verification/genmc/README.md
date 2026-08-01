# Model checking the lock-free surface

GenMC enumerates the executions of a bounded concurrent program under the RC11 memory model. It is
**sound and complete for the program it is given**: it explores every interleaving and every
reads-from choice the model permits, for that thread count, that operation count and those inputs.

That is a different instrument from a stress test, and the difference is the reason this directory
exists. A stress test *samples* interleavings. The engine's determinism gate went **150 runs**
without firing a defect known to be present (`tools/gate_rate.sh 'CausalDeterminism.*' 40`, then 20
more under load, on top of 90 already on record). Sampling can only ever fail to reproduce. A model
checker's clean run is a statement about **all** executions within the bound.

## A harness includes the real header

Every harness here is a small `main()` that includes the engine's own header and calls its own
functions. It is not a hand-written model of the algorithm, and not a re-implementation.

That constraint is load-bearing. A model drifts from the code the moment the code changes, and a
re-implementation proves a property of the re-implementation. These harnesses stop compiling when
the header changes shape, which is exactly the coupling wanted: they track the code.

## Running

```
verification/genmc/run.sh concurrent_map_agreement     # one harness
verification/genmc/run.sh all                          # every harness in this directory
```

`run.sh` exits non-zero if any harness fails, so it can gate a commit.

## Building GenMC

Verified against **GenMC v0.17.0** and **LLVM 18**.

```
git clone --depth 1 https://github.com/MPI-SWS/genmc.git ~/genmc
cd ~/genmc
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/usr/lib/llvm-18/cmake;/usr/lib/llvm-18/lib/cmake/clang" \
    -DCMAKE_IGNORE_PREFIX_PATH="$HOME/miniforge3;$HOME/miniforge3/envs/sage"
cmake --build build -j4
```

`CMAKE_IGNORE_PREFIX_PATH` is not optional on a machine with a conda environment on `PATH`. CMake
otherwise resolves `libxml2`, `zstd`, `zlib` and `libedit` to conda's copies while LLVM 18 was
linked against the system ones, and the build fails at link with undefined references to versioned
`libxml2` symbols. Everything compiles first, so the failure arrives at the very end.

Override locations with `GENMC`, `GENMC_INCLUDE`, `CLANGXX`, `OPT`.

## How a harness is compiled, and why not the obvious way

`genmc -- file.cpp` does not work on this codebase. Driving the compilation itself, GenMC puts its
`runtime-include/c` directory first, replacing `stdlib.h` with a model that declares only what the
checker interprets. libstdc++'s `<string>` then cannot find `std::strtoul` and friends, so any
translation unit reaching `<string>` fails — and `concurrent_map.hpp` reaches it, for its
precondition messages.

So `run.sh` compiles the IR itself:

1. **System headers for the C and C++ libraries**, and GenMC's headers for exactly the four the
   checker must interpret: `pthread.h`, `assert.h`, and the two they include. `pthread_t` becomes
   `__VERIFIER_thread_t` and `assert` routes to the checker's trap, while everything else is the
   real standard library the engine compiles against normally.

2. **An empty `bits/pthreadtypes.h`** on the include path. glibc's `<stdlib.h>` reaches that header,
   which defines `pthread_t` and the mutex/barrier unions GenMC's `pthread.h` has already defined as
   its own types; clang rejects the redefinition. Nothing in glibc's headers uses those types
   itself, so shadowing it is enough.

3. **`clang -O0 -Xclang -disable-O0-optnone`, then `opt` with a chosen pass list.** Both halves are
   forced:

   - `-O0` alone gives the checker an event for every access to every *local*, a state space orders
     of magnitude larger than the shared accesses under test. Locals must be promoted.
   - `-O1` and above cannot do the promoting. The loop-idiom pass rewrites the entry-array
     initialisation as one `memset` spanning several entries, and GenMC's promotion of memory
     intrinsics requires the destination's pointee to be at least as large as the copy — it fails
     an internal check (`typeSizeDst >= len`). `-Os` and `-Oz` instead emit `llvm.umax`, which the
     interpreter does not implement.
   - `-disable-O0-optnone` matters because clang otherwise marks `-O0` functions `optnone`, making
     every subsequent pass a silent no-op. Without it the IR came back 6054 lines → 6048.

   The pass list promotes and inlines and never runs loop-idiom. `instcombine` and `simplifycfg`
   preserve atomic operations, which is what the checker reads the program from.

## What `HG_VERIFICATION` changes, and why each change is sound

`run.sh` defines `HG_VERIFICATION=1`. Three places in the engine compile differently under it. Each
is forced by how LLVM's `ExecutionEngine` materialises module memory **before any thread runs** —
none of them is about how the code behaves.

| Site | Under verification | Why the tool requires it | Why it does not weaken the result |
|---|---|---|---|
| `arena.hpp` — `arena_worker_index()` | constant-initialised `thread_local int` at namespace scope | A `thread_local` of *class* type has no constant representation and the interpreter stops before the first thread. An empty class fails identically. A function-local `thread_local` scalar blocks on its initialisation guard. | The index is read, not exercised; no harness states a property about it. |
| `arena.hpp` — `ArenaWorkerRegistry` | not reached, so not emitted | The registry is an aggregate global of `MAX_ARENA_WORKERS` atomics; the same machinery *faults* on it rather than diagnosing it. An inline function's local static is emitted only when something reaches it. | A counter hands out each index once, so acquire/release recycling is not reproduced. Harnesses run a fixed, bounded thread set whose threads outlive their allocations, so no index is ever reused. |
| `concurrent_map.hpp` — two precondition sites | `assert` instead of `throw std::logic_error` | A `throw` names the exception's typeinfo, an external constant with no definition in the module. **Reaching the throw is not required** — a never-executed throw segfaults the interpreter during memory initialisation. Measured: identical program, `throw` → SIGSEGV, `assert` → verifies. | The precondition is unchanged and still traps. A violated assertion is a safety property the checker *reports*; a throw is a crash it dies on. Strictly more checkable. |

Both `arena.hpp` substitutions concern the worker-index allocator, which no harness in this
directory makes a claim about. If a harness is ever written *for* that allocator, it cannot use
these substitutions and needs a different approach.

## Results

| Harness | Property | Bound | Result |
|---|---|---|---|
| `concurrent_map_agreement` | Two threads offering different values for one key agree on the winner: exactly one reports `was_inserted`, both return the same value, that value is one of the two offered, the winner's own value is what is stored, and a later `lookup` returns it | 2 threads, 1 key, 2 values, capacity 4, no resize | **No errors, 32 complete executions** |

The bound is part of the result. `concurrent_map_agreement` says nothing about three concurrent
inserters, about two different keys colliding in one probe run, or about a resize running
underneath an insert — that last one is a distinct algorithm (rehash, then CAS-install) and belongs
in its own harness rather than inflating this one's state space.

## Adding a harness

Drop a `.cpp` in this directory; `run.sh all` picks it up. Include `genmc_support.hpp` first — it
defines `__dso_handle`, which clang emits for static-destructor registration and the interpreter
cannot otherwise resolve.

Put harness-specific GenMC flags in a `// GENMC-ARGS:` line in the source, so the bound a harness
needs is stated next to the property it bounds rather than in this script.

Useful flags while developing one: `--disable-estimation` skips the state-space estimate;
`--unroll=N` bounds loops; `--sc --bound=N --bound-type=context` bounds context switches (bounding
requires `--sc`).
