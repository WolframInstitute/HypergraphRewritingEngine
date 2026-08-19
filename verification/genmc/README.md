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

Both `arena.hpp` substitutions concern the worker-index allocator. `arena_worker_index()` is
replaced by a monotonic counter that never releases and never reuses, so the registry -- an
aggregate global of `MAX_ARENA_WORKERS` atomics -- never enters the module.

`arena_worker_index_exclusive` is the harness written *for* that allocator, and as this section
predicted it could not use the substitution. It takes the different approach instead: it does not
call `arena_worker_index()` at all, constructs `ArenaWorkerRegistry` as a LOCAL so no aggregate
global is materialised, and drives `acquire()` directly on the shipped type. `HG_MAX_ARENA_WORKERS`
is overridable so the bound can be 2 -- `acquire()` scans every slot and each is an atomic
location, so 256 of them enumerate the scan rather than the property.

What is still NOT covered: index REUSE. The substitution's counter cannot release, and the harness
above holds its indices rather than releasing them, because simultaneous holding is what breaks the
private-cursor invariant -- a registry handing one index to two threads in sequence is correct. A
harness for the release/re-acquire path would need to model thread exit, which the interpreter's
handling of thread_local class types is what blocked in the first place.

## Results

| Harness | Property | Bound | Result |
|---|---|---|---|
| `concurrent_map_agreement` | Two threads offering different values for one key agree on the winner: exactly one reports `was_inserted`, both return the same value, that value is one of the two offered, the winner's own value is what is stored, and a later `lookup` returns it | 2 threads, 1 key, 2 values, capacity 4, no resize | **No errors, 32 complete executions** |
| `concurrent_map_resize` | The same agreement holds across a table replacement, both pre-existing keys survive the rehash, and no key acquires a second entry | 2 threads, capacity 2→4, 3 keys, one resize round | **No errors, 176 complete executions** — after the fix below |
| `deque_no_double_extraction` | A `pop_front` and a `pop_back` racing for the deque's *last* item never both receive it, never invent one, and leave a size consistent with what left | 2 threads, capacity 4, 1 item, 1 pop attempt each | **No errors, 6 complete executions** |
| `job_system_no_lost_wakeup` | A submitter that skips the wake because nobody reads as idle never leaves a worker parked with the job still queued | 1 worker, 1 submitter, 1 job | **No errors, 5 complete executions** — after the fix below |
| `lock_free_list_completeness` | `for_each` visits every pushed node exactly once -- the COMPLETENESS direction no other harness here states, all of which bound exclusivity | 2 pushers, 2 pushes each, distinct values, one walk after join; stub allocator exclusive by construction | **No errors, 184 complete executions** |
| `concurrent_map_double_growth_3t` | exactly one caller is told it inserted, THREE workers across two growths. Found a real defect: a publish that beat the retiring table's seal left a settled entry in a superseded table, so two exchanges won for one key and both callers were told they inserted | 3 threads, 4-context bound, growth target 16 | **No errors, 78,397 complete executions** — after the fix below |
| `lock_free_list_three_meet_once` | THREE concurrent pushers meet pairwise exactly once (total sightings = 3 = C(3,2)). Two threads cannot exhibit a scan running BETWEEN two other pushes; three can, and the branchial relation is built over instances carrying many applications | 3 threads, 1 push each, distinct values | **No errors, 288 complete executions** |
| `key_set_distinct_keys_across_growth` | a key NEVER inserted is never reported as already present, across a growth -- the other half of `key_set_exactly_once`, which bounds one key. A false rejection at `qc_applied_` drops an (instance, match) application, so one raw event never exists and every relation over it changes while the STATE set does not | 2 threads, 2 distinct keys, capacity 1 so growth is on both entry paths | **No errors, 54 complete executions** |
| `quotient_instance_match_rendezvous` | the quotient replay's (instance, match) rendezvous never drops a pair: both sides publish then scan through a ConcurrentMap lookup, and if BOTH scans miss, one raw event never happens and every causal and branchial pair it belonged to goes with it -- while the state and canonical event counts a caller reads stay unchanged | 2 threads, one class, one instance and one match, over the REAL ConcurrentMap and LockFreeList | **No errors, 14 complete executions** |
| `lock_free_list_pairs_meet_once` | two concurrent pushers MEET EXACTLY ONCE under `push` + `for_each_before` -- the pairing rule the quotient branchial relation is built on, which `lock_free_list_completeness` does not state because it walks from the head after the pushers have joined | 2 threads, 1 push each, each scanning the nodes older than its own and counting sightings of the other; their sum must be 1 | **No errors, 8 complete executions** |
| `arena_worker_index_exclusive` | Two live holders never share a worker index, which is the invariant `allocate_local`'s plain non-atomic cursor bump rests on | 2 threads, 1 acquire each, no release, `HG_MAX_ARENA_WORKERS=2` | **No errors, 4 complete executions** |
| `claim_match_rendezvous` | The match-dedup rendezvous claims exactly once (two claimants of one match agree on one winner) and never drops on collision (two matches sharing a 64-bit hash BOTH win — the root of #74) | 2 threads per phase, 2 phases, capacity 8, probe depth 8 | **No errors, 2500 complete executions** |

### What this found

`concurrent_map_resize` reported a safety violation on its first run: both callers inserting one
key across a resize came back with `was_inserted == true` — the split rendezvous the header's
chain-scan comment exists to prevent.

Two resizes could both install. Thread A takes the head from T0 to T1; thread B, whose own
`resize()` loaded `table_` after that, takes it from T1 to T2. A is still holding T1, scans its
ancestors, finds nothing, and claims the key there — while B, working from T2, has already walked
past T1 in its own chain scan and claims the same key at the head. The chain scan is a
point-in-time check and a rival's key can land in an older table after it has passed.

B's second installation was never needed: A had already installed a table with twice the capacity.
`resize()` now grows only if the *current* head still exceeds the load factor, and probe
exhaustion in a superseded table retries at the head rather than growing. Fixed in `29283f7`,
which also records the residual (growth genuinely warranted twice over) as a separate item, and
records a head-re-check alternative that cost +1.9% instructions and did **not** close it.

This is the point of the exercise. That interleaving needs two installations to straddle one
thread's claim; no stress test on this machine had produced it, and the class had been in use
long enough to produce four correctness bugs of the neighbouring kind.

`job_system_no_lost_wakeup` then reported `Non-terminating spinloop: thread 1` on ITS first run.
`wake_one_worker()` skips the wake when `idle_workers_` reads zero, and the header justified that
by "at least one of them must observe the other". The two sides write different locations and read
the other's — store buffering — and that outcome is forbidden only with a sequentially consistent
fence on *both* sides. `idle_workers_` was seq_cst, but the job becomes reachable through the
deque's acquire/release compare-exchange, so both threads could read stale: the submitter skips
the wake, the worker parks, and the job sits queued with nobody awake to take it. Fixed by adding
the two fences; the checker goes from a spinloop to 5 clean executions, and removing either fence
brings the spinloop back.

### One harness is a transcription, not an include

`job_system_no_lost_wakeup` is the exception to the rule above. The protocol lives in JobSystem's
worker loop, which spawns its own threads and blocks in `park_if_equal`, and the interpreter can
run neither libstdc++'s `std::thread` machinery nor `syscall(SYS_futex, ...)`. So the two sides
are transcribed with the same memory orders as `job_system.hpp`, and the park is transcribed as a
spin on the same condition the futex tests — `--check-liveness` reports a spin that can never
exit, which is exactly a worker asleep with a job queued.

Those memory orders are the entire content of the property. If they change in `job_system.hpp` and
not here, this harness verifies something the engine no longer does. Both sides carry a comment
pointing at the other.

The bound is part of the result. `concurrent_map_agreement` says nothing about three concurrent
inserters, about two different keys colliding in one probe run, or about a resize running
underneath an insert — that last one is a distinct algorithm (rehash, then CAS-install) and belongs
in its own harness rather than inflating this one's state space. `deque_no_double_extraction` says
nothing about a push running concurrently with a pop.

### Choose the configuration that actually contends

`deque_no_double_extraction` uses **one** item deliberately. With head=0 and tail=1, `pop_front`
resolves to slot 0 and `pop_back` to slot `(1-1)&mask` = 0 — the same slot. Push *two* items and
the two ends address slots 0 and 1, never meet, and the harness explores a handful of trivially
independent executions while appearing to test the race. Exhaustive exploration of the wrong
configuration is still the wrong configuration.

### Every harness is mutation-checked

A harness that cannot fail proves nothing, and an exhaustive checker reporting "no errors" is
exactly the output a vacuous harness produces. Each one here has been run with its central
assertion inverted, and the checker must report a safety violation:

| Harness | Inverted assertion | Result |
|---|---|---|
| `lock_free_list_completeness` | push's CAS retry loop replaced with a non-retrying publish | `Error: Safety violation!`, exit 42 |
| `quotient_instance_match_rendezvous` | either seq_cst fence removed | `Verification unsuccesful`, exit 42, 0 executions |
| `concurrent_map_double_growth_3t` | the shipped code before the conjunction fix (verdict from the exchange alone) | `Safety violation`, exit 42, at a 4-context bound and clean at 3 |
| `lock_free_list_pairs_meet_once` | `for_each_before(mine, ...)` replaced with `for_each(...)`, the walk from the head | `Verification unsuccesful`, exit 42 |
| `arena_worker_index_exclusive` | `acquire()`'s compare-exchange replaced with check-then-act | `Error: Safety violation!`, exit 42 |
| `concurrent_map_agreement` | both callers report `was_inserted` | `Error: Safety violation!`, exit 42 |
| `concurrent_map_resize` | both callers report `was_inserted` | `Error: Safety violation!`, exit 42 |
| `deque_no_double_extraction` | both consumers receive the item | `Error: Safety violation!`, exit 42 |
| `claim_match_rendezvous` | P1 inverted (both claimants win); P2 inverted (a colliding claim loses) | `Error: Safety violation!`, exit 42, both |

Do this for any harness added here, before believing its clean run.

## Adding a harness

Drop a `.cpp` in this directory; `run.sh all` picks it up. Include `genmc_support.hpp` first — it
defines `__dso_handle`, which clang emits for static-destructor registration and the interpreter
cannot otherwise resolve.

Put harness-specific GenMC flags in a `// GENMC-ARGS:` line in the source, so the bound a harness
needs is stated next to the property it bounds rather than in this script.

Useful flags while developing one: `--disable-estimation` skips the state-space estimate;
`--unroll=N` bounds loops; `--sc --bound=N --bound-type=context` bounds context switches (bounding
requires `--sc`).


## A property that is NOT model-checked here, and why

**Claim: at quiescence, `SegmentedArray::size()` bounds only PUBLISHED elements.**

This is the contract every enumeration in the engine rests on after 36cb9d08. Ids come from an
atomic increment taken before the element exists -- which is what makes them cheap, and why they
are deliberately not repeatable -- so the CLAIM counters behind `num_states()`, `num_edges()` and
`num_raw_events()` run ahead of publication, and an id claimed but never emplaced leaves them
permanently above what the array holds. Readers must use `num_published_*` instead. The defect was
real, reached the paclet's marshaller, and presented as an intermittent whole-suite failure about
one run in three when another process loaded the box.

**A harness for it was written and does not run.** GenMC v0.17.0 aborts on it with
`INTERNAL FAILURE: Internal check failed: size != 0` in `SAddrAllocator::allocate`, before
exploring any execution. This is not a property violation and not a harness bug in the usual sense:
the checker fails to model the structure at all. Established by ground-truthing rather than
assumed -- `run.sh key_set_exactly_once --mode=estimate` exits 0 on the same toolchain in the same
session, so the installation is sound and the failure is specific to `SegmentedArray`. Five
variants were tried: arena over static storage, arena over the modelled heap, one element per
segment, the array as a stack local, and the array in static storage. All abort identically.

**The cause is narrowed but NOT identified, and the first guess was wrong.** Bisected with
throwaway probes, each `run.sh <name> --mode=estimate`:

| probe | result |
|-------|--------|
| include `segmented_array.hpp`, instantiate nothing | exit 0 |
| `std::atomic<int*> a[4096]` with a constructor loop storing nullptr | **exit 0** |
| `std::atomic<int*> a[8]`, same shape | exit 0 |
| `hgcommon::ctz64(1024)` | exit 0 |
| construct `SegmentedArray<uint32_t>(1)`, no threads, no emplaces | **SEGFAULT (139)** |

So the 4,096-entry atomic table is NOT the cause -- a standalone table of exactly that size and
shape checks cleanly -- and neither is `ctz64`, which the constructor calls, nor including the
header. Constructing the object is what crashes the checker, with no threads and no operations.
An earlier version of this note named the table as the likely cause; that is refuted by the second
row above and the guess should not be inherited by whoever picks this up.

Worth noting for that person: the failure MODE differs between the full harness (an internal
check, `size != 0` in `SAddrAllocator::allocate`) and bare construction (a segfault), which
suggests more than one thing here is outside what the checker models.

**What stands in its place.** The fix itself, gated by `all_tests` 275/275 and `cost_matrix` 17/17
ALL EXACT across two consecutive whole-suite runs with the box deliberately loaded by another
tenant -- the condition under which the defect used to reproduce. That is sampling, and this file
argues at length that sampling can only fail to reproduce. The gap is stated rather than papered:
this property has a passing gate and no proof.

**To close it**, either the checker gains the ability to model an atomic array of that size, or
`MAX_SEGMENTS` becomes a template parameter so a harness can instantiate a two-segment array. The
second is a change to shipping code for the benefit of verification, which is a trade worth making
deliberately rather than incidentally.


## Why there is no harness for the DEVICE termination protocol

The persistent kernel's termination detection -- `mark_pushed` before a push, `mark_completed`
after, `rewrites_done` bumped last, and the detector block's quiescence snapshot -- is a
concurrent protocol of exactly the kind this directory exists to check. It has none, and cannot
have one here.

**Every method of `TerminationDetector::DeviceView` is `__device__`** (gpu/include/hg_gpu/
termination.hpp: `mark_pushed`, `mark_completed`, `exit_requested`, `snapshot_quiescent`,
`signal_exit`). GenMC is an LLVM tool that checks HOST C++ against RC11. It cannot compile a CUDA
device function, so the rule this directory is built on -- a harness includes the engine's own
header and calls its own functions, because a model drifts from the code and proves a property of
the model -- cannot be followed for any device-side structure.

Transcribing the protocol into host C++ would produce a harness that runs and proves nothing about
the code that ships. That is the failure mode `README` warns about at the top of this file, and it
is worse than an absent harness because it reads as coverage.

**What the device protocol has instead:** the `wait` phase counter, which measures time spent
spinning on a claimed-but-unpublished record, reads **0.0% on all eight corpus workloads**. That
is evidence the protocol is not livelocking in practice, from an instrument independent of the
argument. It is not a proof and is not offered as one.

**What would change this:** a checker that models the CUDA memory model over device code, or a
device protocol expressed in shared `HG_HD` code that both engines compile -- the same move that
`hgcommon/` already makes for the matcher and the IR core. The second is a design change to the
GPU termination path, not a verification task, and it should be weighed on its own merits rather
than undertaken to satisfy a harness.
