# Verification status

Two model checkers run against this tree, and what each covers is listed here so a gap is a
stated absence rather than an unexamined one.

- **GenMC** enumerates executions of the RC11 memory model for a bounded program. A harness under
  `verification/genmc/` includes the engine's own header and calls its own functions, so it breaks
  when the header breaks. `verification/genmc/run.sh <name>`. TWO DO NOT, and are listed as a gap
  below. A harness marked `// GENMC-LINK: engine` is instead compiled against every engine
  translation unit, linked, so it can call code whose body is in a `.cpp`.
- **GPUMC** is GenMC's scoped-RC11 sibling, for the GPU memory model: threads are organised into
  CTAs and every access carries a SCOPE, so whether two threads synchronise depends on how close
  they are. RC11 has no scopes, so GenMC would check a program the device does not run. It runs
  from a container -- it is a fork of GenMC 0.9 supporting LLVM up to 15, and this tree builds
  against 18. `verification/gpumc/run.sh <name>`. Two harnesses: the termination decision and the
  device work queue.
- **TLA+** models a protocol rather than a translation unit, which is what makes it the right tool
  where the property is about an ordering across many participants rather than about one
  structure's memory operations. `verification/tla/run.sh <config>`.

## What is covered

GenMC, 27 harnesses: the concurrent map (agreement, resize, double growth at two and three
threads, repeated offer, lookup during growth), the key set (exactly-once at two and three
threads, enumeration, contains and distinct keys across growth), both deques (no double
extraction, no double take, tag defeats ABA), the lock-free list (completeness, pairs and triples
meeting once), the job system's wake protocol (no lost wakeup, and the per-domain variant),
the arena's exclusive worker index, frame publication atomicity, depth-relax child registration,
the claim and quotient-instance rendezvous, and the depth join's report ordering.

`depth_report_order` runs `hgcommon/depth_join.hpp` itself. It is the reason that protocol was
lifted out of `ParallelEvolutionEngine`: it touches nothing but its own atomics, so the checker
can be handed the protocol rather than the program around it. It found two defects on its first
run -- a settle cascade with no barrier on either side, and a report order that a per-depth
cursor was not enough to fix. Its `CALIBRATE_REPORT_AFTER_SETTLE` arm reinstates the second and
must report a violation.

WHAT THE COMPOSED LINK REACHES, measured. Linking every engine translation unit and pruning to
what `main` reaches:

| harness | lines after prune | result |
|---|---|---|
| construct a `Hypergraph` | 5,452 | verifies, 118.7s |
| the same, with `HG_SEGMENTED_ARRAY_MAX_SEGMENTS=8` and `HG_CONCURRENT_MAP_INITIAL_CAPACITY=16` | 5,452 | verifies, 4.9s |
| add the engine and a rule | 19,477 | transforms, then the interpreter stops |
| reach `evolve()` | 110,932 | no verdict; transformation alone exceeds 540s |
| construct a `JobSystem` | 2,659 | verifies |
| `JobSystem::start()` | 8,952 | transforms, then GenMC segfaults |

The `JobSystem` row is the same story at a smaller scale: construction is checkable and starting
is not. Getting that far needed two more shims -- GenMC's address allocator refuses a zero-size
request, and libstdc++ declares the ABI's type-info vtables as `[0 x ptr]`, which every
polymorphic class references and `std::thread`'s state class drags in -- and the segfault survives
both those and `-fno-rtti`.

Two ceilings, and they are different. The first is SIZE: GenMC's own transformation phase, before
a single execution is explored, does not finish on the `evolve()` module. The mass is spread --
32% libstdc++, 19% `ConcurrentMap` instantiations -- so no single ablation moves it.

The second is a limitation of GenMC v0.17.0 rather than of the engine: it cannot materialise a
`thread_local` of AGGREGATE type. Measured on minimal programs -- `std::vector`, a seven-field
struct, and `unsigned char[64]` all fail with `Constant unimplemented for type`, while a
`thread_local` scalar or pointer works and a non-TLS aggregate global works. The engine has
fifteen aggregate `thread_local`s, so any harness reaching the evolution engine stops there. It
also hard-errors on any `memset`/`memcpy` whose destination is a heap pointer, because its
promotion pass accepts only Constant/Alloca/GEP; 104 of the 165 intrinsics in the `evolve()`
module are in that class, most of them inside libstdc++.

That is why the protocols are checked as UNITS rather than in situ, and why an extraction like
`hgcommon/depth_join.hpp` is what makes one checkable at all.

TLA+, 7 configurations, all matching their declared verdict:

| model | verdict | distinct states |
|---|---|---|
| `MCSegmentedArray` | PASS | 2,284 |
| `MCSegmentedArrayDeep` | PASS | 27,828,731 |
| `MCSegmentedArrayBroken` | VIOLATION as declared | 341 |
| `MCMatchForwarding` | PASS | 117,005 |
| `MCMatchForwardingEagerFix` | PASS | 479,005 |
| `MCMatchForwardingEagerBroken` | VIOLATION as declared | 305,493 |
| `MCMatchForwardingBatchedBroken` | PASS as declared | 85,777 |
| `MCDepthRelaxation` | PASS | 14 |
| `MCDepthRelaxationBroken` | VIOLATION as declared | 12 |
| `MCQuiescence` | PASS | 22 |
| `MCQuiescenceBroken` | PASS as declared | 22 |
| `MCQuiescenceLateSubmit` | VIOLATION as declared | 34 |

The `Broken` configurations are what make the rest evidence: a model that cannot report a
violation has not been shown to be able to detect one.

## What is NOT covered, and why each matters

~~**Two harnesses check a re-implementation, not the code.**~~ CLOSED. Both now drive
`hgcommon/park_gate.hpp`, which is the park/wake protocol lifted out of `JobSystem`'s worker loop
and `wake_one_worker`. They could not include the old shape for a real reason -- the protocol lived
inside a loop that spawns threads and blocks in a futex, and a JobSystem is not reachable under
GenMC (construction prunes to 2,659 lines and verifies; `start()` prunes to 8,952 and segfaults
v0.17.0) -- so the protocol is a unit and is checked as one.

Both are calibrated against the real path rather than a copy: `HG_PARK_GATE_WEAK_ORDERS` drops the
handshake to release/acquire and `HG_PARK_GATE_NO_REMOTE_SCAN` removes the cross-domain fallback,
and each makes the checker report a non-terminating spinloop -- a worker asleep with a job queued.

EVERY harness under `verification/genmc/` now includes the header it is about.

~~**The job system's completion handshake is barriered on one side.**~~ EXAMINED AND REFUTED.
The completer bumps `quiescence_seq_` (release) and then reads `completion_waiters_` (acquire),
which is not a StoreLoad, and each side writes one location and reads the other -- the shape of
the class. It is not an instance of it, because the two reads do not gate the same thing: missing
the waiter count skips a WAKE, while missing the sequence loses nothing, since the waiter parks on
that very word under a value compare and the write it might have missed is the value it compares
against. What makes it safe is the ORDER -- publish the sequence, then look for a waiter -- and
inverting that is what would lose the wakeup.

Checked rather than argued: the protocol was extracted and run under GenMC with and without the
barrier, and the checker cannot tell the two arms apart. The extraction was then deleted, because
it corrected nothing. What survives is in `job_system.hpp`, where the comment that used to defend
this ordering by a timeout the waiter does not have now states the argument that holds.

The general lesson is recorded in `hgcommon/rendezvous.hpp`: the test is not whether both sides
read, but whether missing the read LOSES THE EVENT with no other path to it.

**`wait_for_completion_with_abort` is dead in the parallel path.** Its only caller is its own
test, which runs it with `serial=true` and so exercises the other branch. The parallel branch
parks with no timeout, so an abort that becomes true while parked is not noticed until something
else moves the sequence.

~~**Termination detection.**~~ COVERED for the HOST by `Quiescence.tla`. The checker reads its two
halves in SEPARATE steps with workers running in between, because TLA+ evaluates a conjunction
atomically and a single-step predicate cannot express the race at all.

What the three cells establish together: the COUNTERS are load-bearing and the queue scan is
defence in depth -- omitting `jobs_executing` still passes, because a worker inside a job has not
completed and so `submitted` and `completed` cannot agree while it runs. And the soundness rests
on a precondition: A JOB SUBMITS ITS CHILDREN BEFORE IT RETURNS. Complete-then-submit opens a
window where the counters agree and every queue is empty while a child is still owed, and no
ordering of the reads defends it, because at that instant there is nothing to see.
`MCQuiescenceLateSubmit` is that defect, and TLC reports it.

~~STILL OPEN: the DEVICE's `TerminationDetector`.~~ COVERED, under scoped-RC11, by
`verification/gpumc/termination_no_early_exit.cpp`. The decision the persistent kernel's detector
makes is `hgcommon/termination_core.hpp` and the harness runs THAT -- the same body both device
detectors drive, not a model of it. It was written twice in `gpu/src/persistent.cu` before this,
once per kernel, which is why it was neither shared nor checkable.

The property: if the detector signals exit through the QUIESCENT path, every piece of work the run
will ever complete has completed. A stall exit is a recorded defect returning partial work
deliberately and claims nothing. 2,265 executions, clean.

Calibrated by `-DCALIBRATE_COMPLETE_THEN_PUSH`, which books a worker complete BEFORE announcing
the child it owes -- the same precondition the host's quiescence rests on, and the one
`MCQuiescenceLateSubmit` reports there. The checker finds the early exit.

Two things about the harness are worth keeping, because both made it report the defect arm as
clean at first: the property must be asserted AT THE INSTANT OF THE DECISION, since after the
threads are joined the worker has always finished; and it must be stated in the COUNTERS rather
than in a "finished" flag, since a flag is set after the last counter write and so lags a state
that is genuinely complete.

**The DEVICE's work queue**, covered under scoped-RC11 by
`verification/gpumc/ring_exactly_once.cpp`. The claim rule is `hgcommon/ring_core.hpp` and the
harness runs THAT -- the same `ring_claim` body `gpu/include/hg_gpu/ring_buffer.hpp` drives for
both of its roles. Producing and consuming are the same rule with two constants (a producer waits
for `seq == pos` and leaves `pos + 1`; a consumer waits for `pos + 1` and leaves `pos + capacity`),
so they are one body rather than two that agree until one is edited.

The property: no item is handed to two consumers, and none is handed out that no producer
published. For this queue that is a TERMINATION property rather than a throughput one -- the
persistent kernel's producers are its own consumers, so an item that vanishes is a completion that
can never be booked, and the detector above then waits for it forever. 8 complete executions, 14
blocked on the retry loop, clean.

Calibrated by `-DCALIBRATE_BUMP_CURSOR`, which reserves a position with an unconditional
`fetch_add` instead of a compare-exchange. The cursor then hands out a position whose slot is not
yet the reserver's, and with producers that are also consumers there is nothing to roll back with.
The checker reports one item handed to two consumers.

The reservation CAS is modelled WEAK, as the device writes it. Modelling it strong would remove
the spurious-failure retries, and removing behaviours from a checker is the unsound direction.

~~**The depth-relaxation cascade.**~~ COVERED by `DepthRelaxation.tla`. The property is that at
quiescence the claimed set is exactly the nodes whose SHORTEST-PATH depth is below the budget, so
a truncated run returns the same subset every time rather than one decided by the order paths were
found. `MCDepthRelaxationBroken` derives a child's depth from the depth its parent carried WHEN
CLAIMED instead of from its live minimum, which freezes an early long path into every descendant,
and TLC reports the violation -- so the model is a gate rather than decoration.

The graph is five nodes and the state counts are 14 and 12, which is small. It is sized to the
shape that makes relaxation matter -- a node reachable both directly and through a longer path,
with a descendant whose place under the budget depends on that lowering arriving -- rather than
to breadth.

`MCMatchForwardingBatchedBroken` is named for the code it models, not for its verdict: the
ownership defect IS present there and the batched gate masks it, which is why the eager variant
is the one that reports a violation.
