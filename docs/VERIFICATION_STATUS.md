# Verification status

Two model checkers run against this tree, and what each covers is listed here so a gap is a
stated absence rather than an unexamined one.

- **GenMC** enumerates executions of the RC11 memory model for a bounded program. A harness under
  `verification/genmc/` includes the engine's own header and calls its own functions, so it breaks
  when the header breaks. `verification/genmc/run.sh <name>`. TWO DO NOT, and are listed as a gap
  below. A harness marked `// GENMC-LINK: engine` is instead compiled against every engine
  translation unit, linked, so it can call code whose body is in a `.cpp`.
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

**Two harnesses check a re-implementation, not the code.** `job_system_no_lost_wakeup.cpp` and
`job_system_no_lost_wakeup_domains.cpp` include nothing from `job_system/`: they re-state the
wake protocol and check the re-statement, which is evidence about the re-statement. Every other
harness includes the header it is about.

**The job system's completion handshake is barriered on one side.** The completer bumps
`quiescence_seq_` (release) and then reads `completion_waiters_` (acquire), which is not a
StoreLoad; the waiter's own registration is `seq_cst` and so is barriered. Each side writes one
location and reads the other, so the completer may see no waiter and skip the wake while the
waiter sees the old sequence and parks. The comment defending it -- that the waiter also polls on
a timeout -- is contradicted by `wait_for_completion`'s own "No timeout and no polling", so the
argument rests entirely on `park_if_equal`'s compare. No harness covers it: the two that name the
job system check a re-implementation, and neither is about the completion half.

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

STILL OPEN: the DEVICE's `TerminationDetector`. This models the host's protocol; the persistent
kernel's detector is a separate mechanism and is not covered.

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
