# Verification status

Two model checkers run against this tree, and what each covers is listed here so a gap is a
stated absence rather than an unexamined one.

- **GenMC** enumerates executions of the RC11 memory model for a bounded program. Every harness
  under `verification/genmc/` includes the engine's own header and calls its own functions, so a
  harness breaks when the header breaks. `verification/genmc/run.sh <name>`.
- **TLA+** models a protocol rather than a translation unit, which is what makes it the right tool
  where the property is about an ordering across many participants rather than about one
  structure's memory operations. `verification/tla/run.sh <config>`.

## What is covered

GenMC, 25 harnesses: the concurrent map (agreement, resize, double growth at two and three
threads, repeated offer, lookup during growth), the key set (exactly-once at two and three
threads, enumeration, contains and distinct keys across growth), both deques (no double
extraction, no double take, tag defeats ABA), the lock-free list (completeness, pairs and triples
meeting once), the job system's wake protocol (no lost wakeup, and the per-domain variant),
the arena's exclusive worker index, frame publication atomicity, depth-relax child registration,
and the claim and quotient-instance rendezvous.

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
