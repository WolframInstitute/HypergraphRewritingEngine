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

The `Broken` configurations are what make the rest evidence: a model that cannot report a
violation has not been shown to be able to detect one.

## What is NOT covered, and why each matters

**Termination detection.** The persistent kernel's `TerminationDetector` and the host's quiescence
protocol together decide when a run is FINISHED across many workers. That is the classic
distributed termination problem and nothing models it. A false positive ends a run early and
returns a smaller multiway system with no indication it is smaller -- the same failure the
container ceiling used to produce, and harder to see, because there is no warning attached to it.

**The depth-relaxation cascade.** `try_lower_explore_depth`, `propagate_explore_depth`,
`claim_canonical_for_expansion` and the budget frontier together are a shortest-path relaxation
racing a claim-that-happens-once. GenMC covers one edge of it, `depth_relax_child_registration`.
The cascade as a whole is not covered, and it is what decides WHICH STATES EXIST when the step
budget is below the closure depth -- a region where a nondeterminism defect has lived before.
