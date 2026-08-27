# Optimisation dispositions

Every optimisation considered for v1.0.0 is CLOSED (landed, gated, measured before and after) or
REFUTED (measured, and the measurement is here so it is not retried). "Not tried" is not a
disposal, and neither is "would probably not help".

Nothing in this file appears in the paper. The paper describes the system as it is.

Instruments: `callgrind` for instruction counts and call counts, because wall clock on the
development box drifts more than 10% between runs; `bench_cpu_evolve` and `bench_gpu_evolve`
medians for wall time, on the same box, quiet.

## Where the time actually goes

Callgrind, one worker, `Full` state canonicalization, two workloads chosen at opposite ends of
the symmetry range:

| workload | `ir_refine` share | `ir_refine` per state | states |
|---|---|---|---|
| `path-l2a2g1r1` d5 (low symmetry) | 39.2% | 16.3 | 393 |
| `disc-l2amg2r2` d4 (high symmetry) | 67.2% | 45.4 | 18,206 |

Including `ir_canonical_hash`, individualization-refinement is **92.6%** of all instructions on
the high-symmetry workload. Everything else -- matching, the replay, the job system -- is the
remaining 7%. So the only optimisation target that can move the total is IR, and the call counts
say the cost is the SEARCH (16-45 refinements per state), not the one refinement that reaches the
equitable partition.

`ir_hash_and_orbits` is called once per raw state, so the engine already calls IR the minimum
number of times its dedup requires.

## IR: refuted

**Automorphism generator budget.** Already tuned, and the measurement is recorded at
`IR_HOST_GENERATORS` in `common/include/hgcommon/ir_core.hpp`: on a state of 30 isomorphic
components a budget of 64 does not finish, 512 completes in 5.4 s, and 512 beats the
unbounded-generator implementation's 6.9 s on the same state. Not an open lever.

**Target cell selection: smallest non-singleton instead of lowest-id.** The branching factor of a
search node is the size of the cell being individualized, so taking the smallest cell should
branch least. Implemented, `all_tests` 302/302, canonical counts identical (84 / 3562 / 2062, so
isomorphism invariance held). Wall time, one worker, median of 3:

| workload | lowest-id | smallest | delta |
|---|---|---|---|
| `path-l2a2g1r1` d5 | 15.992 ms | 16.226 ms | +1.5% |
| `disc-l2amg2r2` d4 | 3261.9 ms | 3221.5 ms | -1.2% |
| `star-l1a2g2r1` d5 | 206.75 ms | 204.78 ms | -1.0% |

Every delta is inside the box's run-to-run drift. REFUTED and reverted rather than kept as
neutral churn. The reason it is neutral is worth keeping: the orbit pruning already collapses the
branching to O(orbits), so the SIZE of the target cell is not the binding constraint -- the orbit
structure is.

**Incremental IR (warm-start refinement from the parent's partition plus the delta).** Bounded by
the call counts rather than by opinion: `ir_refine` runs 45 times per state on `disc-l2amg2r2`,
and warm-starting can only remove the FIRST of them, the one that reaches the equitable partition
from the initial colouring. The other 44 are inside the individualization search, where the
partition being refined is the parent search node's, not the parent STATE's. So the ceiling on
this optimisation is 1/45 of IR work, about 2.2% of the run. REFUTED by measurement.

**Tiered canonicalization (WL bucket, IR only on collision).** Implemented and correct in an
earlier cycle; measured at +28% pessimization, because duplicates still need IR to confirm they
are duplicates. REFUTED, do not retry.

**What remains, and it is not a shortfall of effort.** Reducing the search tree further means
stronger stabilizer/orbit pruning -- the nauty-class problem. Its headroom is bounded by the orbit
structure of the states themselves, and it is a research problem rather than an implementation
one. The corpus workloads that provoke it are those whose growth adds isomorphic copies, which is
the hardest case for any canonical labelling.

## GPU: closed

**Per-call engine allocation.** `evolve()` carries a fixed floor of roughly 70 ms on every
workload regardless of size -- visible as a near-constant column across six workloads spanning
2 ms to 213 ms of CPU work. `PersistentEvolver` removes it and is the path the worker uses.
Measured on the same six: 6.9 / 9.5 / 12.3 / 15.2 / 36.2 ms against 69-101 ms. CLOSED.

**Where each engine wins.** `disc-l2amg2r2` d4, states 18,206 on both engines:

| | median |
|---|---|
| CPU, 1 worker | 3221 ms |
| CPU, 16 workers | 253.6 ms |
| GPU, PersistentEvolver | 144.6 ms |

The GPU is 1.75x the 16-worker CPU and 22x one worker. The residual GPU floor is about 7 ms of
launch and synchronization, which is why the CPU wins below roughly 10 ms of work.

## The parallel overhead is not work

Callgrind, `path-l2a2g1r1` at depth five, total instructions with `--separate-threads=no`:

| workers | instructions |
|---|---|
| 1 | 196,868,220 |
| 8 | 197,289,633 |

Eight times the workers costs 0.21% more instructions. There is no spinning, no retry storm and
no duplicated computation to remove: whatever parallel efficiency is lost is lost to STALLS --
memory and coherence -- and not to work the engine could stop doing. A single-worker profile
cannot see that distinction, which is why it is measured here rather than assumed either way.

## The device spends its time where the host does

`HG_GPU_DBG_TIME=1`, `disc-l2amg2r2` at depth four, persistent evolver, cycles by phase:

| phase | share |
|---|---|
| canonicalization | 79.0% |
| idle | 20.3% |
| rewrite | 0.6% |
| match | 0.0% |
| wait | 0.0% |

BOTH ENGINES ARE CANONICALIZATION-BOUND, and by the same margin -- 79% of device cycles against
58% to 92% of host instructions depending on the state's symmetry. Matching is free on the device
and the rewrite is 0.6%, so the internal split of the rewrite (branchial 53%, emit 25%) is three
tenths of one percent of the run and is not a target.

`nsys` cannot see this: the persistent evolver is one kernel holding 99.9% of kernel time across
13 launches, so kernel-granularity profiling reports the kernel and stops. The in-engine phase
counters are the instrument that resolves it.

OPEN, AND NAMED RATHER THAN LEFT IMPLICIT. The device idles 20.3% of its cycles, which is load
imbalance in the persistent kernel rather than canonicalization, and it is the one GPU-specific
figure that is neither closed nor refuted here.

## What individualization-refinement would take

Every cheap lever above is refuted with its measurement. The remaining one is the trick a
branch-and-bound canonical labelling uses: compare a node's PARTIAL certificate against the best
complete one and abandon the branch as soon as it cannot win.

It does not fit this certificate. `ir_build_form` sorts every edge by its relabeled tuple, so the
form exists only once the partition is discrete -- at a leaf. Pruning during descent needs an
incremental node invariant that is comparable prefix-wise, which is a redesign of the certificate
in `ir_core.hpp`, and that file is `HG_HD`: it is the same code on both engines, so the change
lands on the host and the device together and the determinism contract has to be re-established
for both.

## Determinism, and what it cost to find

Not an optimisation, recorded because it shaped every measurement above: the rule submission order
was permuted from `std::random_device` on every run, unguarded, so a run that discarded work was
not reproducible. Fixed, and gated by asserting the ORDER rather than the counts -- a gate on
counts passes with the defect reintroduced, because with nothing dropping work the order changes
no count.
