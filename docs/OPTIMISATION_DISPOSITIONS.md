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

## Monotonicity: every violation is cross-L3 placement, not the engine

Of 97 generated workloads swept at 1/2/4/8/16/32 workers on the EPYC 9174F, 17 are not monotonic.
Eight of those dip only between 16 and 32 workers, which is simultaneous multithreading on a
16-core part and is the flat-line case rather than a regression. Four are BELOW ONE at two
workers, which is the case that matters.

All four are small, and all four disappear on a machine whose cores share one last-level cache:

| workload | EPYC 9174F, 8 L3 instances of 2 cores | i9-14900K, unified L3 |
|---|---|---|
| `path-l1a2g1r1` | 0.91 | 1.78 |
| `path-l1a2g1r2` | 0.96 | 1.67 |
| `path-l2a2g1r2` | 0.97 | 1.66 |
| `star-l1a2g1r1` | 0.73 | 1.43 |

The mechanism is already measured in this repository: two workers sharing one L3 instance cost
nothing, two on different instances cost 2.7x. Worker threads are NOT pinned by default --
`job_system.hpp` says why, that a binding-derived grouping describes where a thread was rather
than where it is -- so which L3 instances two workers land on is the operating system's choice,
and on a part with eight two-core instances the likely choice is two different ones.

CLOSED BY PLACEMENT. Workers now fill cache domains in order rather than being left to the
scheduler, so the second worker shares the first's cache instead of racing it. Every one of the
four is now faster with two workers than with one, and the large end gains as well:

| workload | 2 workers | 4 workers |
|---|---|---|
| `path-l1a2g1r1` | 0.91 -> 1.69 | 1.04 -> 2.05 |
| `path-l1a2g1r2` | 0.96 -> 1.70 | 1.14 -> 2.19 |
| `path-l2a2g1r2` | 0.97 -> 1.66 | 1.28 -> 2.18 |
| `star-l1a2g1r1` | 0.73 -> 1.50 | 0.85 -> 1.79 |
| `disc-l2amg2r2` | -- | 13.47x -> 18.19x at 32 workers |

Canonical counts are unchanged everywhere, so this moves time and not answers.

The reason it had to be found rather than read off: `performance_cpus()` names the fast cores of a
HETEROGENEOUS part and returns EMPTY on a homogeneous one -- zero cpus on this EPYC -- so a
default built on it fell through its first guard silently. Empty means no core is PREFERABLE, not
that none is usable.

## Compiler-level levers, all three measured

The shipping build is `-O3 -DNDEBUG` with no architecture flag. Three levers were untested; each
was built and measured on the same box against the same three workloads, one worker, median of 3.

| lever | `path-l2a2g1r1` d5 | `disc-l2amg2r2` d4 | `star-l1a2g2r1` d5 |
|---|---|---|---|
| `-march=native -mtune=native` | 0.971x | 0.939x | 0.962x |
| link-time optimisation | 1.019x | 0.996x | -- |
| profile-guided optimisation | 1.004x | 1.019x | 1.036x |

`-march=native` is REFUTED and it is not marginal: it is 3% to 6% SLOWER on all three. The IR
loops are branchy and comparison-heavy rather than vectorizable, so the wider ISA buys nothing and
is paid for anyway. It would also have been wrong to ship, since the artifacts are built once and
run on machines that are not the builder.

LTO is REFUTED as neutral -- 1.019x and 0.996x is the box's own drift.

PGO is the only one that helps, consistently but slightly: +0.4%, +1.9%, +3.6%, trained on four
corpus workloads at depths three and four and measured at four and five. It is MEASURED AND NOT
ADOPTED, and the reason is a cost rather than a doubt: it makes every shipped artifact a two-pass
build with a training run in between, and the release ships fourteen of them across six platforms.
The number is recorded here so the trade is a decision rather than an omission.

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

**The device's idle is available parallelism, not imbalance.** Idle against workload size, same
instrument, three workloads spanning an order of magnitude:

| workload | states | canonicalization | idle |
|---|---|---|---|
| `disc-l2amg2r2` d4 | 18,206 | 77.4% | 22.0% |
| `star-l1a2g2r1` d5 | 5,019 | 32.0% | 67.6% |
| `path-l2a2g1r1` d6 | 1,161 | 15.7% | 84.2% |

Idle falls monotonically as the state count rises, which is what running out of concurrent work
looks like and is not what imbalance looks like -- imbalance would persist at the large end. A
thousand states cannot fill a 4090 whatever the scheduler does, and 22% on the largest workload
is the FLOOR this measurement reaches rather than a defect sitting on top of it. REFUTED as a
scheduling target.

## The default exploration path has a different profile from the one measured

Everything under "Where the time actually goes" was measured with quotient exploration on. That is
not the engine's default -- ParallelEvolutionEngine leaves explore_from_canonical_states_only
false, and full multiway expands every raw state -- so those shares describe a mode the caller has
to ask for. On wpp at depth 6 the two are different workloads: quotient explores 3,867 raw states,
full multiway explores 15,967.

Callgrind, one worker, full multiway, wpp depth 6, 1.43G instructions, inclusive:

| subtree | inclusive |
|---|---|
| the expand task body | 82.6% |
| `execute_rewrite_task` | 29.9% |
| `Rewriter::apply` | 28.8% |
| `create_or_get_canonical_state` | 16.7% |
| `compute_exact_canonical_hash` | 16.2% |

CANONICALIZATION IS 16.2% HERE, against the 58% to 92% recorded above for the corpus workloads
under quotient. The rewrite subtree costs more than it does. So "both engines are
canonicalization-bound" is a statement about the workloads and the mode it was measured in, and
the lever it points at is not the lever on the default path with this rule.

A cycles profile at 32 workers on the quotient path agrees that the mode matters: 14% of the run
is ConcurrentKeySet::insert, and 8.57 of those 14 points are under qc_add_producer -- the
quotient-causal producer-set dedup, which does not execute at all on the default path.

MEASURE THE MODE YOU MEAN TO OPTIMISE. A profile of one is not a profile of the other.

## CLOSED: arena blocks are one huge page

A cycles profile at 32 workers put 9.3% of the run inside the kernel -- 5.99%
`native_queued_spin_lock_slowpath`, 1.70% `down_read_trylock`, 1.65% `clear_page_erms`. That is
the page-fault path: 1,060,520 minor faults on wpp depth 7, which is 4.3 GB arriving 4 KB at a
time, with 32 threads meeting on `mmap_sem`.

Blocks were 1 MB from `operator new`, aligned to nothing, and transparent huge pages run in
`madvise` mode on this box and most distributions -- so an unrequested mapping gets 4 KB pages
however large it is. Both halves were missing: the block is now exactly one 2 MB huge page,
allocated 2 MB aligned, and advised.

| threads | before | after | change |
|---|---|---|---|
| 1 | 3179.3 ms | 2747.2 ms | -13.6% |
| 2 | 1489.1 | 1320.9 | -11.3% |
| 4 | 859.1 | 772.3 | -10.1% |
| 8 | 499.3 | 455.8 | -8.7% |
| 16 | 293.2 | 268.3 | -8.5% |
| 32 | 184.0 | 174.5 | -5.2% |

Minor faults 1,060,520 -> 175,725. Peak resident set 1,313,488 KB -> 1,309,336 KB, so the
footprint did not grow: a 2 MB block carries one header where two 1 MB blocks carried two, and the
alignment slack is usable space. `native_queued_spin_lock_slowpath` and `down_read_trylock` left
the profile entirely.

## REFUTED: spreading workers across cache domains to get more physical cores

The EPYC 9174F is 16 cores over 8 L3 instances, so a domain holds 2 physical cores and 4 logical
CPUs. Domain-major placement therefore puts four workers on CPUs 0,1,16,17 -- two physical cores
and their SMT siblings -- and the obvious reading is that it is leaving two cores idle. Efficiency
does dip there: 0.67 at four workers against 0.80 at sixteen, on wpp depth 7, full multiway.

Measured, wpp depth 6, four workers, medians of five:

| CPUs | physical cores | L3 domains | median |
|---|---|---|---|
| 0,1,16,17 | 2 | 1 | 60.9 ms |
| 0,1,2,3 | 4 | 2 | 73.9 ms |
| 0,2,4,6 | 4 | 4 | 87.3 ms |

Sharing one L3 beats having twice the physical cores, and the penalty grows with the number of
domains spanned. The placement is already right and the dip is a property of the part -- a domain
has two cores, so four workers inside one cannot have four. Four workers on two cores reaching
2.69x is SMT and cache locality doing better than the core count suggests, not worse.

## OPEN: parallelize individualization-refinement WITHIN a state

Both engines run IR one state at a time and parallelize ACROSS states -- the host by giving each
worker whole states, the device by `k_exact_hash_range`, which is a grid-stride loop assigning one
THREAD per state. The refinement itself is serial in that thread.

That is the largest identified win in the codebase and it is open rather than refuted, so it is
stated with what it would attack:

- Canonicalization is 79.0% of device cycles and 58% to 92% of host instructions.
- On the device, 32 lanes of a warp each run an INDEPENDENT search on a different state. Those
  searches differ in length by more than a factor of two -- 16 refinements per state on a
  low-symmetry workload against 45 on a high-symmetry one -- so a warp runs at its slowest lane
  and the lanes that finish early stall. That divergence is inside the warp and the block-level
  idle counter cannot see it: it reports 22.0% idle on the workload where canonicalization is
  77.4% of cycles.
- Refinement is data-parallel over cells and over the vertices in a cell, so a warp cooperating on
  ONE state is the shape that removes the divergence rather than tolerating it.
- The device's IR is already known to be far slower per state than the host's: an isolated
  measurement recorded at the call site in `persistent.cu` puts device IR at 62.9x the host on
  one state. Combined with IR being 72.8% of device cycles, that is where the device's time goes.

MEASURED AND REFUTED ON THE WAY. The block shape is not the lever: `kMatchBlockThreads` is 32
because the MATCHER stripes across the block, and match is 0.0% of cycles, so the shape is set by
a phase that costs nothing. The matcher stripes on `blockDim.x`, so the constant can move -- at
128 the state counts are identical and the run is slower, `disc-l2amg2r2` 144.6 ms to 315.2 ms and
`star-l1a2g2r1` 36.2 ms to 45.6 ms. One warp per block is already right.

WHAT THE CHANGE ACTUALLY IS, from the code rather than from estimate. Each thread calls
`state_key_device` on its OWN child state (`persistent.cu`, the worker loop). Cooperation needs
two things together: a `__shfl_sync` loop so the warp takes one lane's state at a time, AND
lane-strided inner loops inside `ir_refine`. Without the second, the first makes the run ~32x
slower rather than faster, because 31 lanes idle while one works. `ir_refine`'s inner loops carry
sequential accumulators -- the incident-edge count, the epoch stamping, the per-vertex signature
prefix sum, and a heapsort -- so each needs a warp scan or ballot to split, in `HG_HD` code that
is compiled for both engines.

WHY IT IS NOT LANDED HERE. `ir_core.hpp` is `HG_HD`: the same code runs on host and device, so a
warp-cooperative refinement changes both engines at once, and the determinism contract -- the
canonical form must be a function of the state alone -- has to be re-established for both before
any measurement in this file or in the paper can be trusted again.

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
