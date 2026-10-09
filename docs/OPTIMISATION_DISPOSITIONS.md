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

## rc2 pass: thread scaling and the quotient replay (2026-10-08/09)

Instruments: `bench_cpu_evolve` (clang -O3 `build_prof` for A/Bs, gcc -O3 `build_linux` for the
shipping-compiler sweep), `perf record` at 16 and 32 threads to find the cause, callgrind for
one-thread instruction counts, `bench_gpu_evolve` (persistent evolver) with the RTX 4090 idle.
i9-14900K: CPUs 0-15 are the 8 P-cores with their SMT siblings, 16-31 the E-cores. Medians of
3 to 7 runs, box quiet.

### Where the time went at the start (HEAD 9ec5a313, gcc, one thread)

| mode | largest shares (callgrind, inclusive) |
|---|---|
| full multiway (wpp d7, cycle4 d6, multirule d6, wolftri d6) | create_or_get_canonical_state 47-53%, IR 19-33%, same_tokens 9-13% |
| quotient (wpp d8, multirule d7, bigpath n128 d3, growshrink3 d6) | the replay (qr_apply, qc_add_instance, qc_capture_expansion) 60-98%; IR 4-38% |

Thread-scaling defects at the start: full multiway was slower at 32 threads than at 16 (wpp d7
91.7 -> 106.7 ms, multirule d6 35.5 -> 44.2, allfour d5 33.4 -> 36.8); multirule d7 quotient
stopped improving at 8 threads (50.6, 45.2, 60.7 ms at 8, 16, 32); bigpath n128 d3 quotient was
fastest at 8 threads (55.1 ms) and took 101.2 ms at 32.

### CLOSED

| commit | change | measured cause | result |
|---|---|---|---|
| 813f22c8 | SegmentedArray: one elected thread creates the next segment from the second half | 46% of 16-thread samples in memset of segments that lost the install CAS (bigpath q) | bigpath q 16t 43.6 -> 21.8 ms, 32t 53.8 -> 30.7 |
| ec91d506 | claim bits as a chain of blocks (hgcommon::qr_claim_chain, host and device) | instances made before their class's matches claimed in a shared key set: 1,065,597 inserts at 16 threads against 33,722 at 1 (bigpath q) | inserts 766; bigpath q 4t 65.0 -> 36.8 ms; one thread +1.2% instructions; device within spread |
| fefd1f93 | per-worker causal-pair list heads on their own lines | 44% of samples in the push's CAS retry loop (multirule d7 q, 16t) | multirule d7 q 4t 77.8 -> 55.0 ms, 16t 41.4 -> 30.1 |
| 1f24972a | delete matched_raw_states_ | a set every insert won, 23% of 32-thread samples (wpp d7 full) | wpp d7 full 32t 108.2 -> 76.7 ms; multirule d6 full 32t 38.0 -> 22.0 |
| c2749671 | a branchial pair is recorded once by the second event to push into the bucket of the lowest shared edge, without a shared set | record_branchial_overlaps 18% and its set 11% (wpp d7 full, 16t) | wpp d7 full 1t 568 -> 482 ms, 32t 77.3 -> 57.3; arena -1.3% |
| 3d307cc9 | a key-set insert claims at the head it read when another thread holds the growth ticket | inserters looped on the growth trigger: 17% of 32-thread samples | wpp d7 full 32t 62.0 -> 47.2 ms |
| 71d8693f | GPU: a counts-only quotient run reads no relation back; counts come from the replay's counters | 122 ms of a 295 ms call building causal pair vectors nobody asked for (multirule d7 q) | GPU wpp d8 q 668 -> 266 ms, multirule d7 q 302 -> 206, growshrink3 d6 q 58 -> 36 |
| 80a09c28 | the replay's id counters each on their own line | 31% of qc_add_instance samples on the load of qc_inst_blocks_, which shared a line with qc_next_instance_ | multirule d7 q 16t 27.6 -> 26.2 ms; wpp d8 q 32t 231 -> 220 |
| b591bd1d | with several workers, qc_event_sig_ and qc_kept_ create the next segment from the first element | 26 losing 1M-entry segments per run at 32 threads (temporary counter) | multirule d7 q 32t 35.9 -> 22.7 ms; bigpath q 32t 35.9 -> 17.8; 2-8 threads bigpath +1 ms |

| aa6287a8 | an instance point installs its eight padded shard lists only when a push loses the compare-and-swap on `first` | 528 B per (class, depth) point, most holding one or two instances | wpp d8 q arena 2,658 -> 2,509 MB, peak RSS 2,910 -> 2,752 MB; allfour d6 q RSS 1,991 -> 1,887 MB; time within spread |
| 40aaf6b8 | an instance finds its point through the match that made it (child_point, written once) | the point's keyed claim on every instance: 4.5% of instructions (multirule d7 q) | instructions multirule d7 q -7.7%, bigpath q -9.8%; wpp d8 q 1t 2366 -> 2216 ms, 16t 291 -> 264 |
| e1d71d2a | a replay event's slot is its id (the stride-171 permutation removed) | with per-worker id blocks the permutation put different workers' slots on one line: 15% of qr_apply's 16-thread samples on an applied-list head | multirule d7 q 8t 29.6 -> 25.3 ms; bigpath q 16t 18.3 -> 14.8 |
| ad360afd | the early segment creation of b591bd1d only above eight workers | the early segment costs one zero-fill; off is faster at 2-8 threads | bigpath q 8t 15.8 -> 14.6 ms; 16/32 unchanged |
| e1d66677 | a worker appends its replay causal pairs to its own chunks (no node, no compare-and-swap) | qc_record_causal 10% of one-thread instructions (multirule d7 q) | multirule d7 q 1t 114.9 -> 105.5 ms; arena 443 -> 423 MB |
| 263c781d | worker threads and hg_evolve's main thread opt out of Windows power throttling (EcoQoS) | Windows ran every evolve() after the first in a process on efficiency cores at reduced clock: bench wpp d8 q one thread 2,640 then 4,029 / 4,347 ms | 2,655 / 2,474 / 2,481 ms; serve jobs 3,272 then 5,430-5,748 ms -> 3,098-3,189 ms |
| aad7132d | a paclet reply builds the reconstruction's event-identity map only when it lists events or graphs (and an unread content map is deleted) | two std::unordered_maps over all 6.66M replayed applications for every reply, counts-only included | serve job wpp d8 q counts-only: Linux 1,028 -> 306 ms, Windows 3,135 -> 250 ms |
| bbc91dd8 | replies move their WXF trees at last use instead of deep-copying them | the graph marshaller copied each property's lists into GraphData and again into the return value; replies copied every edge list | graph reply (wpp d7 q, StatesGraph+CausalGraph, 188 MB) 2,009 -> 1,422 ms; edge lists reply 846 -> 755 ms |
| 8aefa431 | the WXF writer appends strings, symbols and binary strings in one insert | one push_back per byte: write_string 6.6% and write_byte 5.0% of a graph reply's main thread | graph reply 1,451 -> 1,363 ms; edge lists reply 849 -> 737 ms |
| 544d81cc | arena Block::data aligned to 64 bytes (a defect found on Windows, not a speed change) | alignas(std::max_align_t) is 8 under MSVC: alignas(64) arena objects sat 8 bytes off a line, and after aa6287a8 MSVC's aligned stores faulted on every quotient replay run | Windows quotient runs complete; sizeof(Block) 56 -> 64 |

After (gcc, `build_linux`, median of 3, ms; before aa6287a8):

| workload | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| wpp d8 quotient | 2467 | 1343 | 736 | 421 | 292 | 207 |
| wpp d7 full | 514 | 285 | 157 | 89 | 60 | 49 |
| multirule d6 full | 192 | 108 | 62 | 36 | 25 | 21 |
| allfour d6 quotient | 1731 | 975 | 529 | 313 | 202 | 171 |
| allfour d5 full | 210 | 118 | 66 | 38 | 26 | 22 |
| wolftri d6 full | 191 | 106 | 60 | 34 | 22 | 20 |
| bigcycle n64 d3 full | 958 | 512 | 269 | 148 | 98 | 76 |
| growshrink3 d6 quotient | 304 | 171 | 94 | 57 | 35 | 26 |
| cycle4 d6 full | 121 | 64 | 35 | 20 | 15 | 15 |

At HEAD e1d66677 (gcc, `build_linux`, median of 5, ms):

| workload | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| wpp d8 quotient | 2258 | 1234 | 667 | 376 | 246 | 197 |
| wpp d7 full | 517 | 290 | 160 | 88 | 60 | 46 |
| multirule d7 quotient | 108 | 62 | 36 | 21 | 20 | 24 (min 19) |
| multirule d6 full | 188 | 109 | 62 | 36 | 25 | 22 |
| allfour d6 quotient | 1634 | 916 | 508 | 276 | 192 | 152 |
| allfour d5 full | 214 | 122 | 67 | 38 | 26 | 22 |
| wolftri d6 full | 197 | 106 | 59 | 33 | 23 | 22 |
| bigpath n128 d3 quotient | 76 | 43 | 25 | 14 | 16 | 17 |
| bigcycle n64 d3 full | 964 | 517 | 271 | 147 | 97 | 75 |
| growshrink3 d6 quotient | 296 | 163 | 87 | 48 | 31 | 26 |
| cycle4 d6 full | 115 | 65 | 36 | 20 | 14 | 14 |

Against the start (gcc, HEAD 9ec5a313): wpp d8 quotient 2763 -> 2258 ms at one thread and 364 ->
197 at 32; wpp d7 full 106.7 -> 46.2 at 32; multirule d7 quotient 137 -> 108 at one thread and
45 -> 20 at 16; bigpath n128 d3 quotient 95 -> 76 at one thread and 59 -> 16 at 16; allfour d6
quotient 1942 -> 1634 at one thread and 334 -> 152 at 32. The two few-class quotient workloads
(multirule d7, bigpath n128 d3) stop improving at 8 threads and are within a few ms from 8 to 32.

### REFUTED

- **Descent as a job.** The replay descends depth-first, so a child instance was submitted as
  a job whenever the descending worker's deque was empty. Measured alone, two interleaved
  rounds, quotient wpp d8, multirule d7, bigpath n128 d3, growshrink3 d6 at 4/16/32 threads:
  every pair within spread (multirule 16t 29.4/28.9 against 29.1/28.9 ms). Removed;
  `.scratch/opt/patches/batch1_pre_spawn_revert.patch`.
- **kStale on a sealed key-set slot once the head moved.** Within spread on all six full-multiway
  workloads at 4/16/32 threads (wpp d7 32t 46.4 against 50.3 ms). Not landed.
- **Early segment creation for every SegmentedArray.** Full multiway wpp d7 arena 523 -> 684 MB
  and 32 threads 45.5 -> 50.7 ms. Landed only for the two replay arrays (b591bd1d).
- **Unpadded instance shard heads** (8 or 16 heads sharing lines, before aa6287a8): multirule d7
  quotient 16 threads 22.7 -> 72.8 / 42.0 ms. The lazy `more` keeps the padding.
- **16 or 32 instance shard lists** instead of 8 (after aa6287a8): within spread at 8 and 16
  threads, worse at 32 (bigcycle n128 d3 17.9 -> 25.7 / 40.6 ms).
- **child_point rewritten at every depth**: multirule d7 quotient one thread 112.8 -> 116.9 ms
  despite 7.9% fewer instructions (every worker applying a match writes its line). Landed
  write-once (40aaf6b8).
- **Key-set working capacity 8192 instead of 1024, for every set.** Full multiway one thread
  -2 to -6% (multirule d6 174.3 -> 164.4 ms); not landed: key_set_exactly_once, which grows a
  set built with the default, did not finish in 600 s under GenMC (1,286 executions at 1024).
- **The same for the two causal sets only** (1024 kept under HG_VERIFICATION): one thread
  -2.5 to -4.7% (wpp d7 full 479 -> 467 ms, multirule d6 172 -> 164, wolftri d6 182 -> 174),
  16 and 32 threads within spread or better; peak RSS of a small run +5.7 MB (cycle4 d4 full
  10.3 -> 16.0 MB), cost_matrix arena 233.3 -> 278.7 MB, large runs within 2%. A time-for-
  footprint trade, not landed; `.scratch/opt/patches/causal_set_wc8192_trade.patch`.
- **RewriteRule's zero-fill.** The 7.2 M instructions in `RewriteRule::RewriteRule` are 386
  constructions by the benchmark's corpus generator; the engine constructs one per rule.
  Not an engine cost.

### OPEN, with the measurement

- Full multiway, 32 threads: the causal triple and pair sets (`seen_causal_triples_`,
  `seen_causal_event_pairs_`) are 14-26% of samples (cycle4 d6 26%, wpp d7 14%); cycle4 d6 is
  14.8 ms at 16 threads and 15.3 at 32. Removing the sets needs the producer/consumer
  rendezvous rebuilt on one list per edge, which changes protocol P24 and the online
  reduction's in-edge order.
- Quotient replay per application: about 900 instructions per raw event after this pass,
  spread over claim, mint, content, run signature, causal records, reduction and descent, none
  above 18% of the run. The few-class workloads stop scaling at 8 threads (above).
- GPU full multiway: the result readback copies every event and causal and branchial record
  whether the caller asked for them or not (wpp d7: 8.4 of 34.6 ms), because the counts the
  paclet reports on that route are derived from those vectors.
- Twin check (same_tokens, edge_token): 7-13% of one-thread full-multiway instructions. Comparing
  only the edges the two states do not share would skip the token reads of the shared ones,
  and with them the zero-token and repeated-token checks the test makes today.
- GPU quotient replay: multirule d7 quotient takes 206 ms on the device against 23 ms on 16 CPU
  threads, all of it in the persistent kernel's replay phase (DEVICE_DESIGN 4.6.3).
- Paclet replies with graphs: a wpp depth 7 quotient job asking for States, Events, CausalEdges,
  StatesGraph and CausalGraph took about 2.0 s, 98% of its samples on the main thread building
  WXFValue trees and writing them (malloc/free about 30% of that thread); the engine's part is
  under 0.1 s. After bbc91dd8 and 8aefa431 it takes 1.33-1.36 s (Linux) and 1.85 s (Windows). An unordered set for the marshaller's sent-edge membership instead of std::set
  measured within spread (1,976 / 2,028 against 2,170 / 2,161 ms) and was not landed. Writing the
  graph sections directly to the stream, as the "States" and "Events" sections already are, is
  the change that would remove it.
- Windows native (MSVC Release, HEAD 544d81cc) against WSL (gcc, `build_linux`),
  bench_cpu_evolve, median of 3, ms at 1 / 8 / 16 / 32 threads:

  | workload | Windows | WSL |
  |---|---|---|
  | wpp d8 quotient | 4098 (min 2664) / 570 / 337 / 317 | 2274 / 370 / 255 / 200 |
  | wpp d7 full | 561 / 112 / 64 / 49 | 507 / 87 / 60 / 48 |
  | multirule d7 quotient | 137 / 25 / 22 / 27 | 106 / 27 / 20 / 36 (min 22) |
  | multirule d6 full | 206 / 38 / 25 / 21 | 187 / 36 / 25 / 22 |
  | bigpath n128 d3 quotient | 103 / 20 / 18 / 17 | 76 / 16 / 15 / 23 (min 19) |
  | cycle4 d6 full | 129 / 23 / 17 / 12 | 117 / 20 / 15 / 13 |

  One thread, MSVC is 10-35% slower than gcc (that table predates 263c781d). The slowdown of
  every evolve() after the first in a Windows process was power throttling, closed by
  263c781d; the remaining 3x gap of the counts-only serve job was the reply's identity maps,
  closed by aad7132d (Windows 250 ms against Linux 306 ms per job after both). Serve jobs on
  Windows, the binary built at the start of this pass against HEAD 8aefa431: counts-only wpp
  d8 quotient 3,272 then 5,430-5,748 ms -> 341 then 263 ms; the graph reply 3,704 then 6,528
  ms -> 1,833 then 1,845 ms.
  The ClangCL toolset is not installed on this box, so clang-cl was not measured.
