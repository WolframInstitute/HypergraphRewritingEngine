# v1.0.0 final sweeps

Five items, all v1.0.0 scope (Richard, 2026-08-18: "they are v1.0.0 work"). Each states the
evidence that opened it, what would close it, and its status. The task board is unavailable, so
this file IS the tracker; keep it current in the same commit as the work.

---

## S1 — Quotient-side branchial fan-out: is there a quadratic, and is it removable?

**Status: MEASURED. Verdict: the algorithm is output-optimal; the REPRESENTATION is a 137x
expansion. No implementation defect; a representation decision is owed.**

**Why it is open.** `4587254a` refuted the clique hypothesis on the DIRECT path
(`CausalGraph::record_branchial`): max clique over the whole oracle corpus is 5, and on the only
large case (`cycle4-automorphic`, 68,184 edges) it is 2 — 68,184 buckets against 136,368 members,
so the clique form is BIGGER than the pair form. No quadratic to remove there.

But the number that started this, **133,351,476 branchial pairs at depth 3** on
`disc-l3a2g2r2`, was measured under QUOTIENT reconstruction, which mints pairs somewhere else
entirely: `record_branchial_pair` in `hgcommon/quotient_replay_core.hpp` (~line 156), pairing
SIBLINGS OF ONE INSTANCE whose consumed slots overlap. Different structure, different fan-out.
The quadratic claim was made about a path that was never measured.

Growth on the host: 276 pairs (d1) → 162,996 (d2) → 133,351,476 (d3), ~800x per step.

**Close it by:** measuring the applied-matches-per-instance distribution and the
(instance, slot) bucket sizes, exactly as `HG_CLIQUE_STATS` did for the direct path. Then either
(a) the fan-out is wide and a clique/group representation is a real linearisation, or (b) it is
narrow like the direct path and the pair count tracks the bucket count, in which case the
relation is simply that large and the honest answer is a representation question for the caller,
not an optimisation.

### Measured (host, `disc-l3a2g2r2`, branchial only, 8 threads)

Counters added to the host replay Ctx: one scan per application, one visit per element the scan
touches. `visits/scans` is the mean applications per instance, m.

| depth | scans (applications) | visits | mean m | branchial pairs |
|------:|---------------------:|-------:|-------:|----------------:|
| 1 | 24 | 300 | 12.5 | 276 |
| 2 | 2,904 | 175,117 | 60.3 | 162,996 |
| 3 | 971,064 | 163,391,222 | 168.3 | 133,351,476 |

**The scan is output-optimal, so there is no algorithmic quadratic to remove.** 163,391,222
visits produce 133,351,476 pairs: pairs are **81.6%** of visits, so the overlap test rejects
under a fifth of what it looks at. Nearly every sibling pair of an instance genuinely IS
branchially related. Emitting N pairs cannot cost less than N, so the scan is within 1.23x of
its own output, and an inverted index keyed by (instance, slot) — the structure the DIRECT path
uses — could recover at most that 19%. It is not a linearisation and should not be built as one.

**But the output form is a 137x expansion of a structure the engine already has.** m = 168 at
depth 3, against a maximum clique of 2–5 on the direct path, so the two paths are not comparable
and the earlier refutation (`4587254a`) does not transfer. The per-instance applied list is
**971,064 entries against 133,351,476 pairs — 137.3x** — and it already carries the consumed
slots, which are exactly what decides overlap. So the applied lists ARE the branchial relation in
compressed form, and the pair set is their expansion.

### What is owed

A representation decision, not an optimisation: whether a caller asking for branchial structure
should receive 133M pairs or the ~1M-entry grouped form it is derived from. The engine computes
the compressed form either way. This is the same question `#164` raised on the device, where
holding the full relation needs ~2 GB of pair records.

**Gate:** the decision recorded with these numbers; if grouped output ships, the pair count
derivable from it unchanged and the verified against full capture.

---

## S2 — IR high-symmetry pathology (up to 1100x)

**Status: STATED PATHOLOGY REFUTED; NARROWER QUESTION REOPENED.** The recorded cause is wrong —
the search does not grow with symmetry — and WL is not a usable filter. But "no pathology" is not
"IR is optimal", and the 21-32x per-call gap is unattributed. See S2b.

**Why it is open.** `project_ir_vs_wl_verdict`: IR is exact and subsumes WL in correctness (WL
collides on all 5 1-WL-hard pairs), but blows up on high-automorphism states — cycles measured up
to **1100x slower** — because orbit pruning is weak. `BACKLOG.md` names the mechanism: full
partition copy plus fresh scratch per search node.

Both automorphic corpus cases (`cycle4-automorphic`, `star4-automorphic`) sit at 68,185 raw
states, which is where this bites.

### Measured (2026-08-18)

Two instruments, both of which existed already and neither of which was built. Registered in
CMake as part of this item (and of S4).

**`ir_core_phase_probe` — does the SEARCH grow with symmetry? No.** It calibrates itself first:
C_6 has Aut = D6 of order 12 and an initial refinement of one cell, so the search must branch —
reported `searched=1 leaves=2 nodes=2 depth=1 -> counters respond`, so the counters are live.
Then the sweep, where a cycle is the worst case symmetry can present (every vertex in one cell,
Aut of order 2n):

    C_6  C_12  C_24  C_48  C_96  C_192  C_384   ->  leaves=2  nodes=2  depth=1, every one

Flat. Sixty-four-fold size increase, identical search. On the corpus workload whose automorphism
grows with the state: leaves p50=2, p90=3, p99=3, max=4, deepest individualization 1.
**The search is bounded and does not respond to symmetry.** Whatever the 1100x was, it was not
IR exploring more.

**`ir_vs_wl` — what IS the per-call ratio?**

| graph | edges | WL us | IR us | IR/WL |
|---|---:|---:|---:|---:|
| cycle | 100 | 6.800 | 95.140 | 13.99 |
| cycle | 200 | 13.419 | 285.984 | **21.31** |
| random | 200 | 27.323 | 883.249 | **32.33** |

**Worst measured is 32.33x, not 1100x.** The cycle ratio does grow with size (5.86x at 10 edges
to 21.31x at 200), so IR scales worse per call than WL — by twenty-fold, not by three orders.

**And the per-call ratio is the wrong number for a route choice**, which `ir_vs_wl`'s own header
says before printing it: over a full evolution non-Full/WL costs **68,004,245** instructions
against Full/IR's **59,028,301** (callgrind), because the engine pays the allocation discipline
around the call — 9.74% of the non-Full run is `memset` zeroing WL scratch. IR is CHEAPER in the
engine.

**Correctness runs the same way.** Of 7 1-WL-hard pairs, WL false-collides on 2 (prism vs K3,3,
rook4x4 vs Shrikhande); IR distinguishes all 7.

### Verdict

There is no pathology to fix. IR is exact where WL is not, and cheaper where it counts. The
recorded "1100x, weak orbit pruning" is not reproducible at the sizes the engine sees, and the
BACKLOG line naming per-node partition copy as its cause describes a cost the search does not
incur — it reaches two nodes.

**Reopen only if** a state is exhibited whose `leaves`/`nodes` counters are large. The probe
prints them per state and flags the heaviest 1%, so such a state would be visible rather than
inferred.

---

## S2b — WL is not a usable pre-filter, and it fails worst where IR costs most

**Status: CLOSED. Measured ceiling, plus an implementation that was already tried and refuted.**

`cost_matrix`'s `wlceil` column is the CEILING on IR calls a WL pre-filter could avoid. It is a
ceiling because WL is coarser than IR: distinct WL hashes are at most the canonical classes, and
any state landing in a bucket already seen STILL needs IR, because WL agreement never establishes
isomorphism. So the filter can skip at most canonical/raw of the calls while paying a WL pass on
every raw state.

| workload | wlceil |
|---|---:|
| cycle4-automorphic | **0.0%** |
| star4-automorphic | **0.3%** |
| disconnected-lhs | 0.8% |
| binary-growth | 4.1% |
| multi-rule | 15.1% |
| wolfram-2to4 | 35.4% |

**The filter is useless exactly where IR is expensive.** The two automorphic cases carry 68,185
raw states each and their ceilings are 0.0% and 0.3%. This is structural rather than unlucky:
symmetry is what makes WL coarse AND what makes IR do work, so the two failures are the same
property seen twice.

Already tried, not merely reasoned about: the tiered scheme (WL bucket, IR on collision) was
implemented, was correct, measured **+28% pessimization**, and was reverted. The ceiling above is
why it had to be.

---

## S2b-ii — Where the 21-32x per-call gap actually is (REOPENED, not measured)

**Status: OPEN. This is the part of S2 that survives.**

The search reaches TWO nodes, so the per-call gap is not exploration and cannot be pruned away.
It is in the refinement inner loop — the O(n*e) work each node does — which is a different target
from anything BACKLOG names (its "full-partition copy + fresh scratch per search node" is a
per-node cost on a two-node search).

Nothing here is measured yet. Attributing that loop is what would say whether IR is near its own
bound or has room.

**Gate:** the per-call cost of `ir_canonical_hash` attributed inside the refinement (callgrind or
per-phase counters), with a stated verdict on whether it is improvable; if improvable, the
21-32x re-measured after the change with `all_tests` and `cost_matrix` 17/17 EXACT unchanged.

---

## S3 — Optimality numbers, per phase, against bounds

**Status: PARTIAL.**

**What exists.** Device roofline is in `GPU_REENGINEERING_PLAN.md` §1: no phase is
bandwidth-bound or issue-bound (corpus max 6.82% DRAM, 11.53% SM), occupancy pinned at 16.67%,
and two distinct binding resources — divergence (4.0% lane efficiency) where the device wins,
parallel width where it loses (instances 68,185 over 40 classes on cycle4, a 1,705x deficit).

**What is missing.** The HOST has no equivalent. There is no per-phase statement of which
resource binds and what the algorithmic lower bound is for matching, canonicalization,
rewrite/dedup on CPU. `hgcommon::PhaseTimer` exists and `bench_cpu_evolve` reports phase cycles,
so the instrument is there and the analysis is not.

**Close it by:** per-phase host attribution across the corpus with the binding resource named and
the lower bound stated, recorded beside the device table.

**Gate:** every phase either within 2x of a stated bound or its residual attributed to a named,
evidenced cause.

---

## S4 — Neatness sweep

**Status: OPEN.**

**Evidence.**
- `tools/` is 12,155 lines and 57 of 66 files are built by nothing (project CLAUDE.md), one of
  which already fails to compile. That is more than half the size of the host engine, unbuilt.
- FOUR device headers include `hypergraph/` — `edge_signature.hpp`, `quotient_expansion.hpp`,
  `quotient_causal.hpp`, `hash_table.hpp` — a device→host layering violation. The reverse
  direction is clean (0 files), so the property is otherwise true and worth restoring.
- `RELEASE_CHECKLIST.md` cites `hg_gpu_tests 99/99`; measured today is **88/88**. Either tests
  were consolidated and the checklist is stale, or some are gone. Unexplained either way.

**NOT in scope, and recorded so it is not re-opened:** de-header (#20) is CLOSED (`4614f18`).
Moving the remaining bodies was measured at **+0.21% instructions and 0.00 s compile time**; the
lever was the include closure, which was done (`types.hpp` 757 → 154 ms). A raw header:source
ratio is not evidence of a defect here.

**Gate:** every file under `tools/` either registered in CMake or deleted; zero
`gpu/ -> hypergraph/` includes; the checklist's gate numbers match a measured run.

---

## S5 — The verdict/containment design defect

**Status: OPEN. Richard: v1.0.0 scope, not post-release.**

**The defect.** A claim counter maintained independently of the structure that must contain the
thing claimed. Three instances, all found and fixed on 2026-08-18, all the same shape:

| structure | verdict says | reality |
|---|---|---|
| `SegmentedArray` | `count_` publishes index i | the segment holding i may not exist (`fd78a806`) |
| `ConcurrentMap` | insert returns "you won" | the entry may be stranded in a superseded table |
| `ConcurrentKeySet` | `count_` counts the claim | `migrate_into` may leave the key in neither table (`f694c062`) |

Correctness is derived from a VERDICT while output comes from ENUMERATION, and the two are kept
in step by hand. Each fix re-synchronised one more site; none removed the split.

**The direction.** Make the verdict and the containment the same act. For `ConcurrentKeySet`: the
winning claim places the key in the current head table and that is the whole operation; readers
walk the chain (bounded by growth count, logarithmic); `MIGRATED`/`drained` become advisory
compaction that may fail freely because nothing depends on it; `count_unique()` — the enumeration
— becomes the definition and any counter is a hint.

**Gate:** no state in which a thing is claimed-but-unreachable, in any of the three structures; a
GenMC harness stating the completeness property for each (two now exist —
`lock_free_list_completeness` `33244e82`, `arena_worker_index_exclusive` `11c8cd2b`); all gates
green across 20 consecutive suite runs.

---

## S6 — Hot-path hunt: what the engine profile says, and one refutation

**Status: ONE WIN LANDED, ONE HYPOTHESIS REFUTED, ONE TARGET IDENTIFIED AND UNTRIED.**

Callgrind on `bench_cpu_evolve 7 1 1 wolfram24`, RelWithDebInfo for line attribution, 90.4B
instructions.

| site | share |
|---|---:|
| `bitset.hpp` `contains()` inlined into `execute_expand_task` | **63.35%** |
| `lock_free_list.hpp` inlined into same | 5.05% |
| `ConcurrentKeySet::insert` | 3.03% |
| `ir_refine` | 2.31% |
| malloc family, total | ~0.9% |

**Landed:** IR scratch re-zeroing (`ef73216d`), **1.09% engine** / 15% on the IR-saturated probe.
The headline was corrected in `d8b354bc` — the 15% is the probe's.

### Where the 63% actually is

Not the bit test. Line attribution inside `contains` -> `find_chunk`:

    12,611,066,200 (13.96%)   size_t mid = lo + (hi - lo) / 2;
     8,593,090,998 ( 9.52%)   while (lo < hi) {
        38,409,351 ( 0.04%)   return (words[word_idx] >> bit_idx) & 1;

**23.5% of the engine is the binary search that FINDS the chunk**, against 0.04% for the bit
test it exists to reach. `contains` is called 3.43 billion times.

### REFUTED: a one-entry chunk memo

A chunk spans 512 consecutive edge ids and the matcher walks edges in id order, so consecutive
`contains()` calls should hit the same chunk. Implemented as a `mutable size_t memo_` validated
by the `chunk_id` compare (no invalidation needed: entries are sorted with unique ids, so an
index passing that compare names the right entry however the array has moved).

**Measured: 90,449,530,905 -> 97,957,399,573 instructions, +8.30% WORSE.**

Hit rate says why, and it is the premise not the code: **13,497,293 hits against 1,130,281,656
misses — 1.18%.** Consecutive lookups almost never share a chunk. Reverted; patch kept at
`scratchpad/memo_refuted.patch`.

**Do not retry a locality cache here.** The access pattern is not local, and that is now measured
rather than assumed.

### Also found: two dead filter paths

Instrumenting the candidate call sites to find the hot one showed `SignatureIndex::
for_each_edge_with_signature` (index.hpp:79), `InvertedVertexIndex::for_each_edge` (index.hpp:248)
and `pattern_matcher.hpp:509` all execute **zero times** on wolfram24, wpp and multirule. Matching
goes through the shared join core. Whether those paths are live on any workload, or are dead code,
belongs to S4.

### The untried target

3.43 billion calls is an access-pattern question, not a slow predicate — `contains` is already
force-inlined with its failed variants measured in-place. BACKLOG names the lever: a
match-forwarding edge->matches reverse index, O(|parent matches| x size) -> proportional to what
changed. That is a matcher change, not a data-structure tweak, and it is the one place a large
win plausibly remains.
