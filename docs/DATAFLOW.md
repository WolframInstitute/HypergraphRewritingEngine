# Data flow, parallelism, and where the invariants live

The companion to `CODEMAP.md`. That document says WHERE code is; this one says WHAT FLOWS
THROUGH IT, WHO RUNS CONCURRENTLY, and WHICH INVARIANT EACH STAGE IS TRUSTED TO UPHOLD.

Everything below was read out of the source rather than recalled. Claims that are NOT yet
verified against code are marked `[UNVERIFIED]` and are work items, not documentation.

---

## 1. The pipeline

One evolution runs seven job types (`EvolutionJobType`, `parallel_evolution.hpp:209`) over a
work-stealing scheduler. Four of them form the pipeline:

```
   MATCH  ──▶  SCAN  ──▶  EXPAND  ──▶  REWRITE
 (per state,  (anchor)   (extend one   (apply a RANGE of
  per rule)               edge at a     one state's matches)
                          time)
```

| job | what it does | unit |
|---|---|---|
| `MATCH` | orchestrates matching for one state: spawns SCAN, or falls back to synchronous | (state, rule) |
| `SCAN` | picks anchor candidates for the first pattern edge | (state, rule) |
| `EXPAND` | extends a partial match by one edge; emits a completed match | partial match |
| `REWRITE` | applies a contiguous range of one state's matches | `ExpandChunk` |
| `REDRIVE` | in a continuation (`evolve_more`, a session's `Step`), replays one quotient point that the previous step bound left standing (`quotient_redrive_point`) | (class, depth) |
| `QC_APPLY` | applies one capture to one (depth, list) unit of its class's instances (the replay) | capture x instances |
| `SIDE` | runs a caller's job on the engine's workers (`submit_side_job`): StepStatistics and reply work | caller-defined |

A single history under `EventOrderingFunction` runs these jobs on one worker, since each event
depends on the one before. `"Search" -> "BestFirst"` takes one transition at a time from a heap
and rewrites and matches it on the driver's thread (`best_first_inline`, `best_first_driver`),
with costing in chunks on the workers (`cost_best_first`).

**The work unit is a state expansion, not a match.** `ExpandChunk` carries `{matches, begin,
end, step}` over an arena-resident, immutable match array shared read-only by every chunk of the
same parent. This is deliberate and load-bearing: all the locality lives at the STATE, because
every child of one parent is built from the same edge set, canonicalized against the same
starting adjacency, and every branchial pair among them is internal to that parent. Tasking per
match discards all of it — and that is the recorded reason incremental canonicalization had no
amortization hook and stayed refuted.

Ranges rather than whole states so a parent with thousands of matches still load-balances.

---

## 2. End-to-end flow of one run

```
rules + initial state + steps + options
      │
      ▼
  root state created, full SCAN (the induction base case for completeness)
      │
      ▼
  ┌── MATCH (state, rule) ──────────────────────────────────┐
  │     SCAN                                                │
  │       full  : anchor on rule.match_order[0], the        │
  │               most-constrained pattern edge             │
  │       delta : for each PRODUCED edge, try it at EVERY   │
  │               pattern position  (:1719-1732)            │
  │     EXPAND  : extend one edge at a time                 │
  │     complete_match                                      │
  │     dedup   : claim_match() -- content, not hash        │
  └─────────────────────────────────────────────────────────┘
      │
      ▼
  REWRITE (ExpandChunk = range of one parent's matches)
      │
      ├── child edge set = parent − consumed + produced
      ├── canonical hash  (mode: None / Automatic / Full)
      ├── state dedup     (quotient: expand once per class)
      ├── event created + event identity signature
      ├── causal edges    (edge producer → edge consumer)
      ├── branchial edges (sibling events out of one parent)
      └── optional transitive reduction over causal
      │
      ▼
  artifacts marshalled out (WXF → FFI → paclet, or EvolveResult)
```

**Incremental matching is the two-branch partition — ON THE CPU.** For a child
`C = P − consumed + produced`, every match in `C` either uses only edges surviving from `P` —
hence was already a match in `P`, so it is FORWARDED — or uses at least one produced edge, so
DELTA finds it. The partition is exhaustive, which is what makes incremental matching sound.

**The GPU forwards only under held match sets.** `match_state_rule` takes no produced-edge list
and no delta flag: outside held match sets each (state, rule) pair is matched by a FULL scan.
Under held match sets (`"MatchesPerStep"` with `None` states, `hgcommon::held_matches_apply`) the
child's expand entry keeps the parent's set, drops the matches on consumed edges and joins only
from the produced edges (`held_set.hpp`). So the two devices run different matching algorithms
outside held match sets, and any statement about matching complexity has to name which one
(#78). One
consequence is benign — with no delta path the device cannot find a match twice, so it needs no
match dedup — and one is not: on a deep evolution the host exploits that a child shares almost
all its parent's matches and the device redoes the scan.

---

## 3. Parallelism

**Scheduler.** `job_system/` — lock-free work-stealing deques, per-worker local queues (jobs
submitted from a running job stay node-local), a shared lock-free injector for overflow, and idle
workers parked via `hgcommon/park.hpp`. No mutexes and no condition variables anywhere in the
engine.

**Concurrent data structures**, all lock-free or better:

| structure | guarantee | notes |
|---|---|---|
| `ConcurrentMap` | wait-free lookup, lock-free insert | no tombstones: a claimed slot must never return to EMPTY, or probe chains cut. Keys 0 (EMPTY) and ~0 (LOCKED) are reserved sentinels |
| `LockFreeList` | lock-free push | per-edge consumers, per-state events |
| `SegmentedArray` | lock-free append | bounded `MAX_SEGMENTS` |
| `SparseBitset` | copy-on-write sharing | child edge sets share with the parent |
| work-stealing deque | lock-free | ABA tag width is an open item (#31) |

**Memory.** Per-worker scratch plus persistent arenas take the allocator off the hot path;
matches are arena-resident and immutable once published, which is what lets forwarded copies
share one `MatchCore` by pointer. Reclamation is per-tier arena reset.

**Determinism does not come from the schedule.** It comes from canonical identity: states and
events are keyed by canonical hash, so the observable output is a function of
`(rules, initial state, steps, options)` and is identical at 1 or N workers. This is the same
argument that licenses removing the GPU's per-step barrier.

---

## 4. CPU / GPU split

The shared cores in `common/include/hgcommon/` (among them `ir_core`, `join_core`, `match_core`,
`rewrite_core`, `event_core`, `signature_core`, `sampling_core` and `ring_core`) are the single
definition of everything that is not scheduling.

The `gpu/src/*.cu` files hold scheduling and device-specific layout. Both devices run the same
IR canonicalizer, the same match binding, the same edge signatures, the same rewrite semantics
and the same event identity from these cores.

**The GPU has one scheduler**, the persistent kernel (`persistent.cu`): the whole evolution in
one launch, match and rewrite as roles over an MPMC ring, device-side quiescence, and IR scratch
from a device arena. `compute_state_dedup_keys` on the host and `state_key_device` on the device
compute the same state key.

---

## 5. Where the invariants live, and which are guarded

| # | invariant | upheld by | guarded? |
|---|---|---|---|
| I1 | no match is ever missed | forwarding + delta partition | yes: `test_match_completeness.cpp` runs every corpus case at 1, 2, 4, 8, 16 and 32 workers, twice, in the task-based, batched and eager submission modes, and requires zero matches unclaimed at their state's drain |
| I2 | no two distinct matches are conflated (CPU; the GPU cannot find one twice, see #78) | `seen_match_hashes_`, or step t's `MatchEpoch::seen` when match epochs are on (`parallel_evolution.hpp`), decided by `MatchRecord::operator==` | yes: `claim_match()` compares content on equal hash and probes on collision; gated by `test_match_dedup_exactness.cpp` |
| I3 | isomorphic states share identity | canonical hash (IR, `hgcommon/ir_core.hpp`) | yes: the oracle fuzz, `GoldenMatrix`, `test_ir_canonicalization.cpp` (including 1-WL-hard non-isomorphic pairs) |
| I4 | no dedup key is coarser than the canonical hash | IR on every path | yes on both: the GPU sizes its slot from the state's own counts and reports capacity as partial work with a warning (`gpu/include/hg_gpu/overflow.hpp`) |
| I5 | output independent of worker count | canonical identity | yes: `CausalDeterminism.*` at 1-32 threads, `OracleCorpus.DeterministicAcrossThreadCounts`, and the worker-count fuzz at 1, 4, 16, 32 and one count from 2 to 31 (docs/VERIFICATION_STATUS.md) |
| I6 | CPU and GPU agree | the shared cores | yes: `reference/oracle_fuzz.wls` runs every case on both devices under every state and event identity mode, and the golden corpus gives 12 of 12 cases equal on CPU and GPU |
| I7 | causal/branchial exact under quotient | reconstruction from the skeleton | yes: `cost_matrix` (oracle EXACT) and the golden corpus (docs/SPEC.md §5.5) |
| I8 | TR is order-independent | the reduction of a DAG is unique: `hgcommon::redundant_producers` in the replay, `hgcommon::tr_reduce` on read | yes: `causal_tr_exactness_probe` and `CausalTrExactnessTest.*` at 1-64 workers (docs/SPEC.md §5.5) |

Every downstream artifact (states, events, causal edges, branchial edges, transitive reduction)
is computed from the match set, so I1 and I2 are the invariants the others depend on. A dropped
or conflated match changes all of them consistently, so an end-to-end count cannot find it, and
`test_match_completeness.cpp` checks each state's matches at its drain.

---

## 6. Known structural ceilings

- **GPU occupancy**: the persistent grid is SMs x min(the blocks of `k_persistent_evolve` one SM
  holds, 8) blocks of 32 threads (`default_persistent_grid`). On the RTX 4090 that is 1024
  blocks, 256 of 1,536 thread slots per SM. The block-count sweep over 32 to 3072 blocks has its
  minimum time at 1024 (docs/TUNING.md D1). One match queue (`RingBuffer`, one `head`/`tail`
  pair) serves every block.
- **Matching cost**: the join (`hgcommon/join_core.hpp`) binds one pattern edge at a time by
  backtracking. Its cost has not been compared with the AGM bound.
- **Sampling**: below the branching factor a fixed `TransitionRate` makes a sampled evolution go
  extinct. The spine keeps a state's transitions of smallest seeded rank when every draw at the
  state failed (docs/ARCHITECTURE.md, Sampling). No release gate compares a sampled run's
  observables with the unpruned run's.
