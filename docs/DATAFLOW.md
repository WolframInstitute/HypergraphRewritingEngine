# Data flow, parallelism, and where the invariants live

The companion to `CODEMAP.md`. That document says WHERE code is; this one says WHAT FLOWS
THROUGH IT, WHO RUNS CONCURRENTLY, and WHICH INVARIANT EACH STAGE IS TRUSTED TO UPHOLD.

Everything below was read out of the source rather than recalled. Claims that are NOT yet
verified against code are marked `[UNVERIFIED]` and are work items, not documentation.

---

## 1. The pipeline

One evolution is four job types (`EvolutionJobType`, `parallel_evolution.hpp:167`) over a
work-stealing scheduler:

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

**Incremental matching is the two-branch partition.** For a child `C = P − consumed + produced`,
every match in `C` either uses only edges surviving from `P` — hence was already a match in `P`,
so it is FORWARDED — or uses at least one produced edge, so DELTA finds it. The partition is
exhaustive, which is what makes incremental matching sound in principle.

---

## 3. Parallelism

**Scheduler.** `job_system/` — lock-free work-stealing deques, per-worker local queues (jobs
submitted from a running job stay node-local), a shared lock-free injector for overflow, and idle
workers parked via `hgcommon/park.hpp`. No mutexes and no condition variables anywhere in the
engine.

**Concurrent data structures**, all lock-free or better:

| structure | guarantee | notes |
|---|---|---|
| `ConcurrentMap` | wait-free lookup, lock-free insert | no tombstones: a claimed slot must never return to EMPTY, or probe chains cut. Key 0 is a reserved sentinel |
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

Nine shared cores in `common/include/hgcommon/` are the single definition of everything that is
not scheduling:

`core` · `event_core` · `ir_core` · `match_core` · `park` · `portable_intrinsics` ·
`rewrite_core` · `signature_core` · `wl_core`

The eight `gpu/src/*.cu` files are scheduling and device-specific layout only. Both devices run
the same exact canonicalizer, the same WL hash, the same match binding, the same edge
signatures, the same rewrite semantics and the same event identity — so they agree by
construction rather than by two maintained copies.

**Two GPU schedulers exist on purpose.**

| | role |
|---|---|
| persistent (`persistent.cu`) | DEFAULT. whole evolution in one launch, match + rewrite as roles over an MPMC ring, device-side quiescence, IR scratch from a device arena |
| level-synchronous (`evolve.cu`) | the differential ORACLE the persistent path is checked against; `compute_state_dedup_keys` is the host twin of `state_key_device` |

The level-synchronous loop is retained deliberately. Deleting it would delete the evidence that
makes the persistent default defensible.

---

## 5. Where the invariants live, and which are guarded

| # | invariant | upheld by | guarded? |
|---|---|---|---|
| I1 | no match is ever missed | forwarding + delta partition | **NO** — validator defaults off, commented out in the fuzzer (#75) |
| I2 | no two distinct matches are conflated | `seen_match_hashes_`, decided by `MatchRecord::operator==` | yes — `claim_match()` compares content on equal hash and probes on collision; gated by `test_match_dedup_exactness.cpp` |
| I3 | isomorphic states share identity | canonical hash (IR exact / WL fast) | yes — oracle corpus + 204-row matrix |
| I4 | WL is never a dedup key | IR on the exact path | yes on CPU; GPU degrades to WL above slot bounds and COUNTS it |
| I5 | output independent of worker count | canonical identity | partially — determinism gate fails ~1/30 (#65) |
| I6 | CPU and GPU agree | differential corpus, both schedulers | yes for 8 of 9 identity cells; Automatic diverges (#66) |
| I7 | causal/branchial exact under quotient | reconstruction from the skeleton | yes, verified edge-for-edge across 204 configs |
| I8 | TR is order-independent | — | **NO** — force-disabled under quotient, blocked on a canonical slot tie-break |

**I1 is the remaining root.** Every downstream artifact — states, events, causal edges, branchial
edges, transitive reduction — is computed from the match set. A dropped or conflated match
corrupts all of them silently and self-consistently: the run simply produces less and looks fine.
Nothing else in this table can be trusted above them.

---

## 6. Known structural ceilings

- **GPU occupancy** (#72): 32-thread blocks × SM-count grid ≈ 4096 resident threads, ~4% of a
  4090, and one global `head_`/`tail_` pair every worker CASes. Measured to get WORSE past the SM
  count, so it is a design ceiling, not a tuning parameter.
- **Matching cost** (#21 + #49): the join has never been placed against the AGM bound. "Is
  forwarding optimal" and "is the join worst-case-optimal" are the same question.
- **Sampling** (#63, #64): fixed-rate thinning is a knife-edge (q = 1/8 and 1/4 go extinct at
  depth 1, q = 1/2 accelerates) and no release gate checks that a sampled run's observables match
  the unpruned run's.
