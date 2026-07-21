# Grand design: memory-bounded, incremental, CPU/GPU-shared multiway evolution

## Status (kept current as items land; branch `overhaul/phase0-zero-waste`)

**DONE — landed + verified (full suite green, oracle-exact, multi-thread deterministic):**
- Measurement + verification harness: `reference/oracle_corpus.hpp` (diverse rule-type corpus
  + brute-force oracle), `test_oracle_corpus.cpp` (exactness + 4/8/16-thread determinism +
  causal/branchial-count determinism gates), `tools/cost_matrix.cpp` (arena bytes + heap
  bytes/count counters). This is the "prove, not assume" substrate for everything below.
- Phase-0 zero-waste (§4): dead `state_children_` removed; WL hash-only path (killed the
  per-state persistent-arena leak); pattern F1 (dropped 17 KB context copy); `Event.binding`
  dropped (−132 B/event); `event_canonical_state_map_` gated; **F8 correctness fix** (Bell
  truncation → silently missed matches) + regression test.
- SparseBitset **copy-on-write** (§ representation): child = parent chunks by reference + delta.
- **De-heap of the whole shipping surface (§3c): COMPLETE.** `heapAllocs 1567 → 69` on the
  corpus. Jobs → per-thread slab pool; **every `ConcurrentMap` → per-worker-cursor arena**
  (the arena now scales — per-worker bump cursors, no shared-atomic fast path — solving the
  6× contention a naive single-arena caused); `lockfree_deque` inline payload; IR `std::function`
  removed; WXF serialization streamed (byte-identical, FFI pin-test added). Residual 69 =
  arena backing blocks + one-time single-threaded setup.
- Per-worker-cursor arena (§1, the Tier-allocator fast path): DONE.

**IN PROGRESS:**
- Causal closure memory reduction (§3 base layer): reconvergence-skip + key-only uint32 sets
  + drop-`Anc` + lazy empties. The closure is still O(N²) *bytes* (de-heap moved it heap→arena,
  didn't shrink it); this is the first cut before the reachability oracle.

**NOT STARTED:**
- Causal reachability oracle (§3, the O(N²)→O(N·w) redesign) — after the base layer.
- Tier reclamation (§1 Tier F / §2b) — per-generation arena reset → working memory O(frontier).
- Automorphism reconstruction (§2 pillar 3 / §2b) — don't materialise raw provenance.
- Incremental hashing (§4) — consume the delta already handed to `create_or_get_canonical_state`.
- CPU/GPU shared factoring + persistent-kernel scheduler (§5, Phase 3).

Name/location of this doc still provisional; charter is `OBJECTIVE.md`.

## The objective

Memory is the wall; **maximise how far the multiway frontier can grow** before OOM.
Throughput matters only after we stop blowing up. Today the engine retains, forever:
O(all states ever) state bitsets, **O(events²)** causal closure, O(descendants) forwarding
copies, plus a set of outright leaks and dead-weight fields. The 7-way cost analysis
(2026-07-21) quantified every term; this design turns those findings into one coherent
architecture rather than a pile of patches.

## Invariants the design must enforce structurally (not by patching)

1. **Bare-minimum representation** — store the theoretical minimum needed to reconstruct
   states / events / causal edges / branchial edges.
2. **Bare-minimum computation / incrementalisation is foundational** — the delta path
   (child = parent − consumed + produced) is the ONLY path; full recompute is the
   degenerate first-step case, not a second implementation.
3. **Zero wasted reads or writes** — no dead fields, no leaked allocations, no duplicated
   computation. Every finding below is a violation of this invariant.
4. **Lifetime-matched allocation** — a hierarchical allocator whose tiers match the
   evolution's structure; transient data is reclaimed when its generation quiesces.
5. **Single source of truth** — one implementation per functionally-distinct operation,
   reused at every functionally-equivalent intersection (CPU / GPU-lockstep /
   GPU-persistent; WL / IR; SCAN / EXPAND / SINK; delta / full).
6. **Correctness is absolute** — exact states/events/causal/branchial vs the brute-force
   oracle, at every step, after every change.

## Architecture

### 1. Hierarchical allocator (lifetime tiers)

- **Tier D — Durable / output** (inherently O(result)): shared append-only edge pool,
  states, events, causal edges, branchial edges. These *are* the answer. Lever: stream
  to the FFI as produced and free, for runs whose output is consumed incrementally
  (turns O(result) resident into O(frontier) resident).
- **Tier F — Frontier-generation transient** (target O(active frontier)): match lists,
  partial matches, task records, WL/IR scratch, edge-correspondence arrays. Bulk-`reset()`
  when a frontier band quiesces. This is the new tier — most of today's "persistent arena"
  content actually belongs here and is the source of the unbounded growth.
- **Tier T — Per-task scratch** (exists today): `worker_scratch`, reset per task.

Quiescence signal for Tier-F reset = job-system completion once a frontier band's states
are all expanded (made cheap by per-worker counters, see §4). Complication: quotient +
depth-relaxation can re-open a canonical state at lower depth later — so a band is "closed"
only when no lower-depth arrival is still possible; the depth-relaxation bookkeeping already
tracks this and gives the exact close signal.

### 2. Three pillars for frontier reach

1. **Generational reclamation** — Tier F above.
2. **Quotient exploration** — expand each canonical state once (already in the engine).
3. **Automorphism reconstruction** — the IR canonicalizer already computes automorphism
   generators + edge orbits (`compute_canonical_hash_with_edge_orbits`, `out_generators`)
   for orbit pruning, then **discards them**. Instead: store the quotient (canonical states
   + their single expansion + Aut generators) and **reconstruct the exact raw
   states/events/causal/branchial from the orbit action** at output / on demand, rather
   than materialising every raw provenance during exploration. This is what turns working
   memory from O(all raw states) into O(canonical frontier). Hard obligation: provably
   identical output to full materialisation (oracle-checked).

### 2b. Coupled reclamation + reconstruction [DECIDED: one design]

Reclamation and reconstruction are designed together because **reconstruction changes what
must be retained**, which is exactly what reclamation frees. Treating them separately would
build the generational tier around the wrong retention set.

**Retention model (what each tier holds under the coupled design):**
- **Tier D (durable) shrinks to the QUOTIENT:** canonical states (bitset membership over the
  shared edge pool), one expansion per canonical state (the matches/rewrites found when it
  was expanded), and the **Aut generators** per canonical state. NOT the raw states, raw
  events, or the raw causal/branchial edges — those are *derived*.
- **Tier F (frontier-generation) holds only the active canonical band's** transient working
  data (match lists, partial matches, tasks, WL/IR scratch), bulk-reset when the band closes.
- Raw states/events/causal/branchial are **produced by reconstruction at output time** (or
  streamed as a consumer pulls them), never all resident at once.

**Reconstruction mechanism (the exact, oracle-obligated part):** a raw state is a canonical
state under some group element; the set of raw states sharing a canonical is the orbit of the
initial-condition embedding under Aut. A raw event (raw_in → raw_out) is a canonical event
acted on by a group element; a causal edge (event_p → event_c) holds iff the corresponding
canonical events, under the aligning group elements, share a produced/consumed edge; branchial
siblings likewise. The Aut generators + the recorded canonical expansion determine all of this.

**Open sub-problems to settle in the design (honestly, the hard part):**
- Reconstructing **causal and branchial** edges between RAW events from canonical events +
  group action is subtler than reconstructing states — causal/branchial live on the *event*
  structure, and the group action on events must be pinned down so the reconstructed edge sets
  are exact (not just the state set). Needs a precise formalisation + a proof against the oracle.
- Multiplicity / provenance counts must match (how many raw paths reach a canonical state).
- Concurrency: generators are discovered during canonicalization; capturing them without
  slowing the hot path (they are already computed for orbit pruning — capture, don't recompute).
- When output is **streamed** (Tier D freed on emit), the reconstruction must be
  producible incrementally in the consumer's requested order.

This is the largest and highest-value piece; it is prototyped behind a flag and
differential-tested (reconstructed output vs full materialisation vs brute-force oracle) on the
entire corpus before it becomes a default. Until proven, full materialisation remains available.

### 3. Causal reachability oracle — full redesign to break O(N²) [DECIDED: redesign now]

The Desc/Anc closure exists for ONE query: at each candidate causal edge (p→c), decide
`c ∈ Desc[p]` so the online transitive reduction keeps exactly the non-redundant edges
(online-TR = offline-TR). Storing the full closure to answer it costs O(events²). Replace
the closure with a **reachability oracle** that answers the same query without materialising
all pairs. The representation wins below are folded in as the oracle's base layer, not a
separate step.

**Structure to exploit.** The causal graph is a DAG whose *width* w (largest antichain =
maximum set of mutually-concurrent events) is typically ≪ N. Reachability oracles are cheap
when w is small. Candidate designs, to be prototyped and measured (each must preserve the
exact `c ∈ Desc[p]` decision under concurrent, reverse-topological insertion):

- **Chain decomposition (Jagadish).** Partition the DAG into w chains; label each event by
  (chain id, position) + a per-event vector of "furthest reachable position in each chain."
  Query `c ∈ Desc[p]` = O(w) compare. Memory O(N·w) vs O(N²) — the win is exactly N/w.
  Online insertion updates the reached-position vectors along ancestors of p; the reverse-
  topological add order bounds the update.
- **2-hop labels (Cohen et al.).** Each event carries Lout ⊆ desc, Lin ⊆ anc with
  reachable(p,c) ⇔ Lout(p) ∩ Lin(c) ≠ ∅. Empirically O(N·√M); harder to maintain online
  under concurrency — kept as a fallback if chain width is large.
- **Interval / nested labels** — O(1) for tree-like causal structure, degrades on
  reconvergent DAGs; useful only if the causal graph is near-forest in practice (measure).

**Base-layer representation wins (subsumed into whichever oracle):**
- Drop the persistent `Anc` closure — ancestors come from predecessor adjacency (O(edges)).
- Key-only `uint32` sets where any set survives (16 B → 4 B).
- Lazy / small-buffer sets — no ~608 B/event baseline for always-empty terminal/source nodes.
- Reconvergence skip: `if (!reached(a,c)) …` short-circuits the O(|A|·|D|) update storm to O(|A|).

**GPU note.** GPU.txt flags online TR as "inherently serial." A reachability oracle with
**batched/periodic** relabeling (rebuild labels per frontier band rather than per edge) is the
natural GPU form — and it shares the *query* implementation with the CPU path (single source
of truth), differing only in when labels are refreshed. Branchial detection is unaffected
(overlap of shared consumed edges) and keeps its inverted `(state,edge)` buckets, minus the
FFI-only `state_events_` list and the singleton-bucket overhead.

**Correctness obligation.** Whichever oracle: the emitted causal + branchial edge sets must be
bit-identical to the current engine and to the brute-force oracle across the full corpus,
single- and multi-threaded, at every step. This is the riskiest single piece of the design —
prototype behind a flag, differential-test against the existing closure before switching.

### 3c. De-heap the entire surface — into lifetime tiers, NOT one arena [DECIDED]

Measured (cost_matrix heap counter): heap **dwarfs** the arena (2.8–4.4 MB vs 0.2–1.1 MB
per small evolution) — a ~2.8 MB fixed floor from oversized default `ConcurrentMap` tables
plus ~1 KB/event growth. Every heap allocation contends `malloc` across workers; removing
them is a proven large time win. But **a single global arena is wrong**: it removes the
contention yet leaves the O(N²) closure resident until end-of-run — same wall. Each site
must land in the **tier matched to its lifetime** (§1), so it is reclaimed when its
generation is done. Line-by-line inventory of the shipping engine, with target tier:

| Heap site | Where | Target tier |
|---|---|---|
| `ConcurrentMap` tables (`::operator new`) | `concurrent_map.hpp:66` | per-instance (below) |
| — `canonical_state_map_`, event maps, `seen_*` dedup sets | Hypergraph / CausalGraph | **Tier D** durable (right-sized initial capacity; keys-only where value is dead) |
| — causal `desc_`/`anc_` + per-event `DescAncSet` | CausalGraph | **replaced** by the reachability oracle (§3); residue **Tier F**, reclaimed per generation |
| — `state_matches_`, `state_parent_`, child maps, `seen_match_hashes_`, `missing_match_hashes_` | parallel_evolution | **Tier F** frontier-transient |
| `new T` injector boxing | `lockfree_deque/deque.hpp` ×4 | store `Job*` inline (no alloc) / **Tier T** |
| `FunctionJob` `make_unique` per task | `job_system/job.hpp:50` | inline tagged-union task in the deque / per-worker slab — **no heap** |
| WXF `std::vector<uint8_t>` serialization buffers | `wxf/wxf.hpp` | FFI-boundary, per-call; stream into a reused buffer (Tier D scratch) |
| `std::function` ×11 | job/wxf/ffi/matcher | SBO covers small closures; the load-bearing one is `FunctionJob` (above); rest cold |
| IR `std::map`/union-find, `canonical_types` `unordered_map` | ir_canonicalization / canonical_types | per-canonicalization → worker-scratch (Tier T) |
| `make_unique<WLHash>`/`<JobSystem>`, worker vector | one-time setup | cold — acceptable, or Tier D at construction |

The mechanism: `ConcurrentMap` (and the other structures) take an arena + a tier tag; the
allocator is chosen by lifetime, and superseded tables / completed-generation data are
reclaimed at the generation boundary rather than `::operator delete`d one-by-one or leaked
to end-of-run. Right-size initial capacities to kill the fixed floor. Prove each removal
with the cost_matrix heap counter (heap → 0 on the hot path) and the corpus oracle (exact).
This is gated on the §1 tier design landing first — de-heap and the hierarchical allocator
are one piece of work.

### 4. Zero-waste pass (each collapses a functionally-equivalent intersection)

Canonicalization / hashing:
- WL **hash-only path** (`out_colours=nullptr`) → removes the per-state persistent-arena
  cache leak.
- **Memoize** the `VertexHashCache` / IR canonical form on the `State`; reuse across all
  incident events → collapses the 4×-per-event recompute in `find_edge_correspondence`.
- **Incremental hashing** — consume the `incr_consumed/incr_produced` delta already handed
  to `create_or_get_canonical_state` (today ignored); wire the dead `build_canonical_hash_dense`.

Records / dead weight:
- **Drop `Event.binding`** (132 B × events, never read post-creation).
- **Drop `MatchRecord` dead fields** (`canonical_source`, `source_canonical_hash`,
  `storage_epoch` = 20 B, recomputed per forward, never read).
- **Keys-only sets** (`seen_match_hashes_`, `seen_causal_event_pairs_`, `seen_branchial_pairs_`).
- **Compile-time-configurable rule caps** — default `MAX_VARS=8`, `MAX_PATTERN_EDGES=8`
  (covers typical Wolfram rules; cuts the 132 B binding and every task/MatchRecord). Exposed
  as build-time constants so a user can compile a larger-cap build; a rule exceeding the
  compiled cap hard-fails with a clear error (never silently truncates — cf. the F8 bug).
  Comment the tradeoff at the definition site. Longer-term direction: remove the fixed caps
  entirely via variable-length / arena-backed binding+edge storage sized to the actual rule,
  which also deletes the per-record padding for good (the caps become a fast-path SBO, not a
  hard limit).
- **`Edge.vertices` small-buffer** for arity ≤ 2 (removes an alloc + a cache miss per edge).

Forwarding:
- **Reference, don't deep-copy** — store the immutable match core once at the origin state;
  forward `{const core*, source_state}` + a per-child consumed-edge validity filter; drop
  one of the two redundant (push-recursion / pull-walk) producers.

Storage:
- **SparseBitset copy-on-write** — share immutable parent chunks by pointer, copy only the
  ~2 chunks a consumed/produced edge touches (today: memcpy *all* chunks per child; ~20×).
- **Delete dead `Hypergraph::state_children_`** (reader has zero callers).
- **Gate `event_canonical_state_map_`** on event-canon enabled; alias to `canonical_state_map_`
  in Full mode.
- **`ConcurrentMap`: free superseded resize tables** at quiescence (~2× map memory).

Tasks / scheduler:
- **Inline SINK** (zero fan-out finaliser — direct call, not a scheduled job).
- **Inline tagged-union task** in the deque instead of a per-task heap `FunctionJob` `new`.
- **Per-worker completion counters** (summed at quiescence) instead of two contended
  `seq_cst` globals; **no-box injector** (`Job*` inline, not heap-celled).
- **Hoist per-(state,rule) invariants** into one arena context; tasks carry an index.

Pattern matching:
- **Kill the 17.4 KB context copy** (`sig_caches`/`pattern_sigs`) — read through `ctx.rule`.
- **Remove `signature_compatible` from the bound path** (strictly subsumed by
  `validate_candidate`).
- **Demote the signature index** to repeated-variable pattern edges only; all-distinct
  edges seed from the arity/inverted index (no Bell(k) re-union).
- **Mutate/undo DFS** — bind/unbind + push/pop instead of copying `VariableBinding` (132 B)
  and `PartialMatch` (~224 B) per candidate.
- **`count==1` probe skip** in shortest-list seeding.
- **FIX F8 (correctness):** `MAX_CACHED_SIGS=64` truncates `Bell(6)=203` → silently missed
  matches for arity ≥ 6 all-distinct SCAN seed edges. Must fix regardless of the redesign.

### 5. CPU / GPU shared factoring

- **`hgcommon` widens** to host the reworked hot logic as `HG_HD` (host+device): the
  delta-matcher, WL+IR canonicalization, the causal/branchial closure/redundancy check,
  and the storage membership/COW primitives.
- **CPU, GPU-lockstep, GPU-persistent are three schedulers over one set of shared kernels.**
  Only scheduling differs (host loop vs level-sync device launches vs device-resident
  persistent scheduler); the per-item logic is a single implementation. This is the
  "code reused at every functionally-equivalent intersection" invariant made structural.
- Persistent-kernel work (device-side scheduler + work queue) is de-risked on the laptop
  GPU first, then run on the 4090.

## Sequencing (each phase gated on exact-oracle equivalence + benchmarks)

- **Phase 0 — Zero-waste quick wins** (no structural/naming decisions; each independently
  moves the wall; one commit each): WL hash-only path, `Event.binding`, `MatchRecord` dead
  fields, keys-only sets, dead `state_children_`, gate `event_canonical_state_map_`, pattern
  F1 (context copy) + F2 (redundant signature check), **configurable caps default 8/8**, the
  causal **base-layer** wins (drop `Anc`, key-only uint32 sets, lazy empties, reconvergence
  skip — these are also the foundation the oracle in Phase 2 sits on), and the **F8
  correctness fix**. Held pending Richard's doc review.
- **Phase 1 — Structural memory**: SparseBitset COW, forwarding-by-reference,
  `ConcurrentMap` reclaim, incremental hashing (wire the delta already handed in), Edge SBO,
  task-system rework (inline SINK, inline task union, per-worker counters, no-box injector),
  pattern mutate/undo DFS + signature-index demotion.
- **Phase 2 — The coupled big design** (one design, prototyped behind flags,
  differential-tested vs full materialisation vs oracle): the **hierarchical/generational
  allocator (Tier F)** + **automorphism reconstruction** (Tier D = quotient only), together;
  and the **causal reachability-oracle** redesign (chain-decomposition first candidate) on
  top of the Phase-0 base layer.
- **Phase 3 — CPU/GPU shared factoring** — widen `hgcommon` to the reworked hot logic;
  CPU / GPU-lockstep / GPU-persistent as three schedulers over one kernel set; persistent-
  kernel de-risked on the laptop GPU, then the 4090.

## Decisions — status

Decided (2026-07-21): caps default **8/8, compile-time-configurable, hard-fail on exceed**
(long-term: remove caps via variable-length records); causal = **full reachability-oracle
redesign** (base-layer wins in Phase 0, oracle in Phase 2); reclamation + automorphism
reconstruction = **one coupled design** (Phase 2).

Still open (raise during/after doc review): naming of the allocator tiers / new modules;
whether to stream Tier-D output + free (frontier-resident mode) now or later; which
reachability-oracle candidate to prototype first if chain-decomposition width turns out large.
