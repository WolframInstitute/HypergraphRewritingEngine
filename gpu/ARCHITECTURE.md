# GPU Multiway Hypergraph Evolution — Architecture

**Status:** living design doc; updated as design evolves alongside implementation.
**Owner:** GPU rewrite branch (`gpu-rewrite`).
**Companion:** the code itself; this document captures *why* and *what*, not duplicates of API surface.

---

## 1. Goals and non-goals

**Goals**
- A second, first-class GPU implementation of multiway hypergraph evolution alongside the existing CPU implementation.
- Bit-identical output to the CPU implementation on the same workload (rules, initial state, step count, canonicalization mode, hash strategy, event canon mode), at the level of *equivalence-class sets* — canonical states, events, causal edges, branchial edges all match by isomorphism-key.
- Strong scaling on a single GPU. Throughput should grow with available SMs; no fixed serial bottleneck.
- Configurable across small (< 100 init edges) and large (> 10k init edges) workloads. Auto-tuner picks per-device kernel parameters; cached after first run.
- Modern, clean, well-tested CUDA. Differential test against the CPU reference for every kernel.

**Non-goals**
- Multi-GPU. Single-GPU only for the first pass; multi-GPU is a follow-on.
- Pre-Turing hardware. Baseline is **sm_75 (Turing, 2018)**: independent thread scheduling, modern atomic semantics, full Cooperative Groups. Older hardware would force conservative primitives we don't want as the default.
- Replacing the CPU implementation. Both stand on their own; the user picks per workload.

---

## 2. Workload characterization (what we are building for)

The CPU semantics that the GPU must reproduce (deeper detail in `hypergraph/include/hypergraph/`):

- **State** = per-state edge set over a globally unique, immutable edge pool. Each `Edge` is `{arity, vertex_offset, signature, creator_event, step}`; vertices live in a flat pool.
  - **CPU** uses `hypergraph::SparseBitset` (chunked 512-bit segments, only non-empty chunks allocated — O(1) membership and O(actual-occupancy) memory).
  - **GPU** uses a CSR layout: `state_edge_slices[sid] = (offset, count)` + a flat `state_edge_ids[]` pool of sorted EdgeIds. CSR trades O(log N) binary-search membership for simpler GPU allocation (one `claim_n` per rewrite, no per-chunk atomic pool). See §6.
  - Both representations are O(actual-occupancy) memory, not O(max_edges), so neither has the quadratic-memory wall of a dense bitset.
- **Pattern matching** is Wolfram-style:
  - Variable bindings need **not be distinct** — `{x,y}` may match `{0,0}` with `x=y=0`.
  - Edge consumption is **one-to-one** — within a single match, each pattern edge consumes a different subject edge (tracked via per-match consumed-edges bitmap).
  - Directed: vertex order in an edge matters; `{x,y}` ≠ `{y,x}`.
  - Pattern-matching algorithm: signature-partitioned candidate generation, inverted vertex→edge index, depth-first extension via `PartialMatch`.
- **Rewrite**: copy bitset, remove consumed, allocate fresh vertex IDs (atomic counter), append new edges per RHS pattern with variable resolution.
- **Dedup**: canonical-state hash → first writer wins via lock-free CAS. Hash strategy and canonicalization mode are user-configurable.
- **Causal edges** (event→event): created when one event consumes an edge another event produced. Detected via *rendezvous*: producer marks itself, consumer marks itself, at least one side observes the other and creates the edge. Consumed edges are processed in **descending producer-EventId order** to make online transitive reduction terminate correctly.
- **Branchial edges** (event↔event same input state): created between events from the same input state that share at least one input edge.
- **Event canonicalization**: optional, hashes events by configurable signature (input/output state, step, rule, edge positions). Duplicates point to the first-occurrence canonical event.
- **Append-only invariant**: during evolution, all containers are append-only — no deletions. Bulk-freed at the end.

The workload character that drives the architecture:

- **Embarrassingly parallel state-level**, but per-state work is highly variable (small states have tiny pattern-matching work; large states with many candidate edges per pattern position can have huge work).
- **Producer/consumer streaming**: state production rate, match production rate, and event production rate all vary independently; no fixed batch sizes.
- **No natural step barrier**: the user explicitly does not want global "phase" synchronization. Causal/branchial edges should be created *as states are produced*, not as a separate post-pass.

---

## 3. Architecture overview

The whole evolution runs as **multiple persistent CUDA kernels on independent streams, communicating via lock-free queues, with no inter-phase global synchronization**. Conceptually it mirrors the CPU `ParallelEvolutionEngine`+`JobSystem`: a fixed pool of workers pulls heterogeneous tasks from queues, produces follow-up tasks, and the system terminates when every queue is empty and every worker is idle.

**Target (persistent-kernel streaming — M7 design):**

```
   ┌──────────────┐  match_q   ┌─────────────────┐
   │ MatchKernel  │──────────▶ │ RewriteKernel   │
   │ (persistent) │            │ (persistent)    │
   └──────▲───────┘            └────────┬────────┘
          │                             │ raw new state
          │ new state                   ▼
          │ to match    ┌────────────────────────┐
          │             │ IRCanonKernel          │
          │             │ (persistent)           │
          │             └─────────┬──────────────┘
          │                       │ canonical id  (and: was_new ?)
          │                       ▼
          │             ┌────────────────────────┐
          └─────────────┤ EventRegisterKernel    │
              if new    │ (persistent)           │
              canonical │  - allocate event id   │
                        │  - causal rendezvous   │
                        │  - branchial pair scan │
                        │  - event canon dedup   │
                        └────────────────────────┘

  All kernels share:
    edge pool, vertex pool, state pool   (append-only, atomic-bumped)
    canonical_state_map, event_canon_map (open-addressing CAS hash tables)
    edge_producer_map, branchial_edge_set (rendezvous + dedup tables)
    work queues match_q, rewrite_q, dedup_q, event_q  (lock-free MPMC rings)
    termination_detector                 (CPU JobSystem-equivalent: submitted == completed)
```

**Current (as of Stream 4, 2026-04-23):** the pipeline runs as **batched one-shot kernel launches per step**, not persistent kernels. The host loop in `Engine::Impl::run` does: match → sync → rewrite → sync → IR-canon → dedup → read next_count → swap frontier → repeat. Persistent kernels + streams are M7 (pending); all the shared data structures, the lock-free MPMC ring, and the termination detector are implemented and unit-tested, but the match/rewrite/canon kernels are not yet wrapped in the persistent-outer-loop form. Moving to persistent kernels is additive — the current batched form will stay valid and can be selected via a compile-time or runtime flag for debugging.

Why this works:
- **No barriers**: a worker block in `MatchKernel` can be running step 7's matches while another worker block in `EventRegisterKernel` is still registering causal edges from step 4. They share data structures and synchronize via atomics on the structures themselves, not via global epoch barriers.
- **Bounded register pressure**: each kernel has a single, narrow role. Register footprint per kernel is small, occupancy is high. (The previous megakernel's nvlink "stack size cannot be statically determined" warning was exactly the symptom of stuffing every role into one kernel.)
- **Independent profiling**: each kernel can be benchmarked, tuned, and replaced individually. No 70-KB megafile.

CUDA Graphs is *not* used for the inner loop (would impose phase structure). It may later be used to capture the kernel-launch sequence at startup as a one-shot graph for the entire evolution, but the kernels themselves run persistently.

---

## 4. Concurrency model (no-phase streaming)

This is the load-bearing design choice. Restated:

> Evolution does **not** proceed in synchronized phases (e.g. "first all step-N matches, then all step-N rewrites, then all step-N causal edges"). Instead, work flows freely through the pipeline; any worker on any kernel processes its next available task whenever upstream produces one.

Mechanism:

- **Persistent kernels.** Each role (Match, Rewrite, HashDedup, EventRegister) is a single kernel launched once. It contains an outer loop that pulls work from its input queue and pushes to the next queue, until the termination detector signals exit.
- **Lock-free MPMC queues.** Ring buffers with `head` / `tail` atomic counters. Producers `atomicAdd(tail, 1)` to claim a slot, write payload, then publish via release fence. Consumers `atomicAdd(head, 1)` to claim, read payload after acquire fence. Backpressure handled by capacity check on push (drop-and-stall, with `__nanosleep` retry on Turing+).
- **Termination detection** (`Hong-style`): each kernel maintains `tasks_done` and `tasks_pushed` counters; a small `TerminationDetector` block periodically checks `sum_pushed == sum_done` across all queues. When true, sets a global `should_exit` flag; kernels see it on next loop iteration and exit.
- **Concurrency primitives** (mirroring the CPU layer in `hypergraph/include/hypergraph/`):
  - **Open-addressing hash table with EMPTY/LOCKED sentinels** (mirrors `concurrent_map.hpp`). `atomicCAS(slot, EMPTY, LOCKED)` reserves; if successful, write value, then publish key with release. Readers skip LOCKED slots without spinning. First-writer-wins for canonical ID resolution.
  - **Append-only segmented arrays** (mirrors `segmented_array.hpp`). `atomicAdd` claims an index; pre-allocated mega-pool, no dynamic segment allocation needed at GPU scale (we know upper bounds via auto-tuner).
  - **Causal rendezvous** (mirrors `causal_graph.cpp:85–115`). Producer kernel does `atomicCAS(producer_field, INVALID, my_event_id)` with release; consumer kernel does `lock-free-list append + atomicLoad(producer_field)` with acquire. At-least-one-side-detects invariant preserved.
  - **Edge equivalence union-find** (if/when needed for canonical-event work) — `atomicCAS` on parent array, best-effort path compression.
- **Memory ordering**: `__threadfence()` paired with the atomic ops; `cuda::atomic<T, scope>` with `cuda::std::memory_order_release` / `acquire` from libcu++. No `seq_cst` anywhere — too expensive on GPU and unneeded; we use the same release/acquire pattern the CPU uses.

---

## 5. Kernel roles & decomposition

### 5.1 `MatchKernel` (warp-centric DFS, the heart)

**Current (Stream 1):** batched one-shot kernel, one **block per (state, rule)** pair with 32 threads per block. Threads stride over pattern-edge-0 candidates for their (state, rule); each thread runs its own DFS subtree from its seed candidate. DFS emits `MatchRecord`s into a shared atomic pool.
**Target (M7):** persistent kernels with `(state_id, rule_id)` tasks streamed through an input queue and `MatchRecord`s streamed to an output queue.

Structure follows G2Miner (OSDI'22) warp-centric approach + HGMatch / MaCH (arXiv 2512.10621) Match-and-Filter (our **adapted HGMatch**):

- **LHS is scheduled by connectivity on the host** (see `schedule_lhs_edges` in `match.cu`). Edge 0 is the seed; every subsequent edge carries a `pivot_var` — an LHS variable guaranteed already bound at that DFS depth.
- **At DFS depth 0**, candidates come from the signature index (no bindings yet, so signature-compatible coarsenings are the only pre-filter).
- **At DFS depth ≥ 1**, candidates come from `vertex_inverted_index[binding[pivot_var]]` — typically 2–10 entries instead of the hundreds-to-thousands the signature bucket would return. Duplicates (self-loops, concurrent-insert interleavings) are deduped in a per-DFS-frame `seen[]` buffer (kMaxIncidentSeen = 256); on overflow we fall back to the signature-index walk (correct but slower).
- **Wolfram-specific tweaks**: variable bindings are non-distinct — no "different vertices" constraint. One-to-one edge consumption is checked via a linear scan of `matched_edges[]` (≤ 16 entries).
- **Match dedup** (M4.9, pending): not yet implemented on GPU. Matches are emitted as-is; structural duplicates would show up as duplicate rewrites, which the post-rewrite state dedup collapses. Adds a hash-on-consumed-edges dedup in a follow-up.

**Match-kernel result:** on `n1000_s1` the per-step match kernel went from 944 ms to **4 ms** (236×) after switching from global-signature-walk to inverted-index at depth ≥ 1. This is the single biggest performance fix so far.

**Per-warp shared-memory state** (~2 KB, well within Turing budget):
- partial match (`MAX_PATTERN_EDGES` entries)
- consumed edges bitmap
- variable bindings (`MAX_VARS` entries)
- DFS stack (bounded by `MAX_PATTERN_EDGES`)
- iteration cursors over CHS for each query edge

### 5.2 `IndexKernel` (per-task: when a new state is added)

The signature partitions and inverted-vertex index are not rebuilt per step — they are **incremental**. When `HashDedupKernel` admits a new canonical state, it enqueues an `(state_id, edges)` index task that updates:
- signature → state-edges map (atomic append per bucket)
- vertex → state-edges map (atomic append per vertex)

Per-edge work is small; runs on a single warp or a few threads.

### 5.3 `CHSBuildKernel` (per (state, rule), before MatchKernel can process)

Builds the per-state Candidate Hyperedge Space for one rule against one state. This was MaCH's CHS — the bipartite-like structure that lets MatchKernel run cheaply.

- One warp per (state, rule, query-edge) candidate-set construction.
- Reads the global signature index, filters by signature, runs initial connectivity prune.
- Output: a compact CHS per (state, rule), kept in a memory pool keyed by `(state_id, rule_id)`.

This is the second-most-expensive kernel after MatchKernel. It feeds match tasks into the queue automatically when complete.

### 5.4 `RewriteKernel` (persistent)

**Input queue:** matches.
**Output queue:** raw new states for HashDedupKernel.

For each match: copy the parent state's edge bitset, clear consumed bits, atomic-bump fresh vertex IDs from the global counter, append new edges (atomic append to global edge pool), build the new bitset. One warp per match — bitset operations and edge appends parallelize cleanly.

### 5.5 `HashDedupKernel` + `IRCanonKernel`

**Input queue (target):** raw new states.
**Output queue (target):** dedup decisions `(parent_state, raw_new_state, canonical_id, was_new)`.

**Current:** `compute_state_ir_hashes_range` produces canonical hashes for a contiguous range of states (the "just produced by the current step's rewrite" range). A separate `k_dedup_and_append` kernel runs afterwards to populate the next frontier, first-writer-wins on `canonical_state_map`. Both kernels are one-shot batched launches per step; persistent streaming is M7.

Canonical hash strategy is always IR (the configured-strategy surface is gone — see §7). If IR's fast-path shared-memory scratch can't fit the state, the kernel silently falls back to `wl_hash_state_device` (1-WL) — iso-invariant, so correctness-safe for dedup, just with soft false-positive risk on symmetric graphs.

Events are created for *every* rewrite, regardless of whether the resulting state was a duplicate — currently done inline in the rewrite kernel rather than a separate queue-driven `EventRegisterKernel` step.

### 5.6 `EventRegisterKernel` (persistent)

**Input queue:** `(parent_state, child_state, consumed_edges, produced_edges, rule_id, step)`.

For each event:
1. Allocate a fresh `EventId` (atomic counter).
2. Optionally compute event signature and dedup against `event_canonical_map` (if event canon enabled).
3. **Causal rendezvous**: for each consumed edge in descending producer-EventId order:
   - `lock-free-list append` this event to the edge's consumer list.
   - `atomicLoad(edge_producer_map[edge])` with acquire — if non-INVALID, create causal edge `(producer, this)`.
4. **Causal producer mark**: for each produced edge:
   - `atomicCAS(edge_producer_map[edge], INVALID, this_event_id)` with release.
   - For each existing consumer of this edge (read consumer list), create causal edge `(this, consumer)`.
5. **Branchial scan**: read the parent state's event list (lock-free list keyed by `parent_state`); for each prior sibling event, if they share at least one input edge, atomic-insert branchial edge `(min(this,sib), max(this,sib))` into `branchial_edge_set` for dedup.

The "at least one side detects" invariant from CPU is preserved: producer's `for_each(consumers)` and consumer's `atomicLoad(producer)` together cover both orderings.

### 5.7 Termination detector

Tiny dedicated kernel block. Each task type maintains `(pushed, completed)` counters. Detector polls (with `__nanosleep`) until all pairs equal across all kernels for a stable interval, then sets `should_exit`. All persistent kernels see it and exit.

---

## 6. Memory layout

All data structures are pre-allocated based on auto-tuner sizing or user-passed `MAX_*` hints. No dynamic allocation in device code; all "allocations" are atomic counter bumps.

```
Global state:
  edges: Edge[MAX_EDGES]               // {arity, vert_offset, signature, creator_event, step}
  vertices: VertexId[MAX_VERTICES]     // flat pool, indexed by edges[i].vert_offset
  state_edge_slices: StateEdgeSlice[MAX_STATES]      // {offset, count} per state (CSR row descriptor)
  state_edge_ids:    EdgeId[MAX_STATE_EDGE_TOTAL]    // CSR-packed sorted EdgeId runs per state
  states_meta: State[MAX_STATES]       // {step, parent_event, canonical_id, ...}
  // Replaced the legacy O(MAX_STATES × MAX_EDGES) bitset — the sparse CSR
  // representation is O(total-live-edges) instead of O(states × edges),
  // removing the quadratic memory wall and enabling larger workloads.
  // Slices are kept sorted ascending so state_contains is O(log n) binary
  // search; rewrites preserve sortedness because produced edges (from
  // edge_pool.claim_n) are always consecutive and > any parent edge ID.

Indices (incremental, append-only):
  signature_buckets: CSR (bucket_offsets[N_BUCKETS+1], edge_ids[MAX_EDGES])
  vertex_inverted:   CSR (vert_offsets[MAX_VERTICES+1], inverted_edge_ids[MAX_INV_ENTRIES])

Hash tables (open-addressing, EMPTY/LOCKED, capacity = 2× expected):
  canonical_state_map: hash → StateId
  match_dedup:         hash → MatchId
  event_canonical_map: hash → EventId

Rendezvous structures:
  edge_producer_map[MAX_EDGES]: atomic EventId, default INVALID
  edge_consumers_head[MAX_EDGES]: atomic uint32_t (head of lock-free list per edge)
  edge_consumers_pool: ConsumerNode[MAX_CONSUMER_NODES]

Branchial:
  branchial_edge_set: hash table (event_pair_hash → 1)  -- dedup only

Work queues (MPMC ring, lock-free):
  match_q, chs_build_q, rewrite_q, dedup_q, index_q, event_q
```

CSR indices and bitsets give us **coalesced memory access** for the hot kernels. Per-state edge bitsets are small (~64–256 words for typical workloads), live in L1/shared during use.

---

## 7. Hashing & canonicalization

**IR is the only canonicalization strategy on GPU.** It is exact (no false positives or false negatives), whereas WL has documented false positives on highly symmetric graphs (Cai–Fürer–Immerman, strongly regular, certain Cayley graphs) and UT (Gorard-style uniqueness trees) is a polynomial approximation with known collisions. Because the deduplication invariant requires *isomorphism-correct* equivalence — false positives merge non-isomorphic states and corrupt the multiway graph — IR is the only strategy that is unconditionally safe, and the alternatives have been removed from the public API.

GPU IR canonicalization (mirrors the CPU `IRCanonicalizer` in `hypergraph/src/ir_canonicalization.cpp`):

- **Fast path (≈ 99 % of Wolfram-evolution states):** one block per state, one warp per block. Per-vertex initial colour = hash over the sorted multiset of `(arity, position)` occurrences. 1-WL refinement iterates: `new_colour[v] = hash(sorted[{(arity, position, sorted co-vertex colours)}])`. If the refined colours are all distinct (discrete partition), the per-vertex rank-in-sorted-colours is the canonical labelling. Apply labelling to edges, sort edges lex, emit FNV-1a.
- **Slow path (non-discrete after 1-WL):** individualisation-refinement backtrack. Pick the first non-singleton cell, for each vertex in it individualise → re-refine → recurse; keep the lex-smallest canonical form across the tree. Only triggers on graphs with non-trivial automorphism group (rare in evolutions from asymmetric initial conditions).

Per-state block layout (see `gpu/src/ir_canon.cu`):
- Shared-memory `IRBlock` holds the state's packed edge list, vertex list, per-vertex occurrence CSR, current/next colour arrays, and a per-thread scratch buffer sized for `kMaxIROccs` neighbour-signature entries.
- Scratch bounds: `kMaxIRVerts=128`, `kMaxIREdges=128`, `kMaxIROccs=256`. Cover typical evolution states up to ≈ 100 vertices / 100 edges. **When a state doesn't fit** (large initial states like n=1000, or deep evolutions that accumulate), the kernel silently **falls back to `wl_hash_state_device`** — 1-WL sorted-colour-multiset hash directly on the CSR slice, iso-invariant but weaker than exact IR. No error is recorded for this path; it's a planned graceful degradation.
- The slow-path IR backtrack is S3.5 (pending); closes the `wolfram_canonical_steps5` differential fail.
- A global-memory `IRBlock` variant is the eventual fix for fast-path oversize states (so we get exact IR even on n=1000-class workloads); deferred until after the differential close.

The `HashStrategy` enum / `set_hash_strategy` surface is gone from the public `EvolveInput`. `gpu/src/wl_hash.cu` is retained specifically as the IR fast-path's oversize-state fallback — it is NOT user-selectable.

**CPU-vs-GPU state representation:** CPU uses `hypergraph::SparseBitset` (chunked 512-bit segments, only allocating non-empty chunks) for O(1) membership with sparse-proportional memory. GPU uses CSR (sorted per-state edge-id slice + global flat pool) for O(log state-size) membership with sparse-proportional memory. Both avoid the dense-bitset memory cliff. A GPU-side port of the sparse chunked bitset (pool of 64 B chunks with per-state chunk index) would restore O(1) membership on GPU too and is logged as a future optimisation; CSR is good enough for the current benchmark targets.

---

## 8. Hardware compatibility

Baseline: **sm_75 (Turing, RTX 20-series, 2018+)**.

Required features:
- Independent thread scheduling (post-Volta) — required for correctness of warp-cooperative algorithms with conditional control flow.
- Cooperative Groups (`tiled_partition`, `coalesced_threads`) — used pervasively.
- `__nanosleep` — used in spin-yield loops on hash-table LOCKED slots and termination detector.
- 64-bit atomics on global memory — used in canonical hash CAS.
- Sufficient shared memory per SM (Turing has 64 KB; we use ≤32 KB per block).

Conditional features (compile-time `__CUDA_ARCH__` dispatch):
- **sm_80+** (Ampere+): `cp.async` for overlapping global loads with compute in CHS construction. Fallback: synchronous coalesced loads.
- **sm_90+** (Hopper+): TMA for batched edge loads in MatchKernel hot path. Fallback: vectorized 128-bit loads.

We compile for `sm_75;sm_80;sm_86;sm_89;sm_90` by default; CMake exposes `HG_GPU_ARCHS` to override.

---

## 9. Auto-tuning

Per-device tuning of kernel launch parameters and memory pool sizes. Lives in `<binary_dir>/hg_gpu_tuning_cache.json` (per-binary, not per-user).

Tuned parameters:
- `MatchKernel`: persistent block count, threads per block, virtual-warp size (8/16/32 — Hong et al.), per-warp shared-memory budget.
- `CHSBuildKernel`: block size, candidates-per-warp.
- `HashDedupKernel`: hash table load factor (0.5/0.75), probe-sequence stride.
- `EventRegisterKernel`: per-block consumer-list scan batch size.
- Memory pool sizes: `MAX_EDGES`, `MAX_STATES`, `MAX_INV_ENTRIES`, queue capacities.

Tuning protocol:
- First evolve() call on a new GPU: run a 30-second sweep over a representative workload. Pick best params per kernel by wall-time.
- Cache keyed by `(compute_capability, sm_count, total_global_mem, driver_version)`.
- Cache file format: JSON, hand-editable for debugging.
- We may build our own segmented sort / radix sort optimized for our distribution (Wolfram rules tend to produce specific arity patterns); profile-guided.

---

## 10. Differential testing

**Built first, before any kernel.** Lives in `gpu/tests/test_gpu_vs_cpu_differential.cpp`.

```cpp
struct Workload {
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> initial_state;
    int steps;
    // Canonicalization is always IR (the legacy HashStrategy enum is gone —
    // see §7). canon_mode still selects between None / Automatic / Full.
    StateCanonicalizationMode canon_mode;
    EventCanonicalizationMode event_canon_mode;
    bool transitive_reduction;
    bool explore_from_canonical_states_only;
};

TEST_P(DifferentialEvolution, BitIdenticalCanonicalForm) {
    auto w = GetParam();
    auto cpu = run_cpu(w);
    auto gpu = run_gpu(w);
    ASSERT_EQ_SETS_BY_CANONICAL_HASH(cpu.states, gpu.states);
    ASSERT_EQ_SETS_BY_HASH(cpu.events, gpu.events);
    ASSERT_EQ_SETS(cpu.causal_edges, gpu.causal_edges);
    ASSERT_EQ_SETS(cpu.branchial_edges, gpu.branchial_edges);
}
```

**Initial workload corpus**: lifted from `hypergraph/tests/test_determinism_fuzzing.cpp` (the rule + initial-state combinations there are already vetted). Augmented with:
- **2-edge LHS** patterns (smallest non-trivial).
- **3-edge LHS** patterns.
- **Mixed-arity** rules (arity-2 + arity-3 in the same rule).
- **Wolfram canonical** `{{x,y},{x,z}}->{{x,y},{x,w},{y,w},{z,w}}` over multiple step counts.
- **Self-loop initial states** (`{(0,0)}`) — exercises non-distinct vertex binding.
- **Multi-rule** systems (3-5 rules at once).
- **Stress**: 50 random rules × 100 random initial states × {1,3,5} steps × all canon modes × all hash strategies.

Compare on **canonical equivalence classes**, not raw IDs. Set equality with isomorphism-key.

GPU `run_gpu()` initially returns empty results — every test fails until each kernel lands and closes its column.

---

## 10.5 Current implementation status (2026-04-23)

High-level snapshot of what's done vs what's planned. Detailed sub-task granularity lives in the TaskList; this is the living overview.

### Shipped (on the `gpu-rewrite` branch; commits `62adafe`, `1a5e880`, `5060cd2` pushed to local bare)

- **M1 — Foundation.** `gpu/` scaffold, headers, CMake, differential-test harness, initial determinism-fuzzed workload corpus.
- **M2 — Primitives.** Atomic counter pool, open-addressing concurrent hash table (EMPTY/LOCKED CAS), MPMC ring buffer, append-only segmented pool, lock-free per-key list, termination detector, warp-cooperative helpers, memory-ordering audit. Each with unit + concurrency stress tests.
- **M3 — Indices.** Signature index (bucketed), vertex inverted index, edge-signature computation on device. "Candidate gen matches CPU" differential test passes.
- **M4 — Match pipeline.** PartialMatch device layout, CHSBuildKernel (signature-partitioned candidates), warp-centric DFS MatchKernel, connectivity-filter + intersection-constraint Match-and-Filter, Wolfram non-distinct binding, one-to-one edge consumption. "Matches identical to CPU" passes.
- **M5 — Rewrite + dedup.** RewriteKernel with fresh vertex / edge allocators, first-cut GPU WL hash (now repurposed as IR's oversize-state fallback), device-side HashDedupKernel, full pipeline wire-up, "states match CPU through dedup" passes.
- **M6 — Events.** EventRegisterKernel (inline in rewrite kernel today), producer-side + consumer-side causal rendezvous, branchial scan, online transitive reduction with multiplicity preservation on the (producer, consumer) pair. Event/causal/branchial differential test passes on 11/12 workloads.
- **Stream 1 — HGMatch via inverted-index** (2026-04-22, commit `5060cd2`). Match kernel depth ≥ 1 uses `vertex_inverted_index[binding[pivot_var]]` instead of the global signature walk. LHS connectivity-ordered schedule on the host. 200–250× match-kernel speedup. (The canonical "we built the data structures; now use them" fix.)
- **Stream 2 — CSR per-state edges.** Replaced the O(`max_states` × `max_edges`) dense bitset with `state_edge_slices` + flat `state_edge_ids`. Unblocks large workloads that previously OOM'd or crashed. Binary-search `state_contains`. Sortedness invariant preserved by rewrite (produced edges' IDs always > parent's). In working tree, pending commit.
- **Stream 3 — IR-only canonicalization.** `HashStrategy` enum removed. New `gpu/src/ir_canon.cu` kernel implements the IR fast path (1-WL refinement + discreteness check + canonical labelling + lex-sort + FNV). Falls back to WL multiset hash on oversize states. In working tree, pending commit.
- **Stream 4 — Engine class / call reuse.** `hg_gpu::Engine` PIMPL holds persistent `EngineState` + `Pool<MatchRecord>` + `DedupMap` + per-kernel device buffers. `Engine::run(input)` resets and runs without re-allocating pools. Bench harness constructs one Engine per workload and reuses across warmup + measure. In working tree, pending commit. **This is what made small-workload benchmarks honest** — CUDA init was hiding the real numbers.
- **Stream 5 — Dynamic pool growth on overflow.** `grow_config_for` doubles the relevant `EngineConfig` field(s). Free `evolve(input)` retries up to 6× (64× capacity growth). On a successful retry, `log_winning_config` prints the fields that were grown beyond their initial values so users can pre-size next time (S5.4).
- **S5.6 — Overflow returns partial result, never throws** (2026-04-30). New `gpu/include/hg_gpu/overflow.hpp` CUDA-free header carves out `ErrorKind` + `OverflowWarning`; `EvolveResult` gains a `warnings` field. `Engine::run()` now drains the device error channel into `result.warnings` after each kernel sync (`collect_warnings_into`) and keeps running on the partial budget — capacity overflow is a warning, not an exception. Free `evolve()` reads the warnings list to drive its grow-and-retry loop. Genuine driver failures still throw; capacity limits do not.
- **S3.2b — GPU IR fast-path unit test** (2026-04-30, `gpu/tests/test_ir_canon.cu`). 12 tests covering iso-invariance, distinguishability, edge-ordering, mixed arity. 71/71 unit tests now green.
- **S6.1 — GPU `ExplorationProbability` pruning** (2026-04-30, `gpu/tests/test_exploration_probability.cu`). New `EvolveInput::exploration_probability` + `exploration_seed`; coin flip in `k_dedup_and_append` admits new states to next-step frontier with the configured probability, splitmix64-mixed over (seed, step, sid). Fast-path on `p==1.0`. First CPU-only pruning option to land on GPU; `MaxStatesPerStep` / `MaxSuccessorStatesPerParent` still pending. 77/77 unit tests green.

### Differential test scoreboard

- 11 / 12 workloads pass bit-identical equivalence-class comparison (canonical states, event multiset, causal multiset, branchial multiset).
- **Failing:** `wolfram_canonical_steps5`. Root cause: at 5-step depth the state graph has enough symmetry that 1-WL fails to fully discriminate, and the GPU IR kernel's backtrack fallback (S3.5) isn't written yet — so GPU falls back to 1-WL multiset hash, which has false-positive collisions on the CPU IR's finer partition. Closing this is the `wolfram_canonical_steps5` pass criterion for S3.5.

### Bench numbers (RTX 4090, Engine reuse, warmup 1, measure 1)

| Workload | CPU ms | GPU ms | Speedup | Final states | Events | Causal | Branchial |
|---|---:|---:|---:|---:|---:|---:|---:|
| n10_s1 | 0.89 | 1.54 | 0.58× | 5 | 8 | 0 | 4 |
| n10_s3 | 7.41 | 6.29 | 1.18× | 258 | 664 | 676 | 2288 |
| n50_s1 | 30.43 | 4.76 | **6.39×** | 7 | 132 | 0 | 522 |
| n50_s3 | 1143.45 | 88.93 | **12.86×** | 650 | 9 916 | 2 970 | 52 034 |
| n1000_s1 | 1 485 | 849 | **1.75×** | 63 | 15 136 | 0 | 435 728 |

Interpretation: GPU wins from n10_s3 up. n10_s1 is kernel-launch-overhead-bound (one state with eight events — nothing to parallelise over). Large (n ≥ 200) × deep (s ≥ 3) sweeps run but weren't included in the latest measured row set because the CPU-side multithreaded evolution ALSO takes minutes on those workloads — not a GPU regression, but not a useful comparison until we have compute budget for a full overnight sweep.

Per-kernel decomposition for n1000_s1 (HG_GPU_DBG_TIME=1):
- init 15 ms (once per Engine, amortised across run()s)
- match 4 ms (was 944 ms pre-Stream-1)
- rewrite 89 ms
- IR hash 23 ms
- dedup < 1 ms
- state-readback 570 ms (biggest residual cost — `all_state_edges_host` bulk-copies the whole edge pool + vertex pool + slice table; only matters for benchmarks that verify output)

### Pending

Immediate:
- **S3.5** Individualisation-refinement backtrack on GPU → closes the last differential fail.
- **S3.3b** Warp-parallelise IR refinement (currently thread 0 only within the block).
- **Stream 2/3/4/5 commit** once the user approves the split.

Next quarter of work (roughly in priority order):
- **M7.x** Persistent kernels + independent CUDA streams + backpressure. Kills the per-step host↔device sync overhead.
- **M9.x** Auto-tuning cache: per-device best-fit `EngineConfig` cached next to the binary, first-run sweep to populate, cache load + application on subsequent runs.
- **M10.4** Full benchmark-sweep comparison report (CSV + charts) once streaming is in and the last differential fail closes.
- **Global-memory `IRBlock` variant** so large-state IR is exact rather than falling back to 1-WL.
- **M4.9** Match dedup table on GPU (minor; the rewrite-side dedup collapses most duplicates already).
- **M6.6** Event canonicalisation (Full mode) on GPU.
- **M6.8** Reservoir sampling for width-limited evolutions (user flagged the CPU side needs a correctness review first).
- **M11.x** FFI / paclet bridge so Wolfram Language can call this engine.

### Known limitations

- **Match kernel is one-block-per-(state × rule)** with 32-thread cooperation on pattern-edge-0 candidates. DFS beyond depth 0 is per-thread. Warp-cooperative DFS (G2Miner style) is a future optimisation.
- **IR fast-path is single-thread within the block** (thread 0 does refinement). S3.3b.
- **State-edge readback via `all_state_edges_host`** is O(total live edges) memcpy. Fine for small workloads, expensive on deep n=1000 evolutions. Users who only need the canonical hash set can stay on-device.
- **Oversize states fall back to 1-WL** (soft accuracy loss, not correctness). Will be a hard correctness loss on adversarial symmetric graphs until S3.5 and/or the global-memory IRBlock lands.

---

## 11. Build order (matches the task graph)

See the task list (TaskList) for the full breakdown. High-level milestones (status as of 2026-04-23):

1. ✅ **M1 — Foundation**: gpu/ scaffold, CMake, headers, `evolve()` host API, differential-test harness, initial workload corpus.
2. ✅ **M2 — Primitives**: lock-free MPMC ring, open-addressing hash table (EMPTY/LOCKED), atomic counter pool, append-only segmented pool, lock-free per-key list, termination detector, warp-cooperative helpers, memory-ordering audit.
3. ✅ **M3 — Indices**: signature index, vertex inverted index, edge-signature on device. Candidate-gen differential test passes. Index-update task (M3.5) still pending — would batch index updates instead of per-edge atomic insert.
4. ✅ **M4 — Match pipeline**: PartialMatch device layout, CHSBuildKernel, warp-centric DFS, connectivity + intersection-constraint filter, Wolfram non-distinct binding, one-to-one consumption. M4.3 (CHS connectivity prune kernel) and M4.9 (match dedup) still on the backlog as low-priority refinements.
5. ✅ **M5 — Rewrite + dedup**: RewriteKernel, fresh allocators, GPU WL hash (now repurposed as IR oversize-state fallback), HashDedupKernel, full pipeline wire-up. M5.4-proper (warp-cooperative IR) is partially done via Stream 3 — fast path lands, backtrack S3.5 pending.
6. ✅ **M6 — Events**: EventRegisterKernel (inline today, will move to its own kernel in M7), causal rendezvous, branchial scan, online TR with multiplicity. M6.6 event canonicalisation Full mode and M6.8 reservoir sampling still pending.
7. ⏳ **M7 — Streaming**: persistent kernels, independent CUDA streams, lock-free queue feeding, termination detector wired to all kernels, backpressure on full queues. **In design; primitives ready, refactor pending.** Largest expected perf win after the kernel-algorithm fixes.
8. ❌ **M8 — Hash variants** (CANCELLED): the original idea was WL + UT as alternatives to IR with collision verification. Per Stream 3 decision, IR is the only canonicalisation strategy. The task entries for M8.2 (GPU UT hash), M8.3 (WL/UT collision verification), M8.4 (strategy selector through host API), M8.5 (differential tests across all strategies) were deleted from the task list on 2026-04-23.
9. ⏳ **M9 — Auto-tuning**: tuning cache file format, device characterization, sweep harness, per-kernel tunables, first-run sweep + cache write, cache load + apply on subsequent runs, custom segmented sort.
10. ⏳ **M10 — Benchmark + report**: harness exists (M10.1 ✅); sweep at multiple scales, scaling analysis, comparison report, nsys/ncu profiling pass — pending the streaming refactor so the numbers reflect the final architecture.
11. ⏳ **M11 — Integration**: top-level CMake integration, FFI / WolframLanguage interface, paclet bridge + WL test, README, final compute-sanitizer pass.

Plus the **Stream 1–5 refactor track** (S1.x ✅ already shipped; S2/S3/S4/S5 in working tree, pending commit):
- S1: HGMatch via inverted-index (200× match speedup) — shipped.
- S2: CSR per-state edges (kills the O(N²) bitset).
- S3: IR-only canonicalisation. Sub-tasks S3.5 (backtrack) and S3.3b (warp-parallel refinement) still open.
- S4: Engine class with persistent state and run() reuse.
- S5: Dynamic pool growth on overflow; user-friendly retry in `evolve()`.

Each milestone ends with: differential tests green, no warnings, no memory errors under `compute-sanitizer`, benchmark numbers recorded.

---

## 12. Tweak log

(Append-only record of design changes after first draft. New entries dated.)

- **2026-04-20** — Initial draft. Architecture as above. To be revised as implementation reveals constraints.
- **2026-04-20** — Set IR as default hash strategy (was: configurable, no preference). Reason: WL false positives are a correctness hazard for the dedup invariant, and the user explicitly identified IR as preferred for safety.
- **2026-04-20** — Removed all global phase barriers in favor of fully-streaming persistent kernels. Reason: explicit user requirement that "states first then causal/branchial separately" is not the model.
- **2026-04-20** — Baseline raised from sm_70 to sm_75 (Turing+). Reason: user OK with restricting to modern hardware; Turing covers >90% of deployed CUDA hardware in 2026 and lets us assume `__nanosleep`, modern atomics, full Cooperative Groups without conditional dispatch.
- **2026-04-20** — Auto-tuning cache colocated with binary (was: `~/.cache`). Reason: user preference; keeps cache scoped to the build.
- **2026-04-22** — Stream 1 / commit `5060cd2`: match kernel uses `vertex_inverted_index[binding[pivot_var]]` at DFS depth ≥ 1 instead of the global signature walk. Connectivity-ordered LHS schedule on the host. **Single biggest perf fix so far** — match-kernel time on `n1000_s1` went from 944 ms to 4 ms. Reason: the GPU was using HGMatch in name only; the inverted index was already populated but never consumed by the DFS — the kind of "match the CPU's algorithmic choices, don't skip them just because they're complicated to port" mistake that should never happen again.
- **2026-04-22** — Stream 1 / commit `1a5e880` (precursor): preflight resource reservation in the rewrite kernel + typed `DeviceErrors` channel. Replaces the old "claim mid-kernel, silently early-return on overflow" pattern that caused spurious OOBs in downstream WL/dedup/readback. Every kernel-internal capacity overflow is now a typed exception with a specific `ErrorKind`, not a segfault.
- **2026-04-23** — Stream 2: replaced the dense per-state edge bitset (`state_edges_bits[max_states][max_edges/32]`) with CSR (`state_edge_slices` + flat `state_edge_ids`). Reason: bitset is O(`max_states` × `max_edges`) memory — multi-GB for any realistic workload, which forced aggressive sizing caps that then triggered overflow on medium-large workloads. CSR is O(total live edges); slices stay sorted because produced edges from `edge_pool.claim_n` always have IDs > parent's. CPU's `SparseBitset` (chunked, only allocates non-empty chunks) is the exact analogue on its side and would be the next-better fit for GPU too — logged as a future port.
- **2026-04-23** — Stream 3: `HashStrategy` enum removed from public API. `gpu/src/ir_canon.cu` lands the IR fast path (1-WL refinement + discreteness + canonical labelling + lex-sort + FNV). Backtrack fallback (S3.5) is the remaining IR work. Reason: WL/UT were never going to be selected — user view is that approximations are not strategies, they're bugs in waiting; IR is the only correct hash.
- **2026-04-23** — Stream 4: `hg_gpu::Engine` PIMPL class added. Pools, indices, and per-kernel device buffers are owned by the Engine and reused across `run()` calls. Free `evolve(input)` is a one-shot wrapper around `Engine(cfg).run(in)`. Bench harness now constructs one Engine per workload and reuses across warmup + measure iterations. Reason: every `evolve()` call was paying ~5–500 ms of CUDA init that completely hid the GPU's per-step performance on small workloads. With reuse, n10_s3 went from 150 ms (init-dominated) to 6 ms (real-work-dominated).
- **2026-04-23** — Stream 5: dynamic pool growth on overflow. Typed `DeviceErrors::PoolOverflow` exception carries the offending `ErrorKind`; `grow_config_for(cfg, kind)` doubles the relevant `EngineConfig` field(s); free `evolve()` retries up to 6× (64× max growth). `Engine(cfg).run(in)` does NOT retry — it throws so benchmarks can't hide retry cost in their wall_ms. Reason: users shouldn't have to think about pool sizing; "it just works first time" is the goal, and when it doesn't, the system grows itself rather than crashing.
- **2026-04-30** — S3.2b unit test for the GPU IR fast path (`gpu/tests/test_ir_canon.cu`, 12 cases) verifies iso-invariance + distinguishability + edge-order/direction sensitivity without asserting byte-equality with CPU `IRCanonicalizer` (different mixing). Caught two test-config bugs (max_vertices=64 too tight) and one wrong premise on my part — `{(0,1)}` and `{(1,0)}` ARE iso under directed-hypergraph semantics, replaced with a `{{0,1},{1,2}}` (path) vs `{{0,1},{2,1}}` (double-sink) test that genuinely distinguishes via vertex-1's in/out-degree profile. 71/71 unit tests now green.
- **2026-04-30** — S5.6 (architectural correction to Stream 5): GPU engine no longer throws on capacity overflow. Added `gpu/include/hg_gpu/overflow.hpp` (CUDA-free, exports `ErrorKind` and the new `OverflowWarning {kind, count, context}` POD). `EvolveResult` gains a `std::vector<OverflowWarning> warnings` field. `DeviceErrors::collect_warnings_into(vec, context)` drains the device counters into the vector and clears for the next sync. `Engine::run()` now never throws on capacity overflow — kernels keep running on whatever budget they have, partial result is returned with overflow tagged in `warnings`. Free `evolve()` rewrote its retry loop to inspect the warnings list (no try/catch on `PoolOverflow`); cumulative warning trail returned in the final result. Reason: the user's view that errors are for programmer mistakes only — runtime resource limits should yield partial results plus warnings, not crashes. Closes the philosophical hole the original Stream 5 left.
- **2026-04-30** — S5.4: `log_winning_config()` in `evolve.cu` prints the `EngineConfig` fields that were grown beyond their initial values when grow-and-retry succeeds. Format is stable so users can grep it and copy values into an explicit `Engine(cfg)` construction to skip the retry loop on the next call. Feeds the M9 auto-tune cache when that lands.
- **2026-04-30** — S6.1: GPU `ExplorationProbability` pruning. `EvolveInput` gains `exploration_probability` (float, 1.0 default) and `exploration_seed` (uint64_t, 0 = non-deterministic). The coin flip lives in `k_dedup_and_append`: a freshly-deduped state is admitted to the next-step frontier with probability ≈ `threshold/2^32`, where the per-(seed, step, sid) draw is a `splitmix64` mix. Mirrors CPU `ParallelEvolutionEngine::should_explore()` semantics — the state and its event are still recorded; only expansion from the new state is suppressed. Fast-path (`p == 1.0`) skips the hash work entirely so existing deterministic workloads pay zero overhead. Six unit tests in `gpu/tests/test_exploration_probability.cu` cover baseline parity, full suppression, intermediate shrinking, same-seed determinism, different-seed divergence, and out-of-range clamping. Closes the first feature-parity gap in the CPU pruning option set; `MaxStatesPerStep` and `MaxSuccessorStatesPerParent` remain pending as separate streams.
