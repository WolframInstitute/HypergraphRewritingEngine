# Code map

**This file is the INVENTORY: where code lives.** For what flows through it, who runs
concurrently, and which invariant each stage upholds, see `DATAFLOW.md`. For the CPU/GPU
boundary in detail see `../gpu/ARCHITECTURE.md`; for the observable contract see `SPEC.md`.


Per-file guide to the code-bearing tree: what each file is, and the classes /
structs / functions it contains. For the data-flow and the "start reading here"
path see [ARCHITECTURE.md](ARCHITECTURE.md); for using the paclet see
[QUICKSTART.md](QUICKSTART.md).

> Keep this current. Any change that adds, removes, moves, or splits a file, or
> renames a symbol, must update this map in the same change.

Top-level layout (build dirs, `.git`, `venv` omitted):

```
common/          Shared CPU/GPU foundation (ids, hashes, WL core, intrinsics)
hypergraph/      Core CPU engine: storage, matching, canonicalization, evolution
gpu/             CUDA port of the engine (mirrors the CPU algorithms)
job_system/      Lock-free work-stealing task scheduler
lockfree_deque/  Lock-free MPMC deque
wxf/             Wolfram Exchange Format serialization (the WL boundary)
paclet_source/   FFI: run_rewriting_core, the standalone hg_evolve binary, GPU marshaling
paclet/          The Wolfram Language paclet (kernel code + bundled binaries + doc notebooks)
reference/       Validation oracle (brute-force ground truth) + golden corpus + verify scripts
tools/           Standalone research / validation / profiling probes
testing/         gtest aggregation -> all_tests
benchmarks/ + benchmarking/   Per-area benchmarks + the framework lib
visualisation/   Viz-event interface only (renderer and analyses live in ../hypergraph_viz)
```

The natural reading path for a new developer: `common/core.hpp` ->
`hypergraph/include/hypergraph/{types,hypergraph,parallel_evolution}.hpp` -> the
matcher (`pattern_matcher.hpp`) and canonicalization (`wl_hash.hpp`,
`ir_canonicalization.hpp`) -> then either the GPU mirror
(`gpu/include/hg_gpu/evolve.hpp`) or the WL boundary (`paclet_source/hg_core.hpp`
-> `paclet/Kernel/HypergraphRewriting.wl`).

---

## `common/include/hgcommon/` -- shared CPU/GPU foundation

- **`namespace.hpp`** -- the ONE namespace root, and the knob that renames it. Every namespace this project defines nests under `HG_NAMESPACE` (default `hg`), so linking the engine adds one name to global scope rather than the nine it used to. The root is a MACRO because a library cannot know it has not collided: a host program with its own `hg` builds with `-DHG_NAMESPACE=whatever` and every symbol moves without editing the engine, which is only true while declarations open it as `namespace HG_NAMESPACE { namespace common {` rather than naming the root. The short aliases (`namespace hgcommon = HG_NAMESPACE::common;`) are the MIGRATION SEAM: they keep existing call sites resolving while subsystems move one at a time, they are the only thing still in global scope, and they are what a later pass deletes. Gated by `NamespaceRoot` in `test_slot_core.cpp`, which asserts at compile time that the root resolves AND that the alias names the same entity rather than a second declaration that merely agrees.

- **`phase_timing.hpp`** -- where a run's cycles go, by phase, on the HOST. The device has had this since the persistent kernel (per-block `clock64` deltas summed at exit into `PersistentEvolveStats`); the host had no equivalent, and the substitutes are wrong here -- a wall-clock timer cannot see inside a run where every worker is in some phase at once, and a sampling profiler attributes to the symbol rather than to the phase a symbol serves. `Phase` (the enum the buckets are named by), `PhaseTimer` (RAII, scoped to a phase), `phase_cycles()`, `phase_timing_compiled()`. Compiled out unless asked for, and when it is out `PhaseTimer` is an empty object with no counter read and no branch, so the instrumented and shipping builds are the same code. Read by `bench_cpu_evolve`, which prints the per-phase split beside its timings.

- **`core.hpp`** -- id typedefs, structural limits, integer hash primitives and the small-run sort both canonicalizers use (`HG_HD` host/device).
  - id aliases `VertexId`/`EdgeId`/`StateId`/`EventId`/`MatchId`, `INVALID_ID`; limits `MAX_ARITY`/`MAX_PATTERN_EDGES`/`MAX_VARS`
  - `HG_INLINE` -- force-inline, for functions whose inlining must not track unrelated code size
  - `mix64()` (Murmur3 finalizer), `fnv_hash()` (FNV-1a combine), `splitmix64()` (commutative-sum finalizer), `isort_u64()` (small-run insertion sort -- both canonicalizers sort per-vertex signature multisets and the device has no `std::sort`)
- **`wl_core.hpp`** -- the single shared Weisfeiler-Leman canonical-hash impl, bit-identical CPU/GPU.
  - `WL_MAX_REFINE_ITERS`; `wl_canonical_hash()` (occurrence-CSR build -> initial colouring -> refinement to fixpoint -> commutative fold)
- **`ir_core.hpp`** -- the single shared EXACT canonicalizer (individualization-refinement), one implementation for host and device.
  - `IR_HOST_GENERATORS`/`IR_DEVICE_GENERATORS` (search-budget split), `ir_scratch_words()` (caller-sized span, no allocation), `IrScratch`, `IrPartition`, `ir_heapsort_idx`
  - `ir_canonical_hash()` -- refine, search by individualizing the lowest non-singleton cell, smallest form ACROSS THE TREE'S LEAVES wins (a complete isomorphism invariant, not the minimum over all n! relabellings); optional outputs per input edge: canonical RANK (`out_edge_rank`), automorphism ORBIT and content CLASS (`out_edge_orbit`/`out_edge_class`, computed in input space from the discovered generators -- the quotient-causal DP's keys), and the winning form and labelling (`out_canonical_form`/`out_vertex_label`)
- **`quotient_causal_core.hpp`** -- THE QUOTIENT-CAUSAL DP, one body for host and device. `qc_key`/`qc_rkey`/`qc_seen_key` (the three key spaces the DP indexes) + `qc_reach`/`qc_process_transition`/`qc_add_producer`, the mutually recursive fixpoint. A Ctx supplies storage only -- producer sets, the reached set, transition enumeration, the causal-edge sink and the fence -- so which rendezvous scan follows which publish has one definition. The identification a reconstructed event gets is not a performance property. The producer/transition rendezvous is checked on the core itself by `verification/genmc/dp_producer_meets_transition.cpp` (RC11) and `verification/gpumc/dp_producer_meets_transition.cpp` (scoped RC11).
- **`quotient_replay_core.hpp`** -- THE PER-INSTANCE REPLAY, one body for host and device. `qr_apply` applies one recorded class-frame match to one instance: claims the pair exactly once (`qr_apply_key`), mints the raw event, identifies it twice (`qr_content_hash` for the isomorphism-invariant triple, `event_signature` for the run's mode), records the causal relations in DESCENDING producer order (`qr_collect_producers`), forms the branchial pairs by publish-then-scan on the instance's applied list, and descends to the child instance. `QR_NO_PRODUCER` is the slot-has-no-producer sentinel. A Ctx supplies storage only.
- **`transitive_reduction.hpp`** -- THE TRANSITIVE REDUCTION of a stored relation, one body for every caller. `tr_reduce(enumerate, emit, ids_topological)` returns the pairs of a DAG that no longer path bypasses. Computed from the FINISHED set because a finite DAG's reduction is unique and therefore independent of the order the pairs arrived in; deciding membership as each pair lands answers against the pairs seen so far and keeps or drops the same edge depending on the schedule. One search per SOURCE, not per pair -- every edge leaving `p` shares the set reachable from `p` in two or more steps -- and where ids increase along every edge the search stops at `p`'s largest target. Callers: `CausalGraph::reduced_pairs` (ids not topological), `Hypergraph::for_each_reconstructed_causal_as` and `QeState::reconstructed_pairs_host` (both topological).
- **`rendezvous.hpp`** -- THE SYMMETRIC STORELOAD HANDSHAKE, one inlined definition for every site that needs one. `rendezvous<Tag>(publish, scan)` and `rendezvous_barrier<Tag>()` for the pairs whose two halves live in different functions, plus `rv::` -- the tag list, which IS the list of the engine's handshakes. Two threads that each publish one thing and then scan for the other's may, under acquire/release, both read the value from before the other's write, and then the pair meets nowhere and nothing re-drives it. The tags name the partner, so a pair is greppable and a tag used once is a question. The header also states the three shapes that RESEMBLE the class and are safe -- an asymmetric join whose ordering runs through a shared read-modify-write chain, a rendezvous over a single location where coherence already orders the two, and a read that only skips an optimisation because the other side has a second path to the same information. Costs nothing: both callables inline and the tag generates no code (checked by diffing the emitted assembly of every converted translation unit).
- **`depth_join.hpp`** -- DECIDING THAT A DEPTH CAN RECEIVE NO MORE WORK, and saying so in depth order. `DepthJoin` over caller-owned `Slot`s (`live`/`complete`, one cache line each): `push`/`done`/`settle_from`/`mark_roots_seeded`/`late_arrivals`. A task at depth d submits only ABOVE d, so a settled d-1 means nothing can land at d and the depth needs no barrier to be declared finished. Separate from `ParallelEvolutionEngine` because that is what makes it checkable -- it touches nothing but its own atomics, so `verification/genmc/depth_report_order.cpp` can be handed the protocol rather than the engine around it. Reporting is ordered by a single-holder baton rather than by the settle order, because settling and reporting are two steps and a thread descheduled between them lets the depth above be reported first.
- **`park_gate.hpp`** -- THE PARK/WAKE HANDSHAKE: a worker with nothing to do sleeps, a submitter wakes one. `ParkGate` over caller-owned `Domain`s (one cache line each: a per-domain sequence word and idle count) plus the global idle count: `wake_one`/`wake_all`/`park_unless`. The failure it prevents is a job queued with nobody awake to take it, which is not a slow run but one that never finishes. Each side writes one location and reads the other's -- the worker announces itself idle then takes a last look, the submitter publishes then reads the idle count -- so both may read stale under acquire/release and both conclude there is nothing to do; the seq_cst pairing forbids it. Announcing before sampling the sequence, and the park's compare against that sample, are what make the wake an optimisation rather than the mechanism. The per-domain fallback scan is what keeps a job whose own domain is busy from stranding a worker idle elsewhere. Separate from `JobSystem` because that is what makes it checkable -- constructing a JobSystem under GenMC prunes to 2,659 lines and verifies, starting one prunes to 8,952 and segfaults it -- and `verification/genmc/job_system_no_lost_wakeup{,_domains}.cpp` drive this header, where they used to drive a transcription of it.
- **`termination_core.hpp`** -- THE TERMINATION DECISION for a persistent kernel, one body. `term_detect_loop(ctx, p1, c1, p2, c2)`: a detector block watches per-role pushed/completed counters plus a produced and a consumed cursor, and decides when nothing more can arrive. Written twice in `gpu/src/persistent.cu` before this -- once per kernel, differing only in which cursor counts consumed work and in what each printed. Quiescent once is not enough (a completed match may not have its records visible yet), so it looks again after a backoff and requires every observed quantity UNCHANGED rather than merely satisfying the conditions again; the counters are monotone, so a worker that started and finished inside the window necessarily moved one. The stagnation budget counts rounds in which NOTHING moved, not elapsed rounds, because a round ceiling cannot tell a deadlock from a workload that takes longer than the ceiling and fired on the second. The Ctx supplies where the counters live and what to do at the edges; it supplies no part of the decision. Checked under scoped-RC11 by `verification/gpumc/termination_no_early_exit.cpp`.
- **`ring_core.hpp`** -- THE CLAIM RULE of a bounded MPMC ring, one body for both roles. `ring_claim(ctx, want, leave)`: every slot carries a sequence number that alone says whose turn it is, and a producer waits for `seq == pos` and leaves `pos + 1` while a consumer waits for `pos + 1` and leaves `pos + capacity` -- the same rule with two constants, so they are one body. The reservation is a compare-exchange rather than a bump, which is what makes the queue safe when the same workers both produce and consume: a bump has nothing to undo with, and an item lost from such a queue is not dropped work but a run that never terminates. The Ctx supplies how a cursor or sequence word is touched AND AT WHAT SCOPE, which is exactly what differs between the device and a checker; it supplies no part of the decision. Driven by `gpu/include/hg_gpu/ring_buffer.hpp` and checked under scoped-RC11 by `verification/gpumc/ring_exactly_once.cpp`.
- **`hash_insert_core.hpp`** -- INSERT-IF-ABSENT into an open-addressed table: where to probe, and which of the threads meeting on a key is told it inserted. THE ELECTION IS THE VALUE EXCHANGE, not the key exchange -- unpublished changes to something exactly once per slot, so exactly one thread's exchange succeeds and the thread it elects is by construction the one whose value is stored. The key exchange elects a different thread, which can lose the value exchange and then report inserted while carrying a stranger's value: one signature, two canonical events. Nothing waits -- a thread finding a key claimed but unpublished offers its own value rather than waiting for the claimant. The probe run is EXHAUSTED rather than bounded, because with linear probing a key lives anywhere in its contiguous run, and exhaustion is a third outcome the caller is told about. Driven by `gpu/include/hg_gpu/hash_table.hpp` and checked under scoped-RC11 by `verification/gpumc/hash_insert_elects_one.cpp`.
- **`list_core.hpp`** -- PREPEND TO AN INTRUSIVE LOCK-FREE LIST, AND WALK IT, one body. `list_push(ctx, node)`: link in front of the head read, install by compare-exchange, relink and retry on a refreshed head; the exchange is ACQ_REL because release publishes the node and acquire covers the pusher's own walk below it. `list_for_each` visits exactly the nodes published before the head was loaded; `list_for_each_before(mine)` visits the nodes linked strictly before one node, which is what lets two pushers meet exactly once. The Ctx supplies how the head and next words are touched and at what scope. Driven by `gpu/include/hg_gpu/lock_free_list.hpp` and checked under scoped-RC11 by `verification/gpumc/replay_rendezvous_meets.cpp`.
- **`dedup_claim_core.hpp`** -- CLAIMING A MATCH EXACTLY ONCE over a set keyed by a 64-bit hash that can collide. `dedup_claim(ctx)`: the hash only selects WHERE to look and identity is the CONTENT comparison, applied at both points the walk can conclude -- on the lookup and again on the offer, because the slot can change between them. Two different matches on one key must BOTH win; deciding on hash equality alone silently loses a match and, since forwarding is inductive, its whole subtree. The stable copy is made at most once and only after a probe has actually missed, because it must exist before the exchange that publishes it and a loser is permanent arena. Running out of probes PROCESSES the match: a redundant rewrite is recoverable and a lost one is not. Driven by `ParallelEvolutionEngine::claim_match` and checked by `verification/genmc/claim_match_rendezvous.cpp` over a real ConcurrentMap.
- **`slot_core.hpp`** -- FRAME SLOTS, one definition for host and device: an edge's rank when a state's edges are ordered by (Aut orbit, `EdgeId`). This is the coordinate system a canonical class's matches are recorded in, which is what lets a match found on one raw instance replay against any other instance of the class; two copies drifting by one tie-break would produce replayed events that are wrong and invisible.
- **`join_core.hpp`** -- THE JOIN: one backtracking-join body for host and device.
  - `JoinState<>` (per-thread frame: bound edge and pattern position per depth, binding + mask, `already_taken` edge-injectivity, `bound_pattern_mask`)
  - `join_next_position()` -- which pattern position to bind next: the first UNBOUND one in the schedule, never `order[depth]`, so a seeded join still binds the positions before its anchor
  - `join_dfs()` / `join_seed()` -- the recursion, and the same recursion anchored at one position (which is what delta matching is). The Ctx supplies candidate enumeration and emit; nothing else differs between the two engines.
- **`event_core.hpp` / `match_core.hpp` / `rewrite_core.hpp` / `signature_core.hpp`** -- the shared semantic cores the two engines drive: event-identity lattice (`EventSignatureKeys`, `event_signature`), pattern-edge binding (`bind_pattern_edge`), rewrite vertex resolution (`resolve_rhs_vertices`, `assign_fresh_consecutive`), edge signatures
- **`park.hpp`** -- futex-style parking (`WaitOnAddress` on Windows, futex on Linux) for the job system's idle waits
- **`affinity.hpp`** -- backend selection for pinning a thread to a logical CPU (`pin_this_thread_to_cpu`); Linux and Windows bind, macOS reports refusal. The body lives in `job_system/src/affinity.cpp`.
- **`sampling_core.hpp`** -- the sampling decisions (`TransitionRate`/`RuleWeights` draws, keyed on transition identity), one body for host and device.
- **`portable_intrinsics.hpp`** -- GCC/Clang and MSVC spellings of the intrinsics the engine uses.
  - `popcount`/`popcount64`/`ctz`/`ctz64`
- **`capacity.hpp`** -- `CapacityExhausted`, the error kind for a CONFIGURED limit rather than a programmer mistake. It lives here because the thrower (the engine's containers) and the catcher (the job system) are separate libraries, and the job system must not depend on the engine to name what it caught; classifying on the TYPE is what a rename of the message cannot break.

## `hypergraph/include/hypergraph/` -- core CPU engine (headers)

- **`types.hpp`** -- core value types, IDs, bindings, mode enums.
- **`quotient_types.hpp`** -- what a state, an edge and an event ARE once isomorphic states are identified: `QcEventContent`, `CanonicalEdgeKey`, `EdgeRankTable`, `EdgeOrbitTable`, `CanonicalTransition`, `SlotMatch`. Separate from `types.hpp` because they are the only types needing `hgcommon/quotient_replay_core.hpp`, which `types.hpp` would otherwise hand to every engine header.
  - structs `Edge`, `Event`, `State`, `VariableBinding`, `GlobalCounters` (each counter `alignas(64)`), `CausalEdge`, `BranchialEdge`, `EventSignature`; enums `StateCanonicalizationMode`, `EventSignatureKey(s)`; `AbortedException`
  - quotient reconstruction types: `CanonicalEdgeKey` (the quotient-aware edge identity that meets producers with consumers -- orbit-keyed under quotient, raw `EdgeId` otherwise), `EdgeOrbitTable` (per-state edge orbits + SLOTS), `CanonicalTransition` (orbit-deduplicated), `SlotMatch` (undeduplicated, slot-named)
  - `EMPTY_STATE_CANONICAL_HASH` -- the empty state's own canonical hash; it cannot be 0, which means "not computed" for `State::canonical_hash` and is `ConcurrentMap`'s `EMPTY_KEY`
- **`atomic_compat.hpp`** -- `hypergraph::atomic_ref<T>`: an atomic view over a plain, non-atomic member. `State` keeps some fields as plain scalars so it stays trivially copyable and single-threaded paths touch them directly, while concurrent paths need atomic access to the same words. Selects `std::atomic_ref` where it exists and falls back where it does not -- the OSXCross SDK's bundled libc++ predates C++20, so the macOS cross build cannot name it directly
- **`rule_analysis.hpp`** -- what the RULES decide before a run: `RuleFacts` (edge delta, vertex creation rate, LHS shape) and `RuleSetFacts`/`analyze_rules`. The load-bearing one is `can_branch`, which asks whether two distinct matches can share a CONSUMED edge -- exactly the branchial relation's condition, so a false proves that relation empty for every initial condition. Sound in the false direction only: true means "not ruled out", because the reachability question behind it is undecidable. Holds nothing about termination or global confluence, which are undecidable here
- **`arena.hpp`** -- arena allocators (foundation of off-hot-path, malloc-free allocation). `ConcurrentHeterogeneousArena` blocks are anonymous mappings pooled process-wide across arena lifetimes; each records how far it was dirtied (`Block::dirty_end`) and `allocate_raw` reports whether an allocation's bytes are known zero, so `allocate_array` and the hash tables skip their fills on fresh bytes.
- **`zero_value_init.hpp`** -- `zero_value_init_v<T>`: T() writes only zero bytes; default for trivial types, specialised next to `QcEventContent` and `EdgeSignature`.
  - `Arena<T>`, `ConcurrentArena<T>`, `ConcurrentHeterogeneousArena` (**per-worker bump cursors** — each thread bumps a private non-atomic offset, no shared atomic on the fast path, only a lock-free head CAS to grab a fresh ~1 MB block; scratch arenas keep the shared `allocate_shared` path for `mark/release/reset`; `create`/`create_untracked`), `ArenaWorkerRegistry` (thread→dense index, released at exit), `ArenaVector<T>`; `worker_scratch()` — the recycling scratch arena bumps `current_block_` with plain relaxed accesses (`allocate_single`), not the shared atomic claim: it is single-threaded, and the locked RMW cost 4.35x on the bump path
- **`scratch_alloc.hpp`** -- STL-compatible allocators over the scratch/persistent arenas.
  - `ScratchAlloc<T>`, `PersistAlloc<T>`, `PersistTarget`; `worker_persistent()`; aliases `SVec`/`PVec`/`SSet`/`SUSet`/`SMap`/`SUMap`/`PUMap`
- **`bitset.hpp`** -- sparse chunked bitset for a state's edge set.
  - `SparseBitset` (+ `derive()`, `from_edges()`), nested `Chunk`/`ChunkEntry`
- **`segmented_array.hpp`** -- append-only fixed-segment array, stable pointers, O(1) access.
  - `SegmentedArray<T>` (`emplace`/`emplace_at`/`get_or_default`/`ensure_size`/`operator[]`/`for_each`)
  - `MAX_SEGMENTS` is now enforced in `get_or_create_segment` (throws `std::length_error`); every write path funnels through it, and past it the CAS would land in the adjacent member
  - CONTRACT: `count_` is a high-water mark, so an index may be read only after its own `emplace` returned, or during quiescence
- **`concurrent_map.hpp`** -- lock-free open-addressing append-only hash map.
  - `ConcurrentMap<K,V,EMPTY,LOCKED>` (`insert_if_absent[_waiting]`, `lookup[_waiting]`, `count_unique`, `for_each`, optional arena backing via ctor/`set_arena`, `bytes_allocated`), nested `Entry`/`Table`. The ctor's third argument is the size the FIRST growth jumps to, defaulted -- a parameter so a model checker can bound the protocol, since a growth out of a 1024-slot table is 2048 atomic operations to interleave
  - **No tombstone**: a claimed slot is always resolved to a real key, never back to EMPTY, or the probe run of every key passing through it would be cut. Inserters therefore await a claimed slot IN PLACE before claiming one of their own.
  - **Obstruction-free, not lock-free**, and cannot deadlock: a thread holding a claim only stores, never waits.
  - **`was_inserted` is a CONJUNCTION**: this caller's value exchange won, AND the value the map now answers with is the one it offered. Neither half alone is sound -- a key can be reached in more than one table, so "my exchange won" is not unique to one caller (a publish that beats the retiring table's `ABSENT -> FORWARDED` seal leaves a settled entry in a superseded table), and two callers may offer the SAME value, so a comparison alone calls both of them the inserter. Both directions are gated: `concurrent_map_double_growth_3t` and `concurrent_map_repeated_offer`
  - `reject_sentinel_key` throws on a key equal to EMPTY/LOCKED -- such a key is silently unstorable, which caused four separate correctness bugs.
  - `for_each`/`count_unique` walk the whole resize chain and emit each key once (a key can settle only in a superseded table, since `resize()` skips claimed slots).
- **`concurrent_key_set.hpp`** -- lock-free key-only set (membership, no value word).
  - `ConcurrentKeySet<K, EMPTY, MIGRATED>` (single-CAS `EMPTY->key` claim; growth installs the successor first, then carries each key forward and seals its old slot with `MIGRATED`; a `drained` table is skipped rather than probed, because sealing removes the terminator linear probing stops at)
  - Carries the quotient reconstruction's membership marks (`qc_reached_`, `qc_applied_`, `qc_dsup_seen_`, `seen_transitions_`); model-checked by `verification/genmc/key_set_exactly_once` and `key_set_enumeration`
- **`lock_free_list.hpp`** -- append-only lock-free linked list. `push` returns the node it linked, and `for_each_before(node)` walks the nodes linked strictly earlier; `for_each_node` hands over nodes rather than values so a caller can position itself. Two pushers meet EXACTLY ONCE under this pair -- of any two nodes one is older, so only one scan sees the other -- which is how the quotient branchial relation is formed without a set of pairs to dedup against.
  - `LockFreeList<T>` (`for_each`/`for_each_while`), `SingleThreadedList<T>`
- **`signature.hpp`** -- edge vertex-repetition signatures + compatible-signature enumeration.
  - `EdgeSignature`, `signature_compatible()`, `enumerate_compatible_signatures()`, `CompatibleSignatureCache`
- **`pattern.hpp`** -- rule representation, builder, partial-match state.
  - `PatternEdge`, `RewriteRule` (`compute_var_counts`/`compute_match_order`), `RuleBuilder` + `make_rule()`
  - `RewriteRule::match_order` is a SCHEDULE, not a semantic: every permutation yields the same match set (`JoinCore.EveryBindingOrderYieldsTheSameMatches`)
- **`ancestry.hpp`** -- `AncestryCandidates`: a state's edges by ancestry, for candidate generation. A state's edges are the contributions of its chain of parent states (every edge of the root, the produced edges of each derived state) less what later events consumed; each state indexes its own contribution by vertex once at creation (`State::vertex_index`, `delta_edges`, `parent_state`), and `for_each_edge_at` / `for_each_edge_containing_all` / `for_each_edge_compatible` walk the chain searching each contribution, settling membership on the state's edge set. Nothing is maintained per edge on the write side.
- **`pattern_matcher.hpp`** -- the host's half of the join: candidate enumeration and emit, over `hgcommon/join_core.hpp` (templated on accessors).
  - `PatternMatchingContext<>`, `HostJoinContext<>` (the join's Ctx: `Candidate` carries the edge the enumerator already fetched)
  - free templates `validate_candidate`, `generate_candidates`, `emit_match`, `scan_pattern[_from_edge]`, `find_matches`, `find_delta_matches`
- **`wl_hash.hpp`** -- Weisfeiler-Leman approximate hashing + O(E) edge correspondence; owns `VertexHashCache`, the per-state vertex-hash cache, because it is its only consumer.
  - `WLHash` (`compute_state_hash_with_cache`, `compute_event_signature`)
- **`canonical_types.hpp`** -- shared canonicalization result types.
  - `CanonicalForm`, `VertexMapping`, `CanonicalizationResult` (`are_isomorphic`)
- **`ir_canonicalization.hpp`** -- host face of the McKay individualization-refinement exact canonicalizer; the algorithm is `hgcommon/ir_core.hpp`, shared with the device.
  - `IRCanonicalizer` (`canonicalize_edges`, `compute_canonical_hash[_with_edge_map/_with_edge_rank/_with_edge_orbits]`, `are_isomorphic`)
- **`causal_graph.hpp`** -- online lock-free causal + branchial relationships with online transitive reduction.
  - `CausalGraph` (`set_edge_producer`/`add_edge_consumer` -- both keyed by `CanonicalEdgeKey`, not raw `EdgeId`, so orbit-shared edges meet at one key under quotient; `add_causal_edge`/`add_branchial_edge`; `record_state_event` + `record_branchial_overlaps` (the per-state event list and the branchial pairs it induces, each pair claimed once); the reduction as `record_reduced_edge`/`is_reachable`/`reduces_on_read`/`ids_are_topological`, which is a TAG on one base relation rather than a second graph; `for_each_causal_edge`/`for_each_branchial_edge`)
  - `causal_pair_key(producer, consumer)` offsets both ids so a self-loop on event 0 is not the map's EMPTY sentinel
- **`hypergraph.hpp`** -- central store: edges/states/events, indices, canonicalization, causal graph.
  - `Hypergraph` (`create_edge`/`create_state`/`create_event`, `create_or_get_canonical_state`/`get_canonical_state`, `compute_canonical_hash`/`compute_wl_hash`/`compute_content_ordered_hash`, `try_lower_explore_depth`/`try_claim_expanded` for quotient mode, genesis support), result structs `CanonicalStateResult`/`CreateEventResult`
  - `canonical_hash`/`canonical_id` are published and read through `hg::atomic_ref` (release store / acquire load); a bare fence pair would order nothing
  - **quotient raw causal reconstruction** (`set_quotient_reconstruction`, default OFF): `compute_and_cache_state_orbits` (orbits+slots, piggybacked on the dedup IR pass), `qc_capture_expansion`/`qc_frame_slots` (the representative's full match list in slots, aligned into one pinned reference frame per canonical class), `qc_add_instance`/`qc_apply` (per-instance replay minting raw event ids), `qc_record_causal`/`qc_reachable` (causal base + in-reduction tag), `for_each_reconstructed_causal`, `num_reconstructed_*`, `observable_num_*`
  - the older aggregate producer-set DP (`qc_dsup_`, `qc_reach`, `qc_emit`) is still the DEFAULT quotient causal path; the per-instance reconstruction above is verified against full capture but not yet wired as the default (backlog: S4)
- **`parallel_evolution.hpp`** -- the dataflow parallel multiway evolution engine.
  - `ParallelEvolutionEngine` (`evolve` x2, `add_rule`, `set_*` config; private task methods `execute_*_task`/`submit_*_task`; forwarding `store_match_for_state`/`register_child_with_parent`/`push_match_to_children`/`forward_existing_parent_matches[_eager]`; pruning/RNG/quotient `should_explore`/`sampling_rng`/`propagate_explore_depth`)
  - `last_error()`/`raise_worker_error()` -- `wait_for_completion()` returns the moment a worker latches an error, so every run-completion path raises anything but `Aborted` rather than returning a truncated graph as a complete one
  - `try_claim_budget`/`release_successor_slot` -- pruning budgets are claimed by CAS (a fetch-add-then-rollback publishes a count above the limit, which the readers prune on), and every path that produces no child returns both slots
  - `guard_quotient_transitive_reduction()` -- forces TR off under quotient, because causal edges are emitted between canonical event ids whose assignment is schedule-dependent
  - structs `MatchRecord`, `EvolutionStats` (each counter `alignas(64)`), `MatchContext`, `ScanTaskData`, `ExpandTaskData`, `ChildInfo`, `ParentInfo`; enum `EvolutionJobType`
- **`rewriter.hpp`** -- applies a rule+match to produce a new state (declaration).
  - `RewriteResult`, `Rewriter` (`apply`), `apply_rewrite()`
- **`debug_log.hpp`** -- debug logging routed to an FFI callback or printf (`DEBUG_LOG` macro, no-op unless enabled).

## `hypergraph/src/` -- core engine (out-of-line implementations)

- **`hypergraph.cpp`** -- `Hypergraph` methods: creation + index registration, `create_or_get_canonical_state` dedup, event creation/canonicalization, hashing, edge-correspondence dispatch.
- **`ir_canonicalization.cpp`** -- the adapter from this project's edge lists to `hgcommon/ir_core.hpp`: `ir_core_call` (sorted-unique vertex numbering, depth and generator escalation) + the public hash/canonicalize entries built on it.
- **`causal_graph.cpp`** -- `CausalGraph` methods: lazy slot/list creation, producer/consumer rendezvous, `add_causal_edge`/`record_reduced_edge`/`add_branchial_edge`, `is_reachable` (the backward walk over KEPT predecessors that decides whether a pair is bypassed), `get_causal_edges`/`get_branchial_edges` export.
- **`parallel_evolution.cpp`** -- the engine's implementation: `evolve` loops, the `execute_*`/`submit_*` task engine, match forwarding, pruning/quotient bookkeeping.
- **`rule_analysis.cpp`** -- the rule facts' bodies: `lhs_is_connected`, `lhs_is_acyclic` (GYO), `lhs_edge_cover`, `analyze_rule`, `can_branch`, `analyze_rules`. They run once per rule at registration and once per run at configure time, so they are here rather than in the header every engine translation unit parses.
- **`pattern.cpp`** -- everything behind `pattern.hpp`: `RewriteRule::compute_var_counts`, `edge_constraint_score` and `compute_match_order` (the derived matching data, computed at `add_rule` and read from then on), plus `PatternEdge`, `RewriteRule`'s masks, `RuleBuilder` and `PartialMatch`. Also `pattern_matcher.hpp`'s `validate_candidate`, the matcher's only non-template body.
- **`signature.cpp`** -- `EdgeSignature`'s accessors and both builders (`from_edge`, `from_pattern`), `CompatibleSignatureCache::from_pattern`, `signature_compatible`, and the set-partition ENUMERATION (`enumerate_compatible_signatures` + `detail::enumerate_partitions_recursive`) that `CompatibleSignatureCache::from_pattern` drives once per pattern edge (the device's signature index reads the cache; the host tests each candidate with `signature_compatible` directly). The enumeration hands each signature to a `SignatureVisitor` FUNCTION POINTER, so inlining its outer frame never removed the indirect call that dominates it -- measured at +0.035% when it moved here.
- **`arena.cpp`** -- `ConcurrentHeterogeneousArena`'s bodies (construction, `allocate_raw` and the four bump/grow paths, `register_destructor`, `mark`/`release`/`reset`), `worker_scratch()`, and `scratch_alloc.hpp`'s siblings `worker_persistent_target()`/`worker_persistent()`/`PersistTarget`/`ScratchIdSet` -- the scratch and persistent per-worker arenas are one mechanism at two lifetimes. `ArenaWorkerRegistry::acquire`/`release` are NOT here: `verification/genmc/arena_worker_index_exclusive.cpp` compiles that header alone and links no library, so those two stay inline.
- **`types.cpp`** -- the engine's value types: `VariableBinding`, `Edge`, `Event`, `State`, `GlobalCounters`, `CausalEdge`, `BranchialEdge`, `EventSignature`, and `quotient_types.hpp`'s identity records (`QcEventContent::triple_hash`, `EdgeOrbitTable`, `CanonicalTransition`, `SlotMatch`).
- **`bitset.cpp`** -- `SparseBitset::Chunk` and the bitset's construction, move, `count`, `find_entry_index` and `invalidate_count`. `contains` and `find_chunk` are NOT here: they are `HG_INLINE` in the header with the measurement that pins them (outlining the engine's hottest predicate cost +3.7%).
- **`wl_hash.cpp`** -- `VertexHashCache` and `WLHash`'s `fnv_combine`/`mix64`/`compute_edge_signature`. The refinement is templated on the caller's accessors and stays in `wl_hash.hpp`.
- **`debug_log.cpp`** -- `set_debug_callback`, `clear_debug_callback` and the formatter `debug_output`, and with them `<cstdarg>`/`<sstream>`/`<thread>`: `debug_log.hpp` reaches nearly every engine translation unit through `parallel_evolution.hpp`, `rewriter.hpp` and `concurrent_map.hpp`, and this is the only unit that names libstdc++'s iostream and threading machinery for it.
- **`rewriter.cpp`** -- `RewriteResult`'s constructor, `Rewriter`'s constructor, the free `apply_rewrite`, and `Rewriter::apply`: validate match, derive child edge set, allocate fresh vertices, create RHS edges/state/event, register causal/branchial (consumed edges in descending-producer order for correct online TR).

## `gpu/include/hg_gpu/` -- CUDA port (headers)

- **`cuda_check.hpp`** -- `HG_CUDA_CHECK(err, what)` / `cuda_check_at`: the ONE CUDA error check.
  Throws naming `__FILE__:__LINE__` and the driver string. Every `.cu` and every device
  container routes through it; none carries its own copy.

- **`overflow.hpp`** -- CUDA-free shared overflow types: `ErrorKind`, `OverflowWarning`, `error_kind_name()`
- **`types.hpp`** -- GPU aliases + device storage structs: `DeviceEvent`, `DeviceCausalEdge`, `DeviceBranchialEdge`, `Edge`, `StateEdgeSlice`; enums `CanonicalizationMode`, `EventCanonicalizationMode`
- **`errors.hpp`** -- device error channel: `DeviceErrors` (`DeviceView::record`), `PoolOverflow`
- **`atomic_pool.hpp`** -- `Pool<T>` (pre-allocated device array + atomic bump counter; `DeviceView::claim`/`claim_n`/`at`)
  - CONTRACT: the counter is a RESERVATION high-water mark and can stand ahead of the writes -- a thread that claims slots and then fails a later reservation returns without filling them. Host readbacks bound themselves by the counter, so they copy those slots; the storage is zeroed once in the constructor to make them a defined value. `reset()` is the per-run path and clears only the counter
- **`lock_free_list.hpp`** -- `LockFreeList<T>` (per-key linked-stack over a node Pool; `DeviceView::push` returns the node index, `for_each`, `for_each_before` for the same exactly-once meeting rule the host list documents)
- **`hash_table.hpp`** -- `ConcurrentMap<K,V,EMPTY,LOCKED>` (open-addressing linear probe; `DeviceView::lookup[_waiting]`/`insert_if_absent`)
- **`ring_buffer.hpp`** -- `RingBuffer<T>` (bounded MPMC ring; per-slot sequence numbers + CAS reservation, so producers that are also consumers neither lose nor duplicate an item across wraps)
- **`termination.hpp`** -- `TerminationDetector` (per-role quiescence for a persistent-kernel model)
- **`device_arena.hpp`** -- `DeviceArena` (bump allocator the device claims from; scratch whose size is only known once the work is in hand)
- **`edge_signature.hpp`** -- `EdgeSignature` + device `signature_*` helpers (bit-identical to CPU)
- **`signature_index.hpp` / `vertex_inverted_index.hpp`** -- `SignatureIndex` / `VertexInvertedIndex` (device match-candidate indices)
- **`match.hpp`** -- `DevicePatternEdge`/`DeviceRhsEdge`/`DeviceRule`/`MatchRecord` (carries its `step` and a `published` flag); device `match_state_rule`/`publish_match`/`await_match`; host `make_device_rule`/`run_match_kernel[_batch][_nosync]`
- **`rewrite.hpp`** -- device `apply_one_match` (returns the state it created); host `run_rewrite_kernel[_with][_nosync]`
- **`exploration.hpp`** -- `DedupMap` + device `state_survives_dedup` (which new states get expanded; one predicate for both schedulers)
- **`persistent.hpp`** -- `MatchWorkItem`, `PersistentRunStats`/`PersistentEvolveStats` (incl. the phase-cycle attribution counters); host `run_persistent_match`/`run_persistent_match_rewrite`/`run_persistent_evolve` (the device-resident schedulers), `default_persistent_grid()`/`persistent_arena_words()`
- **`quotient_expansion.hpp`** -- expansion capture and per-instance replay, device side: the twin of the host's `qc_capture_expansion`/`for_each_expansion_match` and the `(instance, match)` rendezvous. `DeviceSlotMatch`/`DeviceQcInstance`/`QeAppliedMatch`, the `QeView`/`QeState` split, `qe_frame_slot_of`/`qe_register_frame`/`qe_apply`/`qe_drive_instance`/`qe_drive_match`. `reconstructed_pairs_host` hands back the causal, reduced and branchial relations, reducing through `hgcommon::tr_reduce` over the id-level relation. Under quotient exploration only one raw state per class is expanded, so the raw events the other instances would have produced are never created; this replays them from the class's captured matches
- **`quotient_causal.hpp`** -- the device's STORAGE for the orbit-keyed quotient-causal DP; the DP itself is `hgcommon/quotient_causal_core.hpp`, the same body the host runs. `DeviceCanonicalTransition`/`QcProducerNode`/`QcTransitionRef`, `QcView`, host `QcState` (engine-lifetime owner, cleared per run); `DeviceQcTransition` (accessors over the packed orbit-word arena) and `DeviceQcCtx` (the core's storage face); `qc_register_transition` (orbit-maps one raw event into a deduplicated canonical transition and drives it), `qc_orbit_of`/`qc_canonical_event`/`qc_emit`/`qc_bucket`. Keys are (state canonical hash, depth, edge orbit) -- no raw ids -- so the causal set under quotient exploration is schedule-independent and equal to the CPU's (gate: `tools/quotient_causal_probe_gpu`)
- **`content_hash.hpp` / `ir_canon.hpp`** -- device `content_hash_state_device` (the content-ordered key `CanonicalizationMode::Automatic` asks for, never a fallback under `Full`), `state_exact_hash_device` (arena-backed, sized per state); host `compute_state_ir_hashes*`
- **`initial_upload.hpp`** -- host `rebuild_indices`/`upload_initial_state[s]`
- **`engine_state.hpp`** -- `DeviceState` (POD passed to kernels) + `EngineState` (host owner of all device pools/indices, readback helpers)
  - CONTRACT: the six scalar counters are slots of one `cudaMalloc`'d block so `counters_snapshot_host()` reads them in a single transfer, and it reads ALL six. Two of them (`event_sig_raw_fallbacks`, `canonical_event_count`) bind to their pointers only when the feature owning them first runs, so the block is zeroed at allocation -- a slot whose feature never runs reads zero rather than whatever the allocator held
  - CONTRACT: `state_edge_ids_counter_` is bumped before the capacity check and before the vertex reservations that can still fail, so `state_edge_ids_` is zeroed at allocation for the same reason `Pool<T>` zeroes its storage. A slot is only ever read through a slice that was written with it
- **`evolve.hpp`** -- the public host API: DTOs `RewriteRule`/`EvolveInput`/`CanonicalState`/`Event`/`CausalEdge`/`BranchialEdge`/`EvolveResult`, `EngineConfig`; classes `Engine` (`run`/`reset`) and `PersistentEvolver` (grow-and-retry reusing one Engine); `evolve()`, `config_from_input()`, `estimated_device_bytes()`

## `gpu/src/` -- CUDA kernels + drivers

- **`engine_state.cu`** -- the HOST bodies of `EngineState` (the constructor's device allocations, `clear`, `device()`, `set_sampling`, the readback helpers) and of the three device-side containers it owns (`DeviceArena`, `SignatureIndex`, `VertexInvertedIndex`). `engine_state.hpp` is included by sixteen translation units and every one of them was compiling all of it; what stays there is `DeviceState` and the `DeviceView` structs the kernels actually use
- **`errors.cu`** -- the port's error reporting: `cuda_fail` (the throw `HG_CUDA_CHECK` jumps to -- `cuda_check_at` stays inline, being one comparison at every CUDA call), `error_kind_name`, `DeviceErrors`' allocation and drain, and `TerminationDetector`'s host side
- **`quotient.cu`** -- the host bodies of `QcState` and `QeState`: allocation, `clear`, the counter readbacks and the `view()` calls that hand a device struct to a kernel. The DP and the replay are `__device__` and stay in their headers
- **`evolve.cu`** -- the driver: `Engine::Impl` sizes the device state, launches `run_persistent_evolve` (which seeds its own roots and owns the dedup map), and assembles the result; device `state_survives_dedup`; host `config_from_input`/`grow_config_for`/`fit_config_to_cap`/`estimated_device_bytes`/`evolve`/`PersistentEvolver::run`
- **`match.cu`** -- device `match_state_rule` + kernel `k_match_batch`. The JOIN is `hgcommon/join_core.hpp`; this file supplies `MatchJoinCtx` (CSR-slice / pivot-inverted / signature-bucket candidate enumeration) and the block-striped depth-0 parallelism. Host `schedule_lhs_edges`/`make_device_rule`/`run_match_kernel*`
- **`rewrite.cu`** -- device `apply_one_match` + kernel `k_rewrite` (preflight-reserve pools, build RHS/new-state CSR, write Event, causal+branchial rendezvous; the raw-edge causal rendezvous is skipped under `DeviceState::quotient_causal`, where the orbit-keyed DP replaces it); online TR is the `preds_list` backward-reachability oracle (`is_reachable_preds`, the device twin of `CausalGraph::is_reachable` -- no stored closure); `try_add_causal_edge` has external linkage (the DP emits through it); host `run_rewrite_kernel*`
- **`event_identity.hpp` / `event_identity.cu`** -- event identity shared by both schedulers: `event_keys_need_ranks`, `edge_rank_in_state_device`, `stamp_event_signature` (computes the signature AND applies it through a signature -> EventId map, so two applications with the same identity are one event), plus `fill_event_identity_inputs` / `stamp_event_identity_range` for the level-synchronous loop's post-hash phase
- **`persistent.cu`** -- the device-resident schedulers: `k_persistent_match` (match role alone), `k_persistent_match_rewrite` (two roles, no barrier), `k_persistent_evolve` (whole evolution in ONE launch chain with ONE host sync: root-hash seed -> counted queue seed -> rewrite -> key + exact hash [+ orbits] -> event stamp -> quotient-causal registration -> dedup -> re-enqueue, block 0 detecting quiescence; workers back off exponentially when idle and flush per-phase clock64 attribution at exit); host `run_persistent_*`, plus `default_persistent_grid()` (eight blocks per SM, from the grid sweep; `HG_GPU_PERSISTENT_BLOCKS` overrides) and `persistent_arena_words()` (IR arena as holders x `ir_arena_share_words`). Identity is answered per mode: `state_key_device` computes what the state mode identifies states by (`None` a per-state unique value with no hashing, `Automatic` the content hash, `Full` the exact IR hash), `stamp_event_signature` fills the event identity from the exact hashes plus the ranks `edge_rank_in_state_device` reads back, then APPLIES it through a signature -> EventId map so two applications with the same identity are one event
- **`ir_canon.cu`** -- device `state_exact_hash_device`, the one exact-hash body (arena-claimed slot sized from the state's own counts, depth and generators from `EngineConfig`, no fallback -- a state it cannot key reports its capacity kind and the wrapper grows and retries; `want_ranks` scatters each edge's canonical rank into `DeviceState::state_edge_rank`, `want_orbits` its automorphism orbit into `state_edge_orbit` + `state_num_orbits`, both riding the pass the hash already runs) + `k_exact_hash_range`/`compute_state_ir_hashes_range`, a grid-stride launch shape over it for batch callers
- **`content_hash.cu`** -- device `content_hash_state_device`, which applies `hgcommon::ContentHasher` over the state's edge slice: the same rule the host applies, so the two agree by construction
- **`initial_upload.cu`** -- `k_init_indices`; host `upload_initial_states`/`rebuild_indices`

## `job_system/include/job_system/` + `lockfree_deque/include/lockfree_deque/`

- **`job.hpp`** -- `Job<JobType>` (abstract), `FunctionJob<>`, `make_job()`, `ScheduleMode`, `CompatibilityAwareJob<>`
- **`job_system.hpp`** -- `JobSystem<JobType>` (the scheduler: per-worker Chase-Lev deques + shared injector, `submit`/`start`/`shutdown`/`wait_for_completion`, `set_on_job_complete` scratch recycle), nested `WorkerData`/`SystemStatistics`; `ErrorType`
  - a WORKER never blocks pushing to the injector (`try_push_back`, else run the job inline): a worker parked in a push cannot pop, so all workers parked there would wedge the system. The inline path passes `recycle_scratch=false`, since the outer job on the same stack still holds live scratch allocations.
- **`job_pool.hpp`** -- `JobSlotPool`: fixed-size job slots recycled through a per-thread free list (`SlotHeader`, `TlsGuard`), so submitting a job does not allocate on the hot path. Bodies in `job_system/src/job_pool.cpp`
- **`work_stealing_deque.hpp`** -- `WorkStealingDeque<T>` (bounded Chase-Lev; owner `push`/`pop`, thief `steal`)
- **`lockfree_deque/deque.hpp`** -- `Deque<T>` (bounded MPMC via one packed {tag,head,tail} atomic, ABA-defeating tag; try/blocking push/pop both ends)

`job_system` is a STATIC library with three translation units, not a header-only INTERFACE target -- it is where the bodies of the headers above live, and it is `POSITION_INDEPENDENT_CODE` because the paclet shared library links it in.

- **`job_system/src/job_pool.cpp`** -- `JobSlotPool`'s allocate/deallocate, the slab `grow`, and the process-lifetime pool registry (`acquire_pool`/`release_pool`/`tls_pool`)
- **`job_system/src/park.cpp`** -- `park_if_equal`/`unpark_one`/`unpark_all` AND the platform headers each backend calls into (`linux/futex.h`, `windows.h`, `os/os_sync_wait_on_address.h`, `<condition_variable>`). `hgcommon/park.hpp` names only the backend SELECTION, so a translation unit that parks does not parse `windows.h`
- **`job_system/src/capacity.cpp`** -- `CapacityExhausted`'s constructor, so one translation unit anchors its vtable. The class is declared in `hgcommon` because the thrower (the engine's containers) and the catcher (the job system) are separate libraries; the body is here because `hgcommon` has no library and everything that throws it links `job_system`

## `wxf/` -- Wolfram Exchange Format serialization

- **`wxf.hpp`** -- `Token` enum; exceptions `WXFException`/`ParseError`/`TypeError`; `WXFValue` variant (+`WXFValueList`/`WXFValueAssociation`); `Parser` (typed readers, `read_association`/`read_function`, `read<T>`); `Writer` (typed writers, `write_association`/`write_function`, `data`/`release_data`); free `serialize`/`deserialize`
- **`wxf.cpp`** -- out-of-line `Parser`/`Writer` methods; `wxf_bswap64` (big-endian path)

## `paclet_source/` -- FFI + standalone binary + GPU marshaling

- **`hg_core.hpp`** -- `HostBridge` (progress callback; abort = process kill), `run_rewriting_core()` declaration
- **`paclet_support.cpp`** -- the bodies behind the support headers: `session.hpp` (`EngineHolder`'s virtual destructor, so one TU anchors the vtable; `SessionSlot`'s open/engine/invalidate/close/require), `cpu_engine_holder.hpp`, `delivery_cursor.hpp` and `graph_marshal.hpp`'s free functions. ONE file rather than one per header, because five targets name their sources explicitly and each new file costs five list edits
- **`hypergraph_ffi.cpp`** -- the marshaling TU. `run_rewriting_core` is the entry point and now holds the engine acquisition, the session bookkeeping and the serialization; the phases whose dependency runs ONE WAY have their own functions, each reading the parsed job and writing one thing: `parse_job` (WXF envelope -> `hgffi::ParsedJob`), `read_back_session_identity` (D16 -- a held verb's identity comes from the SESSION, not its own envelope), `configure_and_evolve` (identity + recording config, rules, initial states, the evolution; a held session skips it). THE RECORD SET IS DERIVED FROM THE REQUEST, not left at its defaults: causal, branchial, per-state events and the RAW UNFOLDING are each switched on only for a request that needs them, which is why a states-only call does not pay the reconstruction (25x on multirule at depth 6). A SESSION is the exception and records everything -- its Open cannot know what a later Query will ask for, and a continuation must not depend on the order the caller asked things in, and `run_gpu_job` (the whole job on the device, behind `HG_GPU_BACKEND`). `ffi_helpers::read_rules_association`, plus the LibraryLink DLL export `performRewriting` + `WolframLibrary_initialize/uninitialize`. Builds the `GraphData` block (for the `*Graph` properties) via the shared `hgmarshal::build_graph_data`, adapting the engine through a `CpuGraphSource`
- **`graph_marshal.hpp`** -- the shared `*Graph` marshaller: `hgmarshal::build_graph_data(source, properties, opts)`, templated over a `Source` that exposes the evolved multiway as effective (canonicalization-collapsed) ids + per-vertex tooltips. Both `hypergraph_ffi.cpp` (CPU engine) and `hg_gpu_backend.cpp` (GPU result) drive it, so CPU and GPU emit identical graph structure for every property -- ONE graph-building code path, no divergent copies. Also carries the STATE-ENDPOINT projection of the branchial relation, which `"BranchialStateEdges"` and `"BranchialStateEdgesAllSiblings"` deliver as raw lists rather than as a `GraphData`: `branchial_state_edges_from_pairs` (from the branchial event pairs, step-filtered on the first event's output state), `branchial_state_edges_all_siblings` (every two events leaving one input state, both filtered) and `push_branchial_state_edges` (the two keys they deliver under). The two rules differ in more than their source and are kept apart deliberately; the caller supplies the traversal, because which order an engine visits its events in is a storage question
- **`hg_evolve_main.cpp`** -- the `hg_evolve` binary: `run_one_shot`, `run_serve` (stdio worker), `run_serve_socket` (loopback-TCP worker), frame I/O helpers, `main` (flag dispatch, progress->stderr)
- **`hg_gpu_backend.hpp`** / **`hg_gpu_backend.cpp`** -- `GpuJob` struct + `run_gpu_evolution` (builds `hg_gpu::EvolveInput`, runs `PersistentEvolver`, regroups the raw GPU result into canonical-class WXF matching the CPU FFI, and builds `GraphData` through the shared `hgmarshal::build_graph_data` via a `GpuGraphSource`); `build_input`. Serves `"BranchialStateEdges"` / `"BranchialStateEdgesAllSiblings"` through the shared rules above, and `"GlobalEdges"` / `"StateBitvectors"` from `EvolveResult::global_edges` / `state_edge_ids`, which the readback fills when `EvolveInput::edge_identity` is set
- **`ffi_job.hpp`** -- `hgffi::ParsedJob`, one parsed WXF envelope: the 38 values `run_rewriting_core` reads out of a job and nothing it derives afterwards, plus `hgffi::FfiWarning`. A field's initialiser IS the option's default, so this is the single place that answers "what happens if I omit it". Extracted because the parse is the one phase in that function whose dependency runs ONE WAY -- it writes these and reads nothing a later phase produces, which is why #12's op boundary had to be drawn by binding references instead
- **`delivery_cursor.hpp`** -- WHAT A SESSION HAS ALREADY BEEN SENT, per graph property, so a `Step` can report what it ADDED rather than the whole accumulated graph. `take_vertex` (carrying a REVISION, because a state's step can be lowered after delivery), `take_edge`, `delivered_before`, `reset`. Keyed by PROPERTY because a caller may ask for StatesGraphStructure on one Step and StatesGraph on the next, and the second has never received those vertices in that shape. A null cursor answers "not sent" to everything, so the full delivery is the delta against an empty record rather than a second walk
- **`build_stamp.hpp`** (`common/include/hgcommon/`) -- the configuration a shipped artifact or a bench was built with, written into it as a scannable literal (`HGBUILDSTAMP/2;commit=<sha>;variant=<name>;stats=<0|1>;phase_timing=<0|1>;ndebug=<0|1>;asan=..;tsan=..;ubsan=..;type=..;compiler=..;flags=..;:HGBUILDSTAMP`). The paclet defines it in `hypergraph_ffi.cpp` (`--version` prints it); the benches define their own and print it first (`--build-info`); `tools/dev/artifact_stamp_check.py` scans shipped artifacts for commit and release configuration, and `paper_tables.py`/`rich_sweep.sh` refuse a binary whose stamp is not the release one.
- **`session.hpp`** -- the session handle space: the `EngineHolder` interface (`extend(steps)` advances the CURRENT frontier -- the host already has this as `ParallelEvolutionEngine::evolve_more`; the device does not) and `SessionSlot` (`open`/`engine`/`close`/`invalidate`, `SessionState`, `SessionError`). Names no device, deliberately: the verbs are served for both, and a holder written around one engine would have to be replaced rather than extended when the other arrived. One session at a time, handles opaque and never reused, and an invalidated handle stays addressable so the next verb can report that its exploration is gone rather than silently serving a fresh engine
- **`cpu_engine_holder.hpp`** -- `hgffi::CpuEngineHolder`, the CPU side of a session: a `Hypergraph` and its `ParallelEvolutionEngine` owned together, because the engine holds a POINTER to the graph so the two have one lifetime and cannot be passed by value. `extend(steps)` calls `evolve_more`, which carries the same run further from its frontier rather than re-evolving and re-minting ids. Constructed non-continuable for a one-shot job, since the frontier a continuation resumes from is not free

## `paclet/` -- the Wolfram Language paclet

- **`PacletInfo.wl`** -- manifest for `WolframInstitute/HypergraphRewriteEngine` v0.0.1 (Kernel context, LibraryLink resources, Documentation, 6 SystemIDs)
- **`Kernel/HypergraphRewriting.wl`** -- the WL layer. Public surface: `HGEvolve`, plus the session verbs `HGSessionOpen`/`HGSessionStep`/`HGSessionQuery`/`HGSessionClose`/`HGSessionFrontier` over an opaque `HGSessionObject`, which continue ONE exploration instead of re-running it. `hgJobOptions` builds the job's Options envelope for both entry points, `hgSendJob` runs a job and surfaces the warning trail, and `hgRunJob` turns a reply into the requested properties -- so `Evolve` and a session's `Step` differ only in the envelope they build. A session verb takes the persistent worker or refuses: the one-shot fallback would mint a handle in a process that exits with the reply. The initial-condition generators (`HGGrid`, topologies, sprinkling, Brill-Lindquist, Poisson, uniform) remain as internal helpers behind HGEvolve's string initial conditions; the physics analyses live in ../hypergraph_viz
- **`Documentation/Source/*.md`** -- markdown doc sources (Symbol/Guide/TechNote) -> notebooks via `tools/build_docs.wls`

## `reference/` -- validation oracle

### What each comparison target is called, and what it actually is

Four different things get compared against here and the short names for them are not
self-explanatory, so they are stated once. The paper names the artifacts directly rather than
using these role-words; this table is the mapping between the two.

| name in code | the actual thing | what it answers | where |
|---|---|---|---|
| **the authority** | `Wolfram/Multicomputation`'s `MultiwaySystem` (Murzin's implementation) | what a user would otherwise run -- both its ANSWER (event identity ground truth) and its RUNNING TIME | `bench_authority.wls`, `authority_properties.wls`, `adjudicate_gap1_authority.wls`, `test_event_identity_authority.cpp` |
| **the reference** | `MultiwayReference.wl`, this project's own brute-force Wolfram evolver | the same definition computed directly and unoptimised; returns DATA, not `Graph` objects, so a ratio against it is not attributable to graph construction | `MultiwayReference.wl`, `validate.wls` |
| **the oracle** | the reference used as ground truth by the C++ gates | correctness only, never timing | `oracle_corpus.hpp`, `HG_REQUIRE_ORACLE`, `ReferenceOracle` tests |
| **the golden corpus** | recorded expected outputs checked into the tree | regression pinning, in milliseconds, without wolframscript | `golden_corpus.wl`, `golden_matrix.hpp` |

The word "authority" therefore covers two distinct uses of ONE object: a performance baseline in
T2 and a ground-truth answer for event identity. Both are `MultiwaySystem`; only the question
differs. "Reference" is the overloaded one -- it is also the directory name -- so in prose the
implementation is written `MultiwayReference` and the directory `reference/`.

- **`oracle_corpus.hpp`** -- the shared measurement substrate: `corpus()` (the named rule cases spanning the rule-type space -- single/mixed arity, productive/idempotent/reductive, self-loop, disconnected LHS, multi-rule, automorphic), `Case`/`Counts`/`LatticeCounts`, the engine drivers `engine_full_count`/`engine_counts`, and the brute-force isomorphism oracle `brute_force_iso_count`/`brute_canonical`/`content_canonical`, which is INDEPENDENT of the engine's WL and IR. One source of truth for what is tested and how it is checked, used by the oracle gate and by `tools/cost_matrix.cpp`
- **`golden_matrix.hpp`** -- the cached identity matrix: every corpus workload across every identity mode, each `Row` carrying the `Provenance` that says what checked it. The brute-force oracle is `O(V! * E log E)` and the WL reference needs wolframscript, so neither runs on every build; caching the expected values lets the gate compare in milliseconds. `event_keys_from_name`/`state_mode_from_name`
- **`MultiwayReference.wl`** -- brute-force ground-truth oracle: `MultiwayEvolve`, `CanonicalForm` (refinement + lex-min), helpers `refineColors`/`findMatches`/`eventSig*`
- **`golden_corpus.wl`** -- `hgGoldenCases`: 12 named cases with expected `{states, rawEvents, causal, branchial}`
- **`verify_paclet.wls` / `verify_paclet_gpu.wls`** -- load the local paclet, check `HGEvolve` against the golden corpus (CPU; GPU via `TargetDevice->"GPU"`)
- **`verify_sessions.wls`** -- the session verbs' LIFETIME (open, refuse a second open, step, close, reopen), with every prefix depth compared against an independent one-shot `HGEvolve`
- **`validate.wls` / `compare_multiwaysystem.wls`** -- validate the reference vs a determinism corpus / cross-check vs the authoritative `MultiwaySystem` paclet
- **`CANONICALIZATION.md`** -- the canonicalization naming cross-map across the four layers

## `tools/` -- standalone probes (each a `main()`)

Validation: `arena_reset_test`, `segmented_array_stress`, `determinism_forwarding_repro`, `causal_tr_determinism_probe`, `causal_tr_exactness_probe`, `canonical_causal_oracle`, `quotient_reconstruction_probe`, `multiplicity_propagation_probe`, `multi_init_rule_2x2_probe`, `quotient_causal_probe_gpu` (quotient causal is schedule-independent at every grid and equal across devices).
Canonicalization research: `ir_vs_wl`, `ir_edge_map_probe`, `ir_edge_orbit_probe`, `ir_incremental_probe`, `ir_malloc_bench`, `incremental_probe`, `incremental_wl_probe`, `wl_engine_incremental`, `wl_sparse_prototype`, `wl_core_bitid_check`.
Physics hunches: `branchial_flux_probe`, `budget_collapse_probe`, `higgs_shadow_probe`.
Profiling: `profile_evolve` (single-threaded, for callgrind/cachegrind), `bench_gpu_evolve` (GPU evolve() vs PersistentEvolver timing; `HG_GPU_DBG_TIME=1` prints the persistent scheduler's phase-cycle attribution), `bench_cpu_evolve` (the CPU twin of the same workload).
Build/docs: `build_paclet.wls` (CreatePacletArchive), `build_docs.wls` (markdown -> paclet notebooks).

## `tools/dev/` -- the measurement pipeline behind the paper

Nothing here is part of the engine; each writes a fragment under `paper/tables/` or drives a
rented box that does.

- **`paper_tables.py`** -- writes every generated fragment. `_fit` shrinks type before geometry, so
  a table's rendered size is a decision rather than a residue of its column count; `provenance()`
  stamps the commit, the machine MEASURED on and (when they differ) the machine the table was
  generated on. One `cost_matrix` run is shared by every table quoting it: they used to invoke it
  separately and disagreed on `multi-rule`, which is nondeterministic there.
- **`corpus_depth_plan.py`** -- picks, per generated workload, the depth whose work lands closest
  to a target in log space, from a `bench_cpu_evolve corpusgrow` run. The corpus spans four orders
  of magnitude in work at a fixed depth, so one depth for all of it measures the large workloads
  and measures process startup on the small ones, and a floor-dominated row reports a speedup that
  is a statement about the harness.
- **`corpus_scale_sweep.sh`** -- runs that plan and prints one row per workload: the depth, the
  capacity scale, the speedup at each thread count, and the raw count the run actually reached.
  A row that hits the container ceiling raises `HG_CAPACITY_SCALE` before it gives up depth,
  because a deeper run that fits is worth more than a shallow one; a run that still does not fit
  says so rather than being dropped.
- **`corpus_determinism_sweep.sh`** -- one worker against many over the same workloads, comparing
  canonical state counts. One worker is the ground truth: with a single worker there is no
  interleaving to depend on. Truncated runs are skipped rather than compared, since past the
  ceiling which states got in is the arrival race.
- **`paper_style_check.py`** -- refuses three things in `paper/main.tex`: LLM writing tells, which
  a reader recognises and discounts the content for; war stories, because the paper describes the
  system AS IT IS rather than the history of arriving at it; and any sentence naming remaining
  performance work, which under the v1.0.0 standard is a work item and not a claim to publish.
  Reference list: claudisms.ai. Exit status, never the finding count -- a count landing on 256
  exits 0, which is what was fixed in three other checkers here. A CI gate.
- **`docs_fresh_check.py`** -- the notebooks under `paclet/Documentation/English` are GENERATED
  from `paclet/Documentation/Source/*.md` and committed, so a commit that edits the markdown
  without rerunning `build_docs.sh` ships a page describing behaviour the engine no longer has.
  Compares the last COMMIT touching each, not mtimes, because a fresh clone gives every file the
  same timestamp. Maps source to notebook through the Template frontmatter, since the generator
  names notebooks after the document title. A CI gate; it has caught this twice.
- **`worker_memory_slope.sh`** -- resident set against worker count on a thread list that fits a
  19 GB box, each measurement alone and under `ulimit -v`. The sweep in `scaling_sweep.py` reaches
  7.5 GB at 32 workers and cannot run there; this asks the same question at 1/2/4.
- **`scaling_sweep.py`** -- the thread-scaling and rule-shape tables (T8, T12, T13) plus their
  figures. Writes `t8_scaling.tex` under the same name `paper_tables` does and runs second, which
  is why the pinned three-depth version is the one that survives.
- **`rich_sweep.sh`** -- collects the left-hand-side data set: a DEPTH phase that saturates the box
  (counts are deterministic, so concurrency is free) and a SCALING phase that runs one job at a
  time behind the quiet gate. Records `measured_on.txt` beside the data so the figures cannot
  claim they were measured where they were drawn.
- **`rich_plots.py`** -- turns that data set into the LHS-space figures and T14. Refuses to run
  rather than guess the measuring machine.
- **`quiet_gate.sh`** -- the single quiet check, shared by `rich_sweep.sh` and the Wolfram
  comparison. Watch-list process names are compared TRUNCATED TO 15 CHARACTERS, which is all
  `comm` holds and all `pgrep -x` can match; a longer pattern matches nothing and turns the gate
  into a rubber stamp.
- **`instruction_profile.py`** -- T15, from `callgrind_annotate` output. Sums the flat
  per-function block only; the per-file annotations that follow it cover the same functions again.
- **`remote_session.sh`** / **`remote_drive.sh`** -- the phase runner on a rented box and the
  driver that pulls each phase's artifacts as soon as it returns. A bare branch name resolves to
  `origin/<name>`: `git fetch` does not advance a local branch, so a reused clone would otherwise
  rebuild whatever it last checked out and report that as the commit under measurement.
- **`evidence.sh`** -- the evidence report, including the four device sanitizers, each scoped in a
  comment to what it actually sees (racecheck is shared memory only).

## `verification/` -- model checking and toolchain cells

No suite here is part of any build; all three are gates run by `ctest`.

- **`verification/genmc/`** -- 24 harnesses over RC11, driven by `run.sh` (compile to LLVM IR with
  clang taking SYSTEM headers, then hand the IR to GenMC) and registered as the `genmc_harnesses`
  ctest, which SKIPS where genmc is absent. Every harness is CALIBRATED: the defect it claims to
  catch is injected and the harness must report a violation. That is the only thing that makes a
  small execution count meaningful -- one of these passed with both of its memory fences deleted
  before it was reworked, and `deque_no_double_extraction` still passes with the deque's ABA tag
  removed entirely, which is why `deque_tag_defeats_aba` exists to cover the tag.
- **`verification/tla/`** -- 7 TLC cells over two specs, driven by `run.sh` and registered as the
  `tla_cells` ctest. Each `.cfg` declares its own expected verdict on line 1 and the runner checks
  against it, because TWO OF THE SEVEN MUST VIOLATE: a spec edit that turns a calibration cell
  green has deleted the calibration while every other signal stays clean, and a table of results
  cannot catch that. State counts are reported but not asserted -- TLC stops a violating cell at
  the first counterexample, so its generated count varies with `-workers`.
- **`verification/mingw/`** -- 7 toolchain cells, driven by `run.sh` and registered as the
  `mingw_tls_teardown` ctest, which SKIPS where the mingw cross-compiler is absent or a Windows
  binary cannot be executed. `tls_teardown.cpp` is the smallest program that makes mingw-w64
  corrupt the heap at worker-thread exit, with NO engine code in it, and the corrupting cell is a
  PINNED REPRODUCER in the same sense as the genmc violation harnesses: it passes while the defect
  is still reachable, and says so when a toolchain fixes it. This is what the shipped Windows
  x86-64 artifacts being MSVC-native rests on; the identical source under MSVC is clean. The
  manifestation is heap-layout sensitive -- the output filename alone flips it -- so every cell is
  built to the same binary name in its own directory, and a CLEAN cell means clean at that layout.

## Tests

- **`testing/`** -- `main.cpp` (gtest entry), `test_helpers.hpp` + **`test_helpers.cpp`** (the canonical-form comparisons and the WolframScript shell-out, in all six test targets), `CMakeLists.txt` (fetches GoogleTest, builds `all_tests` + subset targets `core_tests`/`evolution_tests`/`causal_tests`/`stress_tests`/`integration_tests`).
- **`hypergraph/tests/*.cpp`** -- CPU suites: `test_concurrent_map`, `test_quotient_completeness`, `test_causal_tr_exactness`, `test_ir_canonicalization`, `test_pattern_matching`, `test_parallel_evolution`, `test_multiple_initial_states`, `test_evolution_limits`, `test_sampling_reproducibility`, `test_causal_branchial`, `test_event_canonicalization`, `test_determinism_fuzzing`, `test_blackhole_idempotent`, `test_grid_performance`, `test_repeated_invocation`, `test_reference_oracle`.

## `benchmarks/` + `benchmarking/`

- **`benchmarks/*.cpp`** -> the `benchmark_suite` exe: `canonicalization_`, `pattern_matching_`, `state_management_`, `event_relationship_`, `evolution_`, `job_system_`, `wxf_`, `wolfram_integration_benchmark`.
- **`benchmarking/`** -- the reusable framework (lib `benchmark_framework`): `benchmark_framework.hpp` (the declarations, the `BENCHMARK_*` macros and the templates), **`benchmark_framework.cpp`** (the bodies -- statistics, CSV read/write, the reference-data loader, `BenchmarkRegistry`, and `RandomHypergraphGenerator`'s four generators; nine translation units include the header and each was compiling all of it), `benchmark_main.cpp` (the `main`), `random_hypergraph_generator.hpp`, `plot_benchmarks.py`.

## `visualisation/` -- the viz-event interface

One sub-module: `events/` (viz_event_sink -- the stream the engine emits under
BUILD_VISUALIZATION and an external renderer consumes). The renderer and the physics
analyses live in `../hypergraph_viz`, which consumes this engine as a dependency.
