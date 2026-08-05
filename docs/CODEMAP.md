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

- **`core.hpp`** -- id typedefs, structural limits, integer hash primitives (`HG_HD` host/device).
  - id aliases `VertexId`/`EdgeId`/`StateId`/`EventId`/`MatchId`, `INVALID_ID`; limits `MAX_ARITY`/`MAX_PATTERN_EDGES`/`MAX_VARS`
  - `HG_INLINE` -- force-inline, for functions whose inlining must not track unrelated code size
  - `mix64()` (Murmur3 finalizer), `fnv_hash()` (FNV-1a combine), `splitmix64()` (commutative-sum finalizer)
- **`wl_core.hpp`** -- the single shared Weisfeiler-Leman canonical-hash impl, bit-identical CPU/GPU.
  - `WL_MAX_REFINE_ITERS`; `wl_isort()` (device-safe insertion sort); `wl_canonical_hash()` (occurrence-CSR build -> initial colouring -> refinement to fixpoint -> commutative fold)
- **`ir_core.hpp`** -- the single shared EXACT canonicalizer (individualization-refinement), one implementation for host and device.
  - `IR_HOST_GENERATORS`/`IR_DEVICE_GENERATORS` (search-budget split), `ir_scratch_words()` (caller-sized span, no allocation), `IrScratch`, `IrPartition`, `ir_heapsort_idx`/`ir_isort_u64`
  - `ir_canonical_hash()` -- refine, search by individualizing the lowest non-singleton cell, lexicographically smallest form wins; optional outputs per input edge: canonical RANK (`out_edge_rank`), automorphism ORBIT and content CLASS (`out_edge_orbit`/`out_edge_class`, computed in input space from the discovered generators -- the quotient-causal DP's keys)
- **`slot_core.hpp`** -- FRAME SLOTS, one definition for host and device: an edge's rank when a state's edges are ordered by (Aut orbit, `EdgeId`). This is the coordinate system a canonical class's matches are recorded in, which is what lets a match found on one raw instance replay against any other instance of the class; two copies drifting by one tie-break would produce replayed events that are wrong and invisible.
- **`join_core.hpp`** -- THE JOIN: one backtracking-join body for host and device.
  - `JoinState<>` (per-thread frame: bound edge and pattern position per depth, binding + mask, `already_taken` edge-injectivity, `bound_pattern_mask`)
  - `join_next_position()` -- which pattern position to bind next: the first UNBOUND one in the schedule, never `order[depth]`, so a seeded join still binds the positions before its anchor
  - `join_dfs()` / `join_seed()` -- the recursion, and the same recursion anchored at one position (which is what delta matching is). The Ctx supplies candidate enumeration and emit; nothing else differs between the two engines.
- **`event_core.hpp` / `match_core.hpp` / `rewrite_core.hpp` / `signature_core.hpp`** -- the shared semantic cores the two engines drive: event-identity lattice (`EventSignatureKeys`, `event_signature`), pattern-edge binding (`bind_pattern_edge`), rewrite vertex resolution (`resolve_rhs_vertices`, `assign_fresh_consecutive`), edge signatures
- **`park.hpp`** -- futex-style parking (`WaitOnAddress` on Windows, futex on Linux) for the job system's idle waits
- **`portable_intrinsics.hpp`** -- GCC/Clang and MSVC spellings of the intrinsics the engine uses.
  - `popcount`/`popcount64`/`ctz`/`ctz64`, `cpu_relax()` (PAUSE/YIELD)

## `hypergraph/include/hypergraph/` -- core CPU engine (headers)

- **`types.hpp`** -- core value types, IDs, bindings, mode enums.
  - structs `Edge`, `Event`, `State`, `VariableBinding`, `GlobalCounters` (each counter `alignas(64)`), `CausalEdge`, `BranchialEdge`, `EdgeCorrespondence`, `EventSignature`, `VertexHashCache`, `SubtreeBloomFilter`; enums `StateCanonicalizationMode`, `EventSignatureKey(s)`; `AbortedException`
  - quotient reconstruction types: `CanonicalEdgeKey` (the quotient-aware edge identity that meets producers with consumers -- orbit-keyed under quotient, raw `EdgeId` otherwise), `EdgeOrbitTable` (per-state edge orbits + SLOTS), `CanonicalTransition` (orbit-deduplicated), `SlotMatch` (undeduplicated, slot-named)
  - `EMPTY_STATE_CANONICAL_HASH` -- the empty state's own canonical hash; it cannot be 0, which means "not computed" for `State::canonical_hash` and is `ConcurrentMap`'s `EMPTY_KEY`
- **`atomic_compat.hpp`** -- `hypergraph::atomic_ref<T>`: an atomic view over a plain, non-atomic member. `State` keeps some fields as plain scalars so it stays trivially copyable and single-threaded paths touch them directly, while concurrent paths need atomic access to the same words. Selects `std::atomic_ref` where it exists and falls back where it does not -- the OSXCross SDK's bundled libc++ predates C++20, so the macOS cross build cannot name it directly
- **`rule_analysis.hpp`** -- what the RULES decide before a run: `RuleFacts` (edge delta, vertex creation rate, LHS shape) and `RuleSetFacts`/`analyze_rules`. The load-bearing one is `can_branch`, which asks whether two distinct matches can share a CONSUMED edge -- exactly the branchial relation's condition, so a false proves that relation empty for every initial condition. Sound in the false direction only: true means "not ruled out", because the reachability question behind it is undecidable. Holds nothing about termination or global confluence, which are undecidable here
- **`arena.hpp`** -- arena allocators (foundation of off-hot-path, malloc-free allocation).
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
  - `ConcurrentMap<K,V,EMPTY,LOCKED>` (`insert_if_absent[_waiting]`, `lookup[_waiting]`, `count_unique`, `for_each`, optional arena backing via ctor/`set_arena`, `bytes_allocated`), nested `Entry`/`Table`
  - **No tombstone**: a claimed slot is always resolved to a real key, never back to EMPTY, or the probe run of every key passing through it would be cut. Inserters therefore await a claimed slot IN PLACE before claiming one of their own.
  - **Obstruction-free, not lock-free**, and cannot deadlock: a thread holding a claim only stores, never waits.
  - `reject_sentinel_key` throws on a key equal to EMPTY/LOCKED -- such a key is silently unstorable, which caused four separate correctness bugs.
  - `for_each`/`count_unique` walk the whole resize chain and emit each key once (a key can settle only in a superseded table, since `resize()` skips claimed slots).
- **`concurrent_id_set.hpp`** -- lock-free key-only uint32 set (4 B/slot) for the causal Desc closure.
  - `ConcurrentIdSet<EMPTY>` (single-CAS `EMPTY->key` publication — no LOCKED window, no spin; `insert`, `contains`, `for_each`; arena-backed, superseded-table `prev` chain)
- **`lock_free_list.hpp`** -- append-only lock-free linked list.
  - `LockFreeList<T>` (`for_each`/`for_each_while`), `SingleThreadedList<T>`
- **`signature.hpp`** -- edge vertex-repetition signatures + compatible-signature enumeration.
  - `EdgeSignature`, `signature_compatible()`, `enumerate_compatible_signatures()`, `CompatibleSignatureCache`
- **`pattern.hpp`** -- rule representation, builder, match-identity types.
  - `PatternEdge`, `RewriteRule` (`compute_var_counts`/`compute_match_order`), `RuleBuilder` + `make_rule()`, `MatchIdentity`
  - `RewriteRule::match_order` is a SCHEDULE, not a semantic: every permutation yields the same match set (`JoinCore.EveryBindingOrderYieldsTheSameMatches`)
- **`index.hpp`** -- lock-free matching indices for candidate generation.
  - `SignatureIndex`, `InvertedVertexIndex` (`for_each_edge_containing_all` -- shortest-list-seeded intersection), `PatternMatchingIndex`
- **`pattern_matcher.hpp`** -- the host's half of the join: candidate enumeration and emit, over `hgcommon/join_core.hpp` (templated on accessors).
  - `PatternMatchingContext<>`, `HostJoinContext<>` (the join's Ctx: `Candidate` carries the edge the enumerator already fetched)
  - free templates `validate_candidate`, `generate_candidates`, `emit_match`, `scan_pattern[_from_edge]`, `find_matches`, `find_delta_matches`
- **`wl_hash.hpp`** -- Weisfeiler-Leman approximate hashing + O(E) edge correspondence.
  - `WLHash` (`compute_state_hash_with_cache`, `find_edge_correspondence`, `compute_event_signature`)
- **`canonical_types.hpp`** -- shared canonicalization result types.
  - `CanonicalForm`, `VertexMapping`, `CanonicalizationResult` (`are_isomorphic`)
- **`ir_canonicalization.hpp`** -- host face of the McKay individualization-refinement exact canonicalizer; the algorithm is `hgcommon/ir_core.hpp`, shared with the device.
  - `IRCanonicalizer` (`canonicalize_edges`, `compute_canonical_hash[_with_edge_map/_with_edge_rank/_with_edge_orbits]`, `are_isomorphic`)
- **`causal_graph.hpp`** -- online lock-free causal + branchial relationships with online transitive reduction.
  - `CausalGraph` (`set_edge_producer`/`add_edge_consumer`/`propagate_producers` -- all keyed by `CanonicalEdgeKey`, not raw `EdgeId`, so orbit-shared edges meet at one key under quotient; `add_causal_edge`/`add_branchial_edge`; `record_state_event` + `record_branchial_overlaps` (the per-state event list and the branchial pairs it induces, each pair claimed once); the reduction as `record_reduced_edge`/`is_reachable`/`reduces_on_read`/`ids_are_topological`, which is a TAG on one base relation rather than a second graph; `for_each_causal_edge`/`for_each_branchial_edge`)
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
- **`rewriter.cpp`** -- `Rewriter::apply`: validate match, derive child edge set, allocate fresh vertices, create RHS edges/state/event, register causal/branchial (consumed edges in descending-producer order for correct online TR).

## `gpu/include/hg_gpu/` -- CUDA port (headers)

- **`cuda_check.hpp`** -- `HG_CUDA_CHECK(err, what)` / `cuda_check_at`: the ONE CUDA error check.
  Throws naming `__FILE__:__LINE__` and the driver string. Every `.cu` and every device
  container routes through it; none carries its own copy.

- **`overflow.hpp`** -- CUDA-free shared overflow types: `ErrorKind`, `OverflowWarning`, `error_kind_name()`
- **`types.hpp`** -- GPU aliases + device storage structs: `DeviceEvent`, `DeviceCausalEdge`, `DeviceBranchialEdge`, `Edge`, `StateEdgeSlice`; enums `CanonicalizationMode`, `EventCanonicalizationMode`
- **`errors.hpp`** -- device error channel: `DeviceErrors` (`DeviceView::record`), `PoolOverflow`
- **`atomic_pool.hpp`** -- `Pool<T>` (pre-allocated device array + atomic bump counter; `DeviceView::claim`/`claim_n`/`at`)
- **`lock_free_list.hpp`** -- `LockFreeList<T>` (per-key linked-stack over a node Pool; `DeviceView::push`/`for_each`)
- **`hash_table.hpp`** -- `ConcurrentMap<K,V,EMPTY,LOCKED>` (open-addressing linear probe; `DeviceView::lookup[_waiting]`/`insert_if_absent`)
- **`ring_buffer.hpp`** -- `RingBuffer<T>` (bounded MPMC ring; per-slot sequence numbers + CAS reservation, so producers that are also consumers neither lose nor duplicate an item across wraps)
- **`termination.hpp`** -- `TerminationDetector` (per-role quiescence for a persistent-kernel model)
- **`device_arena.hpp`** -- `DeviceArena` (bump allocator the device claims from; scratch whose size is only known once the work is in hand)
- **`edge_signature.hpp`** -- `EdgeSignature` + device `signature_*` helpers (bit-identical to CPU)
- **`signature_index.hpp` / `vertex_inverted_index.hpp`** -- `SignatureIndex` / `VertexInvertedIndex` (device match-candidate indices)
- **`warp_ops.hpp`** -- `VWarp<N>` (cooperative-groups tile ops: ballot/reduce/scan/compact/sorted-intersect)
- **`match.hpp`** -- `DevicePatternEdge`/`DeviceRhsEdge`/`DeviceRule`/`MatchRecord` (carries its `step` and a `published` flag); device `match_state_rule`/`publish_match`/`await_match`; host `make_device_rule`/`run_match_kernel[_batch][_nosync]`
- **`rewrite.hpp`** -- device `apply_one_match` (returns the state it created); host `run_rewrite_kernel[_with][_nosync]`
- **`exploration.hpp`** -- `DedupMap` + device `state_survives_dedup` (which new states get expanded; one predicate for both schedulers)
- **`persistent.hpp`** -- `MatchWorkItem`, `PersistentRunStats`/`PersistentEvolveStats` (incl. the phase-cycle attribution counters); host `run_persistent_match`/`run_persistent_match_rewrite`/`run_persistent_evolve` (the device-resident schedulers), `run_qc_seed_roots`/`run_qc_register_range` (the step loop's quotient-causal drive), `default_persistent_grid()`/`persistent_arena_words()`
- **`quotient_expansion.hpp`** -- expansion capture and per-instance replay, device side: the twin of the host's `qc_capture_expansion`/`for_each_expansion_match` and the `(instance, match)` rendezvous. `DeviceSlotMatch`/`DeviceQcInstance`/`QeAppliedMatch`/`QePredRef`, the `QeView`/`QeState` split, `qe_frame_slot_of`/`qe_register_frame`/`qe_apply`/`qe_drive_instance`/`qe_drive_match`/`qe_reachable`. Under quotient exploration only one raw state per class is expanded, so the raw events the other instances would have produced are never created; this replays them from the class's captured matches
- **`quotient_causal.hpp`** -- the orbit-keyed quotient-causal DP, device twin of the host's `register_quotient_transition` + `qc_*` propagation: `DeviceCanonicalTransition`/`QcProducerNode`/`QcTransitionRef`, `QcView`, host `QcState` (engine-lifetime owner, cleared per run); device `qc_register_transition`/`qc_add_producer`/`qc_process_transition`/`qc_reach`/`qc_emit`. Keys are (state canonical hash, depth, edge orbit) -- no raw ids -- so the causal set under quotient exploration is schedule-independent and equal to the CPU's (gate: `tools/quotient_causal_probe_gpu`)
- **`wl_hash.hpp` / `ir_canon.hpp`** -- device `wl_hash_state_device`, `state_exact_hash_device` (arena-backed, sized per state); host `compute_state_wl/ir_hashes*`
- **`initial_upload.hpp`** -- host `rebuild_indices`/`upload_initial_state[s]`
- **`engine_state.hpp`** -- `DeviceState` (POD passed to kernels) + `EngineState` (host owner of all device pools/indices, readback helpers)
- **`evolve.hpp`** -- the public host API: DTOs `RewriteRule`/`EvolveInput`/`CanonicalState`/`Event`/`CausalEdge`/`BranchialEdge`/`EvolveResult`, `EngineConfig`; classes `Engine` (`run`/`reset`) and `PersistentEvolver` (grow-and-retry reusing one Engine); `evolve()`, `config_from_input()`, `estimated_device_bytes()`

## `gpu/src/` -- CUDA kernels + drivers

- **`evolve.cu`** -- the driver: `Engine::Impl` level-synchronised step loop (match->rewrite->hash->dedup); kernels `k_seed_roots`/`k_fill_unique_keys`; device `state_survives_dedup`; host `config_from_input`/`grow_config_for`/`fit_config_to_cap`/`estimated_device_bytes`/`evolve`/`PersistentEvolver::run`
- **`match.cu`** -- device `match_state_rule` + kernel `k_match_batch`. The JOIN is `hgcommon/join_core.hpp`; this file supplies `MatchJoinCtx` (CSR-slice / pivot-inverted / signature-bucket candidate enumeration) and the block-striped depth-0 parallelism. Host `schedule_lhs_edges`/`make_device_rule`/`run_match_kernel*`
- **`rewrite.cu`** -- device `apply_one_match` + kernel `k_rewrite` (preflight-reserve pools, build RHS/new-state CSR, write Event, causal+branchial rendezvous; the raw-edge causal rendezvous is skipped under `DeviceState::quotient_causal`, where the orbit-keyed DP replaces it); online TR is the `preds_list` backward-reachability oracle (`is_reachable_preds`, the device twin of `CausalGraph::is_reachable` -- no stored closure); `try_add_causal_edge` has external linkage (the DP emits through it); host `run_rewrite_kernel*`
- **`event_identity.hpp` / `event_identity.cu`** -- event identity shared by both schedulers: `event_keys_need_ranks`, `edge_rank_in_state_device`, `stamp_event_signature` (computes the signature AND applies it through a signature -> EventId map, so two applications with the same identity are one event), plus `fill_event_identity_inputs` / `stamp_event_identity_range` for the level-synchronous loop's post-hash phase
- **`persistent.cu`** -- the device-resident schedulers: `k_persistent_match` (match role alone), `k_persistent_match_rewrite` (two roles, no barrier), `k_persistent_evolve` (whole evolution in ONE launch chain with ONE host sync: root-hash seed -> counted queue seed -> rewrite -> key + exact hash [+ orbits] -> event stamp -> quotient-causal registration -> dedup -> re-enqueue, block 0 detecting quiescence; workers back off exponentially when idle and flush per-phase clock64 attribution at exit); `k_qc_seed_roots`/`k_qc_register_range` (the step loop's quotient-causal drive); host `run_persistent_*`, plus `default_persistent_grid()` (eight blocks per SM, from the grid sweep; `HG_GPU_PERSISTENT_BLOCKS` overrides) and `persistent_arena_words()` (IR arena as holders x `ir_arena_share_words`). Identity is answered per mode: `state_key_device` computes what the state mode identifies states by (`None` a per-state unique value with no hashing, `Automatic` the content hash, `Full` the exact IR hash), `stamp_event_signature` fills the event identity from the exact hashes plus the ranks `edge_rank_in_state_device` reads back, then APPLIES it through a signature -> EventId map so two applications with the same identity are one event
- **`ir_canon.cu`** -- `k_ir_canon_range` (one thread per state over a batch-sized slot pool, exact IR via `hgcommon`, 1-WL fallback with a degraded count) + device `state_exact_hash_device` (arena-claimed slot sized per state, no fallback; `want_ranks` scatters each edge's canonical rank into `DeviceState::state_edge_rank`, `want_orbits` its automorphism orbit into `state_edge_orbit` + `state_num_orbits`, both riding the pass the hash already runs); host `compute_state_ir_hashes_range`
- **`wl_hash.cu`** -- device `wl_hash_state_device`/`content_hash_state_device` (delegates to `hgcommon::wl_canonical_hash`); kernels `k_wl_hash_states`/`k_content_hash_range`
- **`initial_upload.cu`** -- `k_init_indices`; host `upload_initial_states`/`rebuild_indices`

## `job_system/include/job_system/` + `lockfree_deque/include/lockfree_deque/`

- **`job.hpp`** -- `Job<JobType>` (abstract), `FunctionJob<>`, `make_job()`, `ScheduleMode`, `CompatibilityAwareJob<>`
- **`job_system.hpp`** -- `JobSystem<JobType>` (the scheduler: per-worker Chase-Lev deques + shared injector, `submit`/`start`/`shutdown`/`wait_for_completion`, `set_on_job_complete` scratch recycle), nested `WorkerData`/`SystemStatistics`; `ErrorType`
  - a WORKER never blocks pushing to the injector (`try_push_back`, else run the job inline): a worker parked in a push cannot pop, so all workers parked there would wedge the system. The inline path passes `recycle_scratch=false`, since the outer job on the same stack still holds live scratch allocations.
- **`job_pool.hpp`** -- `JobSlotPool`: fixed-size job slots recycled through a per-thread free list (`SlotHeader`, `TlsGuard`), so submitting a job does not allocate on the hot path
- **`work_stealing_deque.hpp`** -- `WorkStealingDeque<T>` (bounded Chase-Lev; owner `push`/`pop`, thief `steal`)
- **`lockfree_deque/deque.hpp`** -- `Deque<T>` (bounded MPMC via one packed {tag,head,tail} atomic, ABA-defeating tag; try/blocking push/pop both ends)

## `wxf/` -- Wolfram Exchange Format serialization

- **`wxf.hpp`** -- `Token` enum; exceptions `WXFException`/`ParseError`/`TypeError`; `WXFValue` variant (+`WXFValueList`/`WXFValueAssociation`); `Parser` (typed readers, `read_association`/`read_function`, `read<T>`); `Writer` (typed writers, `write_association`/`write_function`, `data`/`release_data`); free `serialize`/`deserialize`
- **`wxf.cpp`** -- out-of-line `Parser`/`Writer` methods; `wxf_bswap64` (big-endian path)

## `paclet_source/` -- FFI + standalone binary + GPU marshaling

- **`hg_core.hpp`** -- `HostBridge` (progress callback; abort = process kill), `run_rewriting_core()` declaration
- **`hypergraph_ffi.cpp`** -- the marshaling TU: `run_rewriting_core` (WXF<->engine, parses all options, routes CPU or GPU, serializes States/Events/Causal/Branchial), `ffi_helpers::read_rules_association`, plus the LibraryLink DLL export `performRewriting` + `WolframLibrary_initialize/uninitialize`. Builds the `GraphData` block (for the `*Graph` properties) via the shared `hgmarshal::build_graph_data`, adapting the engine through a `CpuGraphSource`
- **`graph_marshal.hpp`** -- the shared `*Graph` marshaller: `hgmarshal::build_graph_data(source, properties, opts)`, templated over a `Source` that exposes the evolved multiway as effective (canonicalization-collapsed) ids + per-vertex tooltips. Both `hypergraph_ffi.cpp` (CPU engine) and `hg_gpu_backend.cpp` (GPU result) drive it, so CPU and GPU emit identical graph structure for every property -- ONE graph-building code path, no divergent copies
- **`hg_evolve_main.cpp`** -- the `hg_evolve` binary: `run_one_shot`, `run_serve` (stdio worker), `run_serve_socket` (loopback-TCP worker), frame I/O helpers, `main` (flag dispatch, progress->stderr)
- **`hg_gpu_backend.hpp`** / **`hg_gpu_backend.cpp`** -- `GpuJob` struct + `run_gpu_evolution` (builds `hg_gpu::EvolveInput`, runs `PersistentEvolver`, regroups the raw GPU result into canonical-class WXF matching the CPU FFI, and builds `GraphData` through the shared `hgmarshal::build_graph_data` via a `GpuGraphSource`); `build_input`

## `paclet/` -- the Wolfram Language paclet

- **`PacletInfo.wl`** -- manifest for `WolframInstitute/HypergraphRewriteEngine` v0.0.1 (Kernel context, LibraryLink resources, Documentation, 6 SystemIDs)
- **`Kernel/HypergraphRewriting.wl`** -- the WL layer. Public surface: `HGEvolve` alone. The initial-condition generators (`HGGrid`, topologies, sprinkling, Brill-Lindquist, Poisson, uniform) remain as internal helpers behind HGEvolve's string initial conditions; the physics analyses live in ../hypergraph_viz
- **`Documentation/Source/*.md`** -- markdown doc sources (Symbol/Guide/TechNote) -> notebooks via `tools/build_docs.wls`

## `reference/` -- validation oracle

- **`oracle_corpus.hpp`** -- the shared measurement substrate: `corpus()` (the named rule cases spanning the rule-type space -- single/mixed arity, productive/idempotent/reductive, self-loop, disconnected LHS, multi-rule, automorphic), `Case`/`Counts`/`LatticeCounts`, the engine drivers `engine_full_count`/`engine_counts`, and the brute-force isomorphism oracle `brute_force_iso_count`/`brute_canonical`/`content_canonical`, which is INDEPENDENT of the engine's WL and IR. One source of truth for what is tested and how it is checked, used by the oracle gate and by `tools/cost_matrix.cpp`
- **`golden_matrix.hpp`** -- the cached identity matrix: every corpus workload across every identity mode, each `Row` carrying the `Provenance` that says what checked it. The brute-force oracle is `O(V! * E log E)` and the WL reference needs wolframscript, so neither runs on every build; caching the expected values lets the gate compare in milliseconds. `event_keys_from_name`/`state_mode_from_name`
- **`MultiwayReference.wl`** -- brute-force ground-truth oracle: `MultiwayEvolve`, `CanonicalForm` (refinement + lex-min), helpers `refineColors`/`findMatches`/`eventSig*`
- **`golden_corpus.wl`** -- `hgGoldenCases`: 12 named cases with expected `{states, rawEvents, causal, branchial}`
- **`verify_paclet.wls` / `verify_paclet_gpu.wls`** -- load the local paclet, check `HGEvolve` against the golden corpus (CPU; GPU via `TargetDevice->"GPU"`)
- **`validate.wls` / `compare_multiwaysystem.wls`** -- validate the reference vs a determinism corpus / cross-check vs the authoritative `MultiwaySystem` paclet
- **`CANONICALIZATION.md`** -- the canonicalization naming cross-map across the four layers

## `tools/` -- standalone probes (each a `main()`)

Validation: `arena_reset_test`, `segmented_array_stress`, `determinism_forwarding_repro`, `causal_tr_determinism_probe`, `causal_tr_exactness_probe`, `canonical_causal_oracle`, `quotient_reconstruction_probe`, `multiplicity_propagation_probe`, `multi_init_rule_2x2_probe`, `quotient_causal_probe_gpu` (quotient causal is schedule-independent at every grid and equal across devices).
Canonicalization research: `ir_vs_wl`, `ir_edge_map_probe`, `ir_edge_orbit_probe`, `ir_incremental_probe`, `ir_malloc_bench`, `incremental_probe`, `incremental_wl_probe`, `wl_engine_incremental`, `wl_sparse_prototype`, `wl_core_bitid_check`.
Physics hunches: `branchial_flux_probe`, `budget_collapse_probe`, `higgs_shadow_probe`.
Profiling: `profile_evolve` (single-threaded, for callgrind/cachegrind), `bench_gpu_evolve` (GPU evolve() vs PersistentEvolver timing; `HG_GPU_DBG_TIME=1` prints the persistent scheduler's phase-cycle attribution), `bench_cpu_evolve` (the CPU twin of the same workload).
Build/docs: `build_paclet.wls` (CreatePacletArchive), `build_docs.wls` (markdown -> paclet notebooks).

## Tests

- **`testing/`** -- `main.cpp` (gtest entry), `test_helpers.hpp`, `CMakeLists.txt` (fetches GoogleTest, builds `all_tests` + subset targets `core_tests`/`evolution_tests`/`causal_tests`/`stress_tests`/`integration_tests`).
- **`hypergraph/tests/*.cpp`** -- CPU suites: `test_concurrent_map`, `test_quotient_completeness`, `test_causal_tr_exactness`, `test_ir_canonicalization`, `test_pattern_matching`, `test_parallel_evolution`, `test_multiple_initial_states`, `test_evolution_limits`, `test_sampling_reproducibility`, `test_causal_branchial`, `test_event_canonicalization`, `test_determinism_fuzzing`, `test_blackhole_idempotent`, `test_grid_performance`, `test_repeated_invocation`, `test_reference_oracle`.
- **`gpu/tests/*.cu`** -- GPU units (`test_atomic_pool`, `test_lock_free_list`, `test_ring_buffer`, `test_hash_table`, `test_indices`, `test_edge_signature`, `test_warp_ops`, `test_engine_state`, `test_match`, `test_rewrite`, `test_wl_hash`, `test_ir_canon`, `test_termination`, `test_exploration_probability`, `test_smoke`) + harnesses `bench_cpu_vs_gpu.cpp`, `test_gpu_vs_cpu_differential.cpp`.

## `benchmarks/` + `benchmarking/`

- **`benchmarks/*.cpp`** -> the `benchmark_suite` exe: `canonicalization_`, `pattern_matching_`, `state_management_`, `event_relationship_`, `evolution_`, `job_system_`, `wxf_`, `wolfram_integration_benchmark`.
- **`benchmarking/`** -- the reusable framework: `benchmark_framework.hpp`, `benchmark_main.cpp` (lib `benchmark_framework`), `random_hypergraph_generator.hpp`, `plot_benchmarks.py`.

## `visualisation/` -- the viz-event interface

One sub-module: `events/` (viz_event_sink -- the stream the engine emits under
BUILD_VISUALIZATION and an external renderer consumes). The renderer and the physics
analyses live in `../hypergraph_viz`, which consumes this engine as a dependency.
