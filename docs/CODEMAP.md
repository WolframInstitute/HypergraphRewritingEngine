# Code map

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
visualisation/   Optional interactive 3D Vulkan viewer + physics analysis (non-core)
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
  - `mix64()` (Murmur3 finalizer), `fnv_hash()` (FNV-1a combine), `splitmix64()` (commutative-sum finalizer)
- **`wl_core.hpp`** -- the single shared Weisfeiler-Leman canonical-hash impl, bit-identical CPU/GPU.
  - `WL_MAX_REFINE_ITERS`; `wl_isort()` (device-safe insertion sort); `wl_canonical_hash()` (occurrence-CSR build -> initial colouring -> refinement to fixpoint -> commutative fold)
- **`portable_intrinsics.hpp`** -- GCC/Clang and MSVC spellings of the intrinsics the engine uses.
  - `popcount`/`popcount64`/`ctz`/`ctz64`, `cpu_relax()` (PAUSE/YIELD)

## `hypergraph/include/hypergraph/` -- core CPU engine (headers)

- **`types.hpp`** -- core value types, IDs, bindings, mode enums.
  - structs `Edge`, `Event`, `State`, `VariableBinding`, `GlobalCounters`, `CausalEdge`, `BranchialEdge`, `EdgeCausalInfo`, `EdgeOccurrence`, `EdgeCorrespondence`, `EventSignature`, `VertexHashCache`, `SubtreeBloomFilter`; enums `StateCanonicalizationMode`, `EventSignatureKey(s)`; `AbortedException`
- **`arena.hpp`** -- arena allocators (foundation of off-hot-path, malloc-free allocation).
  - `Arena<T>`, `ConcurrentArena<T>`, `ConcurrentHeterogeneousArena` (**per-worker bump cursors** — each thread bumps a private non-atomic offset, no shared atomic on the fast path, only a lock-free head CAS to grab a fresh ~1 MB block; scratch arenas keep the shared `allocate_shared` path for `mark/release/reset`; `create`/`create_untracked`), `ArenaWorkerRegistry` (thread→dense index, released at exit), `ArenaVector<T>`; `worker_scratch()`
- **`scratch_alloc.hpp`** -- STL-compatible allocators over the scratch/persistent arenas.
  - `ScratchAlloc<T>`, `PersistAlloc<T>`, `PersistTarget`; `worker_persistent()`; aliases `SVec`/`PVec`/`SSet`/`SUSet`/`SMap`/`SUMap`/`PUMap`
- **`bitset.hpp`** -- sparse chunked bitset for a state's edge set.
  - `SparseBitset` (+ `derive()`, `from_edges()`), nested `Chunk`/`ChunkEntry`
- **`segmented_array.hpp`** -- append-only fixed-segment array, stable pointers, O(1) access.
  - `SegmentedArray<T>` (`emplace`/`emplace_at`/`get_or_default`/`ensure_size`/`operator[]`/`for_each`)
- **`concurrent_map.hpp`** -- lock-free open-addressing append-only hash map.
  - `ConcurrentMap<K,V,EMPTY,LOCKED>` (`insert_if_absent[_waiting]`, `lookup[_waiting]`, `count_unique`, `for_each`), nested `Entry`/`Table`
- **`lock_free_list.hpp`** -- append-only lock-free linked list.
  - `LockFreeList<T>` (`for_each`/`for_each_while`), `SingleThreadedList<T>`
- **`signature.hpp`** -- edge vertex-repetition signatures + compatible-signature enumeration.
  - `EdgeSignature`, `signature_compatible()`, `enumerate_compatible_signatures()`, `CompatibleSignatureCache`
- **`pattern.hpp`** -- rule representation, builder, match-identity types.
  - `PatternEdge`, `RewriteRule` (`compute_var_counts`/`compute_match_order`), `RuleBuilder` + `make_rule()`, `MatchIdentity`, `PartialMatch`
- **`index.hpp`** -- lock-free matching indices for candidate generation.
  - `SignatureIndex`, `InvertedVertexIndex` (`for_each_edge_containing_all` -- shortest-list-seeded intersection), `PatternMatchingIndex`
- **`pattern_matcher.hpp`** -- header-only HGMatch-style SCAN->EXPAND->SINK join (templated on accessors).
  - `PatternMatchingContext<>`; free templates `validate_candidate`, `generate_candidates`, `expand_match`, `scan_pattern[_from_edge]`, `find_matches`, `find_delta_matches`
- **`wl_hash.hpp`** -- Weisfeiler-Leman approximate hashing + O(E) edge correspondence.
  - `WLHash` (`compute_state_hash_with_cache`, `find_edge_correspondence`, `compute_event_signature`)
- **`canonical_types.hpp`** -- shared canonicalization result types.
  - `CanonicalForm`, `VertexMapping`, `CanonicalizationResult` (`are_isomorphic`)
- **`ir_canonicalization.hpp`** -- McKay individualization-refinement exact canonicalizer (the reference algorithm).
  - `IRPartition`; `IRCanonicalizer` (`canonicalize_edges`, `compute_canonical_hash[_with_edge_map/_with_edge_orbits]`, `are_isomorphic`; private `build_adjacency`/`initial_partition`/`refine`/`individualize`/`find_canonical_labeling`)
- **`causal_graph.hpp`** -- online lock-free causal + branchial relationships with online transitive reduction.
  - `CausalGraph` (`set_edge_producer`/`add_edge_consumer`, `add_causal_edge`/`add_branchial_edge`, `update_transitive_closure`/`is_reachable_via_desc`, `register_event_from_state_with_overlap_check`, `for_each_causal_edge`/`for_each_branchial_edge`)
- **`hypergraph.hpp`** -- central store: edges/states/events, indices, canonicalization, causal graph.
  - `Hypergraph` (`create_edge`/`create_state`/`create_event`, `create_or_get_canonical_state`/`get_canonical_state`, `compute_canonical_hash`/`compute_wl_hash`/`compute_content_ordered_hash`, `try_lower_explore_depth`/`try_claim_expanded` for quotient mode, genesis support), result structs `CanonicalStateResult`/`CreateEventResult`
- **`parallel_evolution.hpp`** -- the dataflow parallel multiway evolution engine.
  - `ParallelEvolutionEngine` (`evolve` x2, `evolve_uniform_random`, `add_rule`, `set_*` config; private task methods `execute_*_task`/`submit_*_task`; forwarding `store_match_for_state`/`register_child_with_parent`/`push_match_to_children`/`forward_existing_parent_matches[_eager]`; pruning/RNG/quotient `should_explore`/`sampling_rng`/`propagate_explore_depth`)
  - structs `MatchRecord`, `EvolutionStats`, `MatchContext`, `ScanTaskData`, `ExpandTaskData`, `ChildInfo`, `ParentInfo`; enum `EvolutionJobType`
- **`rewriter.hpp`** -- applies a rule+match to produce a new state (declaration).
  - `RewriteResult`, `Rewriter` (`apply`), `apply_rewrite()`
- **`debug_log.hpp`** -- debug logging routed to an FFI callback or printf (`DEBUG_LOG` macro, no-op unless enabled).

## `hypergraph/src/` -- core engine (out-of-line implementations)

- **`hypergraph.cpp`** -- `Hypergraph` methods: creation + index registration, `create_or_get_canonical_state` dedup, event creation/canonicalization, hashing, edge-correspondence dispatch.
- **`ir_canonicalization.cpp`** -- `IRCanonicalizer` pipeline: `build_adjacency`/`initial_partition`/`refine`/`individualize`/`find_canonical_labeling` + the public hash/canonicalize entries.
- **`causal_graph.cpp`** -- `CausalGraph` methods: lazy slot/list creation, producer/consumer rendezvous, `add_causal_edge`/`update_transitive_closure`/`add_branchial_edge`, `get_causal_edges`/`get_branchial_edges` export.
- **`parallel_evolution.cpp`** -- the engine's implementation: `evolve` loops, the `execute_*`/`submit_*` task engine, match forwarding, pruning/quotient bookkeeping.
- **`rewriter.cpp`** -- `Rewriter::apply`: validate match, derive child edge set, allocate fresh vertices, create RHS edges/state/event, register causal/branchial (consumed edges in descending-producer order for correct online TR).

## `gpu/include/hg_gpu/` -- CUDA port (headers)

- **`overflow.hpp`** -- CUDA-free shared overflow types: `ErrorKind`, `OverflowWarning`, `error_kind_name()`
- **`types.hpp`** -- GPU aliases + device storage structs: `DeviceEvent`, `DeviceCausalEdge`, `DeviceBranchialEdge`, `Edge`, `StateEdgeSlice`; enums `CanonicalizationMode`, `EventCanonicalizationMode`
- **`errors.hpp`** -- device error channel: `DeviceErrors` (`DeviceView::record`), `PoolOverflow`
- **`atomic_pool.hpp`** -- `Pool<T>` (pre-allocated device array + atomic bump counter; `DeviceView::claim`/`claim_n`/`at`)
- **`lock_free_list.hpp`** -- `LockFreeList<T>` (per-key linked-stack over a node Pool; `DeviceView::push`/`for_each`)
- **`hash_table.hpp`** -- `ConcurrentMap<K,V,EMPTY,LOCKED>` (open-addressing linear probe; `DeviceView::lookup[_waiting]`/`insert_if_absent`)
- **`ring_buffer.hpp`** -- `RingBuffer<T>` (MPMC ring for inter-kernel work queues)
- **`termination.hpp`** -- `TerminationDetector` (per-role quiescence for a persistent-kernel model)
- **`edge_signature.hpp`** -- `EdgeSignature` + device `signature_*` helpers (bit-identical to CPU)
- **`signature_index.hpp` / `vertex_inverted_index.hpp`** -- `SignatureIndex` / `VertexInvertedIndex` (device match-candidate indices)
- **`warp_ops.hpp`** -- `VWarp<N>` (cooperative-groups tile ops: ballot/reduce/scan/compact/sorted-intersect)
- **`partial_match.hpp`** -- `PartialMatch` (per-warp DFS match frame in registers/shared)
- **`match.hpp`** -- `DevicePatternEdge`/`DeviceRhsEdge`/`DeviceRule`/`MatchRecord`; host `make_device_rule`/`run_match_kernel[_batch][_nosync]`
- **`rewrite.hpp`** -- host `run_rewrite_kernel[_with][_nosync]`
- **`wl_hash.hpp` / `ir_canon.hpp`** -- device `wl_hash_state_device`; host `compute_state_wl/ir_hashes*`
- **`initial_upload.hpp`** -- host `rebuild_indices`/`upload_initial_state[s]`
- **`engine_state.hpp`** -- `DeviceState` (POD passed to kernels) + `EngineState` (host owner of all device pools/indices, readback helpers)
- **`evolve.hpp`** -- the public host API: DTOs `RewriteRule`/`EvolveInput`/`CanonicalState`/`Event`/`CausalEdge`/`BranchialEdge`/`EvolveResult`, `EngineConfig`; classes `Engine` (`run`/`reset`) and `PersistentEvolver` (grow-and-retry reusing one Engine); `evolve()`, `config_from_input()`, `estimated_device_bytes()`

## `gpu/src/` -- CUDA kernels + drivers

- **`evolve.cu`** -- the driver: `Engine::Impl` level-synchronised step loop (match->rewrite->hash->dedup); kernels `k_seed_roots`/`k_dedup_and_append`/`k_fill_unique_keys`; host `config_from_input`/`grow_config_for`/`fit_config_to_cap`/`estimated_device_bytes`/`evolve`/`PersistentEvolver::run`
- **`match.cu`** -- match kernels `k_match_one_state`, `k_match_batch` (DFS binding LHS edges, Wolfram non-distinct semantics, CSR-slice/signature/pivot-inverted candidate seeding); host `schedule_lhs_edges`/`make_device_rule`/`run_match_kernel*`
- **`rewrite.cu`** -- `k_rewrite` (preflight-reserve pools, build RHS/new-state CSR, write Event, causal+branchial rendezvous with online TR); host `run_rewrite_kernel*`
- **`ir_canon.cu`** -- `k_ir_canon_range` (one block per state -> shared `IRBlock`, 1-WL refinement, discrete labelling, canonical hash; WL fallback); host `compute_state_ir_hashes_range`
- **`wl_hash.cu`** -- device `wl_hash_state_device`/`content_hash_state_device` (delegates to `hgcommon::wl_canonical_hash`); kernels `k_wl_hash_states`/`k_content_hash_range`
- **`initial_upload.cu`** -- `k_init_indices`; host `upload_initial_states`/`rebuild_indices`

## `job_system/include/job_system/` + `lockfree_deque/include/lockfree_deque/`

- **`job.hpp`** -- `Job<JobType>` (abstract), `FunctionJob<>`, `make_job()`, `ScheduleMode`, `CompatibilityAwareJob<>`
- **`job_system.hpp`** -- `JobSystem<JobType>` (the scheduler: per-worker Chase-Lev deques + shared injector, `submit`/`start`/`shutdown`/`wait_for_completion`, `set_on_job_complete` scratch recycle), nested `WorkerData`/`SystemStatistics`; `ErrorType`
- **`work_stealing_deque.hpp`** -- `WorkStealingDeque<T>` (bounded Chase-Lev; owner `push`/`pop`, thief `steal`)
- **`lockfree_deque/deque.hpp`** -- `Deque<T>` (bounded MPMC via one packed {tag,head,tail} atomic, ABA-defeating tag; try/blocking push/pop both ends)

## `wxf/` -- Wolfram Exchange Format serialization

- **`wxf.hpp`** -- `Token` enum; exceptions `WXFException`/`ParseError`/`TypeError`; `WXFValue` variant (+`WXFValueList`/`WXFValueAssociation`); `Parser` (typed readers, `read_association`/`read_function`, `read<T>`); `Writer` (typed writers, `write_association`/`write_function`, `data`/`release_data`); free `serialize`/`deserialize`
- **`wxf.cpp`** -- out-of-line `Parser`/`Writer` methods; `wxf_bswap64` (big-endian path)

## `paclet_source/` -- FFI + standalone binary + GPU marshaling

- **`hg_core.hpp`** -- `HostBridge` (progress callback; abort = process kill), `run_rewriting_core()` declaration
- **`hypergraph_ffi.cpp`** -- the marshaling TU: `run_rewriting_core` (WXF<->engine, parses all options, routes CPU or GPU, serializes States/Events/Causal/Branchial/analysis), `ffi_helpers::read_rules_association`, plus LibraryLink DLL exports `performRewriting`/`performHausdorffAnalysis`/`performBranchAlignment[Batch]` + `WolframLibrary_initialize/uninitialize`
- **`hg_evolve_main.cpp`** -- the `hg_evolve` binary: `run_one_shot`, `run_serve` (stdio worker), `run_serve_socket` (loopback-TCP worker), frame I/O helpers, `main` (flag dispatch, progress->stderr)
- **`hg_gpu_backend.hpp`/`.cpp`** -- `GpuJob` struct + `run_gpu_evolution` (builds `hg_gpu::EvolveInput`, runs `PersistentEvolver`, regroups the raw GPU result into canonical-class WXF matching the CPU FFI); `build_input`

## `paclet/` -- the Wolfram Language paclet

- **`PacletInfo.wl`** -- manifest for `WolframInstitute/HypergraphRewriteEngine` v0.0.1 (Kernel context, LibraryLink resources, Documentation, 6 SystemIDs)
- **`Kernel/HypergraphRewriting.wl`** -- the WL layer. Public: `HGEvolve`; analysis/plots `HGHausdorffAnalysis`/`HGStateDimensionPlot`/`HGTimestepUnionPlot`/`HGDimensionFilmstrip`/`HGGeodesicPlot`/`HGGeodesicFilmstrip`/`HGLensingPlot`/`HGBranchAlignmentBatch`; `HGToGraph`/`EdgeId`; IC generators `HGGrid`/`HGGridWithHoles`/`HGCylinder`/`HGTorus`/`HGSphere`/`HGKleinBottle`/`HGMobiusStrip`/`HGMinkowskiSprinkling`/`HGBrillLindquist`/`HGPoissonDisk`/`HGUniformRandom`; plus `SyntaxInformation` for all of them
- **`Documentation/Source/*.md`** -- markdown doc sources (Symbol/Guide/TechNote) -> notebooks via `tools/build_docs.wls`

## `reference/` -- validation oracle

- **`MultiwayReference.wl`** -- brute-force ground-truth oracle: `MultiwayEvolve`, `CanonicalForm` (refinement + lex-min), helpers `refineColors`/`findMatches`/`eventSig*`
- **`golden_corpus.wl`** -- `hgGoldenCases`: 12 named cases with expected `{states, rawEvents, causal, branchial}`
- **`verify_paclet.wls` / `verify_paclet_gpu.wls`** -- load the local paclet, check `HGEvolve` against the golden corpus (CPU; GPU via `TargetDevice->"GPU"`)
- **`validate.wls` / `compare_multiwaysystem.wls`** -- validate the reference vs a determinism corpus / cross-check vs the authoritative `MultiwaySystem` paclet
- **`CANONICALIZATION.md`** -- the canonicalization naming cross-map across the four layers

## `tools/` -- standalone probes (each a `main()`)

Validation: `arena_reset_test`, `segmented_array_stress`, `determinism_forwarding_repro`, `causal_tr_determinism_probe`, `causal_tr_exactness_probe`, `canonical_causal_oracle`, `quotient_reconstruction_probe`, `multiplicity_propagation_probe`, `multi_init_rule_2x2_probe`.
Canonicalization research: `ir_vs_wl`, `ir_edge_map_probe`, `ir_edge_orbit_probe`, `ir_incremental_probe`, `ir_malloc_bench`, `incremental_probe`, `incremental_wl_probe`, `wl_engine_incremental`, `wl_sparse_prototype`, `wl_core_bitid_check`.
Physics hunches: `branchial_flux_probe`, `budget_collapse_probe`, `higgs_shadow_probe`.
Profiling: `profile_evolve` (single-threaded, for callgrind/cachegrind), `bench_gpu_evolve` (GPU evolve() vs PersistentEvolver timing).
Build/docs: `build_paclet.wls` (CreatePacletArchive), `build_docs.wls` (markdown -> paclet notebooks).

## Tests

- **`testing/`** -- `main.cpp` (gtest entry), `test_helpers.hpp`, `CMakeLists.txt` (fetches GoogleTest, builds `all_tests` + subset targets `core_tests`/`evolution_tests`/`causal_tests`/`stress_tests`/`integration_tests`).
- **`hypergraph/tests/*.cpp`** -- CPU suites: `test_concurrent_map`, `test_quotient_completeness`, `test_causal_tr_exactness`, `test_ir_canonicalization`, `test_pattern_matching`, `test_parallel_evolution`, `test_multiple_initial_states`, `test_evolution_limits`, `test_sampling_reproducibility`, `test_causal_branchial`, `test_event_canonicalization`, `test_determinism_fuzzing`, `test_blackhole_idempotent`, `test_grid_performance`, `test_repeated_invocation`, `test_reference_oracle`.
- **`gpu/tests/*.cu`** -- GPU units (`test_atomic_pool`, `test_lock_free_list`, `test_ring_buffer`, `test_hash_table`, `test_indices`, `test_edge_signature`, `test_warp_ops`, `test_engine_state`, `test_match`, `test_partial_match`, `test_rewrite`, `test_wl_hash`, `test_ir_canon`, `test_termination`, `test_exploration_probability`, `test_smoke`) + harnesses `bench_cpu_vs_gpu.cpp`, `test_gpu_vs_cpu_differential.cpp`.

## `benchmarks/` + `benchmarking/`

- **`benchmarks/*.cpp`** -> the `benchmark_suite` exe: `canonicalization_`, `pattern_matching_`, `state_management_`, `event_relationship_`, `evolution_`, `job_system_`, `wxf_`, `wolfram_integration_benchmarks`.
- **`benchmarking/`** -- the reusable framework: `benchmark_framework.hpp`, `benchmark_main.cpp` (lib `benchmark_framework`), `random_hypergraph_generator.hpp`, `plot_benchmarks.py`.

## `visualisation/` -- optional 3D viewer (non-core, BUILD_VISUALIZATION only)

Sub-modules: `app/` (executables) · `gal/` (graphics abstraction layer, Vulkan backend) · `scene/` · `camera/` · `events/` · `layout/` (+ optional CUDA backend) · `math/` · `platform/` · `shaders/` (GLSL + SPIR-V) · `blackhole/` (geodesic tracing / particle detection + WL reference) · `tools/` · `tests/`.
