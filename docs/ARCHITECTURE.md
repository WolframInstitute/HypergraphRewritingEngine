# Architecture

A map of the system for developers. Users want [QUICKSTART.md](QUICKSTART.md); this
is "start reading here" for working on the engine.

## What it computes

Multiway hypergraph rewriting (Wolfram Physics model). Given rewrite rules and an
initial hypergraph, apply the rules in **all** possible ways to build the multiway
states graph, deduplicating isomorphic states, and derive the causal graph (which
event enables which) and branchial graph (which events are siblings). The two hard
costs are **subgraph matching** (finding rule applications) and **canonicalization**
(deciding when two states are the same up to isomorphism).

## The path of a call

```
Wolfram Language                C++                                    GPU (optional)
────────────────                ───                                    ──────────────
HGEvolve[...]                                                          
  paclet/Kernel/          WXF over                                     
  HypergraphRewriting.wl  stdin / loopback socket                     
        │  ───────────────────────►  hg_evolve  (paclet_source/hg_evolve_main.cpp)
        │                                 │  run_rewriting_core (hypergraph_ffi.cpp)
        │                                 ▼
        │                            hypergraph/  (the CPU engine)      hg_evolve_gpu
        │                                 │                             hg_gpu_backend.cpp
        │                                 │                                   │
        │                            ParallelEvolutionEngine                  ▼
        │                            + Hypergraph + matcher                 gpu/  (CUDA)
        │                            + WL/IR canonicalization              match/rewrite/
        │                            on job_system + arenas                canon/dedup
        │                                 │                                   │
        │  ◄───────────────────────  WXF result  ◄──────── marshal ──────────┘
  Graph / properties
```

Process isolation: the paclet shells out to the standalone `hg_evolve` binary over
WXF (stdin/stdout, or a persistent worker over a loopback socket). A crash or an
Alt-. abort is a process kill, so it never takes down the notebook. The binary is
shipped for every platform (and `hg_evolve_gpu` on the CUDA platforms); the
LibraryLink library remains only as a last-resort fallback and for the standalone
analysis functions.

## Modules

For a per-file guide — every file with a description and the classes / structs /
functions it contains — see [CODEMAP.md](CODEMAP.md). This section is the
higher-level module overview.

Core engine:
- **`hypergraph/`** — the CPU engine (the heart). `ParallelEvolutionEngine`
  (`parallel_evolution.*`) drives the lock-free dataflow: match → rewrite → dedup as
  states are produced. `Hypergraph` (`hypergraph.*`) stores edges/states/events and
  owns canonical-state dedup. Matching is a worst-case-optimal join
  (`pattern_matcher.hpp`, `ancestry.hpp`, `signature.hpp`). Canonicalization under the exact
  mode is McKay individualization-refinement (`ir_canonicalization.*`) on every
  state; its hash selects the dedup key and its canonical form, stored once per canonical
  state (`hgcommon/canonical_form_core.hpp`), decides a key hit; the coarse modes deduplicate by a
  content-ordered key (`content_core.hpp`), and a key hit compares the content words with the
  class's first state. Event identity claims the signature values the same way: a key hit
  recomputes the class's first event's values and compares them. No filter stands in front of IR. The bound: a
  sound filter's distinct keys are at most the canonical classes, so it can skip at most
  `canonical/raw` of the IR calls while paying its own pass on every state -- measured per case by `tools/cost_matrix`
  across the whole rule-type corpus. Within every rule that keeps rewriting the
  ceiling FALLS with depth (`binary-growth` 29.4% -> 4.1%, `star4-automorphic`
  3.1% -> 0.3%, `cycle4-automorphic` 0.5% -> 0.0% at 68,184 events), because
  reaching the same state along many histories is what multiway evolution does;
  the cases holding a high ceiling are the ones that stop rewriting (`self-loop`
  is 100% at two raw states and one event). Where an option needs the automorphism data (quotient
  exploration, event-canonicalization conventions), the orbits come from that same
  IR pass, so nothing is canonicalized twice. Storage is
  arena-backed and lock-free (`arena.hpp`, `segmented_array.hpp`, `concurrent_map.hpp`,
  `bitset.hpp`, `lock_free_list.hpp`). Causal/branchial in `causal_graph.hpp`.
- **`job_system/`** — work-stealing task scheduler the engine runs on.
- **`lockfree_deque/`** — the lock-free deque backing the scheduler.
- **`common/`** (`hgcommon/`) — shared primitives: portable intrinsics, the WL hash
  core shared by CPU and GPU.
- **`wxf/`** — Wolfram Exchange Format reader/writer for the WL boundary.

GPU (optional, `BUILD_GPU=ON`):
- **`gpu/`** — the CUDA port. Mirrors the CPU algorithms (match/rewrite/canon/dedup)
  in a level-synchronous step loop; `hg_gpu::PersistentEvolver` keeps the device
  engine alive across calls. See [gpu/ARCHITECTURE.md](../gpu/ARCHITECTURE.md).

Boundary + tooling:
- **`paclet/`** — the Wolfram Language paclet (Kernel code, bundled per-platform
  binaries, documentation notebooks).
- **`paclet_source/`** — the FFI: `run_rewriting_core` (host-agnostic body),
  `hg_evolve_main.cpp` (the standalone binary + worker modes), `hg_gpu_backend.cpp`
  (GPU marshaling).
- **`reference/`** — the validation oracle: `MultiwayReference.wl` (brute-force
  ground truth) + golden corpus + paclet verification scripts. See
  [reference/CANONICALIZATION.md](../reference/CANONICALIZATION.md).
- **`tools/`** — standalone research/validation probes (canonicalization vs WL,
  quotient reconstruction, determinism, profiling harnesses). `make host_tools`
  builds every non-GPU one.
- **`testing/`** — the aggregate C++ test target (`all_tests`).
- **`benchmarks/` / `benchmarking/`** — per-area benchmarks and the framework lib.
- **`visualisation/`** — the viz-event interface; the renderer lives in its own
  repository.
- **`verification/`** — GenMC/GPUMC harnesses and TLA+ specifications; coverage
  and bounds are stated in [VERIFICATION_STATUS.md](VERIFICATION_STATUS.md), and
  every optimisation's closed-or-refuted record is
  [OPTIMISATION_DISPOSITIONS.md](OPTIMISATION_DISPOSITIONS.md).

## Key ideas a developer should know

- **Canonicalization is tiered by mode, not tiered per state.** `None` = no dedup;
  `Automatic` = fast content hash (may false-merge); `Full` = exact IR. WL is the
  fast approximate hash; IR is exact. (A WL-bucket + IR-on-collision "tiered exact"
  scheme was tried and profiled as a pessimization — the multiway is dedup-heavy, so
  duplicates still need IR to confirm.)
- **Quotient exploration** (`ExploreFromCanonicalStatesOnly`) expands each canonical
  state once at its shortest depth, so a run costs the canonical closure, not the
  exponentially larger provenance count. The raw events full capture would have fired
  are therefore never created, so causal and branchial structure has to be
  RECONSTRUCTED rather than observed: each canonical state retains its expanded
  representative's full match list named in SLOTS (an edge's rank under a canonical
  content-class ordering, so slot *i* names corresponding edges across every raw
  instance of that state), plus one pinned reference frame per canonical class,
  because two raw states of one class have labelings differing by an automorphism.
  Causal edges key on `CanonicalEdgeKey` — the edge ORBIT, the only edge identity
  invariant across the labelings by which distinct parents reach one canonical state.
  Verified against full capture across the reconstruction matrix; it runs whenever a
  request records causal, branchial or raw-event data.
  A request that reads the raw events and branchial pairs only as counts takes them
  from class multiplicities instead: `m(class, depth)`, the number of raw states a
  class stands for at a depth, times the class's match and overlapping-pair counts
  (`hgcommon/quotient_multiplicity_core.hpp`). Its cost is per (class, depth, match),
  not per raw state.
- **Everything on the hot path is lock-free and arena-allocated.** No mutexes, no
  `std::` heap containers on the hot path. Fixes stay lock-free.
- **Concurrent-structure invariants that are easy to break.** `ConcurrentMap` is open
  addressed with no tombstone, so a claimed slot must never return to EMPTY — that
  cuts the probe run of every key passing through it, and the next insert of a hidden
  key reports itself as newly inserted, which is what every dedup decision reads. A
  key equal to a sentinel is unstorable and is rejected loudly. `SegmentedArray`
  records no extent, so an index is readable only after its writer published it to the
  reader, and its capacity is a CONFIGURED CEILING rather than an assumption: the
  segment table is an inline array of `MAX_SEGMENTS` pointers, so a workload past
  `MAX_SEGMENTS * segment_size` elements raises `CapacityExhausted`, and the engine
  serves the states, events and relations it reached with a warning instead of
  terminating the caller. Past that point WHICH states got in is decided by the arrival
  race, so a truncated run is not a measurement and any two of them may disagree.
  `Hypergraph(capacity_scale)` multiplies the segment size and is the only way past it;
  raising the segment COUNT instead would grow every `Hypergraph` object by eight bytes
  per segment per array, and this type is constructed on stacks with a one-megabyte
  limit. A worker must never block pushing to the injector, since a worker parked
  in a push cannot pop. Election among concurrent participants must key on a CLAIM,
  never on an id ordering that is not the visibility ordering.
- **StepStatistics in the job system.** Each step's record is built by jobs on the
  engine's workers. The step's classes are pooled by one job per 256 classes
  (`stats::pool_classes`): each class adds weighted (slot, value) entries and integer totals
  (`hgcommon::sp_add_class`, `hgcommon/step_pool_core.hpp`), and equal entries add their
  weights. The pool is merged and summarised by one job per slot (`stats::StepPoolFinish`),
  which reads each slot's values in ascending order. One job builds the step's branchial graph
  and its metrics, and the overlap rows are split over up to one job per worker, the last of
  which merges the row counts in key order and writes the overlap keys. With
  "StepStatisticsMatrices" the same jobs run the rows over the states' edge numbers too; each
  pair that shares an element writes its count to its own cell, and each row then fills its row
  of the state matrices (`hgcommon::bo_matrix_row`), so the matrices need no merge and the finish
  copies them into the reply as packed arrays. A step's record is a
  function of its pool and counts, which are the same whichever job ran which part, so the reply
  is byte-identical at any thread count and on either engine.
  Under full capture a step's jobs are submitted when its depth settles (the engine's depth
  join, `set_on_depth_complete`), during the run: the step's raw states are in a per-step
  list the hypergraph fills as it creates them, and its branchial pairs are taken from the
  per-worker branchial lists since the previous step. With "MaxStatesPerStep" step d's
  states are created by step d's selection, which runs on depth d + 1, so step d is closed
  when d + 1 settles. A class whose record its claiming worker has not yet published (it can
  be running at a deeper step) is computed by the step's job and published by
  compare-and-swap; the record is a function of the class, so either copy is the same.
  Under quotient exploration a class's multiplicity at a depth grows until every shallower
  class is expanded, and a depth can still be lowered by a path found later, so no depth is
  final before the run is (see `set_on_depth_complete`); there, and in any run whose classes
  are weighted by the multiplicity count (`Hypergraph::quotient_multiplicity`), every step's
  jobs are submitted when the run ends. A session's steps, and steps the depth join did not
  report, are submitted the same way at the reply. The reply then concatenates finished
  records. The GPU backend runs the same jobs on a host job system after the device run. It
  makes the pool of a run of 1024 classes or more on the device with the same class body
  (`hg_gpu::step_pool_device`), and counts the overlap rows of a step of 2048 states or more on
  the device with the same row body (`hgcommon::br_row`; DEVICE_DESIGN.md 4.8).
- **The GPU mirrors the CPU algorithms.** Never drop a CPU data structure in a kernel
  without justification (an inverted-index skip once cost 200x).

## Where each analysis runs

Every property and option of `HGEvolve` (docs/en/ReferencePages/Symbols/HGEvolve.md), with where it is computed, on which threads, whether during the evolution or after it, and where the GPU engine computes it. "Calling thread" is the thread that runs the job (`run_rewriting_core`, `run_gpu_evolution_once`); "workers" are the engine's job-system workers; "reply jobs" are side jobs on the engine's workers on the CPU engine (`submit_side_job`) and jobs on a `hgmarshal::HostJobs` per reply on the GPU engine. Instruction counts are callgrind totals at 4 workers on the job named.

| Analysis | Code | Threads | When | GPU engine | Instructions |
|---|---|---|---|---|---|
| Per-state record: the 37 invariants, geometry, vertex and edge distributions, ball-growth dimensions and mean log geodesic counts | `hgcommon::state_record` (state_invariants_core.hpp, state_geometry_core.hpp), `Hypergraph::record_state_invariants` (hypergraph.cpp) | workers, the one that creates the class | during evolution | device, the warp that creates the class (`record_state_invariants_device`, ir_canon.cu) | wpp d8 quotient StepStatistics: 118.5e9 of 149.0e9 |
| StepStatistics pool and summaries (`Invariants`, `VertexInvariants`, histograms, `BallGrowthDimension`, multiplicities, entropy) | `stats::pool_classes`, `stats::StepPoolFinish` (paclet_support.cpp), `hgcommon::sp_add_class` (step_pool_core.hpp) | reply jobs: one per 256 classes, one per slot | full capture: as each step's depth settles; quotient exploration and sessions: after evolution | pool on the device for 1024 classes or more (`step_pool_device`, step_pool.cu), summaries in reply jobs | wpp d7 quotient StepStatistics: 0.46e9 of 10.25e9 |
| `StepStatisticsBranchial` | `stats::BranchialStepWork` (paclet_support.cpp), `hgcommon::br_row` | reply jobs: one per step for the graph metrics, up to one per worker for the overlap rows | as the pool | overlap rows of a step of 2048 states or more on the device (`branchial_rows_device`); each step's states and the branchial pairs assembled in reply jobs (`hgmarshal::branchial_steps`, one job per step and one for the pairs) | wpp d6 Automatic, Graph and Overlap: 15.38e9, of which the overlap rows 14.0e9 in up to one job per worker and the graph metrics and index 0.17e9; GPU worker's calling thread 0.196e9 of 0.60e9 |
| `VertexCurvature` ("IncludeVertexCurvature") | `stats::vertex_curvature` through `hgcommon::sg_state_geometry` | reply jobs, in each state's record job | after evolution | device, one warp per state (`vertex_curvature_device`, vertex_curvature.cu); the record edges and each state's vertices in reply jobs; a state past a quarter of the free device memory in its record job | as the records |
| Causal relation and its transitive reduction | `CausalGraph::add_causal_edge` (causal_graph.cpp), online reachability | workers, in the rewrite task | during evolution | device rewrite kernel (rewrite.cu) | part of the evolution |
| Branchial relation | rewriter.cpp, causal_graph.cpp | workers | during evolution | device | part of the evolution |
| Quotient reconstruction of raw events and relations | `hgcommon::qr_apply` (quotient_replay_core.hpp) through `Hypergraph::qc_apply` | workers | during evolution; the reconstructed lists are read after it on the calling thread | device replay (`qe_apply`), pairs read back on the host | part of the evolution |
| `StatesGraph`, `CausalGraph`, `BranchialGraph`, `Evolution*Graph` and the `Structure` variants | `hgmarshal::build_graph_data` (graph_marshal.hpp) | vertex and edge choice on the calling thread; payloads, edge lists and records in reply jobs, 256 per job | after evolution | the same code in reply jobs | wpp d7 quotient StatesGraph and CausalGraph: 6.55e9; the calling thread's share of the reply 1.15e9 |
| `States` (with `CanonicalHash`), `Events` | hypergraph_ffi.cpp, hg_gpu_backend.cpp with `hgmarshal::write_state_record`, `write_event_record`, `write_items` | reply jobs, 256 records per job; under Full each state's IR canonical form, and under None and Automatic its `CanonicalHash`, in its job | after evolution | the same in reply jobs; the host IR hashes of a None or Automatic run in reply jobs | with the next row |
| `CausalEdges`, `BranchialEdges`, `GlobalEdges`, `StateBitvectors` | `hgmarshal::record_list`, `write_items` | pair lists on the calling thread, records in reply jobs | after evolution | the same | wpp d6 Full, every record section: 0.74e9, of which the calling thread 0.10e9 |
| `EventClasses` | the Wolfram kernel groups `"Events"` (HypergraphRewriting.wl) | kernel | after the reply | as the CPU | none in the engine |
| `NumStates`, `NumEvents`, `NumCausalEdges`, `NumBranchialEdges`, `Debug` | counters and multiplicity sums | calling thread | after evolution | host, from the downloaded result | under 1e6 |
| `Device` | build stamp and `probe_device` | calling thread | after evolution | host | under 1e6 |
| The reply's WXF bytes | `wxf::association_pieces`: the reply as `wxf::Pieces`, each top-level `wxf::Encoded` section and the streamed States and Events bytes a run of its own, written by the transport (`hg_evolve`'s pipe and socket workers) without being joined; the LibraryLink adapter copies the runs into its result array once | calling thread | after evolution | host | the runs between the sections: keys and small values |
| Sessions: `Step`, `Query`, deltas | `CpuEngineHolder::extend`; the delivery cursor in `build_graph_data` | workers for the step; the cursor's record on the calling thread | the step runs as an evolution; the delta after it | GPU session step on the device | as the evolution and the graph rows |

These run on the calling thread after evolution: the choice of a graph's vertices and edges (its vertex and pair sorts run on the reply jobs, `hgmarshal::parallel_sort`) and the delivery cursor's record, the relation lists' pair choice, the multiplicity points and events of StepStatistics under quotient exploration and in sessions (`for_each_class_multiplicity`, `events_from_multiplicities`; each step's classes are added in a job of their own) and on the GPU engine the class records' map and the grouping of classes and states by step (each step's classes are added, and each step's branchial states and the branchial pairs are assembled, `hgmarshal::branchial_steps`, in reply jobs).

## Sessions

The FFI serves four verbs over an opaque session — `Open`, `Step`, `Query`, `Close` — beside the
one-shot `Evolve`, on both devices. The decisions the session code encodes, numbered as the code
cites them:

- **D7 — one session at a time.** A single slot, matching the transport (the socket worker
  accepts one connection). A second `Open` while one is live is an error, not an eviction:
  evicting would discard a caller's exploration silently. Both devices refuse with one spelling.
- **D9 — sessions exist on both devices.** The holder is an interface and the slot never names a
  device, so the device half extends the same shape rather than replacing it.
- **D10 — continuation preserves raw vertex labels.** A session that renumbered vertices between
  steps would disconnect a caller's own bookkeeping from the state it names.
- **D11 — the handle is an opaque `uint64`,** minted per worker, with 0 reserved for "no
  session" — absence and a real handle are never confused, the discipline every id space here
  follows.
- **D13 — the device gap was retained *exploration*, not retained *allocation*:** a device
  session extends from its own frontier rather than re-uploading, and `Step` with a frontier
  SUBSET is the remaining device limitation (it is refused with the reason).
- **D14 — a hard overflow invalidates the session,** and the handle then reports that rather
  than silently serving a fresh empty engine — which would return a graph that had lost its
  history and satisfy every internal check.
- **D16 — a held verb takes its identity options from the session,** not from its own envelope:
  a `Query` answered under different identity choices than the session's `Open` would describe a
  different multiway system with the same handle.
- **D17 — a session records every artifact.** An `Open` sets the whole record set (causal,
  branchial, state events, raw events), so a held verb answers any property, whatever the `Open`
  named.

`paclet_source/session.hpp` carries D7/D11/D14 as assertions; `test_session.cpp` and
`verify_sessions.wls` gate the verbs end to end.

## Sampling

The samplers thin the multiway graph while keeping the observables of the unpruned evolution
well-defined; `docs/SPEC.md` §5 states which options are samplers and which are caps.

- **A rate, not a count.** A per-state COUNT needs the state's whole match population, which is
  complete only at the state's drain, and `MatchesPerStateRule` chooses there. A RATE decides per
  match, independently, with no population and no join. `TransitionRate` is that sampler;
  thinning a branching process by a rate yields a branching process with the thinned offspring
  distribution, so branching shape survives. A run under `TransitionRate`, `RuleWeights` or
  `MatchesPerStateRule` turns forwarding off, so each state's own matching finds all of its
  matches.
- **Keyed draws.** Every draw is a function of the transition's isomorphism-invariant identity
  and the seed — never a worker's RNG — so the sampled subgraph is the same at every worker
  count and on either device, and reproducible for a fixed seed.
- **The spine.** Below the branching factor a fixed rate makes the sampled evolution go
  extinct, so a state whose draws all failed keeps its transitions with the smallest seeded rank
  (`hgcommon::transition_rank`). Under Full states the key reads the matched edges' orbits, so
  automorphic transitions share the rank and are kept together. The host
  decides it at the state's drain (`spine_at_drain`), the device in the block that matches all of
  the state's rules (`match_state_rule`).
- **The per-state match-task join.** Matching one state is a tree of tasks, so no single task
  sees all its matches; anything acting on them AS A SET needs to know when the tree drained.
  Two monotone per-state counters — `pushed` incremented before a spawned task is visible,
  `completed` after its effects are — and the task observing `pushed == completed` drains the
  state. The CPU twin of the device's `TerminationDetector`, deliberately the same shape; a join
  over one state's own tasks, not a barrier. The per-depth quiescence signal composes from it.
- **Ranked caps.** `MaxSuccessorStatesPerParent` keeps the k lowest-ranked of a state's own
  transitions at its drain. `MaxStatesPerStep` (and `UniformRandom`+`MatchesPerStep`) adds each
  drain's kept transitions to its step's candidates, and the depth join's report of the step
  submits the N lowest-ranked; step s + 1 holds a token until that selection is in, so a capped
  run's steps complete in order. On the device the block that finishes a step's last piece of work
  makes the same selection. A transition not kept is not taken.

Gates: `SamplingReproducibility.*` (same states at every worker count; reproducible per seed;
the drain fires once per state), `RuleWeights.*`.

## Validation

Ground truth is `reference/MultiwayReference.wl`, cross-checked against the Wolfram
`Multicomputation` `MultiwaySystem` paclet. C++: `ctest`, which is the whole gate --
`all_tests` aggregates the ENGINE suites only, and the job system and the two deques build their
own binaries, so running it alone leaves those unexecuted and green. GPU adds `hg_gpu_tests`,
`gpu_differential_tests` and `gpu_ffi_tests`. Paclet: `reference/verify_paclet.wls` (golden corpus via
wolframscript). The `ReferenceOracle` test compares canonical-state counts to a
brute-force isomorphism oracle — the decisive correctness check.
