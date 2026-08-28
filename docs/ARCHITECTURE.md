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
  (`pattern_matcher.hpp`, `index.hpp`, `signature.hpp`). Canonicalization under the exact
  mode is McKay individualization-refinement (`ir_canonicalization.*`) on every
  state, its hash serving as the dedup key directly; `compute_canonical_hash`
  reaches the Weisfeiler-Leman hash (`wl_hash.hpp`, fast, may collide) only when
  the mode is not `Full`, so WL never stands in front of IR. Bucketing by WL and
  confirming with IR was implemented and measured 28% slower, and the reason is a
  bound rather than an accident: distinct WL hashes are at most the canonical
  classes, so a WL filter can skip at most `canonical/raw` of the IR calls while
  paying a WL pass on every state -- measured per case by `tools/cost_matrix`
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
  quotient reconstruction, determinism, profiling harnesses). Built ad hoc.
- **`testing/`** — the aggregate C++ test target (`all_tests`).
- **`benchmarks/` / `benchmarking/`** — per-area benchmarks and the framework lib.
- **`visualisation/`** — the interactive 3D viewer (Vulkan) and physics analysis.

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
  Verified against full capture across the reconstruction matrix; still default-OFF
  (`set_quotient_reconstruction`) while the older aggregate producer-set DP ships.
- **Everything on the hot path is lock-free and arena-allocated.** No mutexes, no
  `std::` heap containers on the hot path. Fixes stay lock-free.
- **Concurrent-structure invariants that are easy to break.** `ConcurrentMap` is open
  addressed with no tombstone, so a claimed slot must never return to EMPTY — that
  cuts the probe run of every key passing through it, and the next insert of a hidden
  key reports itself as newly inserted, which is what every dedup decision reads. A
  key equal to a sentinel is unstorable and is rejected loudly. `SegmentedArray`'s
  `count_` is a high-water mark, so an index is readable only after its own `emplace`
  returned, and its capacity is a CONFIGURED CEILING rather than an assumption: the
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
- **The GPU mirrors the CPU algorithms.** Never drop a CPU data structure in a kernel
  without justification (an inverted-index skip once cost 200x).

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
- **D17 — an artifact the session was not asked to record is a WARNING on a held verb,** not an
  error: the caller can re-`Open` to widen, and refusing the whole query would discard what it
  did record.

`paclet_source/session.hpp` carries D7/D11/D14 as assertions; `test_session.cpp` and
`verify_sessions.wls` gate the verbs end to end.

## Sampling

The samplers thin the multiway graph while keeping the observables of the unpruned evolution
well-defined; `docs/SPEC.md` §5 states which options are samplers and which are caps.

- **A rate, not a count.** With match forwarding on, a state's match population is not local: a
  parent keeps forwarding matches to a child long after the child's own discovery tree drained,
  so the population closes only when the whole ancestor chain has — most of the run. A per-state
  COUNT therefore cannot be sampled uniformly without a barrier (measured: a count knob bounded
  only new discoveries — 2,038,505 states where the closed population is 1,365). A RATE decides
  per match, independently, with no population and no completeness requirement, so it applies
  identically to a discovered and a forwarded match and needs no join. `TransitionRate` is that
  sampler; thinning a branching process by a rate yields a branching process with the thinned
  offspring distribution, so branching shape survives.
- **Keyed draws.** Every draw is a function of the transition's isomorphism-invariant identity
  and the seed — never a worker's RNG — so the sampled subgraph is the same at every worker
  count and on either device, and reproducible for a fixed seed.
- **The spine.** A fixed rate is a knife-edge (below the branching factor the sampled evolution
  goes extinct), so the minimum-keyed own-found transition of a state survives when none of its
  own draws passed — every term key-deterministic.
- **The per-state match-task join.** Matching one state is a tree of tasks, so no single task
  sees all its matches; anything acting on them AS A SET needs to know when the tree drained.
  Two monotone per-state counters — `pushed` incremented before a spawned task is visible,
  `completed` after its effects are — and the task observing `pushed == completed` drains the
  state. The CPU twin of the device's `TerminationDetector`, deliberately the same shape; a join
  over one state's own tasks, not a barrier. The per-depth quiescence signal composes from it.
- **Arrival-order caps** (`MaxStatesPerStep`, `MaxSuccessorStatesPerParent`,
  `UniformRandom`+`MatchesPerStep`) bound work, not identity; above the cap, which states got in
  is the schedule's.

Gates: `SamplingReproducibility.*` (same states at every worker count; reproducible per seed;
the drain fires once per state), `RuleWeights.*`.

## Validation

Ground truth is `reference/MultiwayReference.wl`, cross-checked against the Wolfram
`Multicomputation` `MultiwaySystem` paclet. C++: `ctest`, which is the whole gate --
`all_tests` aggregates the ENGINE suites only, and the job system and the two deques build their
own binaries, so running it alone leaves those unexecuted and green. GPU adds `hg_gpu_tests` and
`gpu_differential_tests`. Paclet: `reference/verify_paclet.wls` (golden corpus via
wolframscript). The `ReferenceOracle` test compares canonical-state counts to a
brute-force isomorphism oracle — the decisive correctness check.
