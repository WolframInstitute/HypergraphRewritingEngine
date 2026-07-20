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
Alt-. abort is a process kill, so it never takes down the notebook. See
[BINARY_ISOLATION.md](BINARY_ISOLATION.md).

## Modules

Core engine:
- **`hypergraph/`** — the CPU engine (the heart). `ParallelEvolutionEngine`
  (`parallel_evolution.*`) drives the lock-free dataflow: match → rewrite → dedup as
  states are produced. `Hypergraph` (`hypergraph.*`) stores edges/states/events and
  owns canonical-state dedup. Matching is a worst-case-optimal join
  (`pattern_matcher.hpp`, `index.hpp`, `signature.hpp`). Canonicalization is
  Weisfeiler-Leman (`wl_hash.hpp`, fast, may collide) on the hot path with an exact
  McKay individualization-refinement fallback (`ir_canonicalization.*`). Storage is
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
  duplicates still need IR to confirm; see TASKS.md.)
- **Quotient exploration** (`ExploreFromCanonicalStatesOnly`) expands each canonical
  state once at its shortest depth, so a run costs the canonical closure, not the
  exponentially larger provenance count.
- **Everything on the hot path is lock-free and arena-allocated.** No mutexes, no
  `std::` heap containers on the hot path. Fixes stay lock-free.
- **The GPU mirrors the CPU algorithms.** Never drop a CPU data structure in a kernel
  without justification (an inverted-index skip once cost 200x).

## Validation

Ground truth is `reference/MultiwayReference.wl`, cross-checked against the Wolfram
`Multicomputation` `MultiwaySystem` paclet. C++: `all_tests` (+ `hg_gpu_tests`,
`gpu_differential_tests`). Paclet: `reference/verify_paclet.wls` (golden corpus via
wolframscript). The `ReferenceOracle` test compares canonical-state counts to a
brute-force isomorphism oracle — the decisive correctness check.
