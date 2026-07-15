# Task Ledger

Remaining work, ranked by impact and dependency order. Measured numbers are from
the Wolfram rule `{{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}}` on `{{0,1},{0,2}}`
(RTX 4090, WSL2) unless noted; this box is noisy (CV 10-40%), so small margins
need paired-mean measurement, not single samples.

## GPU performance

1. **Attribute the core rewrite cost.** At depth 8 (604k events, TR off) the
   rewrite phase is ~4.6 s of ~5.5 s total; with causal and branchial
   registration compiled out it is still ~2.6 s ≈ 4.3 µs/rewrite for ~30 global
   writes — unexplained. Bisection is exhausted: the 4.5 KB/thread stack frame
   is `update_tr`'s `kTrScratch` arrays (untouched with TR off) and the CSR
   merge already writes directly to global. Needs Nsight Compute on the Windows
   host against the same kernels. No deliberately long kernels (TDR / driver
   crash risk).
2. **Causal + branchial registration contention** (~2.0 s combined at depth 8,
   entangled through shared dedup maps and pool counters). Candidate fix for
   the hub-edge `edge_consumers` head contention: striped sub-heads per key, or
   warp-aggregated pushes (`__match_any_sync` leader links a pre-chained
   segment, one CAS per warp-group). Measure paired.
3. **Deep pruned runs** (analysis deepened; measured floor):
   - Barrier floor measured (job tmp barrier.cu, loaded box so upper-bound):
     4 launch+sync + 1 D2H = **154 us/step** (1 launch+sync+D2H = 61 us). That is
     **1.5 s per 10k steps, 15 s per 100k** of pure barrier, independent of how
     little real work a step does. So level-sync bites exactly when per-step work
     falls below ~150 us — i.e. reservoir-capped narrow-frontier runs over 10^4+
     steps, where each step matches/rewrites a handful of states in microseconds.
   - Breakdown: ~137 us is the 4 launch+sync, ~17 us the D2H. The D2H exists to
     read frontier_count so the host decides whether to continue and how to size
     the next launch — so the loop CONTROL, not just the launches, forces the
     round-trip.
   - Options: (a) CUDA graphs capture the 4-kernel step and replay at ~1-2 us,
     killing launch cost with no TDR exposure, but dynamic per-step grid sizes
     need graph updates or device-side launch, and the frontier-count D2H remains
     unless control moves device-side; (b) persistent kernel / device-resident
     step loop eliminates both launches and D2H (loop condition checked on device)
     but is exactly what WDDM TDR kills — needs chunked/cooperative design.
     Device-side frontier management is the common prerequisite. Pairs with item 6
     (the reservoir sampler's per-step host logic is part of what forces the D2H).
   - Decision + implementation is the remaining large piece.

## Robustness / API

4. **VRAM cap + partial results.** (DONE — GPU-side robustness, cap, and notebook surfacing.)
   - [x] GPU-side robustness (commit 2241b82): `evolve()` grow-and-retry now
     catches a device-out-of-memory throw from an engine at the grown config,
     tags it `kDeviceOutOfMemory` in the warning trail, and returns the last
     completed attempt's partial result. Capacity overflow within a run already
     dropped-and-warned (S5.6). Never crashes; always returns a flagged partial.
     (Mechanism verified by inspection; normal path confirmed clean; a live-OOM
     runtime test is skipped to avoid a 17 GB transient allocation.)
   - [x] User-settable device-memory cap (commit db9694d):
     `EvolveInput::max_device_memory_bytes`, default 90% of total VRAM.
     `estimated_device_bytes(cfg)` + `fit_config_to_cap` shrink an over-provisioned
     initial config and stop grow-and-retry before the cap; a capped run returns a
     flagged partial (verified: 128 MB cap on depth-7 → 15568-state partial, no
     crash). Never allocates past the caller's ceiling.
   - [x] Surface the warnings to the notebook (commit): the GPU marshaler
     (`hg_gpu_backend.cpp`) emits a `Warnings` list (`Kind`/`Count`/`Context`) on
     the WXF result, and `HGEvolve` raises `HGEvolve::overflow` (kinds + total
     count) when it is non-empty — the run still returns the flagged partial
     result, never fails. Verified: the message stays quiet on non-overflow runs
     (golden 12/12).
5. [x] **Watchdog safety** (commit dc1b30b): `EngineConfig::max_blocks_per_launch`
   (EvolveInput override, 0 = single launch) splits the match/rewrite grids into
   consecutive synced chunks, bounding any single kernel's duration below the TDR
   budget. Verified identical to a single launch at caps 3/4/64; a chunked_launch
   differential workload keeps it CPU-cross-checked (20/20). The right cap value
   depends on the target's TDR budget, left to the deployment (not measured here —
   TDR probing forbidden).

## Correctness audits

6. **Pruning / reservoir sampling.** (audited)
   - [x] `evolve_uniform_random` reservoir sampling is uniform within a
     (state,rule) stratum — chi-square test added
     (`test_sampling_reproducibility.cpp::ReservoirUniformWithinStratum`). The
     equal per-stratum cap makes the scheme stratified by design (each state
     fairly represented regardless of branching factor), not flat-uniform.
   - [x] **Fixed** (commit 04b0407): `set_exploration_probability` flipped the coin
     PER TRANSITION under quotient mode, biasing high-in-degree canonical states
     (6-cycle child explored 84%/99% at p=0.25/0.5, matching 1-(1-p)^6). Now flips
     once per canonical state at its claim — P(explored) tracks p (27%/49%),
     matching the GPU's per-deduped-state flip. Isolation gtest added
     (`ExplorationProbabilityIsPerCanonicalState`).
7. **Remaining GPU parity items.**
   - [x] `exploration_probability` coin-placement (item 6) — both engines now
     sample once per canonical state (commit 04b0407).
   - [x] Multi-initial-state (commits c1845d6, 9d60702): `initial_states` on both
     engines; `quotient_initial_states` option (default false = every provided
     root is a distinct entry point, reference MultiwaySystem semantics, CPU==GPU;
     true = isomorphic roots collapse). Validated against the reference oracle
     across the full single/multi initial x single/multi rule 2x2 at depth 3
     (exact states+events, commit be61341); 24/24 differential.
   - [x] WL surfacing (commit 33568a7): `QuotientInitialStates` option threaded
     through the paclet FFI to the CPU engine; FFI compiles, WL parses. (End-to-end
     round-trip needs a Windows DLL rebuild.)
   - [ ] Event canonicalization (GPU reports `canonical_id = INVALID`; the
     differential compares by structural event key, so this is a completeness gap,
     not a correctness one), `MaxStatesPerStep` / `MaxSuccessorStatesPerParent`
     (device-side reservoir), genesis events — each a real feature needing exact
     CPU-semantic matching verified via the differential; deferred to focused
     sessions.

## Interface / longer-term

8. **Shared front-end**: common `RewriteRule`/`EvolveInput`/`EvolveResult` with
   a Backend{CPU,GPU} selector routed through the paclet FFI; match-forwarding
   port to the GPU.
   - [x] `TargetDevice -> "CPU"|"GPU"` option added to the paclet (commit 33568a7),
     mirroring NetTrain[]. CPU runs; GPU issues HGEvolve::gpudev and falls back to
     CPU. The option is a drop-in once the backend is wired.
   - [x] GPU routing done and validated on Linux: `hg_evolve_gpu`
     (`paclet_source/hg_gpu_backend.cpp`, guarded by `HG_GPU_BACKEND`) builds a
     `hg_gpu::EvolveInput` from the parsed job, runs `hg_gpu::evolve`, and marshals
     the `EvolveResult` into the SAME WXF the CPU path emits. The GPU is a raw
     per-provenance space; the marshaler recomputes the host IR canonical hash per
     state, groups into canonical classes (one State per class, as the CPU FFI
     does), keeps events raw (so multiplicity/counts match), dedups causal by
     (from,to), and derives Step/IsInitial from events. `TargetDevice -> "GPU"`
     selects the exe in the paclet WL. Validated: the 6-workload golden subset
     (single/multi-rule, self-loops, hyperedge) matches the CPU golden exactly
     through the Linux GPU binary (`reference/verify_paclet_gpu.wls` +
     `reference/golden_corpus.wl`).
   - [ ] Windows GPU binary: `hg_evolve_gpu.exe` needs the whole stack built with
     MSVC `cl.exe` + `nvcc` (the CUDA lib is MSVC-ABI, cannot link into the MinGW
     CPU binary) — a native MSVC build, not the MinGW cross. Until it exists,
     `TargetDevice -> "GPU"` on Windows falls back to CPU with `HGEvolve::gpudev`
     and `verify_paclet_gpu.wls` SKIPs there. The Linux binary proves the routing.
9. **Process-isolation binary** (CPU path DONE): standalone WXF-over-stdio
   executable that supersedes the LibraryLink DLL — abort is a process kill,
   crashes are isolated from the notebook, and the engine's cooperative-abort
   plumbing is removed.
   - [x] `hg_evolve` (`paclet_source/hg_evolve_main.cpp`): reads the WXF job on
     stdin, writes the WXF result on stdout, progress on stderr. Shares the FFI
     marshaling via `run_rewriting_core` (`hg_core.hpp`); built with
     `HG_STANDALONE_BINARY` so it needs no Wolfram SDK. See docs/BINARY_ISOLATION.md.
   - [x] `HGEvolve` routes through the binary when present via `RunProcess` +
     ISO8859-1 stdout decode (NeuralLearnability's pattern), else the in-process
     DLL. Validated end-to-end: `reference/verify_paclet.wls` golden 6/6 through
     the Windows binary.
   - [x] Cooperative abort removed from the engine: `evolve_with_abort` (both
     overloads) and the dead `abort_flag_`/`set_abort_flag`/`should_abort`
     machinery (`hypergraph.hpp`, `wl_hash.hpp`) are gone; `should_stop_` remains
     as the `max_states`/`max_events` limit flag (and the viz GUI stop button via
     `request_stop`). HAVE_WSTP progress path removed from the FFI.
   - [x] Latent bug fixed along the way: `add_rule` never called
     `compute_var_counts`, so hand-built FFI rules carried uninitialized
     `lhs_sig`/`lhs_cache`; the DLL got lucky zeros, the binary exposed it (0
     matches). `add_rule` now finalizes each rule at registration.
   - [x] GPU device selection (item 8): `hg_evolve_gpu` (nvcc-built) + `TargetDevice
     -> "GPU"` WL routing; Linux-validated. Windows `.exe` (MSVC build) pending.
10. **Paper refresh**: quotient exploration + offline multiplicity
    reconstruction (exact causal and branchial from the skeleton, validated on
    a 24-workload corpus), exact online transitive reduction, GPU speedups
    (depth 7 ~834 -> 252 ms; depth 8 373 s -> ~5.5 s). Final benchmarks need a
    quiet machine; current numbers are rough.
11. [x] **GPU IR host-fallback flag** (commit removes it): the unwired
    `needs_host` flag, `ir_host_fallback.cpp`, and the `hg_gpu -> hypergraph`
    PUBLIC link it forced are all removed. The non-discrete IR fallback gap is
    tracked under S3.5 (GPU-native IR backtrack), the correct fix, not a host
    round-trip.
12. [x] **`gpu/ARCHITECTURE.md` drift** (commit 2241b82): §3-5 now describe the
    actual host-driven level-synchronised step loop, with the persistent-kernel
    streaming design explicitly marked as the M7 target for deep pruned runs.
    §10.5 scoreboard is 19/19 with quotient + forced-index-regime workloads;
    stale 11/12 / open-`wolfram_canonical_steps5` claims removed; tweak log
    carries the 2026-07-10 entries.

## Portability / CI (tracked — deferred native builds)

13. **Native MSVC build (unblocks the Windows GPU `hg_evolve_gpu.exe`).** The
    Windows CUDA toolkit only supports MSVC `cl.exe` as nvcc's host compiler and
    CUDA libs are MSVC-ABI, so the GPU binary must be a native MSVC build (not the
    MinGW cross that makes the CPU binary). `cl.exe` rejects the engine's GCC
    intrinsics. Port surface — cross-builds keep the `__builtin_*` path via `#else`;
    add `#if defined(_MSC_VER)` fallbacks in one
    `common/include/hgcommon/portable_intrinsics.hpp`:
    - popcount / ctz: `bitset.hpp:71,81`, `pattern.hpp:81,209-211,266`,
      `types.hpp:77`, `rewriter.cpp:53` → `__popcnt`/`__popcnt64`,
      `_BitScanForward`/`_BitScanForward64`.
    - CPU-relax spin hint: `concurrent_map.hpp:268-270,357-359`,
      `segmented_array.hpp:82-84` (already x86 `__builtin_ia32_pause` / ARM
      `__asm__("yield")`) → MSVC `_mm_pause()` / `YieldProcessor()`.
    - `__builtin_bswap64`: `wxf.cpp:131,450` → `_byteswap_uint64`.
    - `#pragma GCC diagnostic`: already `__GNUC__`-guarded in `wxf.hpp`, no change.
    - [x] DONE (commit): the intrinsics above are abstracted; the GCC/Clang builds
      take the same `__builtin_*` path and are byte-identical (all gates + golden
      green), with the MSVC branch added.
    - [ ] REMAINING: a native MSVC + nvcc CMake config for the whole stack, driven
      via WSL2 interop. GPU marshaling + routing + WL are already Linux-validated
      (item 8); this is the last toolchain step.

14. **macOS CI (recurring gap: nothing exercises the macOS paclet).** The code
    cross-compiles to macOS via osxcross (clang accepts `__builtin_*`, so item 13
    is NOT needed for macOS), but there is no Mac + Wolfram to load/run the paclet
    and no CI at all (`.github/workflows` absent). Plan: a GitHub Actions workflow
    on the **`macos-14` (ARM64/Apple-Silicon)** hosted runner + free **Wolfram
    Engine** (native ARM64 since 13.1; headless activation via `WOLFRAM_ID`/
    `WOLFRAM_PASS` secrets or on-demand licensing) that builds the paclet natively
    (Xcode clang) and runs `reference/verify_paclet.wls`.
    - **ARM64**: direct on the runner.
    - **x64**: the hosted Intel runner (`macos-13`) was retired Dec 2025 (Apple
      dropped x86_64) — test the x86_64 paclet by running an x64 Wolfram Engine
      under **Rosetta 2** on the ARM64 runner (loads the `MacOSX-x86-64` binary),
      or on a self-hosted Intel Mac.
    The same workflow can add ubuntu (all_tests + GPU gates on a GPU runner) and
    windows-CPU legs; Windows-GPU CI needs a self-hosted GPU runner + item 13.

## Correctness (open)

15. **Quotient exploration completeness gap on non-productive / mixed rulesets.**
    `explore_from_canonical_states_only` can leave a discovered canonical state
    unexpanded, and therefore miss every state reachable only through it. Surfaced
    by the offline-reconstruction corpus: 2 of 24 workloads have an incomplete
    quotient skeleton (mAllThree over a triangle finds all 13 states but leaves one
    it discovered at depth 2 unexpanded, ~10 matches/instance missing; the two-edge
    init discovers only 9 of 17 states). The reconstruction itself is exact wherever
    the skeleton is complete, so this is a defect of the exploration, not the
    propagation (`tools/quotient_reconstruction_probe.cpp` detects and reports it).
    - **Root cause.** Expansion (computing a state's matches, which produces its
      children) is triggered at a state's FIRST creation, gated on the arrival
      depth being within the step budget. A later rewrite that reaches the same
      state is treated as a duplicate: it records the transition and relaxes the
      shortest depth, but does not re-run the expansion decision. Loops (which only
      non-productive / mixed rules create, since productive rules grow the edge
      count so each state occurs at one depth) let a state be created first by a
      long, out-of-budget path (expansion skipped), then reached by a short,
      in-budget path that deduplicates without re-triggering the skipped expansion.
      An unexpanded intermediate state hides its descendants, so one miss cascades.
    - **Fix (bounded, not a redesign).** Make expansion fire on the FIRST in-budget
      arrival regardless of create-vs-dedup: the duplicate-arrival / depth-relaxation
      path also checks whether this arrival brings the state's depth inside the
      budget while it is still unexpanded, and if so takes the one-time expansion
      claim there (carrying that rewrite's context so match forwarding still avoids a
      full rematch). Terminates (depths decrease monotonically, bounded at 0), stays
      expand-once per state, and adds only the correct expansions being skipped, so
      the quotient speedup is preserved. Cruder alternative: a fixpoint re-pass that
      expands any in-budget-but-unexpanded state until none remain.
    - **Confidence / next step.** Reasoned from the commit history and code shape
      (depth-gated expansion at creation + the separate relaxation walk), not a
      fresh trace. Confirm by reproducing the two failing workloads and watching
      where the in-budget-but-unexpanded state loses its expansion. The full multiway
      mode (`explore=false`) has no such gap (it never gates expansion on
      relaxation), which is evidence the gap is in the quotient bookkeeping, not the
      idea of quotienting. Relates to the loops discussion in item's reconstruction
      work (`tools/quotient_reconstruction_probe.cpp`, path-length indexing).

## Status of the remaining open items (why each is not [x])

- **1, 2 (core-rewrite attribution, causal/branchial contention):** BLOCKED on
  the Windows-host Nsight Compute profiler and paired-mean benchmarks on a quiet
  machine. Bisection is exhausted; the next step is hardware counters, which this
  (headless, noisy, no-Windows-Nsight) environment cannot provide. TDR probing is
  forbidden.
- **3 (deep pruned runs):** analysis DONE (154 us/step barrier floor measured, the
  crossover characterised, options laid out). The CUDA-graphs-vs-persistent-kernel
  decision and its device-side-frontier implementation are a large focused effort.
- **4 (memory):** GPU side DONE. Only FFI surfacing remains, BLOCKED on item 8.
- **7 (GPU parity features):** exploration_probability alignment DONE. The rest —
  event canonicalization (`canonical_id`), reservoir/`MaxStatesPerStep`,
  genesis events remain — each a real feature requiring exact CPU-semantic
  matching verified via the differential; deferred to focused sessions.
  **Multi-initial-state DONE** (commit adds it): `EvolveInput::initial_states`,
  22/22 differential incl. distinct-root and iso-root-full workloads. (Quotient
  iso-root seed-dedup differs CPU-vs-GPU; documented, not asserted.)
- **8, 9 (shared front-end, process-isolation binary):** large multi-session
  architecture.
- **10 (paper):** BLOCKED on a quiet machine for final low-variance benchmarks;
  the results content is ready (rough numbers recorded).

Everything that is a bounded, verifiable, unblocked unit of work has been
completed this session: items 4 (GPU), 5, 6, 11, 12, the exploration_probability
correctness fix, the reservoir uniformity test, the item-3 measurement, and the
all-remotes push discipline (post-commit hook).

## Current state (for orientation)

- `master` is the only branch (plus a local-only `gpu-rewrite` safety copy);
  deployed on both remotes. Every commit auto-pushes to `local` + `origin` via a
  `.git/hooks/post-commit` hook (re-install on the laptop; it is not tracked).
- Gates: `gpu_differential_tests` 24/24, `hg_gpu_tests` 77/77, `all_tests`
  172/172 (`PacletTest` needs a Wolfram paclet build absent from the Linux
  configuration).
- `explore_from_canonical_states_only` is the quotient mode: expand each
  canonical state once at its shortest depth; deterministic on both engines;
  exact causal/branchial multisets of the full expansion reconstruct offline
  (`tools/quotient_reconstruction_probe.cpp`). `false` is reference semantics
  (`reference/MultiwayReference.wl`), exact and deterministic.
