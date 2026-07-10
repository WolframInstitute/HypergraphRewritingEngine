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

4. **VRAM cap + partial results.** (GPU-side done; only FFI surfacing remains, blocked on item 8)
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
   - [ ] Surface the warnings through the paclet FFI to the notebook —
     **blocked on item 8**: the FFI (`paclet_source/hypergraph_ffi.cpp`) has no
     GPU path yet, so there is nothing to surface through until the GPU backend
     is wired in.
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
   - [x] Multi-initial-state (commit c1845d6): `EvolveInput::initial_states`;
     22/22 differential incl. distinct-root and iso-root-full workloads. (Quotient
     iso-root seed-dedup differs CPU-vs-GPU; documented, not asserted.)
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
9. **Process-isolation binary**: standalone WXF-output executable replacing the
   paclet DLL (clean aborts by process kill, crash isolation from the notebook
   front end, removes abort plumbing).
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
- Gates: `gpu_differential_tests` 20/20, `hg_gpu_tests` 77/77, `all_tests`
  171/171 (`PacletTest` needs a Wolfram paclet build absent from the Linux
  configuration).
- `explore_from_canonical_states_only` is the quotient mode: expand each
  canonical state once at its shortest depth; deterministic on both engines;
  exact causal/branchial multisets of the full expansion reconstruct offline
  (`tools/quotient_reconstruction_probe.cpp`). `false` is reference semantics
  (`reference/MultiwayReference.wl`), exact and deterministic.
