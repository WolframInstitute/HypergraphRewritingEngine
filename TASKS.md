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
3. **Deep pruned runs**: per-step overhead (~4 launches + syncs + D2H ≈
   50-100 µs) dominates when frontiers are narrow over many steps (reservoir-
   capped runs, 10^4+ steps). Decide CUDA graphs vs persistent kernels: graphs
   amortize launch cost with no TDR exposure; whether that approaches
   persistent-kernel throughput is the open question. The reservoir sampler's
   per-step host logic forces the round-trip, so this pairs with item 6.

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
5. **Watchdog safety.** Per-step kernels are bounded today; depth-9+ launches
   grow. Chunk launches so no single kernel approaches the WDDM TDR budget
   (~2 s when the GPU drives a display).

## Correctness audits

6. **Pruning / reservoir sampling.** (audited)
   - [x] `evolve_uniform_random` reservoir sampling is uniform within a
     (state,rule) stratum — chi-square test added
     (`test_sampling_reproducibility.cpp::ReservoirUniformWithinStratum`). The
     equal per-stratum cap makes the scheme stratified by design (each state
     fairly represented regardless of branching factor), not flat-uniform.
   - [ ] **Finding (moved to item 7):** `set_exploration_probability` places the
     coin flip PER TRANSITION on the CPU (before the quotient claim,
     `parallel_evolution.cpp:1426`), so a canonical state reached by N transitions
     survives with prob 1-(1-p)^N; the GPU flips it once per deduped state
     (`k_dedup_and_append`), i.e. prob p. Both are valid subsampling but they are
     different semantics — a CPU/GPU parity gap. Not fixed blind: pruning compounds
     down depth, which masks a clean before/after measurement, so the fix needs a
     single-step high-in-degree isolation test first. Node-sampling once per
     canonical state (matching the GPU) is the principled target.
7. **Remaining GPU parity items**: event canonicalization (GPU reports
   `canonical_id = INVALID`), `MaxStatesPerStep` / `MaxSuccessorStatesPerParent`,
   multi-initial-state, genesis events, and the `exploration_probability`
   coin-placement discrepancy from item 6 (CPU per-transition vs GPU per-state) —
   align both to per-canonical-state node sampling, with a single-step
   high-in-degree isolation test as the gate.

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

## Current state (for orientation)

- `master` is the only branch (plus a local-only `gpu-rewrite` safety copy);
  deployed on both remotes.
- Gates: `gpu_differential_tests` 19/19, `hg_gpu_tests` 77/77, `all_tests`
  169/169 (`PacletTest` needs a Wolfram paclet build absent from the Linux
  configuration).
- `explore_from_canonical_states_only` is the quotient mode: expand each
  canonical state once at its shortest depth; deterministic on both engines;
  exact causal/branchial multisets of the full expansion reconstruct offline
  (`tools/quotient_reconstruction_probe.cpp`). `false` is reference semantics
  (`reference/MultiwayReference.wl`), exact and deterministic.
