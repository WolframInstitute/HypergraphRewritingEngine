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

4. **VRAM cap + partial results.** `grow_config_for` doubles capacities
   unboundedly on overflow retry; it must cap at free VRAM and return the
   partial result with `OverflowWarning`s. Verify the warnings surface through
   the paclet FFI to the notebook end-to-end.
5. **Watchdog safety.** Per-step kernels are bounded today; depth-9+ launches
   grow. Chunk launches so no single kernel approaches the WDDM TDR budget
   (~2 s when the GPU drives a display).

## Correctness audits

6. **Pruning / reservoir sampling.** Verify uniform-random reservoir sampling
   is unbiased and reproducible under both `explore_from_canonical_states_only`
   settings, on both engines. The GPU `exploration_probability` path has never
   been differentially tested against the CPU's.
7. **Remaining GPU parity items**: event canonicalization (GPU reports
   `canonical_id = INVALID`), `MaxStatesPerStep` / `MaxSuccessorStatesPerParent`,
   multi-initial-state, genesis events.

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
11. **GPU IR host-fallback flag**: `compute_state_ir_hashes_range` can report
    per-state fallback (overflow / non-discrete) but no caller consumes it —
    wire it to a host-side exact-IR pass or remove it.
12. **`gpu/ARCHITECTURE.md` drift**: §3-4 describe the persistent-kernel
    no-phase streaming design in the present tense; the implementation is a
    level-synchronised step loop (which item 3 may or may not change). §10's
    differential-test sketch predates the current 19-workload corpus with
    quotient and forced-index-regime coverage. Rewrite to describe the code.

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
