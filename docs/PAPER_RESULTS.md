# Paper results capture

Material for the paper refresh, produced by the 2026-07 work. Numbers on an
RTX 4090 under WSL2 on a **noisy** box (CV 10-40%) — treat as rough / order-of-
magnitude until re-run on a quiet machine. Validation status is exact.

## 1. Quotient exploration (the headline algorithmic contribution)

`explore_from_canonical_states_only` is respecified as **quotient exploration**:
expand each canonical state exactly once, at its shortest reachable depth, rather
than once per raw provenance. Wall-cost is proportional to the *canonical* state
count, not the (exponentially larger) provenance count.

- **Determinism.** Depth is a shortest-path label — a property of the graph, not
  of arrival order — maintained by lock-free depth relaxation over recorded
  canonical transitions (no re-matching; matches are depth-independent). The
  expanded set and the (input, output, rule) transition multiset are identical
  across thread counts and RNG seeds. (Prior first-discovery semantics were
  ill-defined under an asynchronous frontier and silently non-deterministic.)
- **Speedup, same canonical closure** (Wolfram rule, quotient vs full expansion):
  depth 6 ~4.1x, depth 7 ~16x, widening with depth (provenance branches while the
  canonical closure saturates: 45k vs 290k raw states at depth 7).
- **No perf regression vs the prior (broken) mode:** paired -2.7%, 95% CI
  [-302, +92] ms (spans 0). The relaxation fast-path allocates nothing unless a
  depth actually improves, and the claim is taken at the discovering rewrite so
  match-forwarding still elides a full re-match.

## 2. Exact offline reconstruction of causal and branchial multisets

The quotient skeleton discards multiplicity; it is recovered **exactly** offline.

- **D(s, k, orbit) -> producer-event -> count.** Edges are keyed by canonical
  edge **orbit under the automorphism group**, the only identification valid
  across the differing labelings by which distinct parents reach one canonical
  state. (Content classes suffice only when |Aut| = 1; with nontrivial Aut an
  automorphism permutes edges between classes — verified against brute-force Aut
  on 3000 random hypergraphs, incl. 83 where Aut fuses distinct edge contents.)
  The IR search already discovers the generators for pruning; we surface them.
- **Branchial** is intra-instance (siblings of one parent), so it reconstructs as
  sum_k mult(s,k) * B(s) — no provenance needed.
- **Validation:** exact causal AND branchial multisets on a 24-workload corpus
  (single/multi rule; productive/idempotent/reductive/mixed; 7 initial-state
  shapes; incl. proper-subset-of-orbit consumption and repeated LHS edges), for
  every workload whose skeleton is complete. Indexing is by path length, not the
  state's recorded step, because the multiway states graph has back-edges (a
  canonical transition recurs at several depths).
- **Cost caveat (honest):** D is scratch, ~1.9x the causal triple count, so it is
  NOT a memory win as stored. The wins are exploration (fewer states) and the
  **wire format**: emit per-canonical-event and per-canonical-pair weighted
  counts (385 events / 716 pairs) instead of raw lists (1174 / 2344) — the
  data-efficient FFI path.

## 3. Exact, deterministic online transitive reduction

The online causal TR emits the unique minimal reduction with no redundant edge at
any thread count. Verified two ways: the rendezvous branch that could reorder
edge creation never fires (produced-edge producers are set while private to their
rewrite), and the engine's reduced graph equals the offline minimal TR on three
workloads at 1/2/4/8/16 threads (wolfram5 1868->1332, chain6 872->872,
tri4 201->135). A prior "some redundant edges may slip through in races" comment
was traced and shown false.

## 4. GPU acceleration

Host-driven level-synchronised BFS step loop (four bounded kernel phases/step).
Cumulative on the Wolfram workload, same hardware:

- **Match kernel: ~190 ms -> ~4 ms at depth 7 (~47x).** Small states (<= 256
  edges) enumerate candidates from their own CSR slice instead of the global
  signature / inverted indices, whose buckets grow with the whole evolution.
- **Lazy index maintenance:** indices built only once some state exceeds the
  slice-scan threshold; below it the per-edge inserts were pure hub-bucket CAS
  contention. Depth 8: 11.75 s -> 5.25 s.
- **Branchial via a per-(state,edge) index** (mirrors the CPU's inverted index),
  replacing an O(siblings^2) pairwise consumed-array scan.
- End-to-end: depth 7 ~834 -> ~252 ms; depth 8 373 s -> ~5.5 s.
- Two GPU correctness bugs found + fixed on the way (duplicate matches on
  seen-buffer overflow; state drops against a 32-slot scratch dedup map).
- Robustness: user device-memory cap with graceful partial results; optional
  per-launch block cap to bound kernel duration against the WDDM watchdog.

## 5. Uniform, unbiased subsampling

- Reservoir sampling (`evolve_uniform_random`) is uniform within a (state,rule)
  stratum (chi-square, 3000 seeds); the equal per-stratum cap makes the scheme
  stratified by design.
- `exploration_probability` samples once per canonical state on both engines
  (was per-transition on the CPU under quotient, biasing high-in-degree states by
  1-(1-p)^N; fixed).

## 6. Validation against the reference oracle

`reference/MultiwayReference.wl` (cross-checked against the Wolfram `MultiwaySystem`
paclet) is the ground truth. The CPU engine reproduces it exactly on the full
single/multi-initial x single/multi-rule 2x2 at depth 3, and the GPU matches the
CPU on a 24-workload differential corpus (canonical states, event multiset, and
causal+branchial multisets in reference mode).

## Open for the paper (need a quiet machine)

Final low-variance benchmark tables (paired means + CIs); the current numbers are
single-sample on a loaded box. The GPU profiler-guided rewrite/causal-contention
work (TASKS.md items 1-2) may improve the depth-8 numbers before publication.
