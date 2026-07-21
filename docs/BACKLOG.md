# Overhaul backlog — all remaining work

The full worklist toward the strong release + paper. Compiled from the charter
(`OBJECTIVE.md`), the memory-architecture design (`MEMORY_ARCHITECTURE_DESIGN.md`), the
7-way cost analysis, the recovered planning corpus, and items discovered while working.
Status kept current as items land (branch `overhaul/phase0-zero-waste`).

Legend: [x] done · [~] in progress · [ ] not started.

## 0. Done this session (context)
- [x] Measurement + verification harness: rule-type corpus + brute-force oracle gate,
  4/8/16-thread determinism + causal/branchial determinism gates, `cost_matrix` (arena +
  heap counters), WXF-serialization pin test.
- [x] Phase-0 zero-waste: dead `state_children_`; WL hash-only path (leak); pattern-context
  copy (F1); `Event.binding` (−132 B/event); `event_canonical_state_map_` gating; **F8
  correctness fix** + test.
- [x] SparseBitset copy-on-write.
- [x] **De-heap of the shipping surface** (heapAllocs 1567→69): jobs slab pool; every
  `ConcurrentMap` on the per-worker-cursor arena; inline deque payload; IR `std::function`;
  WXF streamed (byte-identical).
- [x] Causal closure base layer (−28.5% arena): key-only `ConcurrentIdSet`, `Anc` dropped
  (predecessor BFS), reconvergence-skip, lazy empties.

## 1. Memory / allocator — frontier reach
- [x] Pattern-matcher wasted-compute (F2/F4/F5/F6/F7): redundant bound-path signature check
  removed (`SignatureAccessor` dropped); mutate/undo DFS (no per-candidate `binding`/`PartialMatch`
  copies); fetched-edge threaded through the intersection; probe skipped when count==1;
  all-distinct SCAN seeds bypass the Bell enumeration. Exact, no slowdown.
- [ ] **Tier reclamation**: free per-state transient data (match lists `state_matches_`, etc.)
  once a state is fully expanded → working memory O(active frontier), not O(all states). The
  "release what's processed" lever. Needs a generation/completion quiescence signal + a
  reclaimable pool/region.
- [ ] **Causal reachability oracle**: O(N²)→O(N·w) closure via chain decomposition (or 2-hop) —
  the asymptotic fix beyond the −28.5% base layer. Must stay online + exact under concurrency.
- [ ] **Automorphism reconstruction**: store quotient + Aut generators, reconstruct raw
  states/events/causal/branchial from the orbit action instead of materialising them.
  Hard open sub-problem: exact causal/branchial reconstruction under the group action.
- [x] **MatchRecord forward-by-reference**: the immutable `MatchCore` (`rule_index`,
  `num_edges`, `matched_edges[16]`, `binding`) is arena-allocated once on winning the
  `seen_match_hashes_` claim; `MatchRecord` is now `{const MatchCore* core; StateId source_state}`
  (16 B) — forwarding copies two words and shares the core by pointer. Dead fields
  `canonical_source`/`source_canonical_hash`/`storage_epoch` (and the per-forward
  `get_canonical_state` recompute that fed them) dropped. arenaB down every row (−13.7% on
  binary-growth), oracle EXACT, heapAllocs not increased, forwarding validator thread-invariant.
  Regression harness: `tools/forwarding_validator_probe.cpp`.
- [ ] Per-edge storage compaction: `Edge.vertices` SBO for arity≤2 (kill out-of-line alloc +
  cache miss/edge); pack `vertex_adjacency_` as CSR (vs 16 B/occurrence linked list);
  drop/shrink `edge_signatures_` (17 B/edge cache).
- [ ] keys-only sets for `seen_match_hashes_`/`seen_branchial_pairs_`/etc. (apply the causal
  `ConcurrentIdSet` pattern to the remaining dedup sets).
- [ ] `ConcurrentMap` reclaim superseded resize tables at quiescence (~2× map memory retained).
- [ ] SparseBitset empty-chunk compaction; entries-array growth.
- [x] Arena block right-sizing: 64 KB initial block + geometric ×2 growth to the 1 MB cap (per
  `LocalCursor`, no shared fast-path state) killed the uniform ~4.8 MB `heapB` floor — small
  cases −54..62% (self-loop 4.82M→1.81M, below the pre-de-heap 2.79M base), binary-growth
  −21.6%; arenaB byte-identical, +3–5 mallocs (immaterial), oracle EXACT, 184/184.
- [ ] Streaming Tier-D output (free-on-emit) for frontier-resident mode.
- [ ] Configurable rule caps (`MAX_VARS` 32→8, compile-time; path to variable-length records).

## 2. Compute / incrementalisation — no wasted work
- [ ] **Incremental hashing**: consume the delta already handed to
  `create_or_get_canonical_state` (`incr_consumed/produced`, currently ignored); the
  `wl_canonical_hash` fold is patchable (WL colour propagation is non-local — the hard part).
- [ ] Canonicalization event-path redundancy: `find_edge_correspondence` recomputes both states'
  WL 4× per event; both correspondences built when one is keyed (A1); canonical-state cache
  recomputed per incident event (A3) → memoize on the State; drop the `hash1!=hash2` guard.
- [ ] WL stop-check full-sorts all vertices per iteration (O(n²)); `vertex_index` sized by
  ID-span not count; redundant edge-count pass. IR: full-partition copy + fresh scratch per
  search node (the 1100× high-symmetry pathology) → trail mutate/undo + reused scratch.
- [ ] Match-forwarding edge→matches reverse index (REVIEW §12.5): O(|parent matches|×size) →
  O(∑|matches(eᵢ)|).
- [ ] Event edge-map O(E) to remap O(num_consumed) (A4); `MatchRecord::hash/==` iterate
  `MAX_VARS=32` not bound vars.

## 3. Concurrency / hardening
- [ ] Systematic memory-ordering re-verification (the deep audit's cluster — mostly fixed;
  confirm each; audit any structure not yet checked).
- [ ] REVIEW §1 residual bugs: FFI vertex `uint8_t` >255 wrap (1.3); int64→int option narrowing
  (1.4, `checked_narrow`). §2: ConcurrentMap LOCKED spin / `locked_slots[64]` 65th-slot drop.
- [ ] Spin-wait elimination pass — zero spinwait anywhere (charter). TSAN/ASAN CI gates.

## 4. GPU — parity, lockstep + persistent, shared code
- [ ] Apply the whole de-heap + cost analysis to the GPU path.
- [ ] Persistent-kernel / non-lockstep scheduler (de-risk on laptop, then 4090); GPU ringbuffer
  + separate stream interleave.
- [ ] GPU WL-hash O(V·E) bug (`gpu/wl_hash.cu`, broken per REVIEW §7); GPU causal/branchial
  online parity; GPU differential test vs CPU.
- [ ] CPU/GPU shared factoring — widen `hgcommon`; CPU / GPU-lockstep / GPU-persistent as three
  schedulers over one kernel set.
- [ ] `max_blocks_per_launch` WDDM hack → headless A100/H100 target + tuning.

## 5. Visualisation
- [~] Decided: KEEP the header-only monolith (`scene/include/scene/hypergraph_renderer.hpp`),
  dropped the ~25 dead modular `.cpp` refs (+ the `VIZ_BUILD_FULL`/web/tests blocks) from
  `visualisation/CMakeLists.txt`. `BUILD_VISUALIZATION=ON` now configures + builds clean; the
  windowing GAL/apps are gated behind a new non-fatal `VIZ_HAS_WINDOWING` (needs XCB dev pkgs +
  Vulkan). Built here: `viz_camera/layout/events/blackhole` + `analyze_bhdata`. REMAINING: wire
  `BUILD_VISUALIZATION=ON` into CI so it can never silently rot; install XCB on a CI runner to
  cover the interactive apps.
- [ ] NOTE (settled): the 25 CMake-referenced viz `.cpp` were NEVER committed anywhere —
  exhaustive object-DB scan across `_final`, the bare remote, the sibling clones, AND the laptop
  repo/history/disk confirmed zero trace. An interrupted/never-completed modular split, NOT
  recoverable; the renderer logic already exists in `hypergraph_renderer.hpp`. Not a recovery task.
- [ ] GPU Barnes-Hut layout (PLAN.md Phase 3); geodesic/defect overlays ("NOT YET RENDERING" —
  the "for Stephen" items); dimension-colored WL multiway graphs (2 files).
- [ ] Viz UX/animation (convergence-merge, highlight system, event replay, edge bundling);
  WebGPU/WASM port; VR. Open viz bugs (oscillation friction, 2D buffer glitches, scrubber
  endpoints, state boxes→circles).

## 6. Physics analyses (Stephen Wolfram asks)
- [ ] Flat-space/Minkowski equilibrium experiment with closed topology (must-have, never run).
- [ ] Geodesic bundle-divergence metric; classical-limit (bundle_width=1) run; lensing
  quantification. Multispace stacked-node viz (verify shipped). Vertex-lineage tracking.
- [ ] `rotation_analysis.{hpp,cpp}` — recovered, uncommitted, pending review/integration.

## 7. Paper
- [~] Revision plan written: `docs/PAPER_REVISION_PLAN.md` (section-by-section keep/revise/
  rewrite/cut; every UT claim flagged with line numbers vs `reference/CANONICALIZATION.md`;
  perf numbers marked projections; nine benchmark tables T1–T9 specced against real results;
  missing-contribution sections; 10-item priority checklist).
- [ ] Reinstate `paper/` (recovered "Rewriting the Universe"), revise, correct, complete.
- [ ] REAL benchmark numbers (abstract's 50–200× CPU / 5–20× GPU are projections) — CPU here +
  laptop, GPU on laptop then 4090; via `cost_matrix` + a timing harness.
- [ ] Rewrite canonicalization sections (predate UT-removal / IR-is-reference).
- [ ] Ablation numbers via CMake `#define` ablation builds (keep old paths compiled-out).

## 8. Code health / cleanliness / release
- [ ] Finish the WXF streaming (the tail still builds a `WXFValue` tree).
- [ ] Split `hypergraph_ffi.cpp` (~4700 lines) → `ffi/{options,wxf_bridge,evolve,blackhole,
  debug_queue}` (REVIEW R7); split `paclet/Kernel/HypergraphRewriting.wl` (~3800) into contexts.
- [ ] Consolidate the two matching engines (recursive `pattern_matcher` vs task-based join) —
  the duplication flagged as the worst offender; the `hg_evolve_main` frame-length codec dup.
- [ ] gitignore/stale-docs cleanup (`.gitignore` is clean; decide on tracked internal docs
  `IR_VERIFICATION_NOTES`/`TASKS`/`PAPER_RESULTS`); `visualisation/` dead-CMake (`VIZ_BUILD_FULL`).
- [ ] Cross-compilation (OSXCross recipe → macOS CI gap). Ablation `#define` scaffolding.
- [ ] Full CODEMAP/ARCHITECTURE/design-doc sync; comprehensive test coverage incl. the
  feature-combination matrix + TSAN/ASAN + repeated stress.

## 9. Feature-surface cohesion ("one cohesive whole")
- [ ] Reservoir sampling (unbiased) composes with quotient + canonicalization — define
  "unbiased of *what*", preserved by reconstruction. ExplorationProbability, multiplicity,
  multiple-initial-states, genesis, abort, overflow→partial — verify all compose.
- [ ] Resolve the charter conflict matrix (determinism ⊗ all features; sampling ⊗ quotient ⊗
  reconstruction; no-phases ⊗ GPU; ablation flags ⊗ test matrix).
- [ ] Automate the authoritative Wolfram `MultiwaySystem` cross-check + golden-corpus verify.
