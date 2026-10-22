# GPUMC coverage of the device's concurrent accesses

Scope: the device code in `gpu/src/`, `gpu/include/hg_gpu/`, and the `hgcommon` headers whose
bodies run on the device with a warp's lanes (`branchial_rows_core.hpp`, `ir_core.hpp`,
`state_geometry_core.hpp`, `state_invariants_core.hpp`). The host engine's points are
`verification/genmc/COVERAGE.md`'s.

A point is one occurrence in code (comments excluded) of: an `atomic*` call; a `cuda::atomic_ref`
load, store or read-modify-write with an explicit order; `__threadfence`; `__syncthreads`,
`__syncwarp` or a policy's `par.sync()`; a warp collective (`__shfl*_sync`, `__ballot_sync`,
`__any_sync`, `__match_*_sync`, `__activemask`, a cooperative-groups scan or shuffle, a policy's
`par.bcast`); a `__shared__` declaration; a `volatile` access; `__nanosleep`. The inventory at
commit b85b4d3e finds 466 points. They are grouped below into 29 protocols by the function that
contains them.

A point counts as covered when a harness runs the engine function that contains it, compiled
from the engine's source (`run.sh` compiles a copy of `gpu/` with launch configurations deleted;
`device/` maps the CUDA runtime onto GPUMC's threads, scopes and barriers). Two harnesses
transcribe part of the kernel: `evolve_ring_termination` and `replay_task_termination` run the
`hgcommon` ring and termination cores with `k_persistent_evolve`'s accounting written out in the
harness (each header lists what it transcribes); no point is counted on their account alone.

## Protocols

| Protocol | Points | Covered | Harnesses |
|---|---|---|---|
| D01 step selection under MaxStatesPerStep (`step_book`, `step_release`, `select_step`, `run_step_selections`) | 28 | 28 | step_selection_once |
| D02 termination detector and worker exit (`TerminationDetector`, the detector contexts, `block_exit_requested`) | 12 | 12 | termination_no_early_exit, exit_uniform, evolve_ring_termination, replay_task_termination |
| D03 match work ring (`RingBuffer`) | 6 | 6 | ring_exactly_once, evolve_ring_termination |
| D04 work logs and match record publication (`WorkLogView`, `publish_match`, `await_match`, `match_published`, `claim_next_record[s]`) | 15 | 15 | work_log_exactly_once, match_records_exactly_once, replay_task_termination |
| D05 counter reservations and saturating totals (`coalesced_add`, `coalesced_bounded_claim`, `Pool::claim`/`settle`, `DeviceArena::claim`, `saturating_add`, the step pool's totals) | 22 | 22 | counter_claims_disjoint, saturating_totals |
| D06 concurrent map (`ConcurrentMap::DeviceView` insert and lookup) | 7 | 7 | hash_insert_elects_one, and every harness on a map |
| D07 lock-free list (`LockFreeList` push, warp push chain, walks) | 11 | 11 | replay_rendezvous_device, depth_relax_child_registration |
| D08 exploration depth (`DeviceExploreCtx`, `ExploreView::claim`) | 5 | 5 | depth_relax_child_registration |
| D09 state identity claims and hash publication (`keyed_claim_device`, `state_claim_form`, `state_claim_content`, `publish_canonical_hash`, the event identity claim, root seeding) | 9 | 4 | form_claim_classes, keyed_follow_rendezvous, keyed_twin_take |
| D10 keyed twins (`keyed_take_twin`, `keyed_follow`, `keyed_hash_or_follow`, `keyed_close_followers`, `keyed_note_claim`) | 16 | 16 | keyed_twin_take, keyed_follow_rendezvous |
| D11 keyed rewrite interning (`keyed_event_rid`, `keyed_intern_event`, `keyed_state_sum`, `keyed_after_rewrite`) | 6 | 4 | keyed_intern_once, keyed_twin_take |
| D12 causal rendezvous (`register_as_producer`, `register_as_consumer`, `add_causal_edge`, `is_reachable_preds`) | 3 | 1 | causal_rendezvous_device |
| D13 rewrite emission (`apply_one_match`, `copy_kept_edges`) | 6 | 0 | |
| D14a replay instances and match rendezvous (`qe_record_instance`, `qe_add_instance`, `qe_publish_child`, `qe_drive_instance`, `qe_drive_match`) | 6 | 6 | replay_rendezvous_device |
| D14b replay claim chain (`DeviceQrCtx::claim`) | 3 | 3 | replay_claim_once |
| D14c replay run signatures (`qe_claim_runsig`) | 5 | 5 | runsig_claim_once |
| D14d multiplicity cascade (`DeviceQmCtx`, `qe_capture_multiplicity`, `qe_seed_multiplicity`, `qe_redrive`) | 9 | 9 | mass_match_rendezvous, saturating_totals |
| D14e replay capture (`qe_capture_expansion`'s match recording, `qe_alloc_words`) | 13 | 2 | replay_claim_once (`qe_alloc_words`) |
| D14f replay event ids (`DeviceQrCtx::mint_event`) | 2 | 0 | |
| D14g replay kept producers (`DeviceQrCtx::record_kept`) | 1 | 0 | |
| D14h replay branchial count (`k_qe_count_branchial`) | 3 | 3 | branchial_count_block |
| D15 IR on a warp (`ir_core.hpp`, `ir_canon.cu`, `IrTile`, `canonicalise_child`'s broadcasts) | 54 | 0 | |
| D16 branchial rows (`k_branchial_rows`, `br_row`) | 24 | 24 | branchial_rows_warp |
| D17 session frontier append (`session_frontier_append`) | 1 | 0 | |
| D18 persistent kernel blocks (`k_persistent_match`, `k_persistent_match_rewrite`, `k_persistent_evolve`: shared state, barriers, broadcasts, idle backoff) | 104 | 13 | exit_uniform (`k_persistent_match_rewrite`) |
| D19 diagnostic counters (phase cycles, rewrite sub-phase cycles, signature fallbacks, canonical event count, alignment tallies, causal and canon totals, `k_measure_states` maxima) | 24 | 0 | |
| D20 match block cap, draw and parent cap (`emit_admit`, `match_state_rule`, `match_state_parent_capped`, `claim_spill`) | 38 | 17 | match_drain_block |
| D21 state geometry, invariants and vertex curvature (`state_geometry_core.hpp`, `state_invariants_core.hpp`, `record_state_invariants_device`, `k_vertex_curvature`) | 31 | 0 | |
| D22 error counters (`DeviceErrors::record`) | 2 | 0 | |

Covered: 213 of 466 points. 15 protocols are covered in full, 6 in part, 8 not at all.

## Not covered

| Points | Reason |
|---|---|
| D18 `k_persistent_evolve`'s block body and `k_persistent_match`, 91 | No harness runs them. `evolve_ring_termination` and `replay_task_termination` run the termination, ring and log protocols the kernel's loop composes, and exit_uniform runs `k_persistent_match_rewrite`'s loop. |
| D15 IR on a warp, 54 | No harness runs the IR. Its warp primitives (`IrTile::sync`, `bcast`, `fetch_add`) are the ones branchial_rows_warp runs. |
| D21 geometry, invariants and curvature, 31 | GPUMC stops with a segmentation fault executing `sg_geometry_of` on one thread with the serial policy, after `sg_build` has returned (a two-vertex graph); without `--disable-loop-jump-threading` it stops at InstAnnotator.cpp:80 ("Only integer and null constants currently allowed in assume() expressions"). The pool claim `record_state_invariants_device` makes is `atomicAdd` on `invariant_pool_used`, the form D05 checks. |
| D19 diagnostic counters, 24 | Relaxed adds, minima and maxima read by the host after the kernel. |
| D20 parent cap and spill, 21 | match_drain_block runs the per-(state, rule) cap and the draw; MaxSuccessorStatesPerParent's passes (`kParentCount`, `kParentEmit`'s quota bits) and the spill claim are not run. |
| D14e replay capture, 11 | No harness runs `qe_capture_expansion`'s match recording (the warp's ballots and shuffles, the match id and local counters, the class pair total). replay_rendezvous_device starts from a recorded match. |
| D13 rewrite emission, 6 | No harness runs `apply_one_match`. Its reservations are D05's `coalesced_*` calls and its rendezvous D12's functions. |
| D09 content claim, event identity claim, root depth seeding, 5 | `state_claim_content` (Automatic states), the event identity claim's record fence (`event_identity.hpp`) and `k_seed_root_hashes`' depth seeding are not run. |
| D12 TR scratch slot lock, 2 | `is_reachable_preds` takes a scratch slot only when the local search visits more than 512 events; a harness needs a predecessor chain longer than that. |
| D11 `keyed_after_rewrite`, 2 | The switch from armed to interning and its interning call are not run; keyed_intern_once runs `keyed_event_rid`, the same interning. |
| D14f replay event ids, 2 | A relaxed fetch-and-add below `replay_id_limit`; no harness runs `mint_event`. |
| D22 error counters, 2 | A relaxed load, then a relaxed fetch-and-add when the counter is 0; the host reads the counters after the kernel. Every harness asserts its counters stay 0, so none runs `record`. |
| D14g, D17, 2 | `record_kept`'s fence and `session_frontier_append` (a relaxed fetch-and-add; the host reads min(count, capacity) after the kernel) are not run. |

## Bounds that limit what a harness reaches

- A modelled warp has at most 8 lanes. Kernels stride their loops by the warp width, so each
  harness gives every loop at most as many items as it has lanes.
- `mass_match_rendezvous`'s three-block cells (two arrivals) give no verdict in 570 s and are
  `GPUMC-CELL-LONG`; its checked cells race the capture against one arrival.
- `replay_task_termination`'s two-worker cell is `GPUMC-CELL-LONG` (272,243,862 executions).
- Harnesses that include `persistent.cu` run with `--disable-function-inliner`; `match_drain_block`
  runs with `--disable-cast-elimination` (EliminateCastsPass.cpp:214 stops on `match.cu`).
