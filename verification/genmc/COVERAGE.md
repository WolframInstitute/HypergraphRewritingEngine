# GenMC coverage of the host engine's concurrent accesses

Scope: every atomic operation, fence and cross-thread plain-field publication in
`hypergraph/`, `job_system/`, `lockfree_deque/` and `common/include/hgcommon/` (host code; the
device side is GPUMC's, under `verification/gpumc`). The inventory was taken by reading the
source at commit 70c5d508: 837 atomic access points and 39 plain-field publication pairs, grouped
below into the protocols they implement. A point counts as reached when a harness calls the
engine code that contains it; reach was traced per protocol from each harness's calls, not per
instruction by an instrument, and rows marked (inferred) were traced through the composed
engine's call graph without a run that names the site.

Every harness below except engine_evolve and quotient_capture_composition reaches a verdict on
the v0.19 fork with no execution cut at an unroll bound, and every harness carries
`// GENMC-CALIBRATE:` cells that the checker catches (`HG_GENMC_CALIBRATE=1 run.sh`). The two
exceptions run to the end of main (their HG_HARNESS_CALIBRATE_END cell is reported) but their
explorations do not end: engine_evolve explored 23,548 executions in 83 minutes, and
quotient_capture_composition is estimated at 2^97. Their protocols are marked (partial) below.
engine_construct, engine_rule and engine_evolve calibrate reachability of the end only; no
defect calibration is caught at their bounds. Threads are the concurrent threads of the harness,
main included when it races.

Build conditions. Harnesses compile with `HG_VERIFICATION=1 HG_ENGINE_STATS=0`, and composed ones
with `HG_PARK_VERIFICATION=1`. So the 150 statistics-only counters, the `!HG_VERIFICATION` block
pool, and the futex, WaitOnAddress, os_sync and atomic-wait park backends are not compiled in any
harness. The park protocol is checked on its spin backend, which waits on the same word.

## Protocols

| Protocol | Points | Harnesses (threads) |
|---|---|---|
| P01 injector Deque, tagged head/tail CAS | 20 | deque_no_double_extraction (2), deque_tag_defeats_aba (2), job_system_error_wait (2), engine_* (3) |
| P02 WorkStealingDeque take/steal | 25 | work_stealing_deque_no_double_take (3), job_system_error_wait (2), engine_* (3) |
| P03 ParkGate park/wake | 38 | job_system_no_lost_wakeup (2), job_system_no_lost_wakeup_domains (2), job_system_error_wait (2), engine_* (3) |
| P04 JobSystem completion and quiescence | 27 | job_system_error_wait (2), engine_* (3) |
| P05 JobSystem error latch and stop | 18 | job_system_error_wait (2) |
| P06 JobSystem start/shutdown | 36 | job_system_error_wait (2), engine_* (3) |
| P07 job run accounting | 2 | job_system_error_wait (2) |
| P08 JobSlotPool foreign-free list | 11 | job_system_error_wait (2), engine_evolve (inferred, partial) |
| P09 ConcurrentMap insert/settle/lookup/grow/carry/drain | 54 | concurrent_map_agreement (2), concurrent_map_double_growth_2t (2), concurrent_map_double_growth_3t (3, SC), concurrent_map_repeated_offer (2), concurrent_map_resize (2), map_insert_existing_during_double_growth (2), map_lookup_during_growth (2), map_lookup_during_double_growth (3), frame_publication_is_atomic (2), claim_match_rendezvous (2) |
| P10 ConcurrentKeySet claim/grow/migrate | 38 | key_set_contains_during_growth (2), key_set_contains_during_double_growth (3), key_set_distinct_keys_across_growth (2), key_set_enumeration (2), key_set_exactly_once (2), key_set_exactly_once_3t (3, SC), key_set_insert_existing_during_double_growth (3) |
| P11 LockFreeList push/iterate | 21 | lock_free_list_completeness (2), lock_free_list_pairs_meet_once (2), lock_free_list_three_meet_once (3) |
| P12 SegmentedArray publish and count | 16 | causal_in_edge_order (2), engine_* (inferred) |
| P13 arena: cursor, shared bump, block chain, construction fences | 49 | arena_cursor_vs_shared_disjoint (2), arena_worker_index_exclusive (2), engine_* |
| P14 arena block pool (tagged Treiber stack) | 10 | block_pool_exactly_once (2) on `hgcommon/pool_core.hpp`; the engine's instance is `!HG_VERIFICATION` |
| P18 DepthJoin settle cascade and report baton | 25 | depth_report_order (2) |
| P19 explore depth relax and child registration | 3 | depth_relax_child_registration (2) |
| P20 dedup claim | 2 | claim_match_rendezvous (2) |
| P21 MatchJoin drain and child inheritance | 19 | child_inheritance_rendezvous (2) |
| P23 stop and resume flags | 29 | engine_* (3) |
| P24 causal producer/consumer rendezvous | 2 | causal_in_edge_order (2) |
| P25 edge/state/event publication by fence | 12 | causal_in_edge_order (2), engine_evolve (partial) |
| P26 state identity publication (hash, id, rank and orbit tables) | 26 | engine_evolve (partial) |
| P28 quotient instance/match rendezvous | 9 | quotient_instance_match_rendezvous (4) |
| P29 quotient multiplicity mass cascade | 14 | quotient_mass_match_rendezvous (3) |
| P31 quotient id allocation and bounds | 27 | quotient_capture_composition (2, partial) |
| P32 configuration flags (written before workers start) | 36 | engine_* |
| P35-P37, P39 bitset count cache, published-id marks, phase timing slot, id counters | 28 | engine_* (inferred) |

`engine_*` is engine_construct, engine_rule and engine_evolve: the composed engine with two
workers and main. engine_construct and engine_rule are exhaustive (1768 executions each) and
reach construction, rule setup, worker start, parking and shutdown; the matching, rewriting and
registration points are reached only by engine_evolve, which is partial.

## Harnesses that run a copy of the protocol

These harnesses drive the shared `hgcommon` core with a context the harness defines, not the
engine's own context class: P19 (`ExploreCtx` replaced), P20 (`claim_match` with its own probe
key), P21 (both sides of the inheritance rendezvous), P28 (`QrCtx::claim` with a different bit
layout), P29 (`Hypergraph::QmCtx` not called), P14 (`pool_core` on harness arrays). The core
function each one calls is the engine's; the engine's binding of it is reached only through
the composed harnesses.

## Not covered

| Points | Reason |
|---|---|
| P22 spine sampling (`own_min_key`, `own_spawned`), 6 | Reached only with sampling enabled; no harness enables it. |
| P27 keyed rewrite tokens and twins, 17 | Reached only when `keyed_state_` is armed; no harness arms it. |
| P30 quotient replay signature cache, 7 | Reached only with event signature keys set; no harness sets them. |
| P16 `Arena<T>`, `ConcurrentArena<T>`, 17 | Never instantiated anywhere in the tree. |
| P38 matcher early-termination flag, 4 | Every engine caller passes no `should_terminate`. |
| P34 debug callback pointer, 3 | Set by no engine path. |
| P08 `JobSlotPool::release_pool`, 3 | Runs from a thread-exit destructor; the checker runs none (`__cxa_thread_atexit` records nothing). |
| P09 `for_each`, `for_each_in_every_table`, `set_arena`, `bytes_allocated`, `size`, 12 | Read after the workers finish, or before they start. |
| P11 move constructor/assignment, `reset`, 6 | Single-threaded lifecycle. |
| P13 `mark`/`release`/`reset`/`allocate_single`, destructor list, `bytes_allocated`, 17 | Recycle and teardown paths; single-threaded or not on a harness path. |
| P03/P04/P06 statistics getters, 18 | Read for reporting after the run. |
| P33 diagnostic counters, 181 | 150 are compiled only with statistics on; the rest are relaxed counters read after the run. |

## Bounds that limit what a harness reaches

- concurrent_map_double_growth_3t and key_set_exactly_once_3t check sequential consistency with a
  context bound (`--sc --bound`); GenMC's bounding requires `--sc`. They find interleaving
  defects, not weak-memory ones. Their RC11 counterparts are the two-thread harnesses of the same
  protocol.
- depth_report_order asserts `late_arrivals() == 0`, which holds trivially with
  `HG_ENGINE_STATS=0`; its other assertions carry the property.
