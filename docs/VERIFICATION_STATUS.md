# Verification status

Three checkers run against this tree. This page lists what each one covers and what none of them
covers.

- **GenMC** enumerates the executions of a bounded C++ program under the RC11 memory model. A
  harness under `verification/genmc/` includes the engine's own header and calls its own
  functions, so it stops compiling when the header changes. A harness marked
  `// GENMC-LINK: engine` is compiled against every engine translation unit and linked, so it can
  call code whose body is in a `.cpp`. Run with `verification/genmc/run.sh <name>` or `all`.
- **GPUMC** is the scoped-RC11 checker for the GPU memory model: threads are grouped into CTAs and
  every access carries a scope, so whether two threads synchronise depends on the scope. RC11 has
  no scopes. GPUMC is a fork of GenMC 0.9 on LLVM 15 and runs from a container (the CAV 2025
  artifact, with the fixes in `verification/gpumc/checker/`). Run with
  `verification/gpumc/run.sh <name>`, or `run.sh cells` for every declared cell.
- **TLA+** (TLC) checks a model of a protocol at any number of workers. Each spec names the code
  it transcribes. Run with `verification/tla/run.sh [<config>]`.

## GenMC

49 harness sources with a `main` under `verification/genmc/`; `genmc_support.cpp` is support
code. The count is `grep -l "int main" verification/genmc/*.cpp | wc -l`. `COVERAGE.md` in that
directory maps the engine's 837 atomic access points and 39 plain-field publication pairs to the
harnesses that reach them.

### The checker

The suite runs on GenMC v0.19.0 built against LLVM 22, with the fixes in
`verification/genmc/genmc-0.19.0-fixes.patch` (branch `hg-fixes-0.19` of
`github.com/richardassar/genmc`). `verification/genmc/README.md` lists each fix with the defect
in the checker and a reproducer of under thirty lines under `verification/genmc/checker/`, which
`checker/run.sh` runs against the built checker. The fixes cover allocation by `operator new[]`,
the ordering of a failed compare-exchange's read, memory-intrinsic promotion and lowering,
thread-local aggregate initialisers, the error report's message, the transformation time on the
composed engine, and two bounds on the spurious failures of a weak compare-exchange.

### Weak compare-exchange

No weak compare-exchange is made strong. Every successful weak CAS also has its spurious failure
explored. A retry loop whose failure repeats the same attempt has no finite exploration, so a
harness that runs one carries `--weak-cas-spurious-bound=N` in its `GENMC-ARGS`.

- A bound of N explores every execution in which no thread's CAS on one address fails spuriously
  more than N times in a row. Executions with longer runs of spurious failures are not explored.
- The checker prints K, the number of spurious-failure options the bound dropped. K = 0 means
  the run is exhaustive. K > 0 means the verdict is bounded at N, not exhaustive.
- `HG_GENMC_SPURIOUS_BOUND=N run.sh <name>` overrides the harness's bound. The checker refuses
  N = 0, which would make every weak CAS strong.

The bound and K of each harness are in the table under "Weak CAS bounds" below.

### What a run checks

- Every harness carries one or more `// GENMC-CALIBRATE: <defines>` lines. Each reinstates a
  defect the harness must catch, either an `HG_CALIBRATE_*` arm in the engine source (off by
  default) or a `CALIBRATE_*` arm in the harness. `HG_GENMC_CALIBRATE=1 run.sh <name>` runs each
  line and passes only when the checker reports the violation.
- A thread that passes the `--unroll` bound is ended, and the checker counts the execution as
  complete; the fork prints how many were cut. `run.sh` fails a run in which every complete
  execution was cut at the bound, or in which no execution completed.
- Harnesses compile with `HG_VERIFICATION=1 HG_ENGINE_STATS=0`. The substitutions this makes are
  listed in `verification/genmc/README.md`, each with the reason it does not weaken the result.

### Harnesses

All harnesses except `quotient_capture_composition` reach a verdict with no execution cut at an
unroll bound, and every calibration cell is caught. Verdicts are under RC11 unless the table says
otherwise. A harness whose module holds a weak compare-exchange has a bounded verdict; its bound
and K are in "Weak CAS bounds" below.

| protocol | harnesses (threads) |
|---|---|
| injector deque, tagged head/tail CAS | `deque_no_double_extraction` (2), `deque_tag_defeats_aba` (2) |
| work-stealing deque | `work_stealing_deque_no_double_take` (3) |
| park/wake and the searching-worker throttle (`hgcommon/park_gate.hpp`) | `job_system_no_lost_wakeup` (4), `job_system_no_lost_wakeup_domains` (4) |
| job system error wait | `job_system_error_wait` (2) |
| ConcurrentMap | `concurrent_map_agreement` (2), `concurrent_map_resize` (2), `concurrent_map_repeated_offer` (2), `concurrent_map_double_growth_2t` (2), `concurrent_map_double_growth_3t` (3, SC, 4 context switches), `map_lookup_during_growth` (2), `map_lookup_during_double_growth` (3), `map_insert_existing_during_double_growth` (2) |
| ConcurrentKeySet | `key_set_exactly_once` (2), `key_set_exactly_once_3t` (3, SC, 1 context switch), `key_set_enumeration` (2), `key_set_contains_during_growth` (2), `key_set_contains_during_double_growth` (3), `key_set_distinct_keys_across_growth` (2), `key_set_insert_existing_during_double_growth` (3) |
| LockFreeList | `lock_free_list_completeness` (2), `lock_free_list_pairs_meet_once` (2), `lock_free_list_three_meet_once` (3) |
| branchial pair recording | `branchial_pair_once` (2) |
| SegmentedArray publication | `segmented_array_published_read` (3) |
| arena | `arena_cursor_vs_shared_disjoint` (2), `arena_high_water_max` (2), `arena_worker_index_exclusive` (2), `block_pool_exactly_once` (2) |
| frame publication | `frame_publication_is_atomic` (2) |
| causal in-edge registration | `causal_in_edge_order` (2) |
| depth join report order | `depth_report_order` (2) |
| explore depth relaxation | `depth_relax_child_registration` (2) |
| match dedup claim | `claim_match_rendezvous` (2) |
| child inheritance | `child_inheritance_rendezvous` (2) |
| sampling spine | `spine_min_rank` (3) |
| quotient instance/match rendezvous | `quotient_instance_match_rendezvous` (4), `claim_chain_exactly_once` (2) |
| quotient multiplicity mass | `quotient_mass_match_rendezvous` (3) |
| quotient replay signature cache | `runsig_cache_step` (2) |
| keyed rewrites | `keyed_intern_once` (2, SC, 4 context switches), `keyed_twin_rendezvous` (2, SC, 1 context switch) |
| quotient capture | `quotient_capture_register` (2, SC, 1 context switch), `quotient_capture_frame` (2, SC, 1 context switch), `quotient_capture_composition` (2, no verdict) |
| composed engine | `engine_construct` (3), `engine_rule` (3), `engine_evolve` (3, SC, 1 context switch) |

The harnesses added for rc2:

- `spine_min_rank` runs the engine's `MatchJoin` members (`hypergraph/match_join.hpp`). The task
  that balances a join's `pushed` and `completed` counters reads the minimum folded rank and the
  spawned mark of every completed match task. A tree of three tasks over two joins, clean
  (bounded; see "Weak CAS bounds"). Calibrations: completions booked relaxed, and the fold done as
  load-compare-store.
- `segmented_array_published_read`: a `SegmentedArray` element read after its writer published
  the index is non-null and fully constructed, while two writers race to create its segment. 28
  executions, clean. The calibration reads through a high-water extent with no publication.
- `branchial_pair_once`: `CausalGraph::record_branchial_overlaps` records each branchial pair once,
  from the bucket of the lowest shared edge, with no set of recorded pairs. Three events consuming
  the same two edges, clean (bounded; see "Weak CAS bounds").
- `claim_chain_exactly_once`: an instance's claim chain (`hgcommon::qr_claim_chain`) claims each
  pair once while two threads install blocks past the first. 24 executions, clean.
- `runsig_cache_step`: a match's cached run-signature key (`hgcommon::qr_cached_key`,
  `qr_cache_key`) is read only for the output step it was claimed for. 4 executions, clean.
- `keyed_intern_once`: two applications of one rewrite under interning get one rewrite id, and
  exactly one is told the rewrite is new (`Hypergraph::intern_rewrite`, `edge_token`). Under SC
  with 4 context switches (bounded; see "Weak CAS bounds").
- `keyed_twin_rendezvous`: two children with one token set, made at once under Full
  canonicalisation, end in one class with one nonzero key (`claim_twin`, `take_twin`). Under SC
  with 1 context switch (bounded; see "Weak CAS bounds").
- `quotient_capture_register` and `quotient_capture_frame` split `quotient_capture_composition` by
  phase. Registration: two rewrites of one parent through `Rewriter::apply` give two events and
  two children in their own classes with published keys and orbit tables, with every class claim
  on one key. Capture: both matches are recorded on the parent's class. Both under SC with 1
  context switch (bounded; see "Weak CAS bounds").
- `job_system_error_wait`: after a job fails, `wait_for_completion` returns only once no worker is
  inside a job. One worker and one job at `--unroll=64`, clean (bounded; see "Weak CAS bounds"). The calibration
  returns from the wait when the error is seen.

### Weak CAS bounds

22 harnesses run a weak compare-exchange. N is the bound per thread and address
(`--weak-cas-spurious-bound`); M is the bound per execution over all threads
(`--weak-cas-spurious-total`). Before is the count on the checker that made such CASes strong.
K > 0 means the verdict is bounded at that N or M and is not exhaustive. Every listed verdict is
"no errors", and every calibration cell of each harness is reported at the same bound.

| harness | bound | executions | K | before |
|---|---|---|---|---|
| `deque_no_double_extraction` | N = 1 | 40 | 7 | 6 |
| `deque_tag_defeats_aba` | N = 1 | 228 | 131 | 11 |
| `child_inheritance_rendezvous` | N = 1 | 48 | 6 | 12 |
| `depth_relax_child_registration` | N = 1 | 26 | 11 | 7 |
| `lock_free_list_completeness` | N = 1 | 6,248 | 4,420 | 184 |
| `lock_free_list_pairs_meet_once` | N = 1 | 44 | 30 | 8 |
| `lock_free_list_three_meet_once` | N = 1 | 4,848 | 3,341 | 288 |
| `job_system_error_wait` | N = 1 | 2,288 | 345 | 286 |
| `block_pool_exactly_once` | N = 1 | 363,984 | 259,015 | 1,126 |
| `spine_min_rank` | N = 1 | 2,306,832 | 346,821 | 28,224 |
| `quotient_mass_match_rendezvous` | N = 1 | 13,220,608 | 12,740,467 | 19,616 |
| `quotient_instance_match_rendezvous` | N = 1 | 5,958,272 | 571,074 | 663,552 |
| `arena_cursor_vs_shared_disjoint` | N = 1 | 406,912 | 286,343 | 7,680 (larger workload) |
| `engine_construct` | M = 1 | 60,112 | 561 | 1,768 |
| `engine_rule` | M = 1 | running | running | 1,768 |
| `causal_in_edge_order` | M = 1 | running | running | running |
| `branchial_pair_once` | M = 1 | running | running | running |
| `keyed_twin_rendezvous` | M = 1, SC 1 switch | running | running | running |
| `quotient_capture_register` | M = 1, SC 1 switch | running | running | running |
| `quotient_capture_frame` | M = 1, SC 1 switch | running | running | running |
| `keyed_intern_once` | M = 1, SC 4 switches | running | running | running |
| `engine_evolve` | M = 1, SC 1 switch | running | running | 38 |

At N = 1 the composed harnesses gave no verdict in 3,600 s (`engine_construct` and `engine_rule`):
construction and worker start alone run dozens of weak CASes, and each successful one doubles the
executions. `arena_cursor_vs_shared_disjoint` at its earlier workload (three 256-byte allocations
per thread) gave no verdict at N = 1 in 3,600 s; it runs the smaller workload its header states,
and `arena_high_water_max` checks `note_block_grab`'s high-water raise alone: one atomic max
read-modify-write, exhaustive in 8 executions.

### Shared bodies

A harness on a shared `hgcommon` core runs the body both engines call:
`claim_match_rendezvous` runs `hgcommon/dedup_claim_core.hpp`, `depth_relax_child_registration`
runs `hgcommon/explore_depth_core.hpp`, `depth_report_order` runs `hgcommon/depth_join.hpp`,
`quotient_mass_match_rendezvous` runs `hgcommon/quotient_multiplicity_core.hpp`,
`block_pool_exactly_once` runs `hgcommon/pool_core.hpp`, and the park/wake harnesses run
`hgcommon/park_gate.hpp`. Several of these drive the core through a context class the harness
defines rather than the engine's own; `COVERAGE.md` lists which.

### ConcurrentMap lookup across two growths

An absent answer from `ConcurrentMap::lookup` is final only if `table_` is still the head the walk
started from; otherwise the walk restarts from the head it loads. Without that rule a lookup
overtaken by two overlapping growths can skip the table holding a settled key.
`map_lookup_during_double_growth` (one reader, two growers whose growths overlap) reports the
violation in 7,508 executions without the rule and is exhaustively clean with it (18,112,955
executions). The same shape is checked on the map's claim
(`map_insert_existing_during_double_growth`) and on the key set
(`key_set_contains_during_double_growth`, `key_set_insert_existing_during_double_growth`).

### The composed engine

`engine_construct` and `engine_rule` link every engine translation unit and run construction, rule
setup, worker start, parking and shutdown with two workers and main. `engine_evolve` runs one
`evolve()` call under SC with 1 context switch, and its transform needs more than 12,000 MB of
address space (`HG_GENMC_MEM_MB=16000`). All three run with at most one spurious failure per
execution (`--weak-cas-spurious-total=1`); their counts are in "Weak CAS bounds". Each composed harness carries `-DHG_HARNESS_CALIBRATE_END=1`, an
assertion at the end of `main`, so a verdict is reported only with a run that reaches the end. The
three composed harnesses calibrate reachability of the end only; no defect calibration is caught
at their bounds.

`causal_in_edge_order` runs `CausalGraph::consume_edges` on one thread while another forces the
producer map through two growths, over the engine's ConcurrentMap, LockFreeList and arena,
clean (bounded; see "Weak CAS bounds"). `HG_CALIBRATE_IN_EDGE_ORDER_ASCENDING` reverses the recorded order and the
checker reports the redundant pair.

## GPUMC

26 harnesses under `verification/gpumc/`, 24 of them on the device's own code: `run.sh` compiles
a copy of `gpu/` with launch configurations deleted, and `verification/gpumc/device/` maps CUDA's
atomics, `cuda::atomic_ref`, fences, barriers, warp collectives and thread indices onto GPUMC's
threads, scopes and work-group barriers. `evolve_ring_termination` and `replay_task_termination`
run the `hgcommon` ring and termination cores with the kernel's accounting transcribed. Each
harness declares its cells (`// GPUMC-CELL: name PASS|FAIL|BLOCKED defines -- gpumc args`) and
`run.sh cells` fails on a verdict that differs; it is the `gpumc_cells` ctest. Every harness has
a calibration cell that must FAIL: `HG_GPUMC_CALIBRATE_NO_THREADFENCE` (every `__threadfence` a
no-op), `HG_GPUMC_CALIBRATE_SPLIT_RMW` (atomicAdd, atomicMin, atomicMax, atomicCAS and the
`atomic_ref` exchanges and fetch_add a load and a separate store), or a harness's own reordering.
`verification/gpumc/COVERAGE.md` maps the device's 466 concurrency points to 29 protocols: 217
points are covered.

The checker is the CAV 2025 artifact with six fixes (`verification/gpumc/checker/build.sh`, image
`gpumc_docker:hg-fixes`): 32-bit compare-exchange, memcpy and memset promotion, races in included
headers, struct returns, barrier divergence after a pruned spin, and floating point (intrinsics,
C math, external-call arguments). Each fix has a checker cell.

| harness | engine code | property | clean | calibration |
|---|---|---|---|---|
| `step_selection_once` | `persistent.cu` `step_release`, `run_step_selections` | a step's selection runs once after its last unit and keeps the lowest-ranked candidate; a later step's record being written stays for its scan | 112 executions | release_first FAIL, no_threadfence FAIL |
| `exit_uniform` | `persistent.cu` `k_persistent_match_rewrite` | every lane of a worker block leaves at the same iteration | 10 | lane_read FAIL (barrier divergence) |
| `termination_no_early_exit` | `TerminationDetector` | no quiescent exit while work is owed | 3,173 | complete_then_push FAIL |
| `evolve_ring_termination` | `ring_core.hpp`, `termination_core.hpp` | the kernel loop never exits with work in the ring, a record unrewritten or a child owed | 672,126 (1 rule, 2 steps), 2,433,998 (3 rules, 1 step) | done_before_push FAIL |
| `replay_task_termination` | `term_detect_loop` | no quiescent exit while a replay task is claimed, unrun or owed | 174,129 (1 worker); 272,243,862 (2 workers, long cell) | done_before_run FAIL, record_before_capture FAIL |
| `ring_exactly_once` | `RingBuffer` | every item popped once, with drain | 26 | split_rmw FAIL |
| `work_log_exactly_once` | `WorkLogView` | every entry handed out once | 38 | split_rmw FAIL |
| `match_records_exactly_once` | `persistent.cu` `claim_next_record[s]` | every match record claimed once | 34 | split_rmw FAIL |
| `counter_claims_disjoint` | `coalesced_add`, `coalesced_bounded_claim`, `DeviceArena::claim` | claims are disjoint and within the limit | 10, 2, 6 | split_rmw FAIL (each) |
| `bounded_appends_once` | `session_frontier_append`, `DeviceQrCtx::mint_event` | each slot and id handed out once; refusal past the limit is reported | 216 | split_rmw FAIL |
| `saturating_totals` | `saturating_add` | no update lost; the sum stops at `QM_SATURATED` | 6, 6 | split_rmw FAIL |
| `hash_insert_elects_one` | `ConcurrentMap::DeviceView` | one inserter elected, its value stored | 9 | split_rmw FAIL |
| `depth_relax_child_registration` | `persistent.cu` `DeviceExploreCtx` | a child never strands at a stale depth and is logged once | 11 | no_threadfence FAIL |
| `form_claim_classes` | `state_claim_form` | equal forms share a class, a colliding form takes the next key | 696 | split_rmw FAIL |
| `content_claim_classes` | `state_claim_content` | equal content shares a class, other content under the same hash takes the next key | 252 | split_rmw FAIL; no_threadfence PASS (the map's exchange publishes the slice) |
| `keyed_follow_rendezvous` | `keyed_hash_or_follow`, `keyed_close_followers` | a child of an unpublished twin follows once or reads its hash | 23 | no_threadfence FAIL |
| `keyed_twin_take` | `keyed_take_twin`, `state_claim_form` | twins agree on one class key; a taken twin copies the twin's ranks | 400, 1,600, 200 | duplicate_no_threadfence FAIL |
| `keyed_intern_once` | `keyed_event_rid` | one rewrite id per rewrite, also when interned on demand | 335 | split_rmw FAIL |
| `causal_rendezvous_device` | `rewrite.cu` `register_as_producer`/`consumer` | each (producer, consumer, edge) recorded once | 16, 32 (TR) | consumer_first FAIL |
| `match_drain_block` | `match.cu` `match_state_rule` | the cap and the draw emit the lowest-ranked transition once | 3, 3 | split_rmw FAIL (each) |
| `branchial_rows_warp` | `k_branchial_rows` | every pair counted once under its key | 1, 1, 8 | three_split_rmw FAIL |
| `branchial_count_block` | `quotient.cu` `k_qe_count_branchial` | each instance's term summed once | 1 | two_blocks FAIL |
| `replay_rendezvous_device` | `qe_publish_child`, `qe_drive_match`, `qe_seed_root_instance` | an instance and a match never both miss each other | 42 (warp chain), 4 (root) | no_threadfence FAIL (each) |
| `replay_claim_once` | `DeviceQrCtx::claim` | one claimant of a pair wins, past the first claim block too | 12, 4 | split_rmw FAIL (each) |
| `runsig_claim_once` | `qe_claim_runsig` | one key and one win per event class | 146 | split_rmw FAIL |
| `mass_match_rendezvous` | `qe_capture_multiplicity`, `DeviceQmCtx` | class mass passes across a match exactly once | 51 (one arrival); two arrivals: no verdict in 570 s (long cell) | one_arrival_no_threadfence FAIL |

Four checker cells (`cas_width`, `memcpy_promotion`, `struct_return`, `float_math`) PASS on the
patched image, `race_in_header` FAILs as it must, and `barrier_divergence` has an `exited` cell
that FAILs and a `spinning` cell that is BLOCKED.

Device defects these harnesses found, each fixed with the harness that fails before the fix:
`select_step` read candidate records without acquire and could pass over a later step's record
still being written (356bce77); worker blocks read the exit flag per lane before a barrier
(6c19aa10); plain and volatile reads of atomically updated words in the bounded claim, the arena
cursor, the multiplicity cells and the keyed words (48ef4ee8); the saturating add had two bodies
(920d27a8); `quotient.cu` did not compile under clang's C++17 (84e54324).

### Device sanitizers

`tools/dev/release_gate.sh sanitizers` runs compute-sanitizer memcheck, initcheck, synccheck and racecheck over hg_gpu_tests, gpu_differential_tests and gpu_ffi_tests. Three tests are left out of every sanitizer leg; the script header gives the reason for each. One run of the gate, split into gtest shards of at most 590 s each:

| tool | hg_gpu_tests | gpu_differential_tests | gpu_ffi_tests |
|---|---|---|---|
| memcheck | 0 errors (108 tests, before the exclusions; the 8 reports were the intended cudaMalloc failures, which the gate no longer reports) | 0 errors (77 tests, 4 shards) | 0 errors (131 tests, 3 shards) |
| initcheck | 0 errors (106 tests, 2 shards) | 0 errors (77 tests, 3 shards) | 0 errors (131 tests, 2 shards) |
| synccheck | 0 errors (108 tests) | 0 errors (77 tests, 3 shards) | 0 errors (131 tests, 2 shards) |
| racecheck | 0 hazards (106 tests, 4 shards) | not run yet | 0 hazards on 14 of 16 shards (114 tests); shards 1 and 10 did not finish in 590 s |

racecheck sees shared-memory hazards only. Under racecheck one persistent-kernel test of gpu_ffi_tests takes 149 s.

## TLA+

Five modules under `verification/tla/` (`MCMatchForwarding.tla` instantiates `MatchForwarding.tla`
with a bounded universe) and 13 configurations. Line 1 of each `.cfg` declares its expected verdict
and `run.sh` fails on a mismatch; it is the `tla_cells` ctest. Six configurations must report a
violation.

| configuration | expected | distinct states |
|---|---|---|
| `MCMatchForwarding` | PASS | 79,278 |
| `MCMatchForwardingRegisterBroken` | VIOLATION | -- |
| `MCMatchForwardingResume` | PASS | 30,103 |
| `MCMatchForwardingResumeClaimBroken` | VIOLATION | -- |
| `MCDepthRelaxation` | PASS | 14 |
| `MCDepthRelaxationBroken` | VIOLATION | -- |
| `MCDepthRelaxationSteered` | PASS | 24 |
| `MCDepthRelaxationSteeredBroken` | VIOLATION | -- |
| `MCQuiescence` | PASS | 22 |
| `MCQuiescenceBroken` | PASS | 22 |
| `MCQuiescenceLateSubmit` | VIOLATION | -- |
| `MCQuotientContinuation` | PASS | 910,975 |
| `MCQuotientContinuationNoBlocked` | VIOLATION | -- |

TLC stops a violating configuration at its first counterexample, so its state count depends on
the worker count and is not listed. `MCQuiescenceBroken` omits `jobs_executing` from the scan and
passes: the `submitted` and `completed` counters cannot agree while a worker is inside a job. The
host's quiescence rests on one precondition, that a job submits its children before it returns;
`MCQuiescenceLateSubmit` breaks it and TLC reports the violation. The models assume sequentially
consistent memory; the RC11 orderings of the same protocols are the GenMC harnesses'.

## Reference corpus coverage

The engine's results are compared with `reference/MultiwayReference.wl`, this project's
brute-force evolver, which `reference/compare_multiwaysystem.wls` cross-checks against the
`Wolfram/Multicomputation` `MultiwaySystem`. The instruments:

- **OF**: `reference/oracle_fuzz.wls`, HGEvolve on the CPU and GPU against the reference, about
  100 calls per case. Run on Windows Wolfram 15.0: seeds 100-1159, 1053 cases, 81004 raw
  reference events; seeds 0-41 in the local gate.
- **WF**: `tools/dev/worker_fuzz.py`, `hg_evolve` against one `hg_evolve_gpu --serve` worker,
  CPU at 1, 4, 16, 32 and one seed-chosen count of workers, counts-only against full replies.
  5072 cases, seeds 1000-6071.
- **GM**: `reference/golden_matrix.txt` (all_tests `GoldenMatrix`), 17 corpus workloads x 3 state
  modes x 3 event modes, quotient rows under Full.
- **OC**: all_tests `OracleCorpus.*` and `FeatureMatrix.*` on `reference/oracle_corpus.hpp`.
- **GC**: `reference/golden_corpus.wl` through `verify_paclet.wls` and `verify_paclet_gpu.wls`,
  12 cases.
- **SS**: `reference/verify_state_statistics.wls` against `StateGeometryReference.wl`.
- **SE**: `reference/verify_sessions.wls`.

OF draws its case class from the seed, 14 classes, 75 or 76 cases each in the run above. Every
OF case runs every option row below.

| class | instrument | cases |
|---|---|---|
| random rules, arity 1-3, 1-3 rules, 1-4 left edges, 0-4 right edges | OF, WF, GC | 75, 5072, 12 |
| arity 4-16 and mixed arities | OF (class `arity`), WF, OC (`arity4-growth`, `mixed-arity-*`, `arity3-to-2`) | 75, 450, 4 |
| unary edges | OF (`unary`), OC (`arity1-with-binary`) | 76, 1 |
| self-loops and repeated vertices in a pattern | OF (`selfloop`), OC (`self-loop`), GC (3 cases) | 76, 1, 3 |
| disconnected left side | OF (`disconnected`), OC (`disconnected-lhs`) | 76, 1 |
| empty right side, delete-only rules, states emptied | OF (`deleteOnly`) | 76 |
| identical rules, rules equal up to renaming | OF (`identical`), WF | 76, 424 |
| cyclic and recurrent rules | OF (`cyclic`) | 76 |
| non-confluent rule pairs | OF (`nonconfluent`), GC (`TC4`, `TC5`) | 76, 2 |
| several roots, isomorphic or sharing labels | OF (`multiroot`), WF, OC (`FeatureMatrix.MultipleInitialStates*`) | 76, 740 |
| complete digraphs, cycles, stars, grids, tori | OF (`symmetric`), WF (one state in twelve), OC (`cycle4-automorphic`, `star4-automorphic`) | 76, one WF initial state in twelve, 2 |
| named initial conditions (Grid, Grid with holes, Cylinder, Torus, Sphere, Klein, Mobius, Minkowski, BrillLindquist, Poisson, Uniform) | OF (`named`: the name against its edge list and the reference) | 76 |
| large states (12-30 edges) | OF (`large`), WF (15-50 edges) | 68 (6 skipped: reference over 30 s), 698 |
| symbolic vertices, packed arrays | OF (`symbolic`, and packed arrays in 30% of cases), `verify_paclet.wls` | 75 |
| presentation: relabelled and permuted roots, renamed and permuted rules, labels near 2^32 | OF (every case except `named`) | 977 |
| CanonicalizeStates None, Automatic, Full | OF (every case), WF, GM | 1053, 5072, 17 x 3 |
| ExploreFromCanonicalStatesOnly (and its refusal under None and Automatic) | OF, WF, OC (`QuotientMatchesFullClosure*`) | 1053, 1196 |
| CanonicalizeEvents None, Full, Automatic | OF (Automatic under Full: between the marked-form and raw counts), WF, GM | 1053, 5072, 17 x 3 |
| CanonicalizeEvents key lists | OF (3 random lists per case, exact against the marked forms), WF, OC (`CustomEventKeySets*`) | 1053, 1008 |
| CanonicalizeEvents "Positional" | OF (amber against the reference's Positional count), WF | 1053, 259 |
| CausalTransitiveReduction on and off | OF, WF | 1053, 1485 |
| ShowGenesisEvents | OF (20 configurations per case), WF | 1053, 760 |
| EdgeDeduplication (causal graph edges) | OF (one event identity per case) | 1053 |
| MaxStatesPerStep, MaxSuccessorStatesPerParent, MatchesPerStateRule, TransitionRate, ExplorationProbability, RuleWeights, UniformRandom/MatchesPerStep, RandomSeed | OF (a non-binding cap equals the uncapped run; a binding cap agrees across devices and quotient and does not exceed the uncapped counts), WF, OC (`APerParentCap*`, `APerStepCap*`) | 1053, 3055 |
| StepStatistics, weightings, branchial "Graph" and "Overlap" | SS, WF (devices and worker counts) | SS corpus, 3095 |
| sessions: open, step, steer ("From", including {}), query, delta, close | OF (whole steps and a split step against the reference), SE, OC (`Continuing*`, `Steer*`) | 1053, SE corpus |
| TargetDevice CPU and GPU | OF (every case, both), WF, GC | 1053, 5072, 12 |
| worker counts 1 to 32 | WF (1, 4, 16, 32, one of 2-31), OC (`DeterministicAcrossThreadCounts`) | 5072 |

No reference meaning exists for which transitions a binding cap or a sampling draw keeps: the
rank is the engine's transition key (docs/SPEC.md 5), so OF checks the properties listed above.
`Positional` and Automatic events under None or Automatic states read each raw state's own
labeling (docs/SPEC.md 4.2); their counts are compared with the reference's Positional count as
amber, and 11 of the 1053 cases differ from it.

## Open items

- **Bounded spurious failures.** Every harness that runs a weak compare-exchange has K > 0 at its
  bound ("Weak CAS bounds"): executions with more spurious failures than the bound are not
  explored.

- **Bounded under SC.** Seven GenMC harnesses have a verdict under sequential consistency with a
  context bound, not an exhaustive RC11 verdict: `engine_evolve` (1 switch; RC11 gave no verdict
  in 83 minutes), `keyed_intern_once` (4 switches; RC11 gave no verdict in 2,770 s),
  `keyed_twin_rendezvous` (1 switch; RC11 estimate 2^51 executions), `quotient_capture_frame` (1
  switch; RC11 estimate 2^87), `quotient_capture_register` (1 switch; RC11 estimate 2^61),
  `concurrent_map_double_growth_3t` (4 switches) and `key_set_exactly_once_3t` (1 switch). The
  two 3-thread harnesses have RC11 counterparts at two threads.
- **No verdict.** `quotient_capture_composition` runs two rewrites of one parent under quotient
  reconstruction through `Rewriter::apply`, with registration and capture interleaved. It reaches
  the end of `main` (its `HG_HARNESS_CALIBRATE_END` cell is reported), and its RC11 exploration,
  estimated at 2^97 executions, does not end. Its two phases are checked separately by
  `quotient_capture_register` and `quotient_capture_frame`; an interleaving of one rewrite's
  registration with the other's capture is checked by no harness.
- **Partial protocols.** The matching, rewriting and registration points reached only by
  `engine_evolve` are checked at its SC bound. `COVERAGE.md` marks them (partial).
- **Thread exit.** The checker runs no thread-exit destructors, so `JobSlotPool::release_pool`
  (called from `TlsGuard::~TlsGuard`, `job_system/src/job_pool.cpp`) is checked by no harness.
- **Worker index reuse.** Under `HG_VERIFICATION` the arena's worker index comes from a counter
  that never releases, and `arena_worker_index_exclusive` holds its indices. Release and
  re-acquire are not checked.
- **Device points not covered.** `verification/gpumc/COVERAGE.md` lists 249 of the device's 466
  concurrency points that no GPUMC harness runs, with the reason for each. The largest groups:
  `k_persistent_evolve`'s block body (91; `evolve_ring_termination` runs its loop's accounting
  transcribed), the IR on a warp (54), and the geometry, invariants and curvature code (31), on
  which GPUMC stops with a segmentation fault in `sg_geometry_of`.
- **Not reached.** `COVERAGE.md` lists the access points no harness reaches and the reason for each:
  the matcher's early-termination flag and the debug callback pointer, which no engine path sets;
  reads made before workers start or after they finish; single-threaded lifecycle code; and
  statistics counters, which verification builds do not compile.
