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
  artifact). Run with `verification/gpumc/run.sh <name>`.
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

Eight harnesses under `verification/gpumc/`. Each runs a shared `hgcommon` core with the device's
memory orders and scopes.

| harness | core | property | result |
|---|---|---|---|
| `termination_no_early_exit` | `termination_core.hpp` | the detector never takes the quiescent exit while work is owed | 2,265 executions, clean |
| `ring_exactly_once` | `ring_core.hpp` | no ring item is handed to two consumers, none is invented | 8 complete, 14 blocked, clean |
| `hash_insert_elects_one` | `hash_insert_core.hpp` | exactly one thread is told `inserted`, and the stored value is that thread's | 4 executions, clean |
| `replay_rendezvous_meets` | `list_core.hpp` | an instance and a match arriving at once never both miss each other | 3 executions, clean |
| `mass_match_rendezvous` | `quotient_multiplicity_core.hpp` | class mass passes across each match exactly once | 160 executions, clean |
| `replay_task_termination` | `term_detect_loop` | no quiescent exit while a replay task is claimed, unrun or owed | 174,129 executions (1 worker), 272,243,862 (2 workers), clean |
| `depth_relax_child_registration` | `explore_depth_core.hpp` | a child never strands at a stale depth | 9 executions, clean |
| `evolve_ring_termination` | `ring_core.hpp`, `termination_core.hpp` | the kernel loop composing ring, record pool and detector never exits with work in the ring | 672,126 executions (1 rule, 2 steps), 2,433,998 (3 rules, 1 step), clean |

Every GPUMC harness has a calibration arm that the checker reports. The reservation CAS in
`ring_exactly_once` is modelled weak, as the device writes it. A 32-bit compare-exchange always
reports failure in the GPUMC build, so the harnesses use 64-bit words where the device uses 32-bit
ids.

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
- **The device kernel body.** `k_persistent_evolve` (`gpu/src/persistent.cu`) is not run under
  GPUMC from its own source: it is CUDA device code on `cuda::atomic_ref`, `__threadfence`,
  `__syncthreads` and thread indices. Its decisions are the shared cores in the GPUMC table, and
  `evolve_ring_termination` runs the loop that composes them. The per-item work between those
  bookings (matching, rewriting, canonicalisation, the arena) uses storage, monotonic counters,
  bump allocators and one release/acquire publish flag (`match.hpp`), and is not model-checked.
- **Not reached.** `COVERAGE.md` lists the access points no harness reaches and the reason for each:
  the matcher's early-termination flag and the debug callback pointer, which no engine path sets;
  reads made before workers start or after they finish; single-threaded lifecycle code; and
  statistics counters, which verification builds do not compile.
