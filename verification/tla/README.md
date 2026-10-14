# TLA+ layer — algorithm-level correctness at arbitrary N (#80)

GenMC (../genmc) checks the ACTUAL C++ against RC11 at small thread/op bounds.
This layer checks a MODEL of the algorithm at unbounded worker count: every
in-flight protocol step lives in a `pending` bag and any enabled step may fire,
so interleavings subsume every thread count — the bound TLC pays is state count.
The cost is drift: the model is a transcription, and a fix to the code does not
fix the model (see `job_system_no_lost_wakeup` in the GenMC layer for the same
lesson). Each spec header names the code it transcribes; changing that code
means re-checking the transcription.

## MatchForwarding — inheritance completeness (#80 target 1)

Models `register_child_with_parent`, `inherit_from_parent`, the drain in
`note_match_task_done` and the `claim_match` dedup, from
`hypergraph/src/parallel_evolution.cpp`. A state drains once its own matches are
found and it has inherited; at the drain it publishes `drained` and scans its
children, while a child is published in its parent's list and then reads
`drained`. Property: at quiescence every state holds every match discovered at
an ancestor that overlaps none of the edges consumed on the path down, and every
state holds its whole valid set when it drains. A lost match deletes its whole
subtree while the run stays self-consistent.

Run (needs `~/tla/tla2tools.jar`, any Java ≥ 11):

    verification/tla/run.sh            # every cell
    verification/tla/run.sh MCMatchForwardingRegisterBroken

`run.sh` checks each cell against the verdict its own `.cfg` declares on line 1
(`Expected: PASS` or `Expected: VIOLATION`) and exits non-zero on a mismatch. It
is registered as the `tla_cells` ctest, and skips rather than fails
where Java or `tla2tools.jar` is absent. **Seven of the sixteen cells must
VIOLATE** — `MCMatchForwardingRegisterBroken`,
`MCMatchForwardingResumeClaimBroken`, `MCDepthRelaxationBroken`,
`MCDepthRelaxationSteeredBroken`, `MCQuiescenceLateSubmit`,
`MCQuiescenceSearchLeaveLate` and
`MCQuotientContinuationNoBlocked` — and a spec edit that turns one of those into a pass
has disabled a calibration, which is the case a results table cannot catch and
the runner can.

It reports distinct-state counts but does not assert them: TLC stops a violating
cell at the first counterexample, so how far the other workers had gone varies
with `-workers`, so VIOLATION rows are not reproducible numbers.
The PASS rows are, and `run.sh` reproduces them exactly.

The single command underneath, for one cell by hand:

    cd verification/tla
    java -cp ~/tla/tla2tools.jar tlc2.TLC -workers 8 -deadlock \
         -config <cell>.cfg MCMatchForwarding.tla

The four cells (exhaustive at the MC bound — 4 states, 3 matches, 3 edges, a
match found mid-chain). The two resume cells put `s3` in the root's class and
allow one Stop (which cuts every representative still matching) and a Resume:

| cell | RendezvousFix | Stop | cut state's claim | verdict | distinct states |
|---|---|---|---|---|---|
| MCMatchForwarding | TRUE | no | — | PASS | 79,278 |
| MCMatchForwardingRegisterBroken | FALSE | no | — | **VIOLATED** | stops at the first counterexample |
| MCMatchForwardingResume | TRUE | yes | kept | PASS | 30,103 |
| MCMatchForwardingResumeClaimBroken | TRUE | yes | given back | **VIOLATED** | stops at the first counterexample |

- A cut state that gives its class claim back can lose it on resume to another
  state of its class that a resumed rewrite creates first (the counterexample:
  the root finds `mA`, is cut, and the rewrite of `mA` creates `s3` in its class);
  it is then not resumed, never drains, and its child never inherits. The engine
  keeps the claim (`defer_cut_match_task`) and resumes a cut state without
  claiming again (`run_pass`).

- The shipped order (publish the child, then read `drained`; set `drained`, then
  scan) is inheritance-complete at this bound for every interleaving, and every
  state drains holding its whole valid set.
- Reading `drained` before publishing lets a drain fall between the two steps:
  the drain's scan misses the child, the child reads `drained` clear, and the
  child never inherits. The model reaches that loss, which is the calibration.

Model scope: sequentially consistent memory (RC11 is GenMC's job); scans atomic
at fire time (the real `LockFreeList::for_each` tolerates appends mid-walk; the
kept discrepancy is exactly the documented miss window — elements registered
after a scan are covered by the other mechanism); `claim_match` modeled as an
exact (match, state) set, so #74's 64-bit-hash collision class is OUT of model;
no sampling (a run that samples turns forwarding off); no stop and resume (the
resume_pending guard is gated by
MatchCompleteness.AStoppedRunContinuedMatchesOneRunUnderRepetition).

---

## `Quiescence` — termination detection and no lost wakeup under the throttle

`Quiescence.tla` models `JobSystem`'s submit, worker idle path and `is_quiescent()`, from
`job_system/include/job_system/job_system.hpp` and `common/include/hgcommon/park_gate.hpp`:
a submit pushes and then decides whether to wake (skipped while a worker searches, unless the
submitter's own deque holds more than twice the searchers); a worker with an empty deque counts
itself as a searcher or declines (a free choice, so every bound is covered), steals or gives up,
leaves the searching count, announces itself idle, samples the park word, takes an exhaustive
look and parks; a wake moves the park word and unblocks one sleeping worker; a searcher that
found work may wake one more. The caller is a thread that is not a worker: it seeds the run
through the injector, then waits and runs the termination check.

Properties: `NoPrematureQuiescence` (the check never reports done while work remains),
`NoLostWakeup` (no job is queued while every worker is blocked in its park and the caller is not
between a push and its wake decision) and `CountsExact` (the searching and idle counts equal the
workers in those states).

| cell | throttle | searcher leaves the count | other | verdict | distinct states |
|---|---|---|---|---|---|
| MCQuiescence | yes | before the last look | 2 workers | PASS | 133,615 |
| MCQuiescenceThreeWorkers | yes | before the last look | 3 workers | PASS | 5,126,291 |
| MCQuiescencePerSubmit | no | before the last look | 2 workers | PASS | 131,061 |
| MCQuiescenceBroken | yes | before the last look | scan omits `jobs_executing` | PASS | 133,615 |
| MCQuiescenceLateSubmit | yes | before the last look | a job completes before submitting its child | **VIOLATED** (`NoPrematureQuiescence`) | stops at the first counterexample |
| MCQuiescenceSearchLeaveLate | yes | after the last look | 2 workers | **VIOLATED** (`NoLostWakeup`) | stops at the first counterexample |

The `SearchLeaveLate` counterexample: worker 2 parks; worker 1 searches, finds nothing,
announces itself idle and takes its last look while still counted; the caller pushes, reads two
idle workers and one searcher and skips its wake; worker 1 leaves the count and parks. Both
workers sleep with the job in the injector.

Model scope: sequentially consistent memory (the barrier pairing under RC11 is
`verification/genmc/job_system_no_lost_wakeup{,_domains}.cpp`); one park domain; MaxJobs 4.

---

## `QuotientContinuation` — a continued replay reaches what one run reaches

`QuotientContinuation.tla`, cells `MCQuotientContinuation` (shipped) and
`MCQuotientContinuationNoBlocked` (calibration).

Models the quotient replay across continuations, from `hypergraph/src/hypergraph.cpp`
(`qc_add_instance`, `qc_capture_expansion`, `quotient_redrive_point`) and
`ParallelEvolutionEngine::evolve_more`. An instance meets every captured match of its class
through publish-then-scan; a point created at or past the depth bound is pushed on
`qc_blocked_`; between runs the bound is raised and every blocked point with
`old <= depth < new` is submitted as a redrive job, which races the resumed run's new
instances and captures. Captures may happen in any run.

Properties, at the end of the last run: every instance below the final bound has been
applied to every match of its class (`Complete`); the instances are the paths one run to the
final bound creates (`Exact`); no application is at or past the bound in force (`NoneBeyond`).

| cell | runs (bounds) | blocked push | verdict | distinct states |
|---|---|---|---|---|
| MCQuotientContinuation | 3 (1, 2, 3) | yes | PASS | 910,975 |
| MCQuotientContinuationNoBlocked | 3 (1, 2, 3) | no | **VIOLATED** (`Complete`) | stops at the first counterexample |

Two classes, four matches (A→B, A→A, B→A, B→B). Model scope: sequentially consistent memory;
the publish-then-scan ordering under RC11 and the claim are
`verification/genmc/quotient_instance_match_rendezvous.cpp`. The multiplicity path
(`qm_point`, which also pushes on `qc_blocked_`) is not modelled.
