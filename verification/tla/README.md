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
    verification/tla/run.sh --quick    # every cell but the deep one
    verification/tla/run.sh MCMatchForwardingRegisterBroken

`run.sh` checks each cell against the verdict its own `.cfg` declares on line 1
(`Expected: PASS` or `Expected: VIOLATION`) and exits non-zero on a mismatch. It
is registered as the `tla_cells` ctest, `--quick`, and skips rather than fails
where Java or `tla2tools.jar` is absent. **Seven of the sixteen cells must
VIOLATE** — `MCSegmentedArrayBroken`, `MCMatchForwardingRegisterBroken`,
`MCMatchForwardingResumeClaimBroken`, `MCDepthRelaxationBroken`,
`MCDepthRelaxationSteeredBroken`, `MCQuiescenceLateSubmit` and
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

Second target (#80): quiescence liveness — not yet modeled.

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

---

## `SegmentedArray` — the segment-ordering invariant

`SegmentedArray.tla`, run under `MCSegmentedArray.cfg` (shipped) and
`MCSegmentedArrayBroken.cfg` (calibration).

**Why this one is here and not in `verification/genmc/`.** Every other concurrent structure is
model-checked against its own header by a GenMC harness. GenMC v0.17.0 cannot execute this one:
merely CONSTRUCTING a `SegmentedArray<uint64_t,4>` segfaults it inside
`SAddrAllocator::allocate`, in stack and in static storage, with and without the class's throw
path. Isolated — a hand-written `std::atomic<uint64_t*>[4]` plus the same `hgcommon::ctz64` call
verifies in one execution and 0.00 s, so it is this class the tool cannot take. `safe_verify.sh`
already prescribes the alternative: move the argument to TLA+, which is state-bounded rather
than execution-bounded.

**The property.** `count` is a high-water mark advanced independently by each `emplace`, so a
thread claiming an index in segment 2 advances it past segment 1's ENTIRE range. A walk over
`[0, count)` is therefore only safe if the directory is dense below the mark, which is what
`get_or_create_segment` creating predecessors before the segment asked for buys. `DenseBelowCount`
states it; `CompleteWhenQuiescent` states that once no claim is outstanding the mark admits
everything handed out and every index has been written.

| configuration | result |
|---|---|
| `CreateInOrder = TRUE`, 3 threads, 3 segments of 2 (`MCSegmentedArray.cfg`) | **No error. 6,412 states generated, 2,284 distinct, depth 21** |
| `CreateInOrder = TRUE`, 4 threads, 3 segments of 2 | **No error. 62,833 generated, 17,119 distinct** |
| `CreateInOrder = TRUE`, 4 threads, 4 segments of 2 | **No error. 248,425 generated, 65,593 distinct** |
| `CreateInOrder = TRUE`, 5 threads, 4 segments of 2 | **No error. 2,927,021 generated, 626,209 distinct** (3s) |
| `CreateInOrder = TRUE`, 6 threads, 4 segments of 2 | **No error. 27,118,345 generated, 4,926,585 distinct** (17s) |
| `CreateInOrder = TRUE`, 6 threads, 5 segments of 2 (`MCSegmentedArrayDeep.cfg`) | **No error. 157,960,861 generated, 27,828,731 distinct, queue empty** (61s) |
| `CreateInOrder = TRUE`, 7 threads, 4 segments of 2 | **No error. 194,366,005 generated, 31,142,659 distinct** (81s) |
| `CreateInOrder = FALSE` (create only the segment asked for) | **`DenseBelowCount` violated**, 256 states in |

The shipped cell is 2,284 distinct states and TLC exhausts it in under a second, so its bound was
costing nothing and buying correspondingly little — the committed deep cell is **twelve thousand
times** larger and takes a minute. All rows measured 2026-08-26 on this i9-14900K desktop under WSL2, TLC
`-workers 8`.

**A rented 32-core box was tried for this and is not needed**, which is worth writing down because
the opposite is easy to assume: TLC does parallelise, but this spec is nowhere near large enough
for that to matter. The shipped cell takes 3s on the desktop and 5s on the box — JVM startup
dominates at that size — and the deep cell is about a minute either way. Where more machine DOES
help is GenMC, and not through cores: its verification is single-threaded, so what a rented box
buys there is uninterrupted wall time (see `key_set_exactly_once_3t`, whose bound is 1 because
2 and above exceeded a 580s budget, and which completes clean at bound 2 in 847s given the room).

The counterexample is the shape the invariant exists for: `t2` holds index 0 and `t1` holds index
1 — both in segment 0, neither created yet — while `t3` claims index 2, creates only segment 1,
writes and publishes, so the mark reaches 3 with segment 0 absent.

**Two threads are not enough** and the calibration says so: at two threads the broken protocol is
also clean (247 distinct states, identical to the shipped arm), because each thread holds one
claim at a time and index 2 cannot be claimed until a segment-0 holder has published. The bound
has to admit two outstanding claims in the lower segment plus one above it.
