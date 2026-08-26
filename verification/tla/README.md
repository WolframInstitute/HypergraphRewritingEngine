# TLA+ layer — algorithm-level correctness at arbitrary N (#80)

GenMC (../genmc) checks the ACTUAL C++ against RC11 at small thread/op bounds.
This layer checks a MODEL of the algorithm at unbounded worker count: every
in-flight protocol step lives in a `pending` bag and any enabled step may fire,
so interleavings subsume every thread count — the bound TLC pays is state count.
The cost is drift: the model is a transcription, and a fix to the code does not
fix the model (see `job_system_no_lost_wakeup` in the GenMC layer for the same
lesson). Each spec header names the code it transcribes; changing that code
means re-checking the transcription.

## MatchForwarding — forwarding completeness (#80 target 1)

Models `push_match_to_children` + the registration-time ancestor pull
(`forward_existing_parent_matches`) + the `claim_match` dedup, from
`hypergraph/src/parallel_evolution.cpp`. Property: at quiescence every state
holds every match discovered at an ancestor that overlaps none of the edges
consumed on the path down. This is the property whose silent failure produced
#74 and #76 — a lost match deletes its whole subtree while the run stays
self-consistent.

Run (needs `~/tla/tla2tools.jar`, any Java ≥ 11):

    cd verification/tla
    java -cp ~/tla/tla2tools.jar tlc2.TLC -workers 8 -deadlock \
         -config <cell>.cfg MCMatchForwarding.tla

The four cells (2026-08-01, tla2tools 2.19, exhaustive at the MC bound —
4 states, 3 matches, 3 edges, mid-chain original enabling a grandchild):

| cell | OwnershipFix | BatchedGate | verdict | distinct states |
|---|---|---|---|---|
| MCMatchForwarding | TRUE | TRUE | PASS | 117,005 |
| MCMatchForwardingBatchedBroken | FALSE | TRUE | PASS | 85,777 |
| MCMatchForwardingEagerFix | TRUE | FALSE | PASS | 479,005 |
| MCMatchForwardingEagerBroken | FALSE | FALSE | **ForwardingComplete VIOLATED** | trace depth 13 |

What the matrix establishes:

- The SHIPPED protocol (claim-winner owns store AND propagation, batched
  submission) is forwarding-complete at this bound for every interleaving —
  arbitrary worker count by construction.
- Removing the ownership invariant reproduces the D1/#76 loss class, and TLC's
  counterexample is the exact recorded shape: chain s0 -> s1 -> s2, match mB
  discovered at the root after s2's pull ran; s1's pull wins the (mB, s1) claim
  and stores without propagating; the root's push then sees the claim taken and
  skips its recursion; s2 ends quiescent with mB valid for it and not stored.
  The model REACHES the failure it exists to exclude — that is the calibration.
- The batched gate alone (BatchedBroken cell) masks the broken pull at this
  bound: with a state's own matching complete before its children exist, and
  pulls walking the full ancestor chain, every original match is pull-visible.
  This bears on #77's open question about which forwarding work is load-bearing
  under batched submission — at this bound the pull's propagation duty is
  exercised only under eager ordering. A bound, not a proof.

Model scope: sequentially consistent memory (RC11 is GenMC's job); scans atomic
at fire time (the real `LockFreeList::for_each` tolerates appends mid-walk; the
kept discrepancy is exactly the documented miss window — elements registered
after a scan are covered by the other mechanism); `claim_match` modeled as an
exact (match, state) set, so #74's 64-bit-hash collision class is OUT of model;
no transition thinning (sampling draws are downstream of store+propagate in the
code, so completeness of storage is unaffected by them).

Second target (#80): quiescence liveness — not yet modeled.

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
