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
