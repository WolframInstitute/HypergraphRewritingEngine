---------------------------- MODULE MatchForwarding ----------------------------
(* Algorithm-level model of the engine's match-forwarding protocol (#80 target 1):
   push_match_to_children + the registration-time ancestor pull + the claim_match
   dedup, under BATCHED submission (the default: a state's own matching completes
   before any of its rewrites can create children).

   THE PROPERTY. Forwarding completeness: at quiescence, every state holds every
   match that is VALID for it -- discovered at some ancestor (or at the state
   itself) and overlapping none of the edges consumed on the path down. A lost
   match deletes its whole subtree while the run stays self-consistent, which is
   why this property is checked by a model rather than by output inspection
   (#74 and #76 were both silent instances).

   THE CONCURRENCY MODEL. Every in-flight protocol step lives in a `pending` bag;
   any enabled element may fire next. Interleavings of the bag subsume every
   worker count, so the model is arbitrary-N by construction -- the bound TLC
   pays is state count, not thread count. Actions are atomic at the granularity
   of one scan (a push scans the children present AT FIRE TIME; a pull scans the
   matches present at fire time): the real scans tolerate concurrent appends
   inside the list walk (LockFreeList for_each), and what the coarse scan keeps
   is exactly the documented miss window -- an element registered after the scan
   is NOT seen by it and must be covered by the other mechanism. Memory is
   sequentially consistent here; RC11-level questions live in verification/genmc.

   THE OWNERSHIP SWITCH. OwnershipFix = TRUE models the shipped invariant
   (CLAIM-WINNER OWNS THE MATCH AT THIS NODE: a pull that wins a claim stores the
   match AND propagates to already-registered grandchildren). FALSE models the
   pre-4df8c6d protocol, where the pull claimed without propagating; the racing
   push sees the hash taken and skips, so a grandchild whose pull already
   completed is covered by nobody. The broken variant MUST fail the invariant --
   it is the calibration that the model can reach the loss class it exists for. *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  StateIds,      \* the bounded universe of state identities
  Root,          \* the initial state, exists from the start
  Matches,       \* abstract match identities
  MatchEdges,    \* [Matches -> SUBSET Edges]: the edges a match binds
  OrigMatches,   \* [StateIds -> SUBSET Matches]: where a match is discoverable
  Edges,
  OwnershipFix,  \* BOOLEAN: pull claim-winner propagates (shipped) or not (broken)
  BatchedGate    \* BOOLEAN: children only after the parent's matching completes
                 \* (batched submission, the default) vs any time (eager)

ASSUME Root \in StateIds
ASSUME MatchEdges \in [Matches -> SUBSET Edges]
ASSUME OrigMatches \in [StateIds -> SUBSET Matches]

(* State identities are strings in the model configs; "none" is reserved. *)
None == "none"
ASSUME None \notin StateIds

VARIABLES
  exists,        \* SUBSET StateIds: created states
  parentOf,      \* [StateIds -> [par : StateIds \cup {None}, consumed : SUBSET Edges]]
  childrenOf,    \* [StateIds -> SUBSET [c : StateIds, consumed : SUBSET Edges]]
  stored,        \* [StateIds -> SUBSET Matches]: state_matches_
  claimed,       \* SUBSET (Matches \X StateIds): claim_match's exactly-once set
  discovered,    \* [StateIds -> SUBSET Matches]: original discoveries already fired
  matchingDone,  \* SUBSET StateIds: batched phase gate
  pending        \* SUBSET of ops (idempotent, so a set is enough)

vars == <<exists, parentOf, childrenOf, stored, claimed, discovered,
          matchingDone, pending>>

PushOp(s, m) == [type |-> "push", s |-> s, m |-> m]
PullOp(c)    == [type |-> "pull", c |-> c]

Overlaps(m, es) == MatchEdges[m] \cap es /= {}

(* Ancestor chain of s (s excluded), with the consumed edges accumulated from s
   upward -- the pull's accumulated_consumed. Recursion bounded by |StateIds|. *)
RECURSIVE ChainRec(_, _, _)
ChainRec(s, acc, depth) ==
  IF depth = 0 \/ parentOf[s].par = None
  THEN {}
  ELSE LET p == parentOf[s].par
           acc2 == acc \cup parentOf[s].consumed
       IN {[a |-> p, acc |-> acc2]} \cup ChainRec(p, acc2, depth - 1)

AncestorsWithConsumed(s) == ChainRec(s, {}, Cardinality(StateIds))

Init ==
  /\ exists = {Root}
  /\ parentOf = [s \in StateIds |-> [par |-> None, consumed |-> {}]]
  /\ childrenOf = [s \in StateIds |-> {}]
  /\ stored = [s \in StateIds |-> {}]
  /\ claimed = {}
  /\ discovered = [s \in StateIds |-> {}]
  /\ matchingDone = {}
  /\ pending = {}

(* SINK completes a match at s: claim, store, and push toward children
   (PushSite::Discovery). Under batched submission s has no children yet, but the
   push is modeled unconditionally, as in the code. *)
Discover(s, m) ==
  /\ s \in exists
  /\ m \in OrigMatches[s] \ discovered[s]
  /\ discovered' = [discovered EXCEPT ![s] = @ \cup {m}]
  /\ claimed' = claimed \cup {<<m, s>>}
  /\ stored' = [stored EXCEPT ![s] = @ \cup {m}]
  /\ pending' = pending \cup {PushOp(s, m)}
  /\ UNCHANGED <<exists, parentOf, childrenOf, matchingDone>>

(* Batched gate: a state's own matching completes only after every discoverable
   match at it has fired. Children of s cannot be created before this. *)
FinishMatching(s) ==
  /\ s \in exists
  /\ s \notin matchingDone
  /\ discovered[s] = OrigMatches[s]
  /\ matchingDone' = matchingDone \cup {s}
  /\ UNCHANGED <<exists, parentOf, childrenOf, stored, claimed, discovered, pending>>

(* A rewrite of a stored match creates a child: parent link FIRST, then the
   children-list registration, then the registration-time pull
   (register_child_with_parent + forward_existing_parent_matches). *)
CreateChild(p, m, c) ==
  /\ p \in exists /\ (BatchedGate => p \in matchingDone)
  /\ m \in stored[p]
  /\ c \in StateIds \ exists
  /\ exists' = exists \cup {c}
  /\ parentOf' = [parentOf EXCEPT ![c] = [par |-> p, consumed |-> MatchEdges[m]]]
  /\ childrenOf' = [childrenOf EXCEPT ![p] =
       @ \cup {[c |-> c, consumed |-> MatchEdges[m]]}]
  /\ pending' = pending \cup {PullOp(c)}
  /\ UNCHANGED <<stored, claimed, discovered, matchingDone>>

(* Fire a pending push: scan the children of s present NOW; for each non-overlapping
   child, the claim decides one owner; the winner stores at the child and recurses.
   A child registered after this fires is missed here and covered by its own pull. *)
FirePush(op) ==
  /\ op \in pending /\ op.type = "push"
  /\ LET s == op.s
         m == op.m
         won == {ch \in childrenOf[s] :
                   ~Overlaps(m, ch.consumed) /\ <<m, ch.c>> \notin claimed}
     IN /\ claimed' = claimed \cup {<<m, ch.c>> : ch \in won}
        /\ stored' = [t \in StateIds |->
             IF \E ch \in won : ch.c = t THEN stored[t] \cup {m} ELSE stored[t]]
        /\ pending' = (pending \ {op}) \cup {PushOp(ch.c, m) : ch \in won}
  /\ UNCHANGED <<exists, parentOf, childrenOf, discovered, matchingDone>>

(* Fire a pending pull: walk c's ancestors with the accumulated consumed set; for
   each valid ancestor match not yet claimed for c, the pull wins the claim, stores
   -- and, under the shipped invariant, propagates to c's own children exactly as a
   push would (CLAIM-WINNER OWNS THE MATCH AT THIS NODE). The broken variant stores
   without propagating: the racing push sees the hash taken and skips, leaving
   grandchildren whose pulls already ran covered by nobody. *)
FirePull(op) ==
  /\ op \in pending /\ op.type = "pull"
  /\ LET c == op.c
         candidates == UNION {{[m |-> m2, acc |-> aw.acc] : m2 \in stored[aw.a]} :
                              aw \in AncestorsWithConsumed(c)}
         wins == {av \in candidates :
                    /\ ~Overlaps(av.m, av.acc)
                    /\ <<av.m, c>> \notin claimed}
         wonMatches == {av.m : av \in wins}
     IN /\ claimed' = claimed \cup {<<m2, c>> : m2 \in wonMatches}
        /\ stored' = [stored EXCEPT ![c] = @ \cup wonMatches]
        /\ pending' = (pending \ {op}) \cup
             (IF OwnershipFix THEN {PushOp(c, m2) : m2 \in wonMatches} ELSE {})
  /\ UNCHANGED <<exists, parentOf, childrenOf, discovered, matchingDone>>

Next ==
  \/ \E s \in StateIds, m \in Matches : Discover(s, m)
  \/ \E s \in StateIds : FinishMatching(s)
  \/ \E p \in StateIds, m \in Matches, c \in StateIds : CreateChild(p, m, c)
  \/ \E op \in pending : FirePush(op) \/ FirePull(op)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

--------------------------------------------------------------------------------
(* A match is VALID for s when it is discoverable at s itself, or discoverable at
   an ancestor and disjoint from every edge consumed on the path down. Validity
   is judged against OrigMatches (the ground truth), not against what any list
   holds -- that is what makes the property external to the mechanism. *)
ValidFor(s) ==
  OrigMatches[s] \cup
  {m \in Matches :
     \E aw \in AncestorsWithConsumed(s) :
        m \in OrigMatches[aw.a] /\ ~Overlaps(m, aw.acc)}

(* Quiescence: nothing in flight, every discovery fired, every state's matching
   closed. The engine's wait_for_completion() returns exactly here. *)
Quiescent ==
  /\ pending = {}
  /\ \A s \in exists : discovered[s] = OrigMatches[s] /\ s \in matchingDone

(* THE INVARIANT: at quiescence every existing state holds its whole valid set. *)
ForwardingComplete ==
  Quiescent => \A s \in exists : ValidFor(s) \subseteq stored[s]

(* Sanity bound: claims are unique by construction (claimed is a set of pairs),
   and a stored match is always claimed for that state. *)
StoredAreClaimed ==
  \A s \in StateIds : \A m \in stored[s] : <<m, s>> \in claimed

================================================================================
