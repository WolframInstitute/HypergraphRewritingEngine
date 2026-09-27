---------------------------- MODULE MatchForwarding ----------------------------
(* Algorithm-level model of the engine's match inheritance, from
   hypergraph/src/parallel_evolution.cpp: register_child_with_parent,
   inherit_from_parent, the drain in note_match_task_done, and the claim_match dedup.

   THE PROTOCOL. A state finds its own matches (Discover). It drains once every own match is
   found and, unless it is the root, once it has inherited. At the drain it publishes `drained`
   and then scans its children; a child is published in its parent's children list and then
   reads `drained`. Whichever of the two sees the other hands the child the parent's stored
   matches that overlap none of the edges the child's transition consumed, once (the child's
   `inherited` claim). Children are created from any stored match of the parent, at any time,
   so a parent is usually still matching when its first children exist.

   THE PROPERTY. At quiescence every state holds every match VALID for it: discoverable at the
   state itself, or discoverable at an ancestor and disjoint from every edge consumed on the
   path down. And at every drain the draining state already holds its whole valid set, which is
   what validate_state_at_drain checks in the engine.

   THE CONCURRENCY MODEL. Every in-flight step lives in a `pending` set; any enabled element
   may fire next, so interleavings subsume every worker count. The two halves of the
   rendezvous are separate steps: publishing the child and reading `drained` on one side,
   setting `drained` and scanning the list on the other. Memory is sequentially consistent
   here, which is what the seq_cst fences of rv::ChildInheritance provide; RC11-level
   questions live in verification/genmc.

   THE CALIBRATION. RendezvousFix = FALSE reads `drained` BEFORE publishing the child. A drain
   that falls between the two steps sees no child and is not seen, the child never inherits,
   and the invariant must fail. *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
  StateIds,       \* the bounded universe of state identities
  Root,           \* the initial state, exists from the start
  Matches,        \* abstract match identities
  MatchEdges,     \* [Matches -> SUBSET Edges]: the edges a match binds
  OrigMatches,    \* [StateIds -> SUBSET Matches]: matches a state finds by its own matching
  Edges,
  RendezvousFix   \* BOOLEAN: publish then read (shipped) or read then publish (broken)

ASSUME Root \in StateIds
ASSUME MatchEdges \in [Matches -> SUBSET Edges]
ASSUME OrigMatches \in [StateIds -> SUBSET Matches]

None == "none"
ASSUME None \notin StateIds

VARIABLES
  exists,       \* SUBSET StateIds: created states
  parentOf,     \* [StateIds -> [par : StateIds \cup {None}, consumed : SUBSET Edges]]
  childrenOf,   \* [StateIds -> SUBSET [c : StateIds, consumed : SUBSET Edges]]: published
  stored,       \* [StateIds -> SUBSET Matches]: state_matches_
  claimed,      \* SUBSET (Matches \X StateIds): claim_match's exactly-once set
  discovered,   \* [StateIds -> SUBSET Matches]: own discoveries already made
  drained,      \* SUBSET StateIds: MatchJoin::drained
  inherited,    \* SUBSET StateIds: MatchJoin::inherited
  pending       \* SUBSET of ops

vars == <<exists, parentOf, childrenOf, stored, claimed, discovered, drained, inherited,
          pending>>

InheritOp(p, ch)   == [type |-> "inherit", p |-> p, ch |-> ch]
DrainScanOp(s)     == [type |-> "scan", s |-> s]
RegCheckOp(p, ch)  == [type |-> "regcheck", p |-> p, ch |-> ch]
RegPublishOp(p, ch, saw) == [type |-> "regpublish", p |-> p, ch |-> ch, saw |-> saw]

Overlaps(m, es) == MatchEdges[m] \cap es /= {}

RECURSIVE ChainRec(_, _, _)
ChainRec(s, acc, depth) ==
  IF depth = 0 \/ parentOf[s].par = None
  THEN {}
  ELSE LET p == parentOf[s].par
           acc2 == acc \cup parentOf[s].consumed
       IN {[a |-> p, acc |-> acc2]} \cup ChainRec(p, acc2, depth - 1)

AncestorsWithConsumed(s) == ChainRec(s, {}, Cardinality(StateIds))

Drainable(s) ==
  /\ discovered[s] = OrigMatches[s]
  /\ (s = Root \/ s \in inherited)

Init ==
  /\ exists = {Root}
  /\ parentOf = [s \in StateIds |-> [par |-> None, consumed |-> {}]]
  /\ childrenOf = [s \in StateIds |-> {}]
  /\ stored = [s \in StateIds |-> {}]
  /\ claimed = {}
  /\ discovered = [s \in StateIds |-> {}]
  /\ drained = {}
  /\ inherited = {}
  /\ pending = {}

(* complete_match: claim and store one of the state's own matches. *)
Discover(s, m) ==
  /\ s \in exists
  /\ m \in OrigMatches[s] \ discovered[s]
  /\ discovered' = [discovered EXCEPT ![s] = @ \cup {m}]
  /\ claimed' = claimed \cup {<<m, s>>}
  /\ stored' = [stored EXCEPT ![s] = @ \cup {m}]
  /\ UNCHANGED <<exists, parentOf, childrenOf, drained, inherited, pending>>

(* A rewrite of a stored match creates a child. Shipped order: publish in the parent's list,
   then read `drained` (RegCheckOp). Broken order: read `drained` first (RegPublishOp carries
   what was read), publish after. *)
CreateChild(p, m, c) ==
  /\ p \in exists
  /\ m \in stored[p]
  /\ c \in StateIds \ exists
  /\ LET ch == [c |-> c, consumed |-> MatchEdges[m]]
     IN /\ exists' = exists \cup {c}
        /\ parentOf' = [parentOf EXCEPT ![c] = [par |-> p, consumed |-> MatchEdges[m]]]
        /\ IF RendezvousFix
           THEN /\ childrenOf' = [childrenOf EXCEPT ![p] = @ \cup {ch}]
                /\ pending' = pending \cup {RegCheckOp(p, ch)}
           ELSE /\ childrenOf' = childrenOf
                /\ pending' = pending \cup {RegPublishOp(p, ch, p \in drained)}
  /\ UNCHANGED <<stored, claimed, discovered, drained, inherited>>

FireRegCheck(op) ==
  /\ op \in pending /\ op.type = "regcheck"
  /\ pending' = (pending \ {op}) \cup
       (IF op.p \in drained THEN {InheritOp(op.p, op.ch)} ELSE {})
  /\ UNCHANGED <<exists, parentOf, childrenOf, stored, claimed, discovered, drained, inherited>>

FireRegPublish(op) ==
  /\ op \in pending /\ op.type = "regpublish"
  /\ childrenOf' = [childrenOf EXCEPT ![op.p] = @ \cup {op.ch}]
  /\ pending' = (pending \ {op}) \cup
       (IF op.saw THEN {InheritOp(op.p, op.ch)} ELSE {})
  /\ UNCHANGED <<exists, parentOf, stored, claimed, discovered, drained, inherited>>

(* The drain: publish `drained`, then scan the children list as a separate step. *)
Drain(s) ==
  /\ s \in exists
  /\ s \notin drained
  /\ Drainable(s)
  /\ drained' = drained \cup {s}
  /\ pending' = pending \cup {DrainScanOp(s)}
  /\ UNCHANGED <<exists, parentOf, childrenOf, stored, claimed, discovered, inherited>>

FireDrainScan(op) ==
  /\ op \in pending /\ op.type = "scan"
  /\ pending' = (pending \ {op}) \cup {InheritOp(op.s, ch) : ch \in childrenOf[op.s]}
  /\ UNCHANGED <<exists, parentOf, childrenOf, stored, claimed, discovered, drained, inherited>>

(* inherit_from_parent: the child's claim makes the second arrival a no-op. *)
FireInherit(op) ==
  /\ op \in pending /\ op.type = "inherit"
  /\ LET c == op.ch.c
         gets == {m \in stored[op.p] : ~Overlaps(m, op.ch.consumed) /\ <<m, c>> \notin claimed}
     IN IF c \in inherited
        THEN /\ pending' = pending \ {op}
             /\ UNCHANGED <<stored, claimed, inherited>>
        ELSE /\ inherited' = inherited \cup {c}
             /\ claimed' = claimed \cup {<<m, c>> : m \in gets}
             /\ stored' = [stored EXCEPT ![c] = @ \cup gets]
             /\ pending' = pending \ {op}
  /\ UNCHANGED <<exists, parentOf, childrenOf, discovered, drained>>

Next ==
  \/ \E s \in StateIds, m \in Matches : Discover(s, m)
  \/ \E p \in StateIds, m \in Matches, c \in StateIds : CreateChild(p, m, c)
  \/ \E s \in StateIds : Drain(s)
  \/ \E op \in pending :
       FireRegCheck(op) \/ FireRegPublish(op) \/ FireDrainScan(op) \/ FireInherit(op)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

--------------------------------------------------------------------------------
ValidFor(s) ==
  OrigMatches[s] \cup
  {m \in Matches :
     \E aw \in AncestorsWithConsumed(s) :
        m \in OrigMatches[aw.a] /\ ~Overlaps(m, aw.acc)}

(* Nothing in flight, every own match found, every state that can drain has drained. A child
   whose handoff was lost cannot drain, so it counts here as settled with what it holds. *)
Quiescent ==
  /\ pending = {}
  /\ \A s \in exists : discovered[s] = OrigMatches[s] /\ (Drainable(s) => s \in drained)

ForwardingComplete ==
  Quiescent => \A s \in exists : ValidFor(s) \subseteq stored[s]

(* validate_state_at_drain: a state drains holding its whole valid set. *)
CompleteAtDrain ==
  \A s \in drained : ValidFor(s) \subseteq stored[s]

StoredAreClaimed ==
  \A s \in StateIds : \A m \in stored[s] : <<m, s>> \in claimed

================================================================================
