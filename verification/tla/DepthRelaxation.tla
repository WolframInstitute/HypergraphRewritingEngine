------------------------------ MODULE DepthRelaxation ------------------------------
(* Which states exist when the step budget is below the closure depth.

   WHY THIS IS TLA+ AND NOT GENMC. The property is not about one structure's memory operations,
   which is what the GenMC harnesses cover -- depth_relax_child_registration already covers the
   publish/read edge between a child registering itself and its parent's depth being lowered.
   The property here is about an ORDERING ACROSS MANY PARTICIPANTS reaching a fixed point: a
   shortest-path relaxation racing a claim that happens at most once, with a budget deciding
   which nodes are expanded at all. That is a protocol, and TLC is state-bounded rather than
   execution-bounded, so it can close it.

   THE PROTOCOL, from parallel_evolution.cpp. A canonical state carries a depth label, lowered
   monotonically by try_lower_explore_depth, and an expansion claim taken at most once by
   claim_canonical_for_expansion. A rewrite gives a child the depth ONE PAST ITS PARENT'S LIVE
   MINIMUM, not past the depth the parent happened to be expanded at, and propagate_explore_depth
   cascades a lowering down the child list. A child at or past the budget is the FRONTIER: it is
   deferred and NOT claimed, precisely so that a shorter path found later can still pull it into
   budget and expand it.

   THE GATE. Claiming a state submits its match task, and submit_match_task defers a task past
   match_budget(). The claim and the gate compare against one bound (match_depth_bound in
   parallel_evolution.cpp): a state claimed and then deferred at the gate is never matched,
   because the resume of a deferred state takes the claim, finds it taken, and stops. A steered
   continuation lowers the gate for one pass (the continuation ceiling) and the next call raises
   it to the run's budget, so Gate starts at PassBudget and RaiseGate is the call after the pass.

   THE PROPERTY. At quiescence, with the gate at the budget, the EXPANDED set is exactly the nodes
   whose SHORTEST-PATH depth is below the budget. That is what makes the output a function of the
   rules and the budget rather than of the order paths were found, and it is the contract a
   truncated run depends on -- a run whose budget is below the closure depth returns a subset,
   and the subset has to be the same subset every time.

   LiveMinimum is the shipped rule: a child's arrival depth comes from the parent's CURRENT
   label. FALSE models deriving it from the depth the parent carried when it was claimed, which
   freezes an early, longer path into every descendant -- the defect the Broken configuration
   reports. OneBound is the shipped rule for the claim: it compares against the gate. FALSE models
   claiming against the run's budget while the gate stands lower, which strands a claimed state
   unmatched -- the defect 0c1b6461 fixed, and the one the SteeredBroken configuration reports. *)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Budget,        (* steps; nodes at or past this are frontier, not expanded *)
          PassBudget,    (* the gate during a steered pass; Budget when there is none *)
          LiveMinimum,   (* TRUE = shipped, FALSE = derive child depth from the claim depth *)
          OneBound       (* TRUE = shipped, FALSE = claim against Budget while the gate is lower *)

Nodes == 1..5
Root  == 1
(* 2 is reachable at depth 1 directly and at depth 2 through 3, so an interleaving exists that
   labels it 2 first and must then relax it to 1. 4 hangs off 2, so 4's fate under the budget
   depends on that relaxation reaching it. *)
Edges == {<<1,2>>, <<1,3>>, <<3,2>>, <<2,4>>, <<4,5>>}

Children(n) == {c \in Nodes : <<n,c>> \in Edges}
Infinity == 99

VARIABLES depth,       (* depth[n] = shortest known, Infinity until first relaxed *)
          claimDepth,  (* the depth a node carried when it was claimed *)
          claimed,     (* nodes whose expansion claim has been taken *)
          expanded,    (* claimed nodes whose match task passed the gate and ran *)
          gate         (* the bound submit_match_task defers at: PassBudget, then Budget *)

vars == <<depth, claimDepth, claimed, expanded, gate>>

TypeOK == /\ depth \in [Nodes -> 0..Infinity]
          /\ claimDepth \in [Nodes -> 0..Infinity]
          /\ claimed \subseteq Nodes
          /\ expanded \subseteq claimed
          /\ gate \in {PassBudget, Budget}

Init == /\ depth = [n \in Nodes |-> IF n = Root THEN 0 ELSE Infinity]
        /\ claimDepth = [n \in Nodes |-> IF n = Root THEN 0 ELSE Infinity]
        /\ claimed = {Root}
        /\ expanded = {Root}
        /\ gate = PassBudget

(* The depth a parent offers its children. The shipped engine reads the parent's live label; the
   broken variant reads the label frozen at claim time. *)
Offered(n) == IF LiveMinimum THEN depth[n] ELSE claimDepth[n]

(* The bound a claim compares against. Shipped: the gate's own. *)
ClaimBound == IF OneBound THEN gate ELSE Budget

(* Claiming submits the match task at once, and the gate either runs it or defers it. A deferred
   task of a claimed node is never run: the resume takes the claim first, and it is taken. *)
Claim(c, d) == /\ claimed' = claimed \cup {c}
               /\ claimDepth' = [claimDepth EXCEPT ![c] = d]
               /\ expanded' = IF d < gate THEN expanded \cup {c} ELSE expanded

(* An expanded parent offers a child one past its own depth. The child takes it if it improves,
   and is claimed only if that puts it inside the bound -- at or past it the child is frontier,
   kept for a shorter path, or a raised gate, that may arrive later. *)
Relax(n, c) ==
    /\ n \in expanded
    /\ c \in Children(n)
    /\ Offered(n) + 1 < depth[c]
    /\ depth' = [depth EXCEPT ![c] = Offered(n) + 1]
    /\ IF Offered(n) + 1 < ClaimBound /\ c \notin claimed
         THEN Claim(c, Offered(n) + 1)
         ELSE UNCHANGED <<claimed, claimDepth, expanded>>
    /\ UNCHANGED gate

(* A node already labelled inside the bound but not yet claimed is claimed. This is the deferred
   frontier being pulled in by a lowering that arrived after it was first seen, or by the resume
   after the gate rose. *)
ClaimDeferred(n) ==
    /\ n \notin claimed
    /\ depth[n] < ClaimBound
    /\ Claim(n, depth[n])
    /\ UNCHANGED <<depth, gate>>

(* The call after a steered pass: the gate returns to the run's budget. *)
RaiseGate == /\ gate < Budget
             /\ gate' = Budget
             /\ UNCHANGED <<depth, claimDepth, claimed, expanded>>

Next == \/ \E n \in Nodes, c \in Nodes : Relax(n, c)
        \/ \E n \in Nodes : ClaimDeferred(n)
        \/ RaiseGate

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Shortest-path depth, computed here as the fixed point rather than taken from the protocol. *)
RECURSIVE SPFrom(_, _)
SPFrom(n, seen) ==
    IF n = Root THEN 0
    ELSE LET preds == {p \in Nodes : <<p,n>> \in Edges /\ p \notin seen}
         IN IF preds = {} THEN Infinity
            ELSE LET vals == {SPFrom(p, seen \cup {n}) : p \in preds}
                 IN 1 + CHOOSE v \in vals : \A w \in vals : v =< w

SP(n) == SPFrom(n, {})

Quiescent == /\ gate = Budget
             /\ \A n \in Nodes : \A c \in Children(n) :
                 n \in expanded => ~(Offered(n) + 1 < depth[c])
             /\ \A n \in Nodes : ~(n \notin claimed /\ depth[n] < ClaimBound)

(* THE CONTRACT: at quiescence the expanded set is decided by the graph and the budget alone. *)
ExpandedIsShortestPathBounded ==
    Quiescent => \A n \in Nodes : (n \in expanded) <=> (SP(n) < Budget)

(* A label is never below the true shortest path: relaxation may lag, never overshoot. *)
DepthNeverBelowShortestPath == \A n \in Nodes : depth[n] >= SP(n)
=============================================================================
