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

   THE PROPERTY. At quiescence the claimed set is exactly the nodes whose SHORTEST-PATH depth is
   below the budget. That is what makes the output a function of the rules and the budget rather
   than of the order paths were found, and it is the contract a truncated run depends on -- a run
   whose budget is below the closure depth returns a subset, and the subset has to be the same
   subset every time.

   LiveMinimum is the shipped rule: a child's arrival depth comes from the parent's CURRENT
   label. FALSE models deriving it from the depth the parent carried when it was claimed, which
   freezes an early, longer path into every descendant -- the defect this excludes, and the one
   TLC reports for the Broken configuration. *)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Budget,        (* steps; nodes at or past this are frontier, not expanded *)
          LiveMinimum    (* TRUE = shipped, FALSE = derive child depth from the claim depth *)

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
          claimed      (* nodes whose expansion claim has been taken *)

vars == <<depth, claimDepth, claimed>>

TypeOK == /\ depth \in [Nodes -> 0..Infinity]
          /\ claimDepth \in [Nodes -> 0..Infinity]
          /\ claimed \subseteq Nodes

Init == /\ depth = [n \in Nodes |-> IF n = Root THEN 0 ELSE Infinity]
        /\ claimDepth = [n \in Nodes |-> IF n = Root THEN 0 ELSE Infinity]
        /\ claimed = {Root}

(* The depth a parent offers its children. The shipped engine reads the parent's live label; the
   broken variant reads the label frozen at claim time. *)
Offered(n) == IF LiveMinimum THEN depth[n] ELSE claimDepth[n]

(* A claimed parent offers a child one past its own depth. The child takes it if it improves,
   and is claimed only if that puts it inside the budget -- at or past it the child is frontier,
   kept for a shorter path that may arrive later. *)
Relax(n, c) ==
    /\ n \in claimed
    /\ c \in Children(n)
    /\ Offered(n) + 1 < depth[c]
    /\ depth' = [depth EXCEPT ![c] = Offered(n) + 1]
    /\ IF Offered(n) + 1 < Budget /\ c \notin claimed
         THEN /\ claimed' = claimed \cup {c}
              /\ claimDepth' = [claimDepth EXCEPT ![c] = Offered(n) + 1]
         ELSE /\ UNCHANGED claimed
              /\ claimDepth' = claimDepth

(* A node already labelled inside the budget but not yet claimed is claimed. This is the
   deferred frontier being pulled in by a lowering that arrived after it was first seen. *)
ClaimDeferred(n) ==
    /\ n \notin claimed
    /\ depth[n] < Budget
    /\ claimed' = claimed \cup {n}
    /\ claimDepth' = [claimDepth EXCEPT ![n] = depth[n]]
    /\ UNCHANGED depth

Next == \/ \E n \in Nodes, c \in Nodes : Relax(n, c)
        \/ \E n \in Nodes : ClaimDeferred(n)

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

Quiescent == /\ \A n \in Nodes : \A c \in Children(n) :
                 n \in claimed => ~(Offered(n) + 1 < depth[c])
             /\ \A n \in Nodes : ~(n \notin claimed /\ depth[n] < Budget)

(* THE CONTRACT: at quiescence the expanded set is decided by the graph and the budget alone. *)
ClaimedIsShortestPathBounded ==
    Quiescent => \A n \in Nodes : (n \in claimed) <=> (SP(n) < Budget)

(* A label is never below the true shortest path: relaxation may lag, never overshoot. *)
DepthNeverBelowShortestPath == \A n \in Nodes : depth[n] >= SP(n)
=============================================================================
