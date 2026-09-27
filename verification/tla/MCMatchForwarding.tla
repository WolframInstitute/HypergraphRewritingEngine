--------------------------- MODULE MCMatchForwarding ---------------------------
(* TLC harness for MatchForwarding: the concrete bounded universe. Function-valued
   constants cannot live in a .cfg, so they are bound here by instantiation.
   RendezvousFix comes from the .cfg: TRUE = shipped order (expect PASS), FALSE = the child
   reads `drained` before it is published (expect a violation). *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS RendezvousFix

(* mA and mB are found at the root; mC at s1. mA and mB bind different edges, so a child built
   from one inherits the other, and a grandchild built from mC inherits whichever of mA and mB
   its path did not consume. *)
MCStateIds    == {"s0", "s1", "s2", "s3"}
MCRoot        == "s0"
MCMatches     == {"mA", "mB", "mC"}
MCEdges       == {"e1", "e2", "e3"}
MCMatchEdges  == [m \in MCMatches |->
                   CASE m = "mA" -> {"e1"} [] m = "mB" -> {"e2"} [] OTHER -> {"e3"}]
MCOrigMatches == [s \in MCStateIds |->
                   CASE s = "s0" -> {"mA", "mB"} [] s = "s1" -> {"mC"} [] OTHER -> {}]

VARIABLES exists, parentOf, childrenOf, stored, claimed, discovered, drained, inherited,
          pending

INSTANCE MatchForwarding WITH
  StateIds    <- MCStateIds,
  Root        <- MCRoot,
  Matches     <- MCMatches,
  Edges       <- MCEdges,
  MatchEdges  <- MCMatchEdges,
  OrigMatches <- MCOrigMatches

================================================================================
