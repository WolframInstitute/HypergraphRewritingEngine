--------------------------- MODULE MCMatchForwarding ---------------------------
(* TLC harness for MatchForwarding: the concrete bounded universe. Function-valued
   constants cannot live in a .cfg, so they are bound here by instantiation.
   OwnershipFix comes from the .cfg: TRUE = shipped protocol (expect PASS),
   FALSE = pre-4df8c6d pull (expect a ForwardingComplete violation). *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS OwnershipFix, BatchedGate

(* mA and mB originate at the root; mC originates at s1 -- the mid-chain match a
   grandchild is built FROM, so a chain s0 -> s1 -> g can exist while mB is still
   undiscovered. That is the D1 shape: g's pull runs before mB exists anywhere,
   then the pull at s1 claims mB against the root's push; if that claim-winner
   does not propagate, nothing ever reaches g. *)
MCStateIds    == {"s0", "s1", "s2", "s3"}
MCRoot        == "s0"
MCMatches     == {"mA", "mB", "mC"}
MCEdges       == {"e1", "e2", "e3"}
MCMatchEdges  == [m \in MCMatches |->
                   CASE m = "mA" -> {"e1"} [] m = "mB" -> {"e2"} [] OTHER -> {"e3"}]
MCOrigMatches == [s \in MCStateIds |->
                   CASE s = "s0" -> {"mA", "mB"} [] s = "s1" -> {"mC"} [] OTHER -> {}]

VARIABLES exists, parentOf, childrenOf, stored, claimed, discovered,
          matchingDone, pending

INSTANCE MatchForwarding WITH
  StateIds    <- MCStateIds,
  Root        <- MCRoot,
  Matches     <- MCMatches,
  Edges       <- MCEdges,
  MatchEdges  <- MCMatchEdges,
  OrigMatches <- MCOrigMatches

================================================================================
