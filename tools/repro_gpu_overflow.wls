#!/usr/bin/env wolframscript
(* Reproduce the persistent-worker CUDA-context poisoning with the reported parameters:
   Full canonicalization + "Debug", climbing steps until the GPU faults, then coming back
   DOWN to a step that worked. If the low step now fails until a kernel reset, the
   persistent engine's CUDA context was poisoned and never recovered. *)
Needs["HypergraphRewriting`"];

rule = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
init = {{1, 2}, {1, 3}};

$engineMsgs = 0;
Internal`AddHandler["Message",
  If[MatchQ[#, Hold[Message[MessageName[HGEvolve, "enginefail" | "enginemsg" | "overflow"], ___], _]],
    $engineMsgs++] &];

test[s_] := Module[{r},
  $engineMsgs = 0;
  r = Quiet@Check[
    HGEvolve[rule, init, s, "Debug", "CanonicalizeStates" -> Full, "TargetDevice" -> "GPU"],
    $Failed, {HGEvolve::enginefail, HGEvolve::enginemsg}];
  Print["steps=", StringPadLeft[ToString[s], 2], "  -> ",
    If[AssociationQ[r], "NumStates=" <> ToString[r["NumStates"]], ToString[r]],
    "   engineMsgs=", $engineMsgs];
  r];

Print["=== climb ==="];
Do[test[s], {s, {4, 6, 8, 10, 12, 14}}];
Print["=== descend (4 and 6 worked on the way up) ==="];
r6 = test[6];
r4 = test[4];

Print[""];
If[AssociationQ[r6] && AssociationQ[r4],
  Print["RECOVERED: low steps work again after the overflow"],
  Print["POISONED: low steps FAIL after the overflow (persistent bad state reproduced)"]];
