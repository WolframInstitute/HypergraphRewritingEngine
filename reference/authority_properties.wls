#!/usr/bin/env wolframscript
(* Which MultiwaySystem properties return the state set WITHOUT building a Graph?

   The engine is timed at its C++ API: no serialization, no marshalling, no graph object. Timing
   the reference through "StatesGraph" therefore charges it for work the engine side never does
   -- Graph construction, which on this project's own measurements is the dominant cost of a
   Wolfram-side step. A ratio taken that way is inflated by an amount nobody has measured.

   This lists the properties the object offers and times the candidates that could return the
   state set as data, so the comparison can be re-based on one that does. It asserts nothing; it
   prints what each property costs and what it returns, and a property that errors says so.
*)

mwSystem = PacletSymbol["Wolfram/Multicomputation", "Wolfram`Multicomputation`MultiwaySystem"];

rules = {{{1, 2}, {2, 3}} -> {{1, 2}, {2, 4}, {4, 3}}};
init  = {{1, 2}, {2, 3}};
depth = 5;

multi = mwSystem[rules, {init}];

Print["PROPERTIES: ", ToString[multi["Properties"]]];

try[name_] := Module[{t, r, head, size},
  {t, r} = AbsoluteTiming[Quiet[Check[multi[name, depth,
      "CanonicalStateFunction" -> "CanonicalHypergraph"], $Failed]]];
  If[r === $Failed,
    Print["PROP ", name, " -> FAILED"],
    head = ToString[Head[r]];
    size = Which[
      GraphQ[r], "graph V=" <> ToString[VertexCount[r]] <> " E=" <> ToString[EdgeCount[r]],
      ListQ[r],  "list n=" <> ToString[Length[r]],
      AssociationQ[r], "assoc n=" <> ToString[Length[r]],
      True, "scalar"];
    Print["PROP ", name, " ms=", ToString[Round[t*1000, 0.1]], " head=", head, " ", size]]];

Scan[try, {"StatesGraph", "StatesList", "States", "StateList", "AllStatesList",
           "StatesCountsList", "CausalGraph", "EvolutionGraph"}];

Print["AUTHORITY_PROPERTIES_DONE"];
