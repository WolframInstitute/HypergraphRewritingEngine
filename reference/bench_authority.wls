#!/usr/bin/env wolframscript
(* Time the authoritative Wolfram/Multicomputation MultiwaySystem on the workload the C++
   engine's own depth sweep runs, so the two wall times are comparable.

   The rule and initial condition are the corpus's wolfram-2to4, which is what
   tools/quotient_reconstruction_cost_probe times on the engine side. In this paclet's
   convention a rule is lhs -> rhs over edge lists of integer vertex names, and a name on the
   right that does not appear on the left is a fresh vertex:
     {{1,2},{2,3}} -> {{1,2},{2,4},{4,3}},  init {{1,2},{2,3}}

   The state count is printed with every row and is the check that the two sides computed the
   same thing: a timing comparison between different state spaces is not a comparison.

   TWO ROWS PER DEPTH, and the difference between them is the point.
   
     AUTH  the authority, timed the only way its API allows. Every property it exposes is
           graph-valued -- the full list is {AllStatesBranchialGraph, BranchialGraph,
           CausalBranchialGraph, CausalEvolutionGraph, CausalGraph, CausalStatesGraph,
           EvolutionCausalGraph, EvolutionEventsGraph, EvolutionGraph, Graph, Multi, Properties,
           StatesGraph, TokenEventGraph, Type} -- so there is no way to ask it for the state set
           as DATA. Its time therefore includes building Wolfram Graph objects, and no
           subtraction can remove that from the outside.
   
     HGEV  this engine reached the same way a user reaches it: through the paclet, which
           serializes to the worker process, evolves, deserializes and builds the same kind of
           Wolfram structures. This is the LIKE-FOR-LIKE row, and it is the one a ratio against
           AUTH may be taken from, because both sides pay for the Wolfram layer.
   
   The engine's C++ API time is measured separately (tools/quotient_reconstruction_cost_probe)
   and is NOT comparable to AUTH: it excludes marshalling and graph construction, which AUTH
   cannot exclude.
   
   One row per depth, printed as it completes, so a run cut short by the timeout still leaves
   the depths that finished:
     AUTH d=<depth> states=<n> causal=<n> ms=<median of reps>
     HGEV d=<depth> states=<n> causal=<n> ms=<median of reps>

   Usage: wolframscript -file bench_authority.wls [maxDepth] [reps]
*)

maxDepth = If[Length[$ScriptCommandLine] >= 2, ToExpression[ToString[$ScriptCommandLine[[2]]]], 5];
reps     = If[Length[$ScriptCommandLine] >= 3, ToExpression[ToString[$ScriptCommandLine[[3]]]], 3];

mwSystem = PacletSymbol["Wolfram/Multicomputation", "Wolfram`Multicomputation`MultiwaySystem"];

(* This engine through its Wolfram entry point, so the comparison is Wolfram-to-Wolfram. *)
repo = DirectoryName[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

hgRules = {{{1, 2}, {2, 3}} -> {{1, 2}, {2, 4}, {4, 3}}};

rules = {{{1, 2}, {2, 3}} -> {{1, 2}, {2, 4}, {4, 3}}};
init  = {{1, 2}, {2, 3}};

(* One measurement: build the states graph and the causal graph at this depth, from a fresh
   MultiwaySystem object, so nothing is cached across depths or repetitions. *)
once[d_] := Module[{multi, sg, cg, t},
  t = First @ AbsoluteTiming[
    multi = mwSystem[rules, {init}];
    sg = multi["StatesGraph", d, "CanonicalStateFunction" -> "CanonicalHypergraph"];
    cg = multi["CausalGraph", d, "CanonicalStateFunction" -> "CanonicalHypergraph"];
  ];
  {t * 1000., VertexCount[sg], EdgeCount[cg]}
];

(* The same workload through the paclet: states graph and causal edges, so the two rows ask for
   the same artifacts and neither is charged for something the other skipped. *)
hgOnce[d_] := Module[{t, r},
  {t, r} = AbsoluteTiming[
    HGEvolve[hgRules, init, d, {"StatesGraph", "CausalEdges"},
      "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full"]];
  {t * 1000., If[GraphQ[r["StatesGraph"]], VertexCount[r["StatesGraph"]], 0],
   If[ListQ[r["CausalEdges"]], Length[r["CausalEdges"]], 0]}
];

Do[
  Module[{runs, ms, states, causal, hgRuns, hgMs},
    runs = Table[once[d], reps];
    ms = Median[runs[[All, 1]]];
    states = runs[[1, 2]];
    causal = runs[[1, 3]];
    Print["AUTH d=", d, " states=", states, " causal=", causal, " ms=", ToString[Round[ms, 0.1]]];
    hgRuns = Table[hgOnce[d], reps];
    hgMs = Median[hgRuns[[All, 1]]];
    Print["HGEV d=", d, " states=", hgRuns[[1, 2]], " causal=", hgRuns[[1, 3]],
          " ms=", ToString[Round[hgMs, 0.1]]];
  ],
  {d, 2, maxDepth}
];

Print["AUTH_DONE"];
