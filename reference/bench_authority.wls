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
   
     REF   OUR OWN reference implementation, reference/MultiwayReference.wl -- the brute-force
           Wolfram-language multiway evolver this project validates its engine against. It is
           the second thing the engine must be compared with: the authority answers what a
           user's alternative costs, and the reference answers what a DIRECT, unoptimised
           implementation of the same definition costs, with no graph construction in the way.
           It returns DATA, not Graph objects, so its time is the closest Wolfram-side analogue
           of the engine's native number.

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

   Usage: wolframscript -file bench_authority.wls [maxDepth] [reps] [minDepth]

   minDepth EXISTS BECAUSE THE DEEP POINTS ARE THE EXPENSIVE ONES AND THE SHALLOW ONES ARE NOT
   FREE TO REPEAT. The authority's cost grows about sevenfold per depth, so reaching depth 7
   from depth 2 spends minutes re-measuring depths already recorded, and a kernel that exits on
   a licence error part-way through loses the deep point and keeps only what it had already
   printed. Starting at the depth actually wanted makes a deep point one short run instead of a
   long one that must survive the whole ladder.
*)

maxDepth = If[Length[$ScriptCommandLine] >= 2, ToExpression[ToString[$ScriptCommandLine[[2]]]], 5];
reps     = If[Length[$ScriptCommandLine] >= 3, ToExpression[ToString[$ScriptCommandLine[[3]]]], 3];
minDepth = If[Length[$ScriptCommandLine] >= 4, ToExpression[ToString[$ScriptCommandLine[[4]]]], 2];

mwSystem = PacletSymbol["Wolfram/Multicomputation", "Wolfram`Multicomputation`MultiwaySystem"];

(* This engine through its Wolfram entry point, so the comparison is Wolfram-to-Wolfram. *)
repo = DirectoryName[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

(* The versions of both paclets in the comparison, printed for the table's provenance line: a
   ratio against the authority is a statement about one release of it. PacletFind resolves the
   loaded paclet (PacletDirectoryLoad'ed ones included); PacletObject on a bare name can answer
   Missing for a directory-loaded paclet. *)
pacletVersion[name_] := Module[{found = PacletFind[name]},
  If[found === {}, "unknown", First[found]["Version"]]];
Print["PACLET Wolfram/Multicomputation ", pacletVersion["Wolfram/Multicomputation"]];
Print["PACLET HypergraphRewriting ", pacletVersion["HypergraphRewriting"]];

(* OUR reference implementation, loaded from source. Its rules are {lhs, rhs} pairs rather than
   Rule objects, and it returns per-step tuples rather than graphs. *)
Get[FileNameJoin[{repo, "reference", "MultiwayReference.wl"}]];
refRules = {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 4}, {4, 3}}}};

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

(* The reference, asked for the same two quantities. It computes DATA -- no Graph objects -- so
   this row is the Wolfram-side measurement that is not charged for graph construction, and the
   one the engine's native number is most nearly commensurable with. *)
refOnce[d_] := Module[{t, r, st, cz},
  {t, r} = AbsoluteTiming[
    MultiwayReference`MultiwayEvolve[refRules, init, d,
      "StateCanonicalization" -> "Canonical", "EventCanonicalization" -> "States"]];
  (* PerStep's last row is the cumulative answer at this depth: "states" is the canonical state
     count and "causal" the causal-edge count, which are the two the other rows report. *)
  st = If[Length[r["PerStep"]] > 0, Last[r["PerStep"]]["states"], 0];
  cz = If[Length[r["PerStep"]] > 0, Last[r["PerStep"]]["causal"], 0];
  {t * 1000., st, cz}
];

Do[
  Module[{runs, ms, states, causal, hgRuns, hgMs, refRuns, refMs},
    runs = Table[once[d], reps];
    ms = Median[runs[[All, 1]]];
    states = runs[[1, 2]];
    causal = runs[[1, 3]];
    Print["AUTH d=", d, " states=", states, " causal=", causal, " ms=", ToString[Round[ms, 0.1]]];
    refRuns = Table[refOnce[d], reps];
    refMs = Median[refRuns[[All, 1]]];
    Print["REF  d=", d, " states=", refRuns[[1, 2]], " causal=", refRuns[[1, 3]],
          " ms=", ToString[Round[refMs, 0.1]]];
    hgRuns = Table[hgOnce[d], reps];
    hgMs = Median[hgRuns[[All, 1]]];
    Print["HGEV d=", d, " states=", hgRuns[[1, 2]], " causal=", hgRuns[[1, 3]],
          " ms=", ToString[Round[hgMs, 0.1]]];
  ],
  {d, minDepth, maxDepth}
];

Print["AUTH_DONE"];
