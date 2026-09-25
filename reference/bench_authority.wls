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

     SR    SetReplace's WolframModel with "EventSelectionFunction" -> "MultiwaySpacelike" to the
           same number of generations. Its multiway is a token-event graph: states are not
           enumerated or identified, so its event count is its own and is printed beside its time
           rather than compared with the others.

   Usage: wolframscript -file bench_authority.wls [maxDepth] [reps] [minDepth] [rule]
   rule is one of the names in `corpus` below, or All (the default is Chain3, the rule this
   probe was written for). A run that exceeds `limit` seconds is printed as a timeout and the
   deeper depths of that row are skipped.

   minDepth EXISTS BECAUSE THE DEEP POINTS ARE THE EXPENSIVE ONES AND THE SHALLOW ONES ARE NOT
   FREE TO REPEAT. The authority's cost grows about sevenfold per depth, so reaching depth 7
   from depth 2 spends minutes re-measuring depths already recorded, and a kernel that exits on
   a licence error part-way through loses the deep point and keeps only what it had already
   printed. Starting at the depth actually wanted makes a deep point one short run instead of a
   long one that must survive the whole ladder.
*)

Needs["SetReplace`"];
maxDepth = If[Length[$ScriptCommandLine] >= 2, ToExpression[ToString[$ScriptCommandLine[[2]]]], 5];
reps     = If[Length[$ScriptCommandLine] >= 3, ToExpression[ToString[$ScriptCommandLine[[3]]]], 3];
minDepth = If[Length[$ScriptCommandLine] >= 4, ToExpression[ToString[$ScriptCommandLine[[4]]]], 2];

ruleArg  = If[Length[$ScriptCommandLine] >= 5, ToString[$ScriptCommandLine[[5]]], "Chain3"];
limit    = 120;

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
Print["PACLET HypergraphRewriting ", pacletVersion["WolframInstitute/HypergraphRewriteEngine"]];
Print["PACLET SetReplace ", pacletVersion["SetReplace"]];
Print["KERNEL ", $Version, " processors ", $ProcessorCount];

(* OUR reference implementation, loaded from source. Its rules are {lhs, rhs} pairs rather than
   Rule objects, and it returns per-step tuples rather than graphs. *)
Get[FileNameJoin[{repo, "reference", "MultiwayReference.wl"}]];
corpus = <|
  "Chain3" -> {{{{1, 2}, {2, 3}} -> {{1, 2}, {2, 4}, {4, 3}}}, {{1, 2}, {2, 3}}},
  "BinarySplit" -> {{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}},
  "TriangleClose" -> {{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}}, {{1, 2}, {1, 3}}},
  "WPP" -> {{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}, {{1, 2}, {1, 3}}},
  "PathSelfLoop" -> {{{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}}, {{1, 1}, {1, 1}}},
  "Ternary" -> {{{{1, 2, 3}} -> {{1, 2, 4}, {2, 4, 3}}}, {{1, 1, 1}}},
  "TwoRule" -> {{{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}, {{1, 2}}}|>;

(* One measurement: build the states graph and the causal graph at this depth, from a fresh
   MultiwaySystem object, so nothing is cached across depths or repetitions. *)
once[rules_, init_, d_] := Module[{multi, sg, cg, t},
  t = First @ AbsoluteTiming[
    multi = mwSystem[rules, {init}];
    sg = multi["StatesGraph", d, "CanonicalStateFunction" -> "CanonicalHypergraph"];
    cg = multi["CausalGraph", d, "CanonicalStateFunction" -> "CanonicalHypergraph"];
  ];
  {t * 1000., VertexCount[sg], EdgeCount[cg]}
];

(* The same workload through the paclet: states graph and causal edges, so the two rows ask for
   the same artifacts and neither is charged for something the other skipped. *)
hgOnce[rules_, init_, d_] := Module[{t, r},
  {t, r} = AbsoluteTiming[
    HGEvolve[rules, init, d, {"StatesGraph", "CausalEdges"},
      "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full"]];
  {t * 1000., If[GraphQ[r["StatesGraph"]], VertexCount[r["StatesGraph"]], 0],
   If[ListQ[r["CausalEdges"]], Length[r["CausalEdges"]], 0]}
];

(* The reference, asked for the same two quantities. It computes DATA -- no Graph objects -- so
   this row is the Wolfram-side measurement that is not charged for graph construction, and the
   one the engine's native number is most nearly commensurable with. *)
refOnce[rules_, init_, d_] := Module[{t, r, st, cz},
  {t, r} = AbsoluteTiming[
    MultiwayReference`MultiwayEvolve[List @@@ rules, init, d,
      "StateCanonicalization" -> "Canonical", "EventCanonicalization" -> "States"]];
  (* PerStep's last row is the cumulative answer at this depth: "states" is the canonical state
     count and "causal" the causal-edge count, which are the two the other rows report. *)
  st = If[Length[r["PerStep"]] > 0, Last[r["PerStep"]]["states"], 0];
  cz = If[Length[r["PerStep"]] > 0, Last[r["PerStep"]]["causal"], 0];
  {t * 1000., st, cz}
];

(* SetReplace's token-event multiway to d generations, and its event count. *)
srOnce[rules_, init_, d_] := Module[{t, obj},
  {t, obj} = AbsoluteTiming[SetReplace`WolframModel[rules, init, <|"MaxGenerations" -> d|>,
    "EventSelectionFunction" -> "MultiwaySpacelike"]];
  {t * 1000., obj["EventsCount"]}
];

(* reps runs of f, each under the time limit: {median ms, min ms, max ms, counts of the first
   run}, or Missing when a run timed out. *)
timed[f_, args__] := Module[{runs = {}, r},
  Do[r = TimeConstrained[f[args], limit, $TimedOut];
    If[r === $TimedOut, Return[Missing["TimedOut"], Module]]; AppendTo[runs, r], reps];
  Join[{Median[#], Min[#], Max[#]} &[runs[[All, 1]]], Rest[First[runs]]]];
row[tag_, d_, r_, fields_] := Print[tag, " d=", d, " ",
  If[MissingQ[r], "timeout>" <> ToString[limit] <> "s",
    StringRiffle[MapThread[#1 <> "=" <> ToString[#2] &, {fields, Drop[r, 3]}], " "] <>
      " ms=" <> ToString[Round[r[[1]], 0.1]] <> " range=" <> ToString[Round[r[[2]], 0.1]] <> "-" <>
      ToString[Round[r[[3]], 0.1]]]];

(* One untimed run of each engine, so no row is charged for loading a paclet or starting the
   engine worker. *)
Module[{w = corpus["BinarySplit"]},
  once[w[[1]], w[[2]], 2]; refOnce[w[[1]], w[[2]], 2]; hgOnce[w[[1]], w[[2]], 2];
  srOnce[w[[1]], w[[2]], 2]];

Do[
  Module[{rules = corpus[name][[1]], init = corpus[name][[2]], alive = <|"AUTH" -> True, "REF" -> True,
      "HGEV" -> True, "SR" -> True|>, r},
    Print["RULE ", name, " ", rules, " init ", init];
    Do[
      Do[If[alive[tag],
          r = timed[Switch[tag, "AUTH", once, "REF", refOnce, "HGEV", hgOnce, "SR", srOnce], rules, init, d];
          If[MissingQ[r], alive[tag] = False];
          row[StringPadRight[tag, 4], d, r, If[tag === "SR", {"events"}, {"states", "causal"}]]],
        {tag, {"AUTH", "REF", "HGEV", "SR"}}],
      {d, minDepth, maxDepth}]],
  {name, If[ruleArg === "All", Keys[corpus], {ruleArg}]}
];

Print["AUTH_DONE"];
