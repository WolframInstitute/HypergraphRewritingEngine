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

   One row per depth, printed as it completes, so a run cut short by the timeout still leaves
   the depths that finished:
     AUTH d=<depth> states=<n> causal=<n> ms=<median of reps>

   Usage: wolframscript -file bench_authority.wls [maxDepth] [reps]
*)

maxDepth = If[Length[$ScriptCommandLine] >= 2, ToExpression[ToString[$ScriptCommandLine[[2]]]], 5];
reps     = If[Length[$ScriptCommandLine] >= 3, ToExpression[ToString[$ScriptCommandLine[[3]]]], 3];

mwSystem = PacletSymbol["Wolfram/Multicomputation", "Wolfram`Multicomputation`MultiwaySystem"];

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

Do[
  Module[{runs, ms, states, causal},
    runs = Table[once[d], reps];
    ms = Median[runs[[All, 1]]];
    states = runs[[1, 2]];
    causal = runs[[1, 3]];
    Print["AUTH d=", d, " states=", states, " causal=", causal, " ms=", ToString[Round[ms, 0.1]]];
  ],
  {d, 2, maxDepth}
];

Print["AUTH_DONE"];
