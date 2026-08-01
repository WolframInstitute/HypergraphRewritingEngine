#!/usr/bin/env wolframscript
(* Close the Gap 1 adjudication chain against the AUTHORITY, not only our own oracle.

   adjudicate_gap1_automatic.wls established, from our brute-force reference alone:

     engine full capture Automatic == reference eventsAutomaticPositional  (87 / 54 / 10)
     engine quotient replay        == NEITHER reference convention        (86 / 52 / 10)

   Our reference was cross-checked against the Wolfram/Multicomputation MultiwaySystem paclet on
   twelve cases (compare_multiwaysystem.wls), but none of those is a Gap 1 workload, and the
   whole point of these three is that they separate the conventions where high-symmetry states
   appear. So the remaining question is whether the AUTHORITY's Automatic equals our Positional
   column on exactly these workloads -- if yes, the chain is
       authority == our Positional == engine full capture,
   and the replay is the one implementation out of step; if no, our oracle's Positional column
   is itself suspect on symmetric workloads and the earlier verdict rests on one leg.

   Per case and step: authority states / events(Full) / events(Automatic) beside our reference's
   states / eventsStates / eventsAutomaticPositional / eventsAutomaticCanonical. *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

mwSystem = PacletSymbol["Wolfram/Multicomputation", "Wolfram`Multicomputation`MultiwaySystem"];

mwStats[rules_, init_, steps_, evCanon_] := Module[{multi, sg, ecbg},
  multi = mwSystem[rules, {init}];
  sg = multi["StatesGraph", steps, "CanonicalStateFunction" -> "CanonicalHypergraph",
             "CanonicalEventFunction" -> evCanon];
  ecbg = multi["EvolutionCausalBranchialGraph", steps,
               "CanonicalStateFunction" -> "CanonicalHypergraph",
               "CanonicalEventFunction" -> evCanon];
  <|"states" -> VertexCount[sg],
    "events" -> Length[Cases[VertexList[ecbg], _DirectedEdge, {1}]]|>
];

cases = {
  <|"name" -> "wolfram-2to4",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 4}, {4, 3}, {3, 1}}}},
    "init" -> {{1, 2}, {2, 3}}, "steps" -> 4|>,
  <|"name" -> "WPP",
    "rules" -> {{{{1, 2}, {1, 3}}, {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}},
    "init" -> {{1, 2}, {1, 3}}, "steps" -> 4|>,
  <|"name" -> "binary-growth",
    "rules" -> {{{{1, 2}}, {{1, 3}, {3, 2}}}},
    "init" -> {{1, 2}}, "steps" -> 4|>
};

mwRules[r_] := (#[[1]] -> #[[2]]) & /@ r;

Do[
  Module[{c = cases[[ci]], ref},
    Print["==== ", c["name"], " ===="];
    ref = MultiwayEvolve[c["rules"], c["init"], c["steps"], "FullCapture" -> True,
                         "StateCanonicalization" -> "Canonical"];
    Do[
      Module[{rs = ref["PerStep"][[k]], authFull, authAuto},
        authFull = mwStats[mwRules[c["rules"]], c["init"], k, Full];
        authAuto = mwStats[mwRules[c["rules"]], c["init"], k, Automatic];
        Print["  step ", k,
              "  states ref=", rs["states"], " auth=", authFull["states"],
              "   eventsFull ref=", rs["eventsStates"], " auth=", authFull["events"],
              "   eventsAuto auth=", authAuto["events"],
              " refPositional=", rs["eventsAutomaticPositional"],
              " refCanonical=", rs["eventsAutomaticCanonical"]];
      ],
      {k, 1, c["steps"]}
    ];
    Print[""];
  ],
  {ci, Length[cases]}
];
