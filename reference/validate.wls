#!/usr/bin/env wolframscript
(* Validate MultiwayReference against the golden determinism-fuzzing corpus.
   Each expected per-step tuple is {states, eventsBase, eventsFull, eventsAutomatic, causal, branchial}. *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

cases = {
  <|"name" -> "TestCase1_SimpleRule",
    "rules" -> {{{{1, 2}}, {{1, 2}, {2, 3}}}},
    "init" -> {{1, 2}},
    "expected" -> {{2,1,1,1,0,0}, {4,3,3,3,2,0}, {8,9,8,9,8,0}, {17,33,21,26,32,0}}|>,
  <|"name" -> "TestCase2_TwoEdgeRule",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}},
    "init" -> {{1, 2}, {2, 3}, {3, 1}},
    "expected" -> {{2,3,1,3,0,3}, {4,15,3,7,12,18}, {8,75,9,19,72,108}, {13,435,19,47,444,738}}|>,
  <|"name" -> "TestCase3_HyperedgeRule",
    "rules" -> {{{{1, 2, 3}}, {{1, 2}, {1, 3}, {1, 4}}}},
    "init" -> {{1, 2, 3}},
    "expected" -> {{2,1,1,1,0,0}, {2,1,1,1,0,0}, {2,1,1,1,0,0}, {2,1,1,1,0,0}}|>,
  <|"name" -> "TestCase4_MultiRule",
    "rules" -> {{{{1, 2, 3}}, {{1, 2}, {1, 3}, {1, 4}}}, {{{1, 2}}, {{1, 2}, {1, 3}}}},
    "init" -> {{1, 2, 3}},
    "expected" -> {{2,1,1,1,0,0}, {3,4,2,4,3,0}, {4,16,3,8,15,0}, {5,76,4,13,75,0}}|>,
  <|"name" -> "TestCase5_ComplexTwoRuleSystem",
    "rules" -> {{{{1, 2, 3}}, {{1, 2}, {1, 3}, {1, 4}}}, {{{1, 2}, {1, 3}}, {{1, 2}, {1, 3}, {2, 4}}}},
    "init" -> {{1, 2, 3}},
    "expected" -> {{2,1,1,1,0,0}, {3,7,2,7,6,15}, {5,43,4,17,42,105}, {9,283,9,35,282,657}}|>,
  <|"name" -> "TestCase6_TwoEdgeRuleVariant",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}},
    "init" -> {{1, 2}, {2, 3}},
    "expected" -> {{2,1,1,1,0,0}, {3,3,2,3,2,1}, {4,9,3,6,8,7}, {5,33,4,10,32,43}}|>,
  <|"name" -> "TestCase7_TwoEdgeRuleWithSelfLoops",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}},
    "init" -> {{1, 1}, {1, 1}},
    "expected" -> {{2,2,1,2,0,1}, {3,10,2,7,8,13}, {4,58,3,16,56,117}, {5,442,4,29,440,1173}}|>,
  <|"name" -> "TestCase8_ComplexTwoEdgeRule",
    "rules" -> {{{{1, 2}, {2, 3}}, {{4, 1}, {1, 4}, {2, 3}, {4, 3}}}},
    "init" -> {{1, 2}, {2, 3}},
    "expected" -> {{2,1,1,1,0,0}, {5,4,4,4,3,3}, {17,21,16,21,20,37}, {83,182,94,136,181,423}}|>,
  <|"name" -> "TestCase9_ComplexTwoEdgeRuleWithSelfLoops",
    "rules" -> {{{{1, 2}, {2, 3}}, {{4, 1}, {1, 4}, {2, 3}, {4, 3}}}},
    "init" -> {{1, 1}, {1, 1}},
    "expected" -> {{2,2,1,2,0,1}, {6,16,5,9,14,35}, {33,172,36,57,170,489}}|>,
  <|"name" -> "TestCase10_AnotherTwoEdgeRule",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 3}, {2, 3}, {3, 4}}}},
    "init" -> {{1, 2}, {2, 3}},
    "expected" -> {{2,1,1,1,0,0}, {3,3,2,3,2,1}, {6,9,5,8,8,5}, {10,37,12,22,36,35}, {21,203,29,54,210,231}}|>,
  <|"name" -> "TestCase11_AnotherTwoEdgeRuleWithSelfLoops",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 3}, {2, 3}, {3, 4}}}},
    "init" -> {{1, 1}, {1, 1}},
    "expected" -> {{2,2,1,2,0,1}, {4,10,3,6,8,13}, {8,50,7,18,48,81}, {18,282,18,49,280,465}}|>,
  <|"name" -> "TestCase12_ComplexThreeEdgeRule",
    "rules" -> {{{{1, 2, 3}, {5, 1}}, {{1, 5, 6}, {3, 2}, {3, 5}}}},
    "init" -> {{0, 0, 0}, {0, 0}},
    "expected" -> {{2,1,1,1,0,0}, {3,3,2,3,2,1}, {5,9,4,7,8,7}, {9,29,8,14,28,31}, {18,89,17,25,88,99}}|>
};

totalSteps = 0; passSteps = 0; failedCases = {};

Do[
  Module[{c = cases[[ci]], res, perStep, nsteps, actual, ok = True},
    nsteps = Length[c["expected"]];
    res = MultiwayEvolve[c["rules"], c["init"], nsteps];
    perStep = res["PerStep"];
    Do[
      Module[{e = c["expected"][[k]], a, row = perStep[[k]]},
        a = {row["states"], row["eventsNone"], row["eventsStates"], row["eventsAutomaticPositional"], row["causal"], row["branchial"]};
        totalSteps++;
        If[a === e, passSteps++,
          ok = False;
          Print["  ", c["name"], " step ", k, "  expected ", e, "  actual ", a,
                "  {states,base,full,auto,causal,branchial}"]
        ]
      ],
      {k, nsteps}
    ];
    If[ok, Print["PASS  ", c["name"]], AppendTo[failedCases, c["name"]]]
  ],
  {ci, Length[cases]}
];

Print["============================================================"];
Print["Step-tuples passed: ", passSteps, " / ", totalSteps];
Print["Failed cases: ", If[failedCases === {}, "NONE", failedCases]];
