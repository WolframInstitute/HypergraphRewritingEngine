#!/usr/bin/env wolframscript
(* Verify the freshly-built local paclet (HGEvolve) against the golden corpus.
   Checks the engine's observable outputs at the final step, with CanonicalizeStates->Full:
     States      == golden canonical states
     Events      == golden raw events (the "Events" property is the full-capture raw list)
     CausalEdges == golden causal (transitive reduction on)
     BranchialEdges == golden branchial
   Golden numbers are the Full-canon corpus from validate.wls / MultiwaySystem. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

(* {name, rules ({lhs,rhs} form), init, steps, {states, rawEvents, causal, branchial}} *)
cases = {
  <|"name" -> "TC1_SimpleRule",
    "rules" -> {{{{1, 2}}, {{1, 2}, {2, 3}}}}, "init" -> {{1, 2}},
    "steps" -> 4, "expected" -> {17, 33, 32, 0}|>,
  <|"name" -> "TC2_TwoEdgeRule",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}}, "init" -> {{1, 2}, {2, 3}, {3, 1}},
    "steps" -> 4, "expected" -> {13, 435, 444, 738}|>,
  <|"name" -> "TC6_TwoEdgeRuleVariant",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}}, "init" -> {{1, 2}, {2, 3}},
    "steps" -> 4, "expected" -> {5, 33, 32, 43}|>,
  <|"name" -> "TC8_ComplexTwoEdgeRule",
    "rules" -> {{{{1, 2}, {2, 3}}, {{4, 1}, {1, 4}, {2, 3}, {4, 3}}}}, "init" -> {{1, 2}, {2, 3}},
    "steps" -> 4, "expected" -> {83, 182, 181, 423}|>,
  <|"name" -> "TC10_AnotherTwoEdgeRule",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 3}, {2, 3}, {3, 4}}}}, "init" -> {{1, 2}, {2, 3}},
    "steps" -> 5, "expected" -> {21, 203, 210, 231}|>,
  <|"name" -> "TC12_ComplexThreeEdgeRule",
    "rules" -> {{{{1, 2, 3}, {5, 1}}, {{1, 5, 6}, {3, 2}, {3, 5}}}}, "init" -> {{0, 0, 0}, {0, 0}},
    "steps" -> 5, "expected" -> {18, 89, 88, 99}|>,
  <|"name" -> "TC3_HyperedgeRule",
    "rules" -> {{{{1, 2, 3}}, {{1, 2}, {1, 3}, {1, 4}}}}, "init" -> {{1, 2, 3}},
    "steps" -> 4, "expected" -> {2, 1, 0, 0}|>,
  <|"name" -> "TC4_MultiRule",
    "rules" -> {{{{1, 2, 3}}, {{1, 2}, {1, 3}, {1, 4}}}, {{{1, 2}}, {{1, 2}, {1, 3}}}},
    "init" -> {{1, 2, 3}}, "steps" -> 4, "expected" -> {5, 76, 75, 0}|>,
  <|"name" -> "TC5_ComplexTwoRuleSystem",
    "rules" -> {{{{1, 2, 3}}, {{1, 2}, {1, 3}, {1, 4}}}, {{{1, 2}, {1, 3}}, {{1, 2}, {1, 3}, {2, 4}}}},
    "init" -> {{1, 2, 3}}, "steps" -> 4, "expected" -> {9, 283, 282, 657}|>,
  <|"name" -> "TC7_TwoEdgeRuleWithSelfLoops",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}}, "init" -> {{1, 1}, {1, 1}},
    "steps" -> 4, "expected" -> {5, 442, 440, 1173}|>,
  <|"name" -> "TC9_ComplexTwoEdgeRuleWithSelfLoops",
    "rules" -> {{{{1, 2}, {2, 3}}, {{4, 1}, {1, 4}, {2, 3}, {4, 3}}}}, "init" -> {{1, 1}, {1, 1}},
    "steps" -> 3, "expected" -> {33, 172, 170, 489}|>,
  <|"name" -> "TC11_AnotherTwoEdgeRuleWithSelfLoops",
    "rules" -> {{{{1, 2}, {2, 3}}, {{1, 3}, {2, 3}, {3, 4}}}}, "init" -> {{1, 1}, {1, 1}},
    "steps" -> 4, "expected" -> {18, 282, 280, 465}|>
};

cnt[x_] := Which[ListQ[x], Length[x], AssociationQ[x], Length[x], True, Missing[]];

totalChecks = 0; passChecks = 0; failed = {};
Do[
  Module[{c = cases[[ci]], res, a, e, rulesHG},
    e = c["expected"];
    rulesHG = (Rule @@ # &) /@ c["rules"];  (* {lhs, rhs} -> (lhs -> rhs) *)
    res = HGEvolve[rulesHG, c["init"], c["steps"],
      {"States", "Events", "CausalEdges", "BranchialEdges"},
      "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full",
      "CausalTransitiveReduction" -> True];
    a = {cnt[res["States"]], cnt[res["Events"]], cnt[res["CausalEdges"]], cnt[res["BranchialEdges"]]};
    totalChecks++;
    If[a === e, passChecks++; Print["PASS  ", c["name"], "  ", a],
      AppendTo[failed, c["name"]];
      Print["FAIL  ", c["name"], "  expected ", e, "  actual ", a,
            "  {states,rawEvents,causal,branchial}"]
    ]
  ],
  {ci, Length[cases]}
];

Print["============================================================"];
Print["Cases passed: ", passChecks, " / ", totalChecks];
Print["Failed: ", If[failed === {}, "NONE", failed]];

(* TargetDevice consistency: "CPU" and "GPU" must produce the same
   States/Events/CausalEdges/BranchialEdges on a representative workload. The
   GPU device falls back to the CPU until the GPU binary is wired; once wired,
   this validates GPU == CPU end-to-end through the paclet. *)
Print["\n=== TargetDevice consistency ==="];
tdRules = (Rule @@ # &) /@ {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}};
runTD[dev_] := Module[{r},
  r = HGEvolve[tdRules, {{1, 2}, {2, 3}}, 4,
    {"States", "Events", "CausalEdges", "BranchialEdges"},
    "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full",
    "CausalTransitiveReduction" -> True, "TargetDevice" -> dev];
  {cnt[r["States"]], cnt[r["Events"]], cnt[r["CausalEdges"]], cnt[r["BranchialEdges"]]}
];
tdCPU = runTD["CPU"]; tdGPU = Quiet[runTD["GPU"], HGEvolve::gpudev];
Print["CPU: ", tdCPU, "   GPU: ", tdGPU];
Print[If[tdCPU === tdGPU === {5, 33, 32, 43}, "PASS", "FAIL"],
      "  TargetDevice CPU == GPU == golden {5,33,32,43}"];
