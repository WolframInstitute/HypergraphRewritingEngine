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
    "steps" -> 5, "expected" -> {18, 89, 88, 99}|>
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
