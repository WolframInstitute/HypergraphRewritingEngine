(* Shared engine golden corpus: reference/MultiwaySystem values at the final
   step, checked with CanonicalizeStates->Full. Each case's "expected" is
   {states, rawEvents, causal, branchial}. Consumed by verify_paclet.wls (CPU
   binary) and verify_paclet_gpu.wls (GPU binary). *)

hgGoldenCases = {
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
