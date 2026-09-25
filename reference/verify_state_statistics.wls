(* reference/verify_state_statistics.wls

   "StepStatistics" against its definition. The oracle below computes the per-step population
   statistics from a raw run ("CanonicalizeStates" -> None, every raw state listed with its
   exact "CanonicalHash") in Wolfram Language, by the definitions of the multiway-statistics
   probes (mwsCheapInvariants, mwsGraphInvariants, mwsSummary, mwsPopulation). The engine
   computes the same from one state per class and the class's multiplicity, under quotient
   exploration and under full capture. Every key of every step must agree: integers exactly,
   reals to 1e-9 relative.

   Run: wolframscript -file reference/verify_state_statistics.wls
   Output: one PASS/FAIL line per case and route, then STATE STATISTICS VERIFIED or FAILED. *)

PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}]];
Needs["HypergraphRewriting`"];

(* === oracle: the probes' definitions === *)

cheapInvariants[edges_List] := Module[{vs, flat, vertices, n, m, arities, degSeq},
  vs = edges[[All, 2 ;;]];
  flat = Flatten[vs];
  vertices = Union[flat]; n = Length[vertices]; m = Length[edges];
  arities = Length /@ vs;
  degSeq = ReverseSort[Values[Counts[flat]]];
  <|"VertexCount" -> n, "EdgeCount" -> m, "Arities" -> Sort[arities],
    "DegreeSequence" -> degSeq,
    "MaxDegree" -> If[n == 0, 0, Max[degSeq]],
    "MeanDegree" -> If[n == 0, 0., N[Total[arities]/n]],
    "TwoSectionEdgeCount" ->
      Length[DeleteDuplicates[Sort /@ Catenate[Subsets[DeleteDuplicates[#], {2}] & /@ vs]]]|>];

graphInvariants[edges_List] := Module[
  {vs, vertices, n, m, incEdges, g, comps, c, largest, sub, cheap},
  cheap = cheapInvariants[edges];
  vs = edges[[All, 2 ;;]];
  vertices = Union[Flatten[vs]]; n = Length[vertices]; m = Length[edges];
  incEdges = DeleteDuplicates[Catenate[MapIndexed[
    Function[{v, i}, UndirectedEdge[{"e", First[i]}, {"v", #}] & /@ DeleteDuplicates[v]], vs]]];
  g = Graph[Join[{"v", #} & /@ vertices, {"e", #} & /@ Range[m]], incEdges];
  comps = ConnectedComponents[g]; c = Length[comps];
  (* The largest component: most nodes, then the greatest diameter, then the greatest mean
     distance, then the most vertices. The probes took the first of the equally large ones in
     ConnectedComponents order, which depends on the vertex labels; this is the probes' rule
     made an isomorphism invariant, which a per-class computation requires. *)
  largest = First[MaximalBy[comps, {Length[#],
    If[Length[#] <= 1, 0, GraphDiameter[Subgraph[g, #]]],
    If[Length[#] <= 1, 0., N[MeanGraphDistance[Subgraph[g, #]]]],
    Count[#, {"v", _}]} &]];
  sub = If[c == 1, g, Subgraph[g, largest]];
  <|"Components" -> c,
    "CycleRank" -> cheap["TwoSectionEdgeCount"] - n + c,
    "IncidenceCycleRank" -> Length[incEdges] - (n + m) + c,
    "IncidenceDiameter" -> If[VertexCount[sub] <= 1, 0, GraphDiameter[sub]],
    "IncidenceMeanDistance" -> If[VertexCount[sub] <= 1, 0., N[MeanGraphDistance[sub]]],
    "LargestComponentFraction" -> If[n == 0, 0., N[Count[largest, {"v", _}]/n]]|>];

summary[values_List, round_ : 1] := If[values === {}, <|"N" -> 0|>,
  <|"N" -> Length[values], "Mean" -> N[Mean[values]],
    "StandardDeviation" -> If[Length[values] > 1, N[StandardDeviation[values]], 0.],
    "Min" -> Min[values], "Max" -> Max[values], "Median" -> N[Median[values]],
    "Histogram" -> KeySort[Counts[If[round == 1, values, N[Round[values, round]]]]]|>];

entropyBits[counts_List] := Module[{p = N[counts/Total[counts]]}, -Total[p Log2[p]]];

oracleSteps[res_Association] := Module[{states, byStep, events, stepOf, eventStep, ruleOf},
  states = Values[res["States"]];
  stepOf = Association[#Id -> #Step & /@ states];
  byStep = KeySort[GroupBy[states, #Step &]];
  events = Values[res["Events"]];
  eventStep = Association[#Id -> Lookup[stepOf, #OutputState, -1] & /@ events];
  ruleOf = Association[#Id -> #RuleIndex & /@ events];
  KeyValueMap[Function[{k, sts},
    Module[{cheap, ginv, mult, classes, evIds},
      cheap = cheapInvariants[#Edges] & /@ sts;
      ginv = graphInvariants[#Edges] & /@ sts;
      mult = Values[Counts[Lookup[sts, "CanonicalHash"]]]; classes = Length[mult];
      evIds = Keys[Select[eventStep, # == k &]];
      <|"Step" -> k, "RawStates" -> Length[sts], "Classes" -> classes,
        "Redundancy" -> N[Length[sts]/classes], "MaxMultiplicity" -> Max[mult],
        "MultiplicityHistogram" -> KeySort[Counts[mult]],
        "ClassEntropyBits" -> entropyBits[mult],
        "ClassEntropyNormalized" -> If[classes == 1, 1., entropyBits[mult]/Log2[N[classes]]],
        "Events" -> Length[evIds],
        "RuleCounts" -> KeySort[Counts[Lookup[ruleOf, evIds]]],
        "Invariants" -> <|
          "VertexCount" -> summary[Lookup[cheap, "VertexCount"]],
          "EdgeCount" -> summary[Lookup[cheap, "EdgeCount"]],
          "MaxDegree" -> summary[Lookup[cheap, "MaxDegree"]],
          "MeanDegree" -> summary[Lookup[cheap, "MeanDegree"], 0.01],
          "TwoSectionEdgeCount" -> summary[Lookup[cheap, "TwoSectionEdgeCount"]],
          "Components" -> summary[Lookup[ginv, "Components"]],
          "CycleRank" -> summary[Lookup[ginv, "CycleRank"]],
          "IncidenceCycleRank" -> summary[Lookup[ginv, "IncidenceCycleRank"]],
          "IncidenceDiameter" -> summary[Lookup[ginv, "IncidenceDiameter"]],
          "IncidenceMeanDistance" -> summary[Lookup[ginv, "IncidenceMeanDistance"], 0.01],
          "LargestComponentFraction" -> summary[Lookup[ginv, "LargestComponentFraction"], 0.01]|>,
        "ArityHistogram" -> KeySort[Counts[Catenate[Lookup[cheap, "Arities"]]]],
        "AritySignatureHistogram" -> KeySort[Counts[Lookup[cheap, "Arities"]]],
        "DegreeHistogram" -> KeySort[Counts[Catenate[Lookup[cheap, "DegreeSequence"]]]],
        "DegreeSequenceHistogram" -> KeySort[Counts[Lookup[cheap, "DegreeSequence"]]]|>]],
    byStep]];

(* === comparison === *)

(* Associations compare key by key after sorting; numbers compare with a relative tolerance, so
   a histogram keyed 2 and one keyed 2. are the same histogram. *)
normal[a_Association] := KeySort[KeyMap[If[NumberQ[#], N[#], #] &, normal /@ a]];
normal[l_List] := normal /@ l;
normal[x_?NumberQ] := N[x];
normal[x_] := x;
close[a_Association, b_Association] := Keys[a] === Keys[b] && AllTrue[Keys[a], close[a[#], b[#]] &];
close[a_List, b_List] := Length[a] == Length[b] && And @@ MapThread[close, {a, b}];
close[a_?NumberQ, b_?NumberQ] := Abs[a - b] <= 10^-9 Max[1, Abs[a], Abs[b]];
close[a_, b_] := a === b;

firstDifference[o_Association, e_Association, path_ : {}] := Catch[
  If[Keys[o] =!= Keys[e], Throw[{path, "keys", Keys[o], Keys[e]}]];
  Do[If[!close[o[k], e[k]],
      Throw[If[AssociationQ[o[k]] && AssociationQ[e[k]], firstDifference[o[k], e[k], Append[path, k]],
        {Append[path, k], o[k], e[k]}]]], {k, Keys[o]}];
  None];

cases = {
  {"WPP", {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}, {{1, 2}, {1, 3}}, 4},
  {"TriangleClose", {{{1, 2}, {2, 3}} -> {{1, 2}, {2, 3}, {3, 1}}}, {{1, 2}, {2, 3}, {3, 4}}, 3},
  {"BinarySplit", {{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, 6},
  {"TwoRule", {{{1, 2}} -> {{1, 3}, {3, 2}}, {{1, 2}, {2, 3}} -> {{1, 3}}}, {{1, 2}, {2, 3}}, 4},
  {"Ternary", {{{1, 2, 3}} -> {{1, 2, 4}, {4, 3, 2}}}, {{1, 2, 3}}, 4},
  {"SelfLoops", {{{1, 1}, {1, 1}} -> {{1, 1}, {1, 1}, {1, 1}}}, {{1, 1}, {1, 1}}, 4},
  (* Rules that disconnect a state, so a state has several components, some of equal size and
     different diameter: the choice of the largest one is exercised. *)
  {"Disconnect", {{{1, 2}, {2, 3}} -> {{1, 2}, {3, 4}, {4, 5}}}, {{1, 2}, {2, 3}, {3, 4}}, 3},
  {"SplitTies", {{{1, 2}} -> {{1, 3}, {4, 2}}}, {{1, 2}, {2, 3}, {3, 4}}, 3},
  (* Two components of four nodes each, diameters 2 and 3, in either vertex order. *)
  {"TieStarFirst", {{{1, 1}} -> {{1, 1}}}, {{1, 2, 3}, {4, 5}, {5, 5}}, 2},
  {"TiePathFirst", {{{1, 1}} -> {{1, 1}}}, {{1, 2}, {2, 2}, {3, 4, 5}}, 2}
};

failures = 0;
Do[
  Module[{name, rules, init, steps, raw, oracle, routes},
    {name, rules, init, steps} = c;
    raw = HGEvolve[rules, init, steps, {"States", "Events"}, "CanonicalizeStates" -> None,
      "IncludeCanonicalHashes" -> True];
    oracle = normal[oracleSteps[raw]];
    routes = {
      "quotient" -> HGEvolve[rules, init, steps, "StepStatistics", "CanonicalizeStates" -> Full,
        "ExploreFromCanonicalStatesOnly" -> True],
      "full capture" -> HGEvolve[rules, init, steps, "StepStatistics", "CanonicalizeStates" -> None]};
    Do[
      Module[{got = normal[Last[r]], diffs},
        diffs = If[!ListQ[got] || Length[got] =!= Length[oracle], {"step count", Length[got], Length[oracle]},
          DeleteCases[MapThread[firstDifference, {oracle, got}], None]];
        If[diffs === {},
          Print["PASS  ", name, " ", First[r], ": ", Length[oracle], " steps"],
          failures++; Print["FAIL  ", name, " ", First[r], ": ", Short[First[Flatten[{diffs}, 0]], 4]]]],
      {r, routes}]],
  {c, cases}];

Print[If[failures == 0, "STATE STATISTICS VERIFIED", "STATE STATISTICS FAILED: " <> ToString[failures]]];
