(* reference/verify_state_statistics.wls

   "StepStatistics" against its definition. The oracle below computes the per-step population
   statistics from a raw run ("CanonicalizeStates" -> None, every raw state listed with its
   exact "CanonicalHash") in Wolfram Language, by the definitions of the multiway-statistics
   probes (mwsCheapInvariants, mwsGraphInvariants, mwsSummary, mwsPopulation) and, for the
   geometry and branchial metrics, by reference/StateGeometryReference.wl (docs/SPEC.md,
   "StepStatistics geometry" and "StepStatistics branchial metrics"). The engine computes the
   same from one state per class and the class's multiplicity, under quotient exploration and
   under full capture, with "StepStatisticsWeighting" -> "States" and "Classes", and with
   "StepStatisticsBranchial" -> All. Every key of every step must agree: integers exactly, reals
   to 1e-9 relative.

   The two metrics the Wolfram Function Repository defines are also checked against the
   repository's own functions on every connected state of every case with at least two
   vertices: WolframHausdorffDimension (and its per-radius dimensions) and
   WolframRicciCurvatureScalar, to 1e-9 relative. This needs the Internet; without it the check
   prints FUNCTION REPOSITORY UNAVAILABLE and fails.

   The cases are the fixed corpus below and random rules (genCase, the generator of the
   oracle fuzz) for the seeds given on the command line.

   Run: wolframscript -file reference/verify_state_statistics.wls [seedFrom seedTo]
   Setting HG_VERIFY_LINUX_ENGINE to the Linux path of an hg_evolve binary runs every HGEvolve
   call through wsl.exe on that binary, so a Windows kernel checks a Linux build.
   Output: one PASS/FAIL line per case and route, then STATE STATISTICS VERIFIED or FAILED. *)

PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}]];
Needs["HypergraphRewriting`"];
Get[FileNameJoin[{DirectoryName[$InputFileName], "StateGeometryReference.wl"}]];

With[{engine = Environment["HG_VERIFY_LINUX_ENGINE"]},
  If[StringQ[engine],
    HypergraphRewriting`Private`hgEngineAvailableQ[] := True;
    Clear[HypergraphRewriting`Private`hgCallEngine];
    HypergraphRewriting`Private`hgCallEngine[bytes_, ___] :=
      Module[{p = RunProcess[{"wsl.exe", "-e", engine}, All, bytes]},
        If[p["ExitCode"] =!= 0, Print["engine exit ", p["ExitCode"], ": ", p["StandardError"]];
          Return[$Failed]];
        ByteArray[ToCharacterCode[p["StandardOutput"], "ISO8859-1"]]];
    Print["engine: ", engine, " through wsl.exe"]]];

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

(* The geometry of a state from its record's edges, memoised on the edge list. *)
geometry[edges_List] := geometry[edges] = sgrStateGeometry[edges[[All, 2 ;;]]];
ballGrowth[edges_List] := ballGrowth[edges] = sgrBallGrowth[edges[[All, 2 ;;]]];

summary[values_List, round_ : 1] := If[values === {}, <|"N" -> 0|>,
  <|"N" -> Length[values], "Mean" -> N[Mean[values]],
    "StandardDeviation" -> If[Length[values] > 1, N[StandardDeviation[values]], 0.],
    "Min" -> Min[values], "Max" -> Max[values], "Median" -> N[Median[values]],
    "Histogram" -> KeySort[Counts[If[round == 1, values, N[Round[values, round]]]]]|>];

entropyBits[counts_List] := Module[{p = N[counts/Total[counts]]}, -Total[p Log2[p]]];

geometryKeys = {"GraphRadius", "MeanEccentricity", "WolframHausdorffDimension",
  "WolframRicciCurvatureScalar", "OllivierRicciCurvature", "DegreeEntropy", "LocalEntropy",
  "MutualInformation", "FisherInformation"};

(* The branchial keys of one step from the raw run: the step's states, the branchial pairs
   whose first event's output state is at the step, and each state's vertex set. *)
branchialKeys[sts_List, pairs_List] := Module[{gm, ov},
  gm = sgrBranchialGraphMetrics[Lookup[sts, "Id"], pairs];
  ov = sgrOverlapMetrics[Union[Flatten[#Edges[[All, 2 ;;]]]] & /@ sts];
  Join[
    <|"BranchialDegree" -> summary[gm["Degrees"]],
      "BranchialDistance" -> summary[gm["Distances"]],
      "BranchialComponents" -> gm["Components"]|>,
    If[MissingQ[gm["Dimension"]], <||>, <|"BranchialDimension" -> gm["Dimension"]|>],
    <|"StateOverlap" -> summary[ov["StateOverlap"], 0.01],
      "VertexSharpness" -> summary[ov["VertexSharpness"], 0.01],
      "BranchEntropy" -> summary[ov["BranchEntropy"], 0.01]|>]];

(* `byClass`: each class of a step once ("StepStatisticsWeighting" -> "Classes"); the counts of
   raw states, classes and events stay those of the raw states. `branchial`: add the
   "StepStatisticsBranchial" -> All keys. *)
oracleSteps[res_Association, byClass_ : False, branchial_ : False] := Module[
  {states, byStep, events, stepOf, eventStep, ruleOf, outOf, pairsByStep},
  states = Values[res["States"]];
  stepOf = Association[#Id -> #Step & /@ states];
  byStep = KeySort[GroupBy[states, #Step &]];
  events = Values[res["Events"]];
  eventStep = Association[#Id -> Lookup[stepOf, #OutputState, -1] & /@ events];
  ruleOf = Association[#Id -> #RuleIndex & /@ events];
  outOf = Association[#Id -> #OutputState & /@ events];
  pairsByStep = If[branchial,
    GroupBy[{outOf[#From], outOf[#To]} & /@ res["BranchialEdges"], stepOf[First[#]] &], <||>];
  KeyValueMap[Function[{k, sts},
    Module[{pop, cheap, ginv, geo, balls, mult, classes, evIds, rec},
      pop = If[byClass, DeleteDuplicatesBy[sts, #CanonicalHash &], sts];
      cheap = cheapInvariants[#Edges] & /@ pop;
      ginv = graphInvariants[#Edges] & /@ pop;
      geo = geometry[#Edges] & /@ pop;
      balls = ballGrowth[#Edges] & /@ pop;
      mult = Values[Counts[Lookup[sts, "CanonicalHash"]]]; classes = Length[mult];
      evIds = Keys[Select[eventStep, # == k &]];
      rec = <|"Step" -> k, "RawStates" -> Length[sts], "Classes" -> classes,
        "Redundancy" -> N[Length[sts]/classes], "MaxMultiplicity" -> Max[mult],
        "MultiplicityHistogram" -> KeySort[Counts[mult]],
        "ClassEntropyBits" -> entropyBits[mult],
        "ClassEntropyNormalized" -> If[classes == 1, 1., entropyBits[mult]/Log2[N[classes]]],
        "Events" -> Length[evIds],
        "RuleCounts" -> KeySort[Counts[Lookup[ruleOf, evIds]]],
        "Invariants" -> Join[<|
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
          AssociationMap[summary[DeleteMissing[Lookup[geo, #]], If[# === "GraphRadius", 1, 0.01]] &,
            geometryKeys]],
        "ArityHistogram" -> KeySort[Counts[Catenate[Lookup[cheap, "Arities"]]]],
        "AritySignatureHistogram" -> KeySort[Counts[Lookup[cheap, "Arities"]]],
        "DegreeHistogram" -> KeySort[Counts[Catenate[Lookup[cheap, "DegreeSequence"]]]],
        "DegreeSequenceHistogram" -> KeySort[Counts[Lookup[cheap, "DegreeSequence"]]],
        "BallGrowthDimension" -> Association[Table[
          r -> summary[Select[balls, Length[#] >= r &][[All, r]], 0.01],
          {r, Max[0, Length /@ balls]}]]|>;
      If[branchial, Join[rec, branchialKeys[sts, Lookup[pairsByStep, k, {}]]], rec]]],
    byStep]];

(* === comparison === *)

(* Associations compare key by key after sorting; numbers compare with a relative tolerance, so
   a histogram keyed 2 and one keyed 2. are the same histogram. *)
normal[a_Association] := KeySort[KeyMap[If[NumberQ[#], N[#], #] &, normal /@ a]];
normal[l_List] := normal /@ l;
normal[x_?NumberQ] := N[x];
normal[x_] := x;
close[a_Association, b_Association] := Keys[a] === Keys[b] &&
  AllTrue[Keys[a], If[# === "Histogram", histogramClose[a[#], b[#]], close[a[#], b[#]]] &];
(* A histogram of reals rounded to 0.01 bins a value that lies on a bin's edge by the last bit
   of its computation: the oracle's exact rational and the engine's double can round to
   neighbouring bins. Two histograms agree when they bin the same number of values and the
   k-th smallest binned values differ by at most one bin; the values themselves are compared
   by Mean, Min, Max and Median. Integer histograms have bins 1 apart, so they must agree
   exactly. *)
histogramClose[a_Association, b_Association] := close[Normal[a], Normal[b]] || Module[
  {ea = Catenate[ConstantArray[#1, Round[#2]] & @@@ Normal[a]],
   eb = Catenate[ConstantArray[#1, Round[#2]] & @@@ Normal[b]]},
  Length[ea] == Length[eb] && Max[Abs[ea - eb]] <= 0.0100001];
histogramClose[a_, b_] := close[a, b];
close[a_List, b_List] := Length[a] == Length[b] && And @@ MapThread[close, {a, b}];
close[a_?NumberQ, b_?NumberQ] := Abs[a - b] <= 10^-9 Max[1, Abs[a], Abs[b]];
close[a_, b_] := a === b;

firstDifference[o_Association, e_Association, path_ : {}] := Catch[
  If[Keys[o] =!= Keys[e], Throw[{path, "keys", Complement[Keys[o], Keys[e]], Complement[Keys[e], Keys[o]]}]];
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
  {"TiePathFirst", {{{1, 1}} -> {{1, 1}}}, {{1, 2}, {2, 2}, {3, 4, 5}}, 2},
  (* The rules of the multiway-statistics research corpus not covered above. *)
  {"CorpusTriangle", {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}}, {{1, 2}, {1, 3}}, 4},
  {"CorpusLoops", {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}}, {{1, 1}, {1, 1}}, 4},
  {"CorpusTernary", {{{1, 2, 3}} -> {{1, 2, 4}, {2, 4, 3}}}, {{1, 1, 1}}, 4},
  {"CorpusTwoRule", {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}, {{1, 1}, {1, 1}}, 3}
};

(* Random rules: the generator of the oracle fuzz. A rule is {lhs, rhs}; vertices of the right
   side not on the left are new. *)
genCase[seed_] := Module[{arities, nR, rules, init, nv},
  SeedRandom[seed];
  arities = RandomChoice[{{2}, {2}, {2}, {1, 2}, {2, 3}, {3}, {1}, {1, 3}, {1, 2, 3}}];
  nR = RandomChoice[{1, 1, 1, 2, 2, 3}];
  rules = Table[
    Module[{nL, p, lhs, lvars, nRhs, rhs, fresh},
      nL = RandomChoice[{1, 1, 2, 2, 2, 3}];
      p = RandomInteger[{1, Max[1, nL*Max[arities] - 1]}];
      lhs = Table[RandomInteger[{1, p}, RandomChoice[arities]], {nL}];
      lvars = DeleteDuplicates[Flatten[lhs]];
      lhs = lhs /. Thread[lvars -> Range[Length[lvars]]];
      lvars = Range[Length[lvars]];
      nRhs = RandomChoice[{0, 1, 1, 2, 2, 2, 3, 3, 4}];
      fresh = Length[lvars] + Range[RandomInteger[{0, 2}]];
      rhs = Table[Table[If[fresh =!= {} && RandomReal[] < 0.3, RandomChoice[fresh], RandomChoice[lvars]],
          {RandomChoice[arities]}], {nRhs}];
      {lhs, rhs}], {nR}];
  nv = RandomInteger[{1, 4}];
  init = Module[{lhs = RandomChoice[rules][[1]], vs, sub},
    vs = DeleteDuplicates[Flatten[lhs]];
    sub = Thread[vs -> RandomInteger[{1, Max[nv, 1]}, Length[vs]]];
    RandomSample[Join[lhs /. sub,
      Table[RandomInteger[{1, nv}, RandomChoice[arities]], {RandomChoice[{0, 0, 1, 1, 2}]}]]]];
  {"Random" <> ToString[seed], Rule @@@ rules, init, RandomChoice[{2, 3, 3, 4}]}];

seeds = If[Length[$ScriptCommandLine] >= 3,
  Range[ToExpression[$ScriptCommandLine[[2]]], ToExpression[$ScriptCommandLine[[3]]]], {}];
(* A random case whose raw run exceeds 400 states is cut to fewer steps. *)
randomCases = DeleteCases[Table[Module[{c = genCase[s], n},
    While[c[[4]] >= 1 &&
        (n = Quiet[HGEvolve[c[[2]], c[[3]], c[[4]], "NumStates", "CanonicalizeStates" -> None]];
         !IntegerQ[n] || n > 400), c[[4]]--];
    If[c[[4]] >= 1, c, Nothing]], {s, seeds}], Nothing];

failures = 0;
allStates = {};
Do[
  Module[{name, rules, init, steps, raw, oracle, oracleClasses, oracleBranchial, routes},
    {name, rules, init, steps} = c;
    raw = HGEvolve[rules, init, steps, {"States", "Events", "BranchialEdges"},
      "CanonicalizeStates" -> None, "IncludeCanonicalHashes" -> True];
    allStates = Join[allStates, #Edges[[All, 2 ;;]] & /@ Values[raw["States"]]];
    oracle = normal[oracleSteps[raw]];
    oracleClasses = normal[oracleSteps[raw, True]];
    oracleBranchial = normal[oracleSteps[raw, False, True]];
    routes = {
      {"quotient", oracle, HGEvolve[rules, init, steps, "StepStatistics",
        "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]},
      {"full capture", oracle, HGEvolve[rules, init, steps, "StepStatistics",
        "CanonicalizeStates" -> None]},
      {"classes, quotient", oracleClasses, HGEvolve[rules, init, steps, "StepStatistics",
        "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True,
        "StepStatisticsWeighting" -> "Classes"]},
      {"classes, full capture", oracleClasses, HGEvolve[rules, init, steps, "StepStatistics",
        "CanonicalizeStates" -> None, "StepStatisticsWeighting" -> "Classes"]},
      {"branchial", oracleBranchial, HGEvolve[rules, init, steps, "StepStatistics",
        "CanonicalizeStates" -> None, "StepStatisticsBranchial" -> All]}};
    Do[
      Module[{want = r[[2]], got = normal[r[[3]]], diffs},
        diffs = If[!ListQ[got] || Length[got] =!= Length[want], {"step count", Length[got], Length[want]},
          DeleteCases[MapThread[firstDifference, {want, got}], None]];
        If[diffs === {},
          Print["PASS  ", name, " ", r[[1]], ": ", Length[want], " steps"],
          failures++; Print["FAIL  ", name, " ", r[[1]], ": ", Short[First[Flatten[{diffs}, 0]], 6]]]],
      {r, routes}]],
  {c, Join[cases, randomCases]}];

(* === the Function Repository's definitions === *)

$AllowInternet = True;
whd = Quiet[ResourceFunction["WolframHausdorffDimension"]];
wrc = Quiet[ResourceFunction["WolframRicciCurvatureScalar"]];
If[Head[whd] =!= ResourceFunction || Head[wrc] =!= ResourceFunction,
  failures++; Print["FUNCTION REPOSITORY UNAVAILABLE"],
  Module[{dev, worst = <|"WolframHausdorffDimension" -> 0., "BallGrowthDimension" -> 0.,
      "WolframRicciCurvatureScalar" -> 0.|>, n = 0},
    dev[a_, b_] := Abs[a - b]/Max[1, Abs[a], Abs[b]];
    Do[
      Module[{g = sgrStateGraph[s], geo, d},
        If[VertexCount[g] >= 2 && ConnectedGraphQ[g],
          geo = sgrStateGeometry[s];
          d = <|
            "WolframHausdorffDimension" -> dev[geo["WolframHausdorffDimension"],
              whd[g, All, "AllDimensions", "VertexMethod" -> Mean]],
            "BallGrowthDimension" -> Max[MapThread[dev, {sgrBallGrowth[s],
              whd[g, All, GraphRadius[g], "AllDimensions", "DimensionMethod" -> Identity,
                "VertexMethod" -> Mean]}]],
            "WolframRicciCurvatureScalar" -> dev[geo["WolframRicciCurvatureScalar"],
              wrc[g, geo["WolframHausdorffDimension"], All, "AllCurvatures",
                "VertexMethod" -> Mean]]|>;
          worst = Merge[{worst, d}, Max]; n++]],
      {s, DeleteDuplicates[allStates]}];
    If[Max[worst] <= 10^-9,
      Print["PASS  Function Repository: ", n, " states, largest relative deviation ", worst],
      failures++; Print["FAIL  Function Repository: ", n, " states, largest relative deviation ", worst]]]];

Print[If[failures == 0, "STATE STATISTICS VERIFIED", "STATE STATISTICS FAILED: " <> ToString[failures]]];
