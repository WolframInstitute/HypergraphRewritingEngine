(* ::Package:: *)

(* StateGeometryReference.wl

   An independent Wolfram Language implementation of the "StepStatistics" geometry and
   branchial metrics defined in docs/SPEC.md ("StepStatistics geometry" and "StepStatistics
   branchial metrics"). The engine computes them in common/include/hgcommon/state_geometry_core.hpp
   and paclet_source/paclet_support.cpp; reference/verify_state_statistics.wls compares the two,
   and compares the two Function Repository metrics here against ResourceFunction itself.

   Every function here uses Graph built-ins (VertexOutComponent, GraphRadius,
   VertexEccentricity, ConnectedComponents, GraphDistanceMatrix) and LinearProgramming, and none
   of the engine's algorithms: the transport cost is an exact linear program, not successive
   shortest paths, and the ball sizes come from VertexOutComponent per radius, not from one
   search per vertex. *)

BeginPackage["StateGeometryReference`"];

sgrStateGraph::usage = "sgrStateGraph[edges] is the undirected simple graph of a state: its vertices, and an edge between the consecutive vertices of each hyperedge (sgrStatisticsGraph \"Consecutive\") or between every two vertices of a hyperedge (\"AllPairs\"), without self-loops.";
sgrStatisticsGraph::usage = "sgrStatisticsGraph is the rule of sgrStateGraph, the value of HGEvolve's \"StatisticsGraph\": \"Consecutive\" (the default) or \"AllPairs\". Every function of this package reads it.";
sgrStateGeometry::usage = "sgrStateGeometry[edges] gives an association of the per-state geometry metrics; an undefined metric is Missing[].";
sgrBallGrowth::usage = "sgrBallGrowth[edges] gives the per-radius ball-growth dimensions for r = 1..R, or {} when undefined.";
sgrVertexDimensions::usage = "sgrVertexDimensions[g] gives each vertex's own Hausdorff dimension over r = 1..GraphRadius[g].";
sgrBranchialGraphMetrics::usage = "sgrBranchialGraphMetrics[nodes, pairs] gives the per-step branchial graph metrics for a step's states and its branchial state pairs.";
sgrOverlapMetrics::usage = "sgrOverlapMetrics[vertexSets, edgeLists, initial] gives the per-step overlap values for a step's states, each given as its vertex set and its list of edges (each edge its vertex list); initial is the vertex set of step 0's states.";
sgrOverlapByDistance::usage = "sgrOverlapByDistance[vertexSets, distanceMatrix] gives, for each finite branchial distance d > 0, the Jaccard index of every two states at distance d.";
sgrVertexCurvature::usage = "sgrVertexCurvature[edges] gives, for each vertex of the state's graph with a neighbour, ascending, the mean Ollivier-Ricci curvature of its edges.";
sgrLargestIncidenceComponent::usage = "sgrLargestIncidenceComponent[edges] gives the incidence graph of a state (a node {\"v\", x} per vertex and {\"e\", i} per edge), its connected components, and the largest one: the most nodes, then the greatest diameter, then the greatest mean distance, then the most vertices.";
sgrVertexInvariants::usage = "sgrVertexInvariants[edges] gives the per-vertex values \"VertexInvariants\" pools for one state: \"LocalDimension\" (d_v over the largest component G'), \"OllivierRicciCurvature\" (k_v over the vertices with a neighbour) and \"WolframRicciCurvatureScalar\" (K_v over G'), each a list of numbers of 50 digits or exact rationals.";
sgrPooledDistribution::usage = "sgrPooledDistribution[values, name] gives the summary \"VertexInvariants\" reports for the pooled values of the distribution name: N, Mean, StandardDeviation, Min, Max, Median, Q1, Q3, P10, P90, Skewness, Kurtosis and the 32-bin Histogram.";
sgrVolumeGrowth::usage = "sgrVolumeGrowth[g, v] gives the six volume-growth values of vertex v of graph g (BallDimension, BallScalarCurvature, RiemannianBallDimension, RiemannianBallScalarCurvature, SphereDimension, SphereScalarCurvature), or Missing[] when v has no fit.";
sgrLogGeodesicCount::usage = "sgrLogGeodesicCount[edges] gives, for d = 1..D, the mean over the ordered pairs of vertices at distance d of the natural logarithm of their number of shortest paths, or {} for a state with no edge.";
sgrEdgeInvariants::usage = "sgrEdgeInvariants[edges] gives the per-edge values \"EdgeInvariants\" pools for one state: \"NonLazyOllivierRicciCurvature\", the curvature of each edge of the state's graph with the walk of idleness 0, as exact rationals.";
sgrCombinatorial::usage = "sgrCombinatorial[edges] gives the EulerCharacteristic, RimFraction and CombinatorialCurvatureTotal of a state, and \"Curvatures\", 1 - k/6 for each vertex whose link is a k-cycle, ascending by vertex.";
sgrParticle::usage = "sgrParticle[edges] gives the per-vertex clustering coefficients, Euler excesses and topological charges of a state, ascending by vertex, as exact numbers.";
sgrStateMatrices::usage = "sgrStateMatrices[vertexSets, edgeLists] gives the six state x state matrices of a step's states in the given order: the Jaccard index, the cosine similarity and the mutual information of every two states, over their vertex sets and over their edge sets (each edge its vertex list).";

Begin["`Private`"];

sgrStatisticsGraph = "Consecutive";

sgrStateGraph[edges_List] := Graph[
  Union[Flatten[edges]],
  UndirectedEdge @@@ DeleteDuplicates[Sort /@ Select[
    Catenate[If[sgrStatisticsGraph === "AllPairs", Subsets[#, {2}], Partition[#, 2, 1]] & /@
      edges], #[[1]] =!= #[[2]] &]]];

ball[g_, v_, r_] := Length[VertexOutComponent[g, v, r]];

term[g_, v_, r_] := (Log[ball[g, v, r]] - Log[ball[g, v, r - 1]])/(Log[r + 1] - Log[r]);

sgrVertexDimensions[g_Graph] := Module[{R = GraphRadius[g]},
  Association[# -> N[Mean[Table[term[g, #, r], {r, R}]]] & /@ VertexList[g]]];

connectedQ[g_Graph] := VertexCount[g] >= 1 && ConnectedGraphQ[g];

entropyBits[counts_List] := Module[{p = N[counts/Total[counts]]}, -Total[p Log2[p]]];

degreeEntropy[g_Graph, vs_List] := entropyBits[Values[Counts[VertexDegree[g, #] & /@ vs]]];

(* The exact optimum of the transport linear program from masses mu on the vertices ax to
   masses nu on the vertices by, under the graph distance. *)
transportCost[dm_, index_, ax_, mu_, by_, nu_] := Module[{a = Length[ax], b = Length[by], cost, eqs},
  cost = Flatten[Table[dm[[index[ax[[i]]], index[by[[j]]]]], {i, a}, {j, b}]];
  eqs = Join[
    Table[Flatten[Table[If[i == k, 1, 0], {i, a}, {j, b}]], {k, a}],
    Table[Flatten[Table[If[j == k, 1, 0], {i, a}, {j, b}]], {k, b}]];
  cost . LinearProgramming[cost, eqs, Join[{#, 0} & /@ mu, {#, 0} & /@ nu]]];

(* The lazy walk: 1/2 at x, 1/(2 deg x) at each neighbour. *)
ollivierEdge[g_Graph, dm_, index_, x_, y_] := Module[{ax, by},
  ax = Prepend[AdjacencyList[g, x], x]; by = Prepend[AdjacencyList[g, y], y];
  1 - transportCost[dm, index,
    ax, Prepend[ConstantArray[1/(2 Length[ax] - 2), Length[ax] - 1], 1/2],
    by, Prepend[ConstantArray[1/(2 Length[by] - 2), Length[by] - 1], 1/2]]];

(* The non-lazy walk (idleness 0): 1/deg x at each neighbour, none at x. *)
nonLazyEdge[g_Graph, dm_, index_, x_, y_] := Module[
  {ax = AdjacencyList[g, x], by = AdjacencyList[g, y]},
  1 - transportCost[dm, index, ax, ConstantArray[1/Length[ax], Length[ax]],
    by, ConstantArray[1/Length[by], Length[by]]]];

(* The exact non-lazy curvature of each edge of G. *)
nonLazyCurvatures[g_Graph] := Module[{vs = VertexList[g], dm, index},
  If[EdgeCount[g] == 0, Return[{}]];
  dm = GraphDistanceMatrix[g]; index = AssociationThread[vs -> Range[Length[vs]]];
  nonLazyEdge[g, dm, index, #[[1]], #[[2]]] & /@ EdgeList[g]];

sgrEdgeInvariants[edges_List] := <|"NonLazyOllivierRicciCurvature" ->
  nonLazyCurvatures[sgrStateGraph[edges]]|>;

(* Combinatorial curvature (docs/SPEC.md, "StepStatistics combinatorial curvature"): the link
   of v is the subgraph on v's neighbours, a k-cycle when it has k >= 3 vertices, each of link
   degree 2, and is connected. The triangles are FindCycle's cycles of length 3. *)
linkCycleLength[g_Graph, v_] := Module[{link = Subgraph[g, AdjacencyList[g, v]]},
  If[VertexCount[link] >= 3 && ConnectedGraphQ[link] && Union[VertexDegree[link]] === {2},
    VertexCount[link], Missing[]]];

sgrCombinatorial[edges_List] := Module[{g = sgrStateGraph[edges], vs, n, ks, cyc},
  vs = VertexList[g]; n = Length[vs];
  If[n == 0, Return[<|"EulerCharacteristic" -> Missing[], "RimFraction" -> Missing[],
    "CombinatorialCurvatureTotal" -> Missing[], "Curvatures" -> {}|>]];
  ks = linkCycleLength[g, #] & /@ Sort[vs];
  cyc = DeleteMissing[ks];
  <|"EulerCharacteristic" -> n - EdgeCount[g] + Length[FindCycle[g, {3}, All]],
    "RimFraction" -> If[AllTrue[vs, FindCycle[{g, #}, {3}, 1] =!= {} &],
      N[Count[ks, _Missing]/n], Missing[]],
    "CombinatorialCurvatureTotal" -> If[cyc === {}, Missing[], N[Total[1 - cyc/6]]],
    "Curvatures" -> (1 - cyc/6)|>];

(* Particle measures (docs/SPEC.md, "StepStatistics particle measures"). The triangles at v are
   the edges among v's neighbours; B(v, 3) is VertexOutComponent to depth 3. *)
sgrParticle[edges_List] := Module[{g = sgrStateGraph[edges], vs},
  vs = Sort[VertexList[g]];
  Transpose[Function[v, Module[{d = VertexDegree[g, v], t, c, ball, nV, nE, p, x},
    t = EdgeCount[Subgraph[g, AdjacencyList[g, v]]];
    c = If[d < 2, 0, t/(d (d - 1)/2)];
    ball = VertexOutComponent[g, v, 3];
    nV = Length[ball]; nE = EdgeCount[Subgraph[g, ball]];
    p = If[nV < 3, nV (nV - 1)/2, 3 nV - 6];
    x = If[p > 0 && nE > p, (nE - p)/p, 0];
    {c, x, Log2[Max[1, d]] (1 - c) + 10 x}]] /@ vs]];

growthNames = {"BallDimension", "BallScalarCurvature", "RiemannianBallDimension",
  "RiemannianBallScalarCurvature", "SphereDimension", "SphereScalarCurvature"};

(* Geodesic multiplicity (docs/SPEC.md, "StepStatistics geodesic multiplicity"). sigma(p, q) is
   the (p, q) entry of A^d with d = d(p, q), and the segment's density at w is
   sigma(p, w) sigma(w, q) over the vertices w with d(p, w) + d(w, q) = d. Each ordered pair of
   distinct vertices joined by a path gives {d, sigma, H}. *)
geodesicPairs[g_Graph] := Module[{vs = VertexList[g], n, a, dm, powers, top, sigma},
  n = Length[vs];
  If[EdgeCount[g] == 0, Return[{}]];
  a = Normal[AdjacencyMatrix[g]];
  dm = GraphDistanceMatrix[g];
  top = Max[Cases[Flatten[dm], _Integer]];
  powers = NestList[#.a &, IdentityMatrix[n], top];
  sigma[x_, y_] := powers[[dm[[x, y]] + 1, x, y]];
  Catenate[Table[If[p == q || dm[[p, q]] === Infinity, Nothing,
      Module[{d = dm[[p, q]], rho},
        rho = sigma[p, #] sigma[#, q] & /@ Select[Range[n], dm[[p, #]] + dm[[#, q]] == d &];
        {d, sigma[p, q], -Total[(rho/Total[rho]) Log2[rho/Total[rho]]]}]],
    {p, n}, {q, n}]]];

geodesicValues[edges_List] := geodesicValues[edges, sgrStatisticsGraph];
geodesicValues[edges_List, rule_String] := geodesicValues[edges, rule] = Module[{pairs = geodesicPairs[sgrStateGraph[edges]]},
  If[pairs === {},
    <|"GeodeticFraction" -> Missing[], "SegmentEntropy" -> Missing[], "LogGeodesicCount" -> {}|>,
    <|"GeodeticFraction" -> N[Count[pairs[[All, 2]], 1]/Length[pairs]],
      "SegmentEntropy" -> N[Mean[pairs[[All, 3]]], 20],
      "LogGeodesicCount" -> Table[N[Mean[Log[Select[pairs, #[[1]] == d &][[All, 2]]]], 20],
        {d, Max[pairs[[All, 1]]]}]|>]];

sgrLogGeodesicCount[edges_List] := N[geodesicValues[edges]["LogGeodesicCount"]];
geometryNames = Join[{"DimensionGradientCurvature", "ClusteringCoefficient", "LocalEulerExcess", "TopologicalCharge",
  "TopologicalChargeMax", "EulerCharacteristic", "RimFraction", "CombinatorialCurvatureTotal",
  "NonLazyOllivierRicciCurvature", "GeodeticFraction", "SegmentEntropy", "GraphRadius", "MeanEccentricity", "WolframHausdorffDimension",
  "WolframRicciCurvatureScalar", "OllivierRicciCurvature", "DegreeEntropy", "LocalEntropy",
  "MutualInformation", "FisherInformation", "LargestComponentDimension", "LocalDimensionMax",
  "LocalDimensionStandardDeviation", "OllivierMoranI", "OllivierDegreeCorrelation"},
  growthNames];

(* Volume growth (docs/SPEC.md, "StepStatistics volume growth"). The profiles come from one
   GraphDistance per vertex; the Riemannian measure counts, by its definition, the vertices of
   the ball whose neighbours all lie in the ball. Each window's fit is LeastSquares on the
   window's points, and its residual standard error the norm of the residuals. *)
growthFit[q_List, probe_] := Module[{k = Length[q], x, fit, rse, windows, tol, sel, c1, c2},
  If[k < 2, Return[Missing[]]];
  x = N[Range[k] (Range[k] + 1)];
  fit[i_, j_] := LeastSquares[Transpose[{ConstantArray[1., j - i + 1], x[[i ;; j]]}], q[[i ;; j]]];
  rse[i_, j_] := Module[{c = fit[i, j]},
    Sqrt[Total[(q[[i ;; j]] - (c[[1]] + c[[2]] x[[i ;; j]]))^2]/(j - i - 1)]];
  sel = If[k < 5, {1, k},
    tol = Max[2 Sort[Table[rse[i, i + 4], {i, k - 4}]][[Ceiling[(k - 4)/4]]], 10.^-10];
    windows = Catenate[Table[{i, i + len - 1}, {len, k, 5, -1}, {i, 1, k - len + 1}]];
    SelectFirst[windows, rse @@ # <= tol &, {1, k}]];
  {c1, c2} = fit @@ sel;
  If[probe === "Ball", {c1, -3 (c1 + 2) c2}, {c1 + 1, -3 (c1 + 1) c2}]];

sgrVolumeGrowth[g_Graph, v_] := Module[{d = GraphDistance[g, v], vs = VertexList[g], within, e,
    shell, ball, riem, p, q},
  within = Pick[vs, Thread[d < Infinity]];
  d = AssociationThread[vs -> d];
  e = Max[Lookup[d, within]];
  shell = Table[Count[Lookup[d, within], r], {r, 0, e}];
  ball = Accumulate[shell];
  riem = Table[Count[within, u_ /; d[u] <= r && AllTrue[AdjacencyList[g, u], d[#] <= r &]],
    {r, 0, e}];
  p = Last[Flatten[Position[shell, Max[shell]]]] - 1;
  If[p < 3, Return[Missing[]]];
  q[f_] := Table[(Log[N[f[[r + 2]]]] - Log[N[f[[r + 1]]]])/(Log[r + 1.] - Log[N[r]]), {r, 1, p - 1}];
  Join[growthFit[q[ball], "Ball"], growthFit[q[riem], "Ball"], growthFit[q[shell], "Sphere"]]];

(* Each vertex of G' with a fit: vertex -> its six values. *)
growthOfLargest[edges_List] := Module[{g = sgrStateGraph[edges], gp},
  gp = Cases[sgrLargestIncidenceComponent[edges]["Largest"], {"v", x_} :> x];
  DeleteMissing[AssociationMap[sgrVolumeGrowth[g, #] &, Sort[gp]]]];

sgrLargestIncidenceComponent[edges_List] := Module[{vertices, m, incEdges, g, comps},
  vertices = Union[Flatten[edges]]; m = Length[edges];
  incEdges = DeleteDuplicates[Catenate[MapIndexed[
    Function[{v, i}, UndirectedEdge[{"e", First[i]}, {"v", #}] & /@ DeleteDuplicates[v]], edges]]];
  g = Graph[Join[{"v", #} & /@ vertices, {"e", #} & /@ Range[m]], incEdges];
  comps = ConnectedComponents[g];
  <|"Graph" -> g, "IncidenceEdges" -> incEdges, "Components" -> comps,
    "Largest" -> If[comps === {}, {}, First[MaximalBy[comps, {Length[#],
      If[Length[#] <= 1, 0, GraphDiameter[Subgraph[g, #]]],
      If[Length[#] <= 1, 0., N[MeanGraphDistance[Subgraph[g, #]]]],
      Count[#, {"v", _}]} &]]]|>];

(* The exact Ollivier-Ricci curvature of each edge of G, keyed by its sorted endpoints. *)
edgeCurvatures[g_Graph] := Module[{vs = VertexList[g], dm, index},
  If[EdgeCount[g] == 0, Return[<||>]];
  dm = GraphDistanceMatrix[g]; index = AssociationThread[vs -> Range[Length[vs]]];
  Association[Sort[List @@ #] -> ollivierEdge[g, dm, index, #[[1]], #[[2]]] & /@ EdgeList[g]]];

(* k_v, exact: the mean curvature of v's edges, for each vertex of G with a neighbour, ascending. *)
vertexCurvatureExact[g_Graph, kE_Association] := KeySort[Association[Function[v,
  v -> Mean[Function[w, kE[Sort[{v, w}]]] /@ AdjacencyList[g, v]]] /@
    Select[VertexList[g], VertexDegree[g, #] > 0 &]]];

(* G', d_v over G' (50 digits), D and K_v over G' (50 digits); Missing[] when G' has fewer than
   2 vertices. *)
largestComponentGeometry[edges_List] := Module[{g = sgrStateGraph[edges], gp, vs, R, dv, D},
  gp = Subgraph[g, Cases[sgrLargestIncidenceComponent[edges]["Largest"], {"v", x_} :> x]];
  vs = VertexList[gp];
  If[Length[vs] < 2, Return[Missing[]]];
  R = GraphRadius[gp];
  dv = Association[# -> N[Mean[Table[term[gp, #, r], {r, R}]], 50] & /@ vs];
  D = Mean[Values[dv]];
  <|"Dimensions" -> dv, "D" -> D,
    "Ricci" -> Association[# -> Mean[Table[
      6 (D + 2)/r^2 (1 - ball[gp, #, r] Gamma[D/2 + 1]/(Pi^(D/2) r^D)), {r, R}]] & /@ vs]|>];

(* L_v over G': the mean over v's neighbours of d_u - d_v, with the 50-digit d_v of
   largestComponentGeometry. *)
dimensionGradient[edges_List] := Module[{g = sgrStateGraph[edges], lc},
  lc = largestComponentGeometry[edges];
  If[MissingQ[lc], Return[{}]];
  Function[v, Mean[(lc["Dimensions"][#] - lc["Dimensions"][v]) & /@ AdjacencyList[g, v]]] /@
    Keys[lc["Dimensions"]]];

sgrVertexInvariants[edges_List] := Module[{g = sgrStateGraph[edges], lc, growth},
  lc = largestComponentGeometry[edges];
  growth = Values[growthOfLargest[edges]];
  Join[<|"LocalDimension" -> If[MissingQ[lc], {}, Values[lc["Dimensions"]]],
    "OllivierRicciCurvature" -> Values[vertexCurvatureExact[g, edgeCurvatures[g]]],
    "WolframRicciCurvatureScalar" -> If[MissingQ[lc], {}, Values[lc["Ricci"]]]|>,
    AssociationThread[growthNames -> If[growth === {}, ConstantArray[{}, 6], Transpose[growth]]],
    <|"CombinatorialCurvature" -> sgrCombinatorial[edges]["Curvatures"],
      "ClusteringCoefficient" -> If[VertexCount[g] == 0, {}, sgrParticle[edges][[1]]],
      "TopologicalCharge" -> If[VertexCount[g] == 0, {}, N[sgrParticle[edges][[3]], 30]],
      "DimensionGradientCurvature" -> dimensionGradient[edges]|>]];

(* The per-vertex distributions' fixed ranges [lo, hi), each cut into 32 equal bins; a value
   outside counts in the end bin. *)
distributionRange = <|"LocalDimension" -> {0, 8}, "OllivierRicciCurvature" -> {-2, 1},
  "WolframRicciCurvatureScalar" -> {-24, 8}, "BallDimension" -> {0, 8},
  "BallScalarCurvature" -> {-24, 8}, "RiemannianBallDimension" -> {0, 8},
  "RiemannianBallScalarCurvature" -> {-24, 8}, "SphereDimension" -> {0, 8},
  "SphereScalarCurvature" -> {-24, 8}, "NonLazyOllivierRicciCurvature" -> {-2, 1},
  "CombinatorialCurvature" -> {-7, 1}, "ClusteringCoefficient" -> {0, 1},
  "TopologicalCharge" -> {0, 16}, "DimensionGradientCurvature" -> {-8, 8}|>;

sgrPooledDistribution[values_List, name_String] := Module[
  {n = Length[values], lo, hi, w, bin, counts, mn, mx, quantile, out},
  If[n == 0, Return[<|"N" -> 0|>]];
  {lo, hi} = distributionRange[name]; w = (hi - lo)/32;
  (* A value less than 10^-9 bin widths below a bin's lower edge counts in that bin: d_v of 50
     digits whose exact value is the edge, as (Log[9] - Log[4])/(Log[3] - Log[2]) = 2, and a
     machine fit whose exact value is the edge. *)
  bin[x_] := Clip[Floor[(x - lo)/w + 10^-9], {0, 31}];
  counts = Counts[bin /@ values];
  mn = Min[values]; mx = Max[values];
  (* The bin holding the fraction q of the values, linear inside it from its lower edge to its
     upper edge, the edges clipped to [Min, Max]; the first bin starts at Min and the last ends
     at Max, since they hold the values outside the range. *)
  quantile[q_] := Module[{t = q n, seen = 0, c, a, z},
    Catch[
      Do[c = Lookup[counts, b, 0];
        If[c == 0 || seen + c < t, seen += c; Continue[]];
        a = If[b == 0, mn, Max[lo + b w, mn]];
        z = If[b == 31, mx, Min[lo + (b + 1) w, mx]];
        Throw[Min[Max[a + (t - seen)/c (z - a), mn], mx]], {b, 0, 31}];
      mx]];
  out = <|"N" -> n, "Mean" -> N[Mean[values]],
    "StandardDeviation" -> If[n > 1, N[StandardDeviation[values]], 0.],
    "Min" -> N[mn], "Max" -> N[mx], "Median" -> N[quantile[1/2]], "Q1" -> N[quantile[1/4]],
    "Q3" -> N[quantile[3/4]], "P10" -> N[quantile[1/10]], "P90" -> N[quantile[9/10]]|>;
  If[mn != mx, out["Skewness"] = N[Skewness[values]]; out["Kurtosis"] = N[Kurtosis[values]]];
  out["Histogram"] = Association[N[lo + # w] -> counts[#] & /@ Sort[Keys[counts]]];
  out];

sgrStateGeometry[edges_List] := Module[
  {g, n, conn, R, vs, dims, d, ricci, dm, index, ollivier, local, mi, fisher, b2, kE, kv, lc,
   moran, corr},
  g = sgrStateGraph[edges];
  vs = VertexList[g]; n = Length[vs];
  If[n == 0, Return[AssociationMap[Missing[] &, geometryNames]]];
  conn = ConnectedGraphQ[g];
  R = If[conn, GraphRadius[g], Infinity];
  b2[v_] := VertexOutComponent[g, v, 2];
  local = N[Mean[degreeEntropy[g, b2[#]] & /@ vs]];
  mi = Module[{withNbr = Select[vs, VertexDegree[g, #] > 0 &]},
    If[withNbr === {}, Missing[],
      N[Mean[Function[v, Mean[Function[w, Module[{bv = b2[v], bw = b2[w], both, uni},
        both = Length[Intersection[bv, bw]]; uni = Length[Union[bv, bw]];
        Max[0, Log2[both uni/(Length[bv] Length[bw])]]]] /@ AdjacencyList[g, v]]] /@ withNbr]]]];
  kE = edgeCurvatures[g];
  ollivier = If[EdgeCount[g] == 0, Missing[], N[Mean[Values[kE]]]];
  (* Moran's I of k_v over the edges of G, and the correlation of k_v with the degree, over the
     vertices with a neighbour; undefined when k_v (or the degree) is the same at each. *)
  kv = vertexCurvatureExact[g, kE];
  {moran, corr} = If[EdgeCount[g] == 0 || Min[kv] == Max[kv], {Missing[], Missing[]},
    Module[{z = kv - Mean[kv], degs = VertexDegree[g, #] & /@ Keys[kv]},
      {N[Length[kv]/EdgeCount[g] Total[z[#[[1]]] z[#[[2]]] & /@ EdgeList[g]]/Total[Values[z]^2]],
       If[Min[degs] == Max[degs], Missing[], N[Correlation[Values[kv], degs]]]}]];
  lc = largestComponentGeometry[edges];
  If[conn && R >= 1,
    dims = sgrVertexDimensions[g];
    d = N[Mean[Values[dims]]];
    ricci = N[Mean[Function[v, Mean[Table[
      6 (d + 2)/r^2 (1 - ball[g, v, r] Gamma[d/2 + 1]/(Pi^(d/2) r^d)), {r, R}]]] /@ vs]];
    fisher = N[Mean[Function[v, Module[{us = DeleteCases[b2[v], v], du},
      du = Lookup[dims, us];
      (1 + Mean[Abs[du - dims[v]]])/(Mean[(du - Mean[du])^2] + 1/100)]] /@ vs]],
    d = Missing[]; ricci = Missing[]; fisher = Missing[]];
  <|"GraphRadius" -> If[conn, R, Missing[]],
    "MeanEccentricity" -> If[conn, N[Mean[VertexEccentricity[g, #] & /@ vs]], Missing[]],
    "WolframHausdorffDimension" -> d,
    "WolframRicciCurvatureScalar" -> ricci,
    "OllivierRicciCurvature" -> ollivier,
    "DegreeEntropy" -> degreeEntropy[g, vs],
    "LocalEntropy" -> local,
    "MutualInformation" -> mi,
    "FisherInformation" -> fisher,
    "LargestComponentDimension" -> If[MissingQ[lc], Missing[], N[lc["D"]]],
    "LocalDimensionMax" -> If[MissingQ[lc], Missing[], N[Max[Values[lc["Dimensions"]]]]],
    "LocalDimensionStandardDeviation" -> If[MissingQ[lc], Missing[],
      N[Sqrt[Mean[(Values[lc["Dimensions"]] - lc["D"])^2]]]],
    "OllivierMoranI" -> moran,
    "OllivierDegreeCorrelation" -> corr,
    Sequence @@ With[{pv = sgrParticle[edges]},
      Thread[{"ClusteringCoefficient", "LocalEulerExcess", "TopologicalCharge"} -> N[Mean /@ pv]]],
    "TopologicalChargeMax" -> N[Max[sgrParticle[edges][[3]]]],
    "DimensionGradientCurvature" -> With[{l = dimensionGradient[edges]},
      If[l === {}, Missing[], N[Mean[l]]]],
    "EulerCharacteristic" -> sgrCombinatorial[edges]["EulerCharacteristic"],
    "RimFraction" -> sgrCombinatorial[edges]["RimFraction"],
    "CombinatorialCurvatureTotal" -> sgrCombinatorial[edges]["CombinatorialCurvatureTotal"],
    "NonLazyOllivierRicciCurvature" -> With[{k = nonLazyCurvatures[g]},
      If[k === {}, Missing[], N[Mean[k]]]],
    "GeodeticFraction" -> geodesicValues[edges]["GeodeticFraction"],
    "SegmentEntropy" -> N[geodesicValues[edges]["SegmentEntropy"]],
    Sequence @@ With[{growth = Values[growthOfLargest[edges]]},
      Thread[growthNames -> If[growth === {}, ConstantArray[Missing[], 6], Mean[growth]]]]|>];

sgrBallGrowth[edges_List] := Module[{g = sgrStateGraph[edges], R},
  If[!connectedQ[g] || VertexCount[g] < 2, Return[{}]];
  R = GraphRadius[g];
  N[Mean[Table[term[g, #, r], {r, R}] & /@ VertexList[g]]]];

(* A step's branchial graph: the step's states, joined when a branchial pair at the step has
   them as output states; a pair whose two output states are one state adds nothing. *)
sgrBranchialGraphMetrics[nodes_List, pairs_List] := Module[
  {g, comps, dm, dists, largest, sub, dim},
  g = Graph[nodes, UndirectedEdge @@@ DeleteDuplicates[Sort /@ Select[pairs, #[[1]] =!= #[[2]] &]]];
  comps = ConnectedComponents[g];
  dm = GraphDistanceMatrix[g];
  dists = Select[Flatten[Table[dm[[i, j]], {i, Length[nodes]}, {j, i + 1, Length[nodes]}]],
    # =!= Infinity &];
  (* The largest component: most states, then the greatest dimension, then the least state. *)
  dim[c_] := If[Length[c] < 2, Missing[],
    Mean[Values[sgrVertexDimensions[Subgraph[g, c]]]]];
  largest = MaximalBy[SortBy[comps, Min],
    {Length[#], Replace[dim[#], _Missing -> -Infinity]} &];
  <|"Degrees" -> (VertexDegree[g, #] & /@ nodes),
    "Distances" -> dists,
    "DistanceMatrix" -> dm,
    "Components" -> Length[comps],
    "Dimension" -> If[largest === {}, Missing[], dim[First[largest]]],
    "Measures" -> If[largest === {} || Length[First[largest]] < 2, <||>,
      branchialMeasures[Subgraph[g, First[largest]]]]|>];

(* The volume growth and the two Ollivier-Ricci curvatures of a connected branchial component. *)
branchialMeasures[c_Graph] := Module[{growth, vs = VertexList[c], dm, index},
  growth = DeleteMissing[sgrVolumeGrowth[c, #] & /@ vs];
  dm = GraphDistanceMatrix[c]; index = AssociationThread[vs -> Range[Length[vs]]];
  Join[
    If[growth === {}, <||>,
      AssociationThread[("Branchial" <> #) & /@ growthNames -> Mean[growth]]],
    <|"BranchialOllivierRicciCurvature" ->
        N[Mean[ollivierEdge[c, dm, index, #[[1]], #[[2]]] & /@ EdgeList[c]]],
      "BranchialNonLazyOllivierRicciCurvature" ->
        N[Mean[nonLazyEdge[c, dm, index, #[[1]], #[[2]]] & /@ EdgeList[c]]]|>]];

(* The mutual information in bits of the indicators of two sets of sizes a and b sharing n11
   elements, inside a universe of u elements. *)
mutualInformation[n11_, a_, b_, u_] := If[u == 0, 0., Max[0., N[Total[MapThread[
    If[#1 == 0, 0, #1/u Log2[#1 u/(#2 #3)]] &,
    {{n11, a - n11, b - n11, u - a - b + n11}, {a, a, u - a, u - a}, {b, u - b, b, u - b}}]]]]];

sgrOverlapMetrics[vertexSets_List, edgeLists_List, initial_List] := Module[
  {k = Length[vertexSets], sets = DeleteDuplicates /@ vertexSets, counts, edgeCounts, pairs,
   shared, u},
  counts = Counts[Catenate[sets]];
  edgeCounts = Counts[Catenate[DeleteDuplicates /@ edgeLists]];
  u = Length[counts];
  pairs = Flatten[Table[{i, j}, {i, k}, {j, i + 1, k}], 1];
  shared[{i_, j_}] := Length[Intersection[sets[[i]], sets[[j]]]];
  <|"StateOverlap" -> (With[{w = Length[Union[sets[[#[[1]]]], sets[[#[[2]]]]]]},
        (* Two states with no vertices share nothing: overlap 0. *)
        If[w == 0, 0., N[shared[#]/w]]] & /@ pairs),
    "StateCosineSimilarity" -> (With[{a = Length[sets[[#[[1]]]]], b = Length[sets[[#[[2]]]]]},
        If[a b == 0, 0., N[shared[#]/Sqrt[a b]]]] & /@ pairs),
    "StateMutualInformation" -> (mutualInformation[shared[#], Length[sets[[#[[1]]]]],
        Length[sets[[#[[2]]]]], u] & /@ pairs),
    "InitialStateMutualInformation" -> With[{w = Length[Union[Keys[counts], initial]]},
      mutualInformation[Length[Intersection[#, initial]], Length[#], Length[initial], w] & /@
        sets],
    "VertexSharpness" -> N[1/Values[counts]],
    "BranchEntropy" -> N[Log2[Values[counts]]],
    "EdgeSharpness" -> N[1/Values[edgeCounts]],
    "EdgeBranchEntropy" -> N[Log2[Values[edgeCounts]]]|>];

sgrOverlapByDistance[vertexSets_List, dm_] := Module[
  {k = Length[vertexSets], sets = DeleteDuplicates /@ vertexSets},
  KeySort[GroupBy[Select[Flatten[Table[{dm[[i, j]],
        With[{w = Length[Union[sets[[i]], sets[[j]]]]},
          If[w == 0, 0., N[Length[Intersection[sets[[i]], sets[[j]]]]/w]]]},
      {i, k}, {j, i + 1, k}], 1], #[[1]] =!= Infinity &], First -> Last]]];

sgrVertexCurvature[edges_List] := Module[{g = sgrStateGraph[edges]},
  N[vertexCurvatureExact[g, edgeCurvatures[g]]]];

sgrStateMatrices[vertexSets_List, edgeLists_List] := Module[{level, v, e},
  level[sets0_] := Module[{sets = DeleteDuplicates /@ sets0, u},
    u = Length[Union @@ sets];
    Table[Module[{a = sets[[i]], b = sets[[j]], n, w},
        n = Length[Intersection[a, b]]; w = Length[Union[a, b]];
        {If[w == 0, 0., N[n/w]],
         If[Length[a] Length[b] == 0, 0., N[n/Sqrt[Length[a] Length[b]]]],
         mutualInformation[n, Length[a], Length[b], u]}],
      {i, Length[sets]}, {j, Length[sets]}]];
  v = level[vertexSets]; e = level[edgeLists];
  <|"StateOverlap" -> v[[All, All, 1]], "StateCosineSimilarity" -> v[[All, All, 2]],
    "StateMutualInformation" -> v[[All, All, 3]], "StateEdgeOverlap" -> e[[All, All, 1]],
    "StateEdgeCosineSimilarity" -> e[[All, All, 2]],
    "StateEdgeMutualInformation" -> e[[All, All, 3]]|>];

End[];
EndPackage[];
