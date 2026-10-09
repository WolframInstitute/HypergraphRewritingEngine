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

sgrStateGraph::usage = "sgrStateGraph[edges] is the undirected simple graph of a state: its vertices, and an edge between the consecutive vertices of each hyperedge, without self-loops.";
sgrStateGeometry::usage = "sgrStateGeometry[edges] gives an association of the per-state geometry metrics; an undefined metric is Missing[].";
sgrBallGrowth::usage = "sgrBallGrowth[edges] gives the per-radius ball-growth dimensions for r = 1..R, or {} when undefined.";
sgrVertexDimensions::usage = "sgrVertexDimensions[g] gives each vertex's own Hausdorff dimension over r = 1..GraphRadius[g].";
sgrBranchialGraphMetrics::usage = "sgrBranchialGraphMetrics[nodes, pairs] gives the per-step branchial graph metrics for a step's states and its branchial state pairs.";
sgrOverlapMetrics::usage = "sgrOverlapMetrics[vertexSets, edgeLists, initial] gives the per-step overlap values for a step's states, each given as its vertex set and its list of edges (each edge its vertex list); initial is the vertex set of step 0's states.";
sgrOverlapByDistance::usage = "sgrOverlapByDistance[vertexSets, distanceMatrix] gives, for each finite branchial distance d > 0, the Jaccard index of every two states at distance d.";

Begin["`Private`"];

sgrStateGraph[edges_List] := Graph[
  Union[Flatten[edges]],
  UndirectedEdge @@@ DeleteDuplicates[Sort /@ Select[
    Catenate[Partition[#, 2, 1] & /@ edges], #[[1]] =!= #[[2]] &]]];

ball[g_, v_, r_] := Length[VertexOutComponent[g, v, r]];

term[g_, v_, r_] := (Log[ball[g, v, r]] - Log[ball[g, v, r - 1]])/(Log[r + 1] - Log[r]);

sgrVertexDimensions[g_Graph] := Module[{R = GraphRadius[g]},
  Association[# -> N[Mean[Table[term[g, #, r], {r, R}]]] & /@ VertexList[g]]];

connectedQ[g_Graph] := VertexCount[g] >= 1 && ConnectedGraphQ[g];

entropyBits[counts_List] := Module[{p = N[counts/Total[counts]]}, -Total[p Log2[p]]];

degreeEntropy[g_Graph, vs_List] := entropyBits[Values[Counts[VertexDegree[g, #] & /@ vs]]];

(* The lazy walk: 1/2 at x, 1/(2 deg x) at each neighbour; the cost is the exact optimum of the
   transport linear program under the graph distance. *)
ollivierEdge[g_Graph, dm_, index_, x_, y_] := Module[
  {ax, by, mu, nu, a, b, cost, eqs, rhs, sol},
  ax = Prepend[AdjacencyList[g, x], x]; by = Prepend[AdjacencyList[g, y], y];
  mu = Prepend[ConstantArray[1/(2 Length[ax] - 2), Length[ax] - 1], 1/2];
  nu = Prepend[ConstantArray[1/(2 Length[by] - 2), Length[by] - 1], 1/2];
  a = Length[ax]; b = Length[by];
  cost = Flatten[Table[dm[[index[ax[[i]]], index[by[[j]]]]], {i, a}, {j, b}]];
  eqs = Join[
    Table[Flatten[Table[If[i == k, 1, 0], {i, a}, {j, b}]], {k, a}],
    Table[Flatten[Table[If[j == k, 1, 0], {i, a}, {j, b}]], {k, b}]];
  rhs = Join[{#, 0} & /@ mu, {#, 0} & /@ nu];
  sol = LinearProgramming[cost, eqs, rhs];
  1 - cost . sol];

sgrStateGeometry[edges_List] := Module[
  {g, n, conn, R, vs, dims, d, ricci, dm, index, ollivier, local, mi, fisher, b2},
  g = sgrStateGraph[edges];
  vs = VertexList[g]; n = Length[vs];
  If[n == 0, Return[<|"GraphRadius" -> Missing[], "MeanEccentricity" -> Missing[],
    "WolframHausdorffDimension" -> Missing[], "WolframRicciCurvatureScalar" -> Missing[],
    "OllivierRicciCurvature" -> Missing[], "DegreeEntropy" -> Missing[],
    "LocalEntropy" -> Missing[], "MutualInformation" -> Missing[],
    "FisherInformation" -> Missing[]|>]];
  conn = ConnectedGraphQ[g];
  R = If[conn, GraphRadius[g], Infinity];
  b2[v_] := VertexOutComponent[g, v, 2];
  local = N[Mean[degreeEntropy[g, b2[#]] & /@ vs]];
  mi = Module[{withNbr = Select[vs, VertexDegree[g, #] > 0 &]},
    If[withNbr === {}, Missing[],
      N[Mean[Function[v, Mean[Function[w, Module[{bv = b2[v], bw = b2[w], both, uni},
        both = Length[Intersection[bv, bw]]; uni = Length[Union[bv, bw]];
        Max[0, Log2[both uni/(Length[bv] Length[bw])]]]] /@ AdjacencyList[g, v]]] /@ withNbr]]]];
  dm = GraphDistanceMatrix[g]; index = AssociationThread[vs -> Range[n]];
  ollivier = If[EdgeCount[g] == 0, Missing[],
    N[Mean[ollivierEdge[g, dm, index, #[[1]], #[[2]]] & /@ EdgeList[g]]]];
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
    "FisherInformation" -> fisher|>];

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
  (* The largest component: most states, then the greatest dimension. *)
  dim[c_] := If[Length[c] < 2, Missing[],
    Mean[Values[sgrVertexDimensions[Subgraph[g, c]]]]];
  largest = MaximalBy[comps, {Length[#], Replace[dim[#], _Missing -> -Infinity]} &];
  <|"Degrees" -> (VertexDegree[g, #] & /@ nodes),
    "Distances" -> dists,
    "DistanceMatrix" -> dm,
    "Components" -> Length[comps],
    "Dimension" -> If[largest === {}, Missing[], dim[First[largest]]]|>];

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

End[];
EndPackage[];
