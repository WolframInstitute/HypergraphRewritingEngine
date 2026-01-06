Options[WolframHausdorffDimension] = {"TransitivelyReduce" -> False, 
   "UndirectedGraph" -> False, "VolumeMethod" -> Mean, 
   "DimensionMethod" -> Mean, "VertexMethod" -> Identity};

WolframHausdorffDimension[causalGraph_Graph, vertex_, "Volume", 
  options : OptionsPattern[]] := 
 Module[{transitiveReduction, graphRadius},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  graphRadius = GraphRadius[UndirectedGraph[transitiveReduction]];
  WolframHausdorffDimension[causalGraph, vertex, graphRadius, 
   "Volume", options]]

WolframHausdorffDimension[causalGraph_Graph, vertex_, 
  maxRadius_Integer, "Volume", options : OptionsPattern[]] := 
 Module[{res, transitiveReduction},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  transitiveReduction = 
   If[TrueQ[OptionValue["UndirectedGraph"]], 
    UndirectedGraph[transitiveReduction], transitiveReduction];
  res = Length[VertexOutComponent[transitiveReduction, vertex, #]] & /@
     Range[maxRadius];
  Switch[OptionValue["VolumeMethod"],
   Max,
   N[Max[res]],
   Min,
   N[Min[res]],
   Mean,
   N[Mean[res]],
   _,
   res]]

WolframHausdorffDimension[causalGraph_Graph, All, "AllVolumes", 
  options : OptionsPattern[]] := 
 Module[{transitiveReduction, graphRadius},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  graphRadius = GraphRadius[UndirectedGraph[transitiveReduction]];
  WolframHausdorffDimension[causalGraph, All, graphRadius, 
   "AllVolumes", options]]

WolframHausdorffDimension[causalGraph_Graph, All, maxRadius_Integer, 
  "AllVolumes", options : OptionsPattern[]] := 
 Module[{res, transitiveReduction, vL},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  vL = VertexList[transitiveReduction];
  res = WolframHausdorffDimension[causalGraph, #, maxRadius, "Volume",
       options] & /@ vL;
  Switch[OptionValue["VertexMethod"],
   Max,
   N[First[MaximalBy[res, Total]]],
   Min,
   N[First[MinimalBy[res, Total]]],
   Mean,
   N[Mean[res]],
   _,
   AssociationThread[vL -> res]]]

WolframHausdorffDimension[causalGraph_Graph, 
  vertex_, {minRadius_Integer, maxRadius_Integer}, "Volume", 
  options : OptionsPattern[]] := Module[{res, transitiveReduction},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  transitiveReduction = 
   If[TrueQ[OptionValue["UndirectedGraph"]], 
    UndirectedGraph[transitiveReduction], transitiveReduction];
  res = Length[VertexOutComponent[transitiveReduction, vertex, #]] & /@
     Range[minRadius, maxRadius];
  Switch[OptionValue["VolumeMethod"],
   Max,
   N[Max[res]],
   Min,
   N[Min[res]],
   Mean,
   N[Mean[res]],
   _,
   res]]

WolframHausdorffDimension[causalGraph_Graph, 
  All, {minRadius_Integer, maxRadius_Integer}, "AllVolumes", 
  options : OptionsPattern[]] := Module[{res, transitiveReduction, vL},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  vL = VertexList[transitiveReduction];
  res = WolframHausdorffDimension[
      causalGraph, #, {minRadius, maxRadius}, "Volume", options] & /@ 
    vL;
  Switch[OptionValue["VertexMethod"],
   Max,
   N[First[MaximalBy[res, Total]]],
   Min,
   N[First[MinimalBy[res, Total]]],
   Mean,
   N[Mean[res]],
   _,
   AssociationThread[vL -> res]]]

WolframHausdorffDimension[causalGraph_Graph, vertex_, 
   options : OptionsPattern[]] /; ! 
   MatchQ[vertex, 
    "Volume" | "AllVolumes" | "Dimension" | "AllDimensions" | All] := 
 WolframHausdorffDimension[causalGraph, vertex, "Dimension", options]

WolframHausdorffDimension[causalGraph_Graph, vertex_, "Dimension", 
  options : OptionsPattern[]] := 
 Module[{transitiveReduction, graphRadius},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  graphRadius = GraphRadius[UndirectedGraph[transitiveReduction]];
  WolframHausdorffDimension[causalGraph, vertex, graphRadius, 
   "Dimension", options]]

WolframHausdorffDimension[causalGraph_Graph, vertex_, 
  maxRadius_Integer, options : OptionsPattern[]] := 
 WolframHausdorffDimension[causalGraph, vertex, maxRadius, 
  "Dimension", options]

WolframHausdorffDimension[causalGraph_Graph, vertex_, 
  maxRadius_Integer, "Dimension", options : OptionsPattern[]] := 
 Module[{res, transitiveReduction},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  transitiveReduction = 
   If[TrueQ[OptionValue["UndirectedGraph"]], 
    UndirectedGraph[transitiveReduction], transitiveReduction];
  res = (Log[
         Length[VertexOutComponent[transitiveReduction, vertex, #]]] -
         Log[Length[
          VertexOutComponent[transitiveReduction, 
           vertex, # - 1]]])/(Log[# + 1] - Log[#]) & /@ 
    Range[maxRadius];
  Switch[OptionValue["DimensionMethod"],
   Max,
   N[Max[res]],
   Min,
   N[Min[res]],
   Mean,
   N[Mean[res]],
   _,
   N[res]]]

WolframHausdorffDimension[causalGraph_Graph, All, 
  options : OptionsPattern[]] := 
 WolframHausdorffDimension[causalGraph, All, "AllDimensions", options]

WolframHausdorffDimension[causalGraph_Graph, All, "AllDimensions", 
  options : OptionsPattern[]] := 
 Module[{transitiveReduction, graphRadius},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  graphRadius = GraphRadius[UndirectedGraph[transitiveReduction]];
  WolframHausdorffDimension[causalGraph, All, graphRadius, 
   "AllDimensions", options]]

WolframHausdorffDimension[causalGraph_Graph, All, maxRadius_Integer, 
  options : OptionsPattern[]] := 
 WolframHausdorffDimension[causalGraph, All, maxRadius, 
  "AllDimensions", options]

WolframHausdorffDimension[causalGraph_Graph, All, maxRadius_Integer, 
  "AllDimensions", options : OptionsPattern[]] := 
 Module[{transitiveReduction, vL},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  vL = VertexList[transitiveReduction];
  Switch[OptionValue["VertexMethod"],
   Max,
   N[First[
     MaximalBy[
      WolframHausdorffDimension[causalGraph, #, maxRadius, 
         "Dimension", options] & /@ vL, Total]]],
   Min,
   N[First[
     MinimalBy[
      WolframHausdorffDimension[causalGraph, #, maxRadius, 
         "Dimension", options] & /@ vL, Total]]],
   Mean,
   N[Mean[
     WolframHausdorffDimension[causalGraph, #, maxRadius, "Dimension",
         options] & /@ vL]],
   _,
   Association[(# -> 
        WolframHausdorffDimension[causalGraph, #, maxRadius, 
         "Dimension", options]) & /@ vL]]]

WolframHausdorffDimension[causalGraph_Graph, 
  vertex_, {minRadius_Integer, maxRadius_Integer}, 
  options : OptionsPattern[]] := 
 WolframHausdorffDimension[causalGraph, 
  vertex, {minRadius, maxRadius}, "Dimension", options]

WolframHausdorffDimension[causalGraph_Graph, 
  vertex_, {minRadius_Integer, maxRadius_Integer}, "Dimension", 
  options : OptionsPattern[]] := Module[{res, transitiveReduction},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  transitiveReduction = 
   If[TrueQ[OptionValue["UndirectedGraph"]], 
    UndirectedGraph[transitiveReduction], transitiveReduction];
  res = (Log[
         Length[VertexOutComponent[transitiveReduction, vertex, #]]] -
         Log[Length[
          VertexOutComponent[transitiveReduction, 
           vertex, # - 1]]])/(Log[# + 1] - Log[#]) & /@ 
    Range[minRadius, maxRadius];
  Switch[OptionValue["DimensionMethod"],
   Max,
   N[Max[res]],
   Min,
   N[Min[res]],
   Mean,
   N[Mean[res]],
   _,
   N[res]]]

WolframHausdorffDimension[causalGraph_Graph, 
  All, {minRadius_Integer, maxRadius_Integer}, 
  options : OptionsPattern[]] := 
 WolframHausdorffDimension[causalGraph, All, {minRadius, maxRadius}, 
  "AllDimensions", options]

WolframHausdorffDimension[causalGraph_Graph, 
  All, {minRadius_Integer, maxRadius_Integer}, "AllDimensions", 
  options : OptionsPattern[]] := Module[{res, transitiveReduction, vL},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  vL = VertexList[transitiveReduction];
  res = WolframHausdorffDimension[
      causalGraph, #, {minRadius, maxRadius}, "Dimension", 
      options] & /@ vL;
  Switch[OptionValue["VertexMethod"],
   Max,
   N[First[MaximalBy[res, Total]]],
   Min,
   N[First[MinimalBy[res, Total]]],
   Mean,
   N[Mean[res]],
   _,
   AssociationThread[vL -> res]]]

WolframHausdorffDimension[causalGraph_Graph, vertex_, 
  "HighlightedGraph", options : OptionsPattern[]] := 
 Module[{transitiveReduction, graphRadius},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[UndirectedGraph[causalGraph]], 
    causalGraph];
  graphRadius = GraphRadius[UndirectedGraph[transitiveReduction]];
  WolframHausdorffDimension[causalGraph, vertex, graphRadius, 
   "HighlightedGraph", options]]

WolframHausdorffDimension[causalGraph_Graph, vertex_, 
  maxRadius_Integer, "HighlightedGraph", options : OptionsPattern[]] :=
  Module[{transitiveReduction},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  transitiveReduction = 
   If[TrueQ[OptionValue["UndirectedGraph"]], 
    UndirectedGraph[transitiveReduction], transitiveReduction];
  HighlightGraph[transitiveReduction, 
   Style[GraphDifference[
       Subgraph[transitiveReduction, 
        VertexOutComponent[transitiveReduction, vertex, #]], 
       Subgraph[transitiveReduction, 
        VertexOutComponent[transitiveReduction, vertex, # - 1]]], 
      Directive[AbsoluteThickness[5], ColorData[1, #]]] & /@ 
    Range[maxRadius]]]

WolframHausdorffDimension[causalGraph_Graph, 
  vertex_, {minRadius_Integer, maxRadius_Integer}, "HighlightedGraph",
   options : OptionsPattern[]] := Module[{transitiveReduction},
  transitiveReduction = 
   If[TrueQ[OptionValue["TransitivelyReduce"]], 
    TransitiveReductionGraph[causalGraph], causalGraph];
  transitiveReduction = 
   If[TrueQ[OptionValue["UndirectedGraph"]], 
    UndirectedGraph[transitiveReduction], transitiveReduction];
  HighlightGraph[transitiveReduction, 
   Style[GraphDifference[
       Subgraph[transitiveReduction, 
        VertexOutComponent[transitiveReduction, vertex, #]], 
       Subgraph[transitiveReduction, 
        VertexOutComponent[transitiveReduction, vertex, # - 1]]], 
      Directive[AbsoluteThickness[5], ColorData[1, #]]] & /@ 
    Range[minRadius, maxRadius]]]
