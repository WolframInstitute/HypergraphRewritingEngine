---
Template: TechNote
Name: ScienceWithStepStatistics
Title: Science with Step Statistics
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/tutorial/ScienceWithStepStatistics
Keywords: [hypergraph, multiway, step statistics, dimension, curvature, Ollivier-Ricci, Moran's I, Brill-Lindquist, torus, branchial, mutual information, tutorial]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [GettingStarted, InitialConditions, SamplingAndPruning]
---

`"StepStatistics"` gives one association per step of an evolution, with counts of states and classes, summaries of per-state invariants, distributions over vertices and, on request, measures of the branchial graph and of the overlap between states. The [HGEvolve]() reference page defines every key. This tutorial uses them to answer six questions about rules of the Wolfram Physics Project. Every number plotted comes from the engine.

Each plot reads one key of every step with this function:

```wl
perStep[stats_, keys__] := Table[{s["Step"], s[keys]}, {s, stats}]
```

## States and classes per step

How many of the states of the multiway system are distinct up to isomorphism? The rule is a rule of the Wolfram Physics Project, from two self-loops on one vertex:

```wl
wolfram = {{x, y}, {x, z}} -> {{x, y}, {x, w}, {y, w}, {z, w}};
classes = HGEvolve[wolfram, {{0, 0}, {0, 0}}, 5, "StepStatistics", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True];
Lookup[classes, {"Step", "RawStates", "Classes"}]
```

<!-- => {{0, 1, 1}, {1, 2, 1}, {2, 24, 3}, {3, 408, 18}, {4, 9504, 156}, {5, 280080, 1776}} -->

With `"ExploreFromCanonicalStatesOnly" -> True`, the engine expands one state per class and counts the states of each class, so `"RawStates"` is the number of states of the full multiway system. The number of classes grows more slowly:

```wl
ListLogPlot[{perStep[classes, "RawStates"], perStep[classes, "Classes"]}, Joined -> True, PlotMarkers -> Automatic, AxesLabel -> {"step", "count"}, PlotLegends -> {"states", "classes"}, ImageSize -> 400]
```

![With "ExploreFromCanonicalStatesOnly" -> True, the engine expands one state per class and counts the states of each class, so "RawStates" is the number of states of the full multiway system.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-3.png)

`"MultiplicityHistogram"` maps a class size to the number of classes of that size. The class sizes at steps 4 and 5:

```wl
ListLogLogPlot[Table[Normal[classes[[s + 1]]["MultiplicityHistogram"]] /. Rule -> List, {s, {4, 5}}], PlotMarkers -> Automatic, AxesLabel -> {"states in the class", "classes"}, PlotLegends -> {"step 4", "step 5"}, ImageSize -> 400]
```

!["MultiplicityHistogram" maps a class size to the number of classes of that size.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-4.png)

## Dimension of a growth history

Does a growth rule settle at a fixed dimension? `"UniformRandom" -> True` with `"MatchesPerStep" -> 20` keeps 20 states at each step:

```wl
growth = {{x, y}, {x, z}} -> {{x, z}, {x, w}, {y, w}, {z, w}};
history = HGEvolve[growth, {{1, 2}, {2, 3}, {3, 4}, {2, 4}}, 200, "StepStatistics", "UniformRandom" -> True, "MatchesPerStep" -> 20, "RandomSeed" -> 1];
```

The mean over the step's states of the Wolfram-Hausdorff dimension, and of the largest local dimension of a vertex in each state:

```wl
ListLinePlot[{perStep[history, "Invariants", "WolframHausdorffDimension", "Mean"], perStep[history, "Invariants", "LocalDimensionMax", "Mean"]}, AxesLabel -> {"step", "dimension"}, PlotLegends -> {"mean over vertices", "largest vertex"}, ImageSize -> 400]
```

![The mean over the step's states of the Wolfram-Hausdorff dimension, and of the largest local dimension of a vertex in each state](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-6.png)

The vertex count grows by one per step:

```wl
ListLinePlot[perStep[history, "Invariants", "VertexCount", "Mean"], AxesLabel -> {"step", "vertices"}, ImageSize -> 400]
```

![The vertex count grows by one per step](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-7.png)

The 20 states of a step are 20 branches. The `"Histogram"` of `"LargestComponentDimension"` in `"Invariants"` maps each dimension, rounded to 0.01, to the number of states with that dimension. The dimension over the branches at steps 50, 100 and 200:

```wl
Histogram[Table[With[{h = history[[s + 1]]["Invariants", "LargestComponentDimension", "Histogram"]}, WeightedData[Keys[h], Values[h]]], {s, {50, 100, 200}}], {0.01}, "Probability", AxesLabel -> {"dimension", "fraction of states"}, ChartLegends -> {"step 50", "step 100", "step 200"}, ImageSize -> 400]
```

![The 20 states of a step are 20 branches.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-8.png)

`"BallGrowthDimension"` maps each radius r to the summary over the states of the dimension measured from the growth of balls between radius r - 1 and r. The mean at steps 50, 100 and 200:

```wl
ListLinePlot[Table[KeyValueMap[{#1, #2["Mean"]} &, history[[s + 1]]["BallGrowthDimension"]], {s, {50, 100, 200}}], PlotMarkers -> Automatic, AxesLabel -> {"radius", "dimension"}, PlotLegends -> {"step 50", "step 100", "step 200"}, ImageSize -> 400]
```

!["BallGrowthDimension" maps each radius r to the summary over the states of the dimension measured from the growth of balls between radius r - 1 and r.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-9.png)

## Free space and a gravitational field

Does the same rule give a different dimension and curvature in free space and near two black holes? The rule replaces a path of four edges with four edges through a new vertex. One run starts from a torus of 81 vertices, the other from a Brill-Lindquist point cloud of 80 points around two black holes, both described in [Generated Initial Conditions](paclet:WolframInstitute/HypergraphRewriteEngine/tutorial/InitialConditions):

```wl
ruleA = {{x, y}, {y, z}, {z, w}, {w, v}} -> {{y, u}, {u, v}, {w, x}, {x, u}};
torus = HGEvolve[ruleA, <|"Type" -> "Torus", "Resolution" -> 9, "Seed" -> 1|>, 40, "StepStatistics", "UniformRandom" -> True, "MatchesPerStep" -> 10, "RandomSeed" -> 1];
blackHoles = HGEvolve[ruleA, <|"Type" -> "BrillLindquist", "Density" -> 80, "Separation" -> 6., "BoxX" -> {-10., 10.}, "BoxY" -> {-10., 10.}, "EdgeThreshold" -> 3.5, "Seed" -> 1|>, 40, "StepStatistics", "UniformRandom" -> True, "MatchesPerStep" -> 10, "RandomSeed" -> 1];
```

The Brill-Lindquist state has two components, and `"WolframHausdorffDimension"` is undefined for a disconnected state:

```wl
{First[torus]["Invariants", "Components", "Mean"], First[blackHoles]["Invariants", "Components", "Mean"]}
```

<!-- => {1., 2.} -->

`"LargestComponentDimension"` is the dimension of the largest component, so it is defined for both:

```wl
ListLinePlot[{perStep[torus, "Invariants", "LargestComponentDimension", "Mean"], perStep[blackHoles, "Invariants", "LargestComponentDimension", "Mean"]}, AxesLabel -> {"step", "dimension"}, PlotLegends -> {"torus", "Brill-Lindquist"}, ImageSize -> 400]
```

!["LargestComponentDimension" is the dimension of the largest component, so it is defined for both](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-12.png)

The mean Ollivier-Ricci curvature of the edges:

```wl
ListLinePlot[{perStep[torus, "Invariants", "OllivierRicciCurvature", "Mean"], perStep[blackHoles, "Invariants", "OllivierRicciCurvature", "Mean"]}, AxesLabel -> {"step", "curvature"}, PlotLegends -> {"torus", "Brill-Lindquist"}, ImageSize -> 400]
```

![The mean Ollivier-Ricci curvature of the edges](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-13.png)

## The distribution of curvature

How is curvature spread over the vertices of all branches at a step? `"VertexInvariants"` pools the vertices of every state at the step. Its `"Histogram"` maps the lower edge of each bin to the number of vertices in the bin. Curvature has 32 bins of width 3/32 from -2 to 1. The curvature of the vertices at steps 5, 20 and 40, as a fraction of the vertices:

```wl
vertexHistogram[stats_, step_, key_] := With[{h = stats[[step + 1]]["VertexInvariants", key, "Histogram"]}, WeightedData[Keys[h], Values[h]]];
Row[Table[Histogram[{vertexHistogram[torus, s, "OllivierRicciCurvature"], vertexHistogram[blackHoles, s, "OllivierRicciCurvature"]}, {-2, 1, 3/32}, "Probability", PlotRange -> {{-1, 1}, {0, 0.6}}, AxesLabel -> {"curvature", "fraction"}, PlotLabel -> "step " <> ToString[s], ChartLegends -> If[s == 40, {"torus", "Brill-Lindquist"}, None], ImageSize -> 260], {s, {5, 20, 40}}], Spacer[30]]
```

![How is curvature spread over the vertices of all branches at a step? "VertexInvariants" pools the vertices of every state at the step.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-14.png)

Each column of this plot is the curvature histogram of one step, shaded by the fraction of the vertices in each bin:

```wl
curvatureColumn[s_] := With[{h = s["VertexInvariants", "OllivierRicciCurvature", "Histogram"]}, Lookup[KeyMap[Round[32 (# + 2)/3] &, h/Total[h]], Range[10, 31], 0]];
Row[Table[ArrayPlot[Reverse[Transpose[curvatureColumn /@ run[[1]]]], DataRange -> {{0, 40}, {-1.0625, 1}}, FrameTicks -> {{Range[-1, 1, 1/2], None}, {Range[0, 40, 10], None}}, FrameLabel -> {"curvature", "step"}, PlotLabel -> run[[2]], ColorFunction -> "SunsetColors", PlotLegends -> Automatic, AspectRatio -> 1, ImageSize -> 300], {run, {{torus, "torus"}, {blackHoles, "Brill-Lindquist"}}}], Spacer[30]]
```

![Each column of this plot is the curvature histogram of one step, shaded by the fraction of the vertices in each bin](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-15.png)

The local dimension of the vertices has 32 bins of width 1/4 from 0 to 8. At steps 5, 20 and 40:

```wl
Row[Table[Histogram[{vertexHistogram[torus, s, "LocalDimension"], vertexHistogram[blackHoles, s, "LocalDimension"]}, {0, 8, 1/4}, "Probability", PlotRange -> {{0, 4}, {0, 0.8}}, AxesLabel -> {"dimension", "fraction"}, PlotLabel -> "step " <> ToString[s], ChartLegends -> If[s == 40, {"torus", "Brill-Lindquist"}, None], ImageSize -> 260], {s, {5, 20, 40}}], Spacer[30]]
```

![The local dimension of the vertices has 32 bins of width 1/4 from 0 to 8.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-16.png)

`"DegreeHistogram"` maps a degree to the number of vertices with that degree. At step 40:

```wl
Histogram[Table[With[{h = run[[41]]["DegreeHistogram"]}, WeightedData[Keys[h], Values[h]]], {run, {torus, blackHoles}}], {1/2, 30, 1}, "Probability", AxesLabel -> {"degree", "fraction"}, ChartLegends -> {"torus", "Brill-Lindquist"}, ImageSize -> 400]
```

!["DegreeHistogram" maps a degree to the number of vertices with that degree.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-17.png)

The quantiles of the vertex curvature of the Brill-Lindquist run:

```wl
ListLinePlot[Table[perStep[blackHoles, "VertexInvariants", "OllivierRicciCurvature", q], {q, {"P10", "Q1", "Median", "Q3", "P90"}}], AxesLabel -> {"step", "curvature"}, PlotLegends -> {"10th percentile", "first quartile", "median", "third quartile", "90th percentile"}, ImageSize -> 400]
```

![The quantiles of the vertex curvature of the Brill-Lindquist run](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-18.png)

The skewness of the same distribution in both runs:

```wl
ListLinePlot[{perStep[torus, "VertexInvariants", "OllivierRicciCurvature", "Skewness"], perStep[blackHoles, "VertexInvariants", "OllivierRicciCurvature", "Skewness"]}, AxesLabel -> {"step", "skewness"}, PlotRange -> All, PlotLegends -> {"torus", "Brill-Lindquist"}, ImageSize -> 400]
```

![The skewness of the same distribution in both runs](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-19.png)

## Is curvature local?

Do vertices of high curvature sit next to each other, or does curvature follow the degree? `"OllivierMoranI"` is Moran's I of the vertex curvature over the edges of a state, positive when neighbours have similar curvature. It is undefined when every vertex has the same curvature, as on the torus at step 0:

```wl
ListLinePlot[{perStep[torus, "Invariants", "OllivierMoranI", "Mean"], perStep[blackHoles, "Invariants", "OllivierMoranI", "Mean"]}, AxesLabel -> {"step", "Moran's I"}, PlotLegends -> {"torus", "Brill-Lindquist"}, ImageSize -> 400]
```

![Do vertices of high curvature sit next to each other, or does curvature follow the degree? "OllivierMoranI" is Moran's I of the vertex curvature over the edges of a state, positive when neighbours have similar curvature.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-20.png)

`"OllivierDegreeCorrelation"` is the correlation of the vertex curvature with the vertex degree:

```wl
ListLinePlot[{perStep[torus, "Invariants", "OllivierDegreeCorrelation", "Mean"], perStep[blackHoles, "Invariants", "OllivierDegreeCorrelation", "Mean"]}, AxesLabel -> {"step", "correlation"}, PlotLegends -> {"torus", "Brill-Lindquist"}, ImageSize -> 400]
```

!["OllivierDegreeCorrelation" is the correlation of the vertex curvature with the vertex degree](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-21.png)

## Branches and the initial state

How similar are the states at a step, and how much of the initial state do they keep? `"StepStatisticsBranchial" -> All` adds the branchial graph measures and the overlap of the states' vertex sets. `"MaxStatesPerStep" -> 400` keeps 400 states at each step:

```wl
branches = HGEvolve[ruleA, <|"Type" -> "Grid", "Width" -> 8, "Height" -> 8, "Seed" -> 1|>, 10, "StepStatistics", "MaxStatesPerStep" -> 400, "StepStatisticsBranchial" -> All];
```

The mean over every two states of their cosine similarity and their mutual information, and the mean over the states of their mutual information with the initial state, in bits. At step 0 the only state is the initial state, so every vertex is in both sets and the mutual information is 0:

```wl
ListLinePlot[{perStep[branches, "StateCosineSimilarity", "Mean"], perStep[branches, "StateMutualInformation", "Mean"], perStep[branches, "InitialStateMutualInformation", "Mean"]}, AxesLabel -> {"step", None}, PlotLegends -> {"cosine similarity", "mutual information", "mutual information with the initial state"}, ImageSize -> 400]
```

![The mean over every two states of their cosine similarity and their mutual information, and the mean over the states of their mutual information with the initial state, in bits.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-23.png)

The mean branch entropy of a vertex and of an edge, log2 k for an item held by k states:

```wl
ListLinePlot[{perStep[branches, "BranchEntropy", "Mean"], perStep[branches, "EdgeBranchEntropy", "Mean"]}, AxesLabel -> {"step", "bits"}, PlotLegends -> {"vertex", "edge"}, ImageSize -> 400]
```

![The mean branch entropy of a vertex and of an edge, log2 k for an item held by k states](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-24.png)

`"OverlapByBranchialDistance"` gives, for each branchial distance d, the summary of the Jaccard index of the vertex sets of the pairs of states at distance d. At steps 1 and 2:

```wl
ListLinePlot[Table[KeyValueMap[{#1, #2["Mean"]} &, branches[[s + 1]]["OverlapByBranchialDistance"]], {s, {1, 2}}], PlotMarkers -> Automatic, AxesLabel -> {"branchial distance", "Jaccard index"}, PlotLegends -> {"step 1", "step 2"}, ImageSize -> 400]
```

!["OverlapByBranchialDistance" gives, for each branchial distance d, the summary of the Jaccard index of the vertex sets of the pairs of states at distance d.](../images/ScienceWithStepStatistics/ScienceWithStepStatistics-25.png)
