(* =============================================================================
   Branch Alignment via Curvature-Weighted Moment of Inertia

   Reference implementation in Wolfram Language.
   C++ implementation should produce equivalent results.

   Purpose: Compare curvature distributions across branches by aligning them
   using principal component analysis on curvature-weighted spectral embeddings.
   ============================================================================= *)

(* -----------------------------------------------------------------------------
   Main alignment function

   Input: stateId - ID of state in evolution result
          result - HGEvolve result with CurvatureData

   Output: Association with:
     "Vertices" -> list of vertex IDs in canonical order
     "PC1", "PC2", "PC3" -> principal component positions
     "Curvature" -> curvature values in canonical order
     "Rank" -> normalized rank [0,1] for each vertex
   ----------------------------------------------------------------------------- *)

alignBranch[stateId_, result_] := Module[
  {edges, curvature, graph, vertices, pos, masses, centroid, centered,
   inertia, evals, evecs, aligned, curvValues, ordering, n},

  (* Extract edges and curvature from result *)
  edges = result["States"][stateId]["Edges"];
  curvature = result["CurvatureData"]["PerState"][stateId]["WolframRicci"];

  (* Validate inputs *)
  If[!ListQ[edges] || !AssociationQ[curvature] || Length[edges] < 4,
    Return[<|"Error" -> "Insufficient data"|>]];

  (* Build graph and get vertices *)
  graph = Graph[UndirectedEdge @@@ (List @@@ edges)];
  vertices = VertexList[graph];
  n = Length[vertices];

  If[n < 4, Return[<|"Error" -> "Too few vertices"|>]];

  (* Step 1: Spectral embedding using Laplacian eigenvectors *)
  (* This embeds vertices into R^3 respecting graph structure *)
  pos = GraphEmbedding[graph, "SpectralEmbedding", 3];

  If[!MatrixQ[pos], Return[<|"Error" -> "Embedding failed"|>]];

  (* Step 2: Get curvature values as masses *)
  (* Use absolute value since curvature can be negative *)
  curvValues = Lookup[curvature, vertices, 0.];
  masses = Abs[curvValues] + 0.01;  (* Small offset to avoid zero mass *)

  (* Step 3: Compute curvature-weighted centroid *)
  centroid = Total[masses * pos] / Total[masses];

  (* Step 4: Center the positions *)
  centered = # - centroid & /@ pos;

  (* Step 5: Compute moment of inertia tensor *)
  (* I_ij = sum_k m_k * (|r_k|^2 * delta_ij - r_k,i * r_k,j) *)
  inertia = Total[Table[
    masses[[i]] * (Norm[centered[[i]]]^2 * IdentityMatrix[3] -
      Outer[Times, centered[[i]], centered[[i]]]),
    {i, n}
  ]];

  (* Step 6: Eigendecomposition to get principal axes *)
  {evals, evecs} = Eigensystem[N[inertia]];

  (* Step 7: Project onto principal axes *)
  aligned = centered . Transpose[evecs];

  (* Step 8: Sort by PC1 to get canonical vertex ordering *)
  ordering = Ordering[aligned[[All, 1]]];

  (* Return structured result *)
  <|
    "Vertices" -> vertices[[ordering]],
    "PC1" -> aligned[[ordering, 1]],
    "PC2" -> aligned[[ordering, 2]],
    "PC3" -> aligned[[ordering, 3]],
    "Curvature" -> curvValues[[ordering]],
    "Rank" -> Table[(i - 1)/(n - 1), {i, n}],
    "Eigenvalues" -> evals,
    "Centroid" -> centroid,
    "NumVertices" -> n
  |>
];

(* -----------------------------------------------------------------------------
   Batch alignment for all branches at a timestep
   ----------------------------------------------------------------------------- *)

alignAllBranches[result_, step_] := Module[
  {statesAtStep},

  statesAtStep = Select[
    Keys[result["CurvatureData"]["PerState"]],
    result["States"][#]["Step"] == step &
  ];

  AssociationMap[alignBranch[#, result] &, statesAtStep]
];

(* -----------------------------------------------------------------------------
   Visualization: 1D - Curvature vs Vertex Rank
   ----------------------------------------------------------------------------- *)

plotCurvatureVsRank[alignedBranches_] := Module[
  {data},

  data = KeyValueMap[
    Thread[{#2["Rank"], #2["Curvature"]}] &,
    Select[alignedBranches, AssociationQ[#] && !KeyExistsQ[#, "Error"] &]
  ];

  ListLinePlot[data,
    PlotLabel -> "Curvature vs Vertex Rank (PC1 ordered)",
    AxesLabel -> {"Vertex Rank", "Curvature"},
    PlotStyle -> Opacity[0.5],
    PlotRange -> All
  ]
];

(* -----------------------------------------------------------------------------
   Visualization: 2D - PC1 vs PC2, colored by curvature
   ----------------------------------------------------------------------------- *)

plotPC1vsPC2[alignedBranches_] := Module[
  {allPoints, curvMin, curvMax, normalize},

  allPoints = Flatten[
    Values[KeyValueMap[
      Thread[{#2["PC1"], #2["PC2"], #2["Curvature"]}] &,
      Select[alignedBranches, AssociationQ[#] && !KeyExistsQ[#, "Error"] &]
    ]], 1];

  If[Length[allPoints] == 0, Return[$Failed]];

  curvMin = Min[allPoints[[All, 3]]];
  curvMax = Max[allPoints[[All, 3]]];
  normalize[c_] := If[curvMax == curvMin, 0.5, (c - curvMin)/(curvMax - curvMin)];

  Graphics[{
    PointSize[Medium], Opacity[0.6],
    {ColorData["TemperatureMap"][normalize[#[[3]]]], Point[{#[[1]], #[[2]]}]} & /@ allPoints
  },
    Axes -> True,
    AxesLabel -> {"PC1", "PC2"},
    PlotLabel -> "Aligned Shape Space (color = curvature)",
    PlotRange -> All,
    Frame -> True
  ]
];

(* -----------------------------------------------------------------------------
   Visualization: 3D - PC1 vs PC2 vs PC3, colored by curvature
   ----------------------------------------------------------------------------- *)

plotPC3D[alignedBranches_] := Module[
  {allPoints, curvMin, curvMax, normalize},

  allPoints = Flatten[
    Values[KeyValueMap[
      Thread[{#2["PC1"], #2["PC2"], #2["PC3"], #2["Curvature"]}] &,
      Select[alignedBranches, AssociationQ[#] && !KeyExistsQ[#, "Error"] &]
    ]], 1];

  If[Length[allPoints] == 0, Return[$Failed]];

  curvMin = Min[allPoints[[All, 4]]];
  curvMax = Max[allPoints[[All, 4]]];
  normalize[c_] := If[curvMax == curvMin, 0.5, (c - curvMin)/(curvMax - curvMin)];

  Graphics3D[{
    PointSize[Medium], Opacity[0.6],
    {ColorData["TemperatureMap"][normalize[#[[4]]]], Point[{#[[1]], #[[2]], #[[3]]}]} & /@ allPoints
  },
    Axes -> True,
    AxesLabel -> {"PC1", "PC2", "PC3"},
    PlotLabel -> "Aligned Shape Space 3D (color = curvature)",
    Boxed -> True,
    ViewPoint -> {1.5, -2, 1.5}
  ]
];

(* -----------------------------------------------------------------------------
   Example usage:

   result = HGEvolve[
     {{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}},
     {{0, 1}, {1, 2}, {2, 3}, {3, 4}},
     50, "All",
     "UniformRandom" -> True, "MatchesPerStep" -> 30,
     "CurvatureAnalysis" -> True, "CurvaturePerVertex" -> True,
     "CurvatureMethod" -> "WolframRicci"
   ];

   aligned = alignAllBranches[result, 30];

   plotCurvatureVsRank[aligned]
   plotPC1vsPC2[aligned]
   plotPC3D[aligned]
   ----------------------------------------------------------------------------- *)
