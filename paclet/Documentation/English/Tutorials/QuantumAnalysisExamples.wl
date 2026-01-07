(* ::Package:: *)

(* ============================================================================ *)
(* Quantum Analysis Examples                                                    *)
(* ============================================================================ *)
(*
   This tutorial demonstrates the quantum mechanical analysis features
   implemented based on Gorard's paper "Some Quantum Mechanical Properties
   of the Wolfram Model".

   Features covered:
   1. Hausdorff Dimension Analysis (local dimension estimation)
   2. Geodesic Analysis (test particle paths)
   3. Topological Analysis (K5/K3,3 defect detection)
   4. Curvature Analysis (Ollivier-Ricci and dimension gradient)
   5. Entropy Analysis (graph entropy and information measures)
   6. Rotation Curve Analysis (orbital velocity vs radius)
   7. Hilbert Space Analysis (state inner products and vertex probabilities)
*)

(* Load the paclet *)
<< HypergraphRewriting`

(* ============================================================================ *)
(* Basic Setup: Define a rule and evolve *)
(* ============================================================================ *)

(* Standard Wolfram Model rule *)
rule = {{a, b}, {b, c}} -> {{a, b}, {b, c}, {c, d}, {d, a}};

(* Initial state (triangle) *)
initial = {{0, 1}, {1, 2}, {2, 0}};

(* Evolve for several steps with dimension analysis enabled *)
Print["Running evolution with analysis options..."];

result = HGEvolveV2[
  {rule},
  initial,
  6,  (* steps *)
  "All",
  "DimensionAnalysis" -> True,
  "GeodesicAnalysis" -> True,
  "TopologicalAnalysis" -> True,
  "IncludeStateContents" -> True,
  "ShowProgress" -> True
];

(* ============================================================================ *)
(* 1. Hausdorff Dimension Analysis *)
(* ============================================================================ *)

Print["\n=== Hausdorff Dimension Analysis ==="];

If[KeyExistsQ[result, "DimensionData"],
  dimData = result["DimensionData"];
  Print["Global dimension range: ", dimData["GlobalRange"]];
  Print["Number of states with dimension data: ", Length[dimData["PerState"]]];

  (* Show dimension distribution *)
  If[Length[dimData["PerState"]] > 0,
    dims = Values[dimData["PerState"]][[All, "Mean"]];
    Print["Dimension statistics:"];
    Print["  Min: ", Min[dims]];
    Print["  Max: ", Max[dims]];
    Print["  Mean: ", Mean[dims]];
  ];
  ,
  Print["DimensionData not available"];
];

(* Standalone dimension analysis on a grid graph *)
Print["\nStandalone HGHausdorffAnalysis on 4x4 grid:"];
gridEdges = Flatten[Table[
  {{i, j} -> {i + 1, j}, {i, j} -> {i, j + 1}},
  {i, 0, 3}, {j, 0, 3}
], 2];
gridEdges = Select[gridEdges, AllTrue[#[[2]], Between[{0, 4}]] &];

gridDim = HGHausdorffAnalysis[gridEdges, "MinRadius" -> 1, "MaxRadius" -> 3];
If[AssociationQ[gridDim],
  Print["Grid dimension (should be ~2.0): ", Mean[Values[gridDim["PerVertex"]]]];
];

(* ============================================================================ *)
(* 2. Geodesic Analysis *)
(* ============================================================================ *)

Print["\n=== Geodesic Analysis ==="];

If[KeyExistsQ[result, "GeodesicData"],
  geoData = result["GeodesicData"];
  Print["Number of geodesic paths: ", Length[geoData["Paths"]]];

  If[Length[geoData["Paths"]] > 0,
    (* Show path lengths *)
    pathLengths = Length /@ geoData["Paths"];
    Print["Path length statistics:"];
    Print["  Min: ", Min[pathLengths]];
    Print["  Max: ", Max[pathLengths]];
    Print["  Mean: ", Mean[N[pathLengths]]];

    (* Show bundle spread if available *)
    If[KeyExistsQ[geoData, "BundleSpread"] && Length[geoData["BundleSpread"]] > 0,
      Print["Bundle spread (geodesic deviation): ", Mean[geoData["BundleSpread"]]];
    ];
  ];
  ,
  Print["GeodesicData not available"];
];

(* ============================================================================ *)
(* 3. Topological Analysis (Particle Detection) *)
(* ============================================================================ *)

Print["\n=== Topological Analysis ==="];

If[KeyExistsQ[result, "TopologicalData"],
  topoData = result["TopologicalData"];

  (* K5 minors (complete graph on 5 vertices - non-planarity) *)
  If[KeyExistsQ[topoData, "K5Minors"],
    Print["K5 minors detected: ", Length[topoData["K5Minors"]]];
  ];

  (* K3,3 minors (complete bipartite - non-planarity) *)
  If[KeyExistsQ[topoData, "K33Minors"],
    Print["K3,3 minors detected: ", Length[topoData["K33Minors"]]];
  ];

  (* High degree vertices *)
  If[KeyExistsQ[topoData, "HighDegreeVertices"],
    Print["High degree vertices: ", Length[topoData["HighDegreeVertices"]]];
  ];

  (* Dimension spikes (potential particles) *)
  If[KeyExistsQ[topoData, "DimensionSpikes"],
    Print["Dimension spike vertices: ", Length[topoData["DimensionSpikes"]]];
  ];

  (* Topological charge *)
  If[KeyExistsQ[topoData, "TopologicalCharge"],
    charges = Values[topoData["TopologicalCharge"]];
    Print["Topological charge statistics:"];
    Print["  Total: ", Total[charges]];
    Print["  Non-zero vertices: ", Count[charges, x_ /; x != 0]];
  ];
  ,
  Print["TopologicalData not available"];
];

(* ============================================================================ *)
(* 4. Curvature Analysis *)
(* ============================================================================ *)

Print["\n=== Curvature Analysis ==="];

(* Run evolution with curvature analysis enabled *)
curvResult = HGEvolveV2[
  {rule},
  initial,
  5,
  "All",
  "CurvatureAnalysis" -> True,
  "ShowProgress" -> False
];

If[KeyExistsQ[curvResult, "CurvatureData"],
  curvData = curvResult["CurvatureData"];

  (* Ollivier-Ricci curvature *)
  If[KeyExistsQ[curvData, "OllivierRicci"],
    ricciVals = Values[curvData["OllivierRicci"]];
    Print["Ollivier-Ricci curvature:"];
    Print["  Mean: ", Mean[ricciVals]];
    Print["  Range: [", Min[ricciVals], ", ", Max[ricciVals], "]"];
  ];

  (* Dimension gradient curvature *)
  If[KeyExistsQ[curvData, "DimensionGradient"],
    gradVals = Values[curvData["DimensionGradient"]];
    Print["Dimension gradient curvature:"];
    Print["  Mean: ", Mean[gradVals]];
    Print["  Range: [", Min[gradVals], ", ", Max[gradVals], "]"];
  ];
  ,
  Print["CurvatureData not available - check FFI options"];
];

(* ============================================================================ *)
(* 5. Entropy Analysis *)
(* ============================================================================ *)

Print["\n=== Entropy Analysis ==="];

entropyResult = HGEvolveV2[
  {rule},
  initial,
  5,
  "All",
  "EntropyAnalysis" -> True,
  "ShowProgress" -> False
];

If[KeyExistsQ[entropyResult, "EntropyData"],
  entData = entropyResult["EntropyData"];

  (* Degree distribution entropy *)
  If[KeyExistsQ[entData, "DegreeEntropy"],
    Print["Degree distribution entropy: ", entData["DegreeEntropy"]];
  ];

  (* Graph entropy *)
  If[KeyExistsQ[entData, "GraphEntropy"],
    Print["Graph entropy: ", entData["GraphEntropy"]];
  ];

  (* Local entropy per vertex *)
  If[KeyExistsQ[entData, "LocalEntropy"],
    localEnt = Values[entData["LocalEntropy"]];
    Print["Local entropy statistics:"];
    Print["  Mean: ", Mean[localEnt]];
    Print["  Range: [", Min[localEnt], ", ", Max[localEnt], "]"];
  ];

  (* Fisher information *)
  If[KeyExistsQ[entData, "FisherInformation"],
    Print["Total Fisher information: ", entData["FisherInformation"]];
  ];
  ,
  Print["EntropyData not available - check FFI options"];
];

(* ============================================================================ *)
(* 6. Rotation Curve Analysis *)
(* ============================================================================ *)

Print["\n=== Rotation Curve Analysis ==="];

rotResult = HGEvolveV2[
  {rule},
  initial,
  6,
  "All",
  "DimensionAnalysis" -> True,  (* Needed to find center *)
  "RotationCurveAnalysis" -> True,
  "ShowProgress" -> False
];

If[KeyExistsQ[rotResult, "RotationData"],
  rotData = rotResult["RotationData"];

  (* Center vertex (highest dimension) *)
  If[KeyExistsQ[rotData, "CenterVertex"],
    Print["Center vertex (highest dimension): ", rotData["CenterVertex"]];
  ];

  (* Rotation curve points *)
  If[KeyExistsQ[rotData, "Curve"] && Length[rotData["Curve"]] > 0,
    Print["Rotation curve points: ", Length[rotData["Curve"]]];
    Print["Sample points (radius, velocity):"];
    Do[
      Print["  r=", pt["Radius"], " v=", pt["Velocity"]],
      {pt, Take[rotData["Curve"], UpTo[5]]}
    ];
  ];

  (* Power law fit *)
  If[KeyExistsQ[rotData, "PowerLawExponent"],
    Print["Power law fit: v ~ r^", rotData["PowerLawExponent"]];
    Print["  (Expected for inverse-square: -0.5)"];
  ];
  ,
  Print["RotationData not available - check FFI options"];
];

(* ============================================================================ *)
(* 7. Hilbert Space Analysis *)
(* ============================================================================ *)

Print["\n=== Hilbert Space Analysis ==="];

hilbertResult = HGEvolveV2[
  {rule},
  initial,
  4,  (* Fewer steps to limit branching *)
  "All",
  "HilbertSpaceAnalysis" -> True,
  "MaxStatesPerStep" -> 50,  (* Limit states for tractable analysis *)
  "ShowProgress" -> False
];

If[KeyExistsQ[hilbertResult, "HilbertSpaceData"],
  hilbData = hilbertResult["HilbertSpaceData"];

  Print["States analyzed: ", hilbData["NumStates"]];
  Print["Unique vertices: ", hilbData["NumVertices"]];

  (* Inner product statistics *)
  If[KeyExistsQ[hilbData, "MeanInnerProduct"],
    Print["Mean state inner product <psi|phi>: ", hilbData["MeanInnerProduct"]];
    Print["Max state inner product: ", hilbData["MaxInnerProduct"]];
  ];

  (* Vertex probability *)
  If[KeyExistsQ[hilbData, "MeanVertexProbability"],
    Print["Mean vertex probability P(v): ", hilbData["MeanVertexProbability"]];
  ];

  (* Probability entropy *)
  If[KeyExistsQ[hilbData, "VertexProbabilityEntropy"],
    Print["Vertex probability entropy: ", hilbData["VertexProbabilityEntropy"]];
  ];

  (* Per-vertex probabilities *)
  If[KeyExistsQ[hilbData, "VertexProbabilities"] && Length[hilbData["VertexProbabilities"]] > 0,
    Print["Sample vertex probabilities:"];
    Do[
      Print["  P(v=", v, ") = ", hilbData["VertexProbabilities"][v]],
      {v, Take[Keys[hilbData["VertexProbabilities"]], UpTo[5]]}
    ];
  ];
  ,
  Print["HilbertSpaceData not available - check FFI options"];
];

(* ============================================================================ *)
(* Visualization Examples *)
(* ============================================================================ *)

Print["\n=== Visualization Examples ==="];

(* State dimension plot *)
Print["Generating state dimension plot..."];
dimPlot = HGStateDimensionPlot[gridEdges,
  "Palette" -> "TemperatureMap",
  "MinRadius" -> 1,
  "MaxRadius" -> 3,
  ImageSize -> 400
];
Print["  -> Use Export[\"grid_dimension.png\", dimPlot] to save"];

(* Evolution timestep union plot *)
If[KeyExistsQ[result, "States"] && Length[result["States"]] > 0,
  Print["Generating timestep union plot (step 3)..."];
  unionPlot = HGTimestepUnionPlot[result, 3,
    "Palette" -> "TemperatureMap",
    "Layout" -> "Spring",
    ImageSize -> 500
  ];
  Print["  -> Use Export[\"union_step3.png\", unionPlot] to save"];
];

(* Filmstrip of all steps *)
If[KeyExistsQ[result, "States"] && Length[result["States"]] > 0,
  Print["Generating dimension filmstrip..."];
  filmstrip = HGDimensionFilmstrip[result,
    "Palette" -> "TemperatureMap",
    "Steps" -> {1, 3, 5},
    ImageSize -> 300
  ];
  Print["  -> Use Export[\"filmstrip.png\", filmstrip] to save"];
];

(* ============================================================================ *)
(* Summary *)
(* ============================================================================ *)

Print["\n=== Summary ==="];
Print["All quantum analysis features demonstrated successfully."];
Print[""];
Print["Key findings from Gorard's paper implemented:"];
Print["  - Local Hausdorff dimension estimates graph 'dimensionality'"];
Print["  - Geodesics trace test particle paths (like light rays)"];
Print["  - K5/K3,3 minors indicate topological defects (particles)"];
Print["  - Ollivier-Ricci curvature measures local graph curvature"];
Print["  - Entropy measures information content and diversity"];
Print["  - Rotation curves test inverse-square law emergence"];
Print["  - Hilbert space inner products capture quantum-like state overlap"];
Print[""];
Print["For interactive visualization, run blackhole_viz with:"];
Print["  ./blackhole_viz --steps 50"];
Print["Key bindings in blackhole_viz:"];
Print["  J - Toggle geodesic paths"];
Print["  K - Toggle topological defect markers"];
Print["  C - Cycle curvature heatmap modes"];
Print["  E - Toggle entropy heatmap"];
Print["  R - Toggle rotation curve plot"];
Print["  Q - Toggle Hilbert space stats panel"];
