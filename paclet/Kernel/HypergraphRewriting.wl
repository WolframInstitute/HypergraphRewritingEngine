(* ::Package:: *)

BeginPackage["HypergraphRewriting`"]

PackageExport["HGEvolve"]
PackageExport["HGHausdorffAnalysis"]
PackageExport["HGStateDimensionPlot"]
PackageExport["HGTimestepUnionPlot"]
PackageExport["HGDimensionFilmstrip"]
PackageExport["EdgeId"]

(* Public symbols *)
HGEvolve::usage = "HGEvolve[rules, initialEdges, steps, property] performs multiway rewriting evolution."
HGHausdorffAnalysis::usage = "HGHausdorffAnalysis[edges, opts] computes local Hausdorff dimension for each vertex in a graph."
HGStateDimensionPlot::usage = "HGStateDimensionPlot[edges, opts] plots a hypergraph with vertices colored by local dimension."
HGTimestepUnionPlot::usage = "HGTimestepUnionPlot[evolutionResult, step, opts] plots the union graph at a timestep with dimension coloring."
HGDimensionFilmstrip::usage = "HGDimensionFilmstrip[evolutionResult, opts] shows a grid of timestep union graphs with dimension coloring."
EdgeId::usage = "EdgeId[id] wraps an edge identifier."

Options[HGEvolve] = {
  "HashStrategy" -> "WL",
  "CanonicalizeStates" -> None,  (* None, Automatic, Full *)
  "CanonicalizeEvents" -> None,  (* None, Full, Automatic, or {keys...} *)
  "CausalTransitiveReduction" -> True,
  "MaxSuccessorStatesPerParent" -> 0,
  "MaxStatesPerStep" -> 0,
  "ExplorationProbability" -> 1.0,
  "ExploreFromCanonicalStatesOnly" -> False,  (* Only explore from canonical state representatives *)
  "ShowProgress" -> False,
  "ShowGenesisEvents" -> False,
  "AspectRatio" -> None,
  "DebugFFI" -> False,
  "IncludeStateContents" -> False,
  "IncludeEventContents" -> False,
  "BranchialStep" -> Automatic,  (* Automatic: BranchialGraph->-1 (final), Evolution*Branchial*->All; or explicit: -1, All, 1-based step *)
  "EdgeDeduplication" -> True,  (* True: one edge per event pair; False: N edges for N shared hypergraph edges *)
  (* Uniform Random Evolution (reservoir sampling like blackhole_viz) *)
  "UniformRandom" -> False,  (* True: use uniform random match selection with reservoir sampling *)
  "MatchesPerStep" -> 0,  (* How many matches to apply per step in uniform random mode (0 = all) *)
  (* Dimension Analysis Options *)
  "DimensionAnalysis" -> False,  (* True: compute Hausdorff dimensions for all states *)
  "DimensionColorBy" -> "Mean",  (* "Mean", "Variance", "Min", "Max" *)
  "DimensionPalette" -> "TemperatureMap",  (* ColorData palette name for dimension coloring *)
  "DimensionRange" -> Automatic,  (* {min, max} or Automatic - color scale range *)
  "DimensionFormula" -> "LinearRegression",  (* "LinearRegression" or "DiscreteDerivative" *)
  "DimensionRadius" -> {1, 5},  (* {minR, maxR} for dimension computation *)
  (* Geodesic Analysis Options - trace test particles through graph *)
  "GeodesicAnalysis" -> False,  (* True: trace geodesic paths through the graph *)
  "GeodesicSources" -> Automatic,  (* List of vertex IDs or Automatic (auto-select near high-dim regions) *)
  "GeodesicMaxSteps" -> 50,  (* Maximum path length *)
  "GeodesicBundleWidth" -> 5,  (* Number of paths in each bundle *)
  "GeodesicFollowGradient" -> False,  (* Follow dimension gradient vs random walk *)
  "GeodesicDimensionPercentile" -> 0.9,  (* For auto-selecting sources near high-dim regions *)
  (* Topological/Particle Analysis Options - Robertson-Seymour defect detection *)
  "TopologicalAnalysis" -> False,  (* True: detect topological defects (K5/K3,3 minors) *)
  "TopologicalCharge" -> False,  (* True: compute per-vertex topological charge *)
  "DetectK5Minors" -> True,  (* Look for K5 graph minors (non-planarity) *)
  "DetectK33Minors" -> True,  (* Look for K3,3 bipartite minors (non-planarity) *)
  "DetectDimensionSpikes" -> True,  (* Detect via dimension anomalies *)
  "DetectHighDegree" -> True,  (* Detect high-degree vertices *)
  "DimensionSpikeThreshold" -> 1.5,  (* Multiplier above mean to flag as spike *)
  "DegreePercentile" -> 0.95,  (* Top 5% by degree *)
  "ChargeRadius" -> 3.0,  (* Radius for local charge computation *)
  (* Curvature Analysis Options - Ollivier-Ricci and dimension gradient *)
  "CurvatureAnalysis" -> False,  (* True: compute per-vertex curvature *)
  "CurvatureMethod" -> "Both",  (* "OllivierRicci", "DimensionGradient", or "Both" *)
  (* Entropy Analysis Options - graph entropy and information measures *)
  "EntropyAnalysis" -> False,  (* True: compute entropy measures *)
  (* Rotation Curve Analysis Options - orbital velocity vs radius *)
  "RotationCurveAnalysis" -> False,  (* True: compute rotation curve *)
  "RotationNumRadii" -> 10,  (* Number of radius samples *)
  "RotationSamplesPerRadius" -> 20,  (* Geodesics per radius *)
  (* Hilbert Space Analysis Options - state bitvector inner products *)
  "HilbertSpaceAnalysis" -> False,  (* True: compute Hilbert space analysis *)
  "HilbertStep" -> -1,  (* Step to analyze: -1 = final, or specific step *)
  (* Branchial Analysis Options - distribution sharpness and branch entropy *)
  "BranchialAnalysis" -> False,  (* True: compute branchial analysis *)
  (* Multispace Analysis Options - vertex/edge probabilities across branches *)
  "MultispaceAnalysis" -> False,  (* True: compute multispace analysis *)
  (* Initial Condition Options - alternative to InitialEdges *)
  "InitialCondition" -> "Edges",  (* "Edges" (use provided), "Sprinkling" (Minkowski causal set) *)
  "SprinklingDensity" -> 500,  (* Number of spacetime points for sprinkling *)
  "SprinklingTimeExtent" -> 10.0,  (* Time dimension extent *)
  "SprinklingSpatialExtent" -> 10.0  (* Spatial dimension extent *)
};

Begin["`Private`"]

(* ============================================================================ *)
(* Library Loading *)
(* ============================================================================ *)

$HypergraphLibrary = Quiet[FindLibrary["HypergraphRewriting"]];

If[$HypergraphLibrary === $Failed,
  Module[{libraryName, libraryPath, pacletRoot},
    libraryName = If[StringMatchQ[$SystemID, "Windows*"], "HypergraphRewriting", "libHypergraphRewriting"];
    pacletRoot = DirectoryName[$InputFileName, 2];
    libraryPath = FileNameJoin[{pacletRoot, "LibraryResources", $SystemID,
      libraryName <> "." <> Internal`DynamicLibraryExtension[]}];
    $HypergraphLibrary = Quiet[FindFile[libraryPath]];
  ]
];

If[$HypergraphLibrary =!= $Failed,
  performRewriting = LibraryFunctionLoad[$HypergraphLibrary, "performRewriting",
    {LibraryDataType[ByteArray]}, LibraryDataType[ByteArray]];
  performHausdorffAnalysis = LibraryFunctionLoad[$HypergraphLibrary, "performHausdorffAnalysis",
    {LibraryDataType[ByteArray]}, LibraryDataType[ByteArray]];
];

(* ============================================================================ *)
(* Property -> Required Data Mapping *)
(* ============================================================================ *)

(* Base data requirements for each property - truly minimal *)
(* Structure graphs use even less data; optional content controlled by Include* options *)
propertyRequirementsBase = <|
  (* Raw data - minimal *)
  "States" -> {"States"},
  "Events" -> {"Events"},
  "CausalEdges" -> {"CausalEdges"},
  "BranchialEdges" -> {"BranchialEdges"},
  (* All graph properties - FFI handles via GraphProperty option, no WL-side data needed *)
  "StatesGraph" -> {}, "StatesGraphStructure" -> {},
  "CausalGraph" -> {}, "CausalGraphStructure" -> {},
  "BranchialGraph" -> {}, "BranchialGraphStructure" -> {},
  "EvolutionGraph" -> {}, "EvolutionGraphStructure" -> {},
  "EvolutionCausalGraph" -> {}, "EvolutionCausalGraphStructure" -> {},
  "EvolutionBranchialGraph" -> {}, "EvolutionBranchialGraphStructure" -> {},
  "EvolutionCausalBranchialGraph" -> {}, "EvolutionCausalBranchialGraphStructure" -> {},
  (* Counts - request specific count from FFI *)
  "NumStates" -> {"NumStates"},
  "NumEvents" -> {"NumEvents"},
  "NumCausalEdges" -> {"NumCausalEdges"},
  "NumBranchialEdges" -> {"NumBranchialEdges"},
  (* Global edge list and state bitvectors *)
  "GlobalEdges" -> {"GlobalEdges"},
  "StateBitvectors" -> {"StateBitvectors"},
  (* Debug/All *)
  "Debug" -> {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"},
  "All" -> {"States", "Events", "CausalEdges", "BranchialEdges", "NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"},
  (* Analysis data - computed in C++ FFI *)
  "DimensionData" -> {},
  "GeodesicData" -> {},
  "TopologicalData" -> {},
  "CurvatureData" -> {},
  "EntropyData" -> {},
  "RotationData" -> {},
  "HilbertSpaceData" -> {},
  "BranchialData" -> {},
  "MultispaceData" -> {}
|>;

(* Compute union of required data for a list of properties *)
(* Graph properties have empty requirements - FFI handles them via GraphProperty option *)
computeRequiredData[props_List, includeStateContents_, includeEventContents_, canonicalizeStates_:None] := Module[
  {unknown, requirements},

  unknown = Complement[props, Keys[propertyRequirementsBase]];
  If[Length[unknown] > 0,
    Message[HGEvolve::unknownprop, unknown];
    Return[$Failed]
  ];

  requirements = Lookup[propertyRequirementsBase, props];
  DeleteDuplicates[Flatten[requirements]]
]

computeRequiredData[prop_String, includeStateContents_, includeEventContents_, canonicalizeStates_:None] :=
  computeRequiredData[{prop}, includeStateContents, includeEventContents, canonicalizeStates]

HGEvolve::unknownprop = "Unknown property(s): `1`. Valid properties are: States, Events, CausalEdges, BranchialEdges, StatesGraph, CausalGraph, BranchialGraph, EvolutionGraph, their Structure variants, DimensionData, GeodesicData, TopologicalData, CurvatureData, EntropyData, RotationData, HilbertSpaceData, BranchialData, MultispaceData, GlobalEdges, StateBitvectors, All.";
HGEvolve::missingdata = "FFI did not return requested data: `1`. This indicates a bug in the FFI layer.";

(* ============================================================================ *)
(* Graph Creation Helpers *)
(* ============================================================================ *)

(* Vertex styles for Structure variants *)
stateVertexStyle = Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]];
eventVertexStyle = Directive[LightYellow, EdgeForm[RGBColor[0.8, 0.8, 0.4]]];

(* Helper function to format edges for display: bold edge ID, truncate if over limit *)
formatEdgesForDisplay[stateEdges_List, maxEdges_Integer:5] := Module[
  {displayed = Take[stateEdges, UpTo[maxEdges]], formatted},
  formatted = Map[Prepend[Rest[#], Style[First[#], Bold]] &, displayed];
  If[Length[stateEdges] > maxEdges, Append[formatted, "..."], formatted]
];

(* Format state tooltip with full info *)
formatStateTooltip[stateData_Association] := Column[{
  Row[{Style["State", Bold]}],
  Grid[{
    {"Id:", stateData["Id"]},
    {"CanonicalId:", stateData["CanonicalId"]},
    {"Step:", stateData["Step"]},
    {"IsInitial:", stateData["IsInitial"]},
    {Row[{"Edges (", Length[stateData["Edges"]], "):"}], formatEdgesForDisplay[stateData["Edges"]]}
  }, Alignment -> Left, Spacings -> {1, 0.5}]
}, Spacings -> 0.5];

(* Format event tooltip with full info *)
formatEventTooltip[eventData_Association] := Module[
  {rows = {
    {"Id:", eventData["Id"]},
    {"CanonicalId:", eventData["CanonicalId"]},
    {"RuleIndex:", eventData["RuleIndex"]},
    {"InputState:", eventData["InputState"]},
    {"OutputState:", eventData["OutputState"]}
  }},
  If[!MissingQ[eventData["ConsumedEdges"]], AppendTo[rows, {"ConsumedEdges:", eventData["ConsumedEdges"]}]];
  If[!MissingQ[eventData["ProducedEdges"]], AppendTo[rows, {"ProducedEdges:", eventData["ProducedEdges"]}]];
  Column[{
    Row[{Style["Event", Bold]}],
    Grid[rows, Alignment -> Left, Spacings -> {1, 0.5}]
  }, Spacings -> 0.5]
];

(* Format causal edge tooltip *)
formatCausalEdgeTooltip[data_Association] := Column[{
  Row[{Style["Causal Edge", Bold]}],
  Grid[{
    {"Producer Event:", data["ProducerEvent"]},
    {"Consumer Event:", data["ConsumerEvent"]}
  }, Alignment -> Left, Spacings -> {1, 0.5}]
}, Spacings -> 0.5];

(* Format branchial edge tooltip - shows states or events depending on context *)
formatBranchialEdgeTooltip[data_Association] := Column[{
  Row[{Style["Branchial Edge", Bold]}],
  Grid[
    If[KeyExistsQ[data, "State1"],
      {{"State 1:", data["State1"]}, {"State 2:", data["State2"]}},
      {{"Event 1:", data["Event1"]}, {"Event 2:", data["Event2"]}}
    ],
    Alignment -> Left, Spacings -> {1, 0.5}]
}, Spacings -> 0.5];

(* ============================================================================ *)
(* Graph Creation Functions *)
(* ============================================================================ *)

(* Edge styles by type for GraphData-based graphs *)
graphDataEdgeStyles = <|
  "Directed" -> Directive[Gray, Arrowheads[0.02]],
  "Causal" -> Directive[Orange, Arrowheads[0.02]],
  "Branchial" -> ResourceFunction["WolframPhysicsProjectStyleData"]["BranchialGraph"]["EdgeStyle"],
  "StateEvent" -> Directive[Gray],  (* Same gray as EventState for consistency *)
  "EventState" -> Directive[Gray, Arrowheads[0.02]]
|>;

(* Check if vertex data represents a state (has Edges but no RuleIndex) *)
isStateVertexData[data_Association] := KeyExistsQ[data, "Edges"] && !KeyExistsQ[data, "RuleIndex"];

(* State vertex shape function for styled mode using GraphData *)
makeStyledStateVertexShapeFn[vertexData_] := Function[{pos, v, size},
  With[{data = vertexData[v]},
    If[AssociationQ[data] && KeyExistsQ[data, "Edges"],
      Inset[Framed[
        ResourceFunction["WolframModelPlot"][Rest /@ data["Edges"], ImageSize -> {32, 32}],
        Background -> LightBlue, RoundingRadius -> 3
      ], pos, {0, 0}],
      (* Fallback for missing data *)
      Inset[Framed[v, Background -> LightBlue], pos, {0, 0}]
    ]
  ]
];

(* Event vertex shape function for styled mode using GraphData *)
makeStyledEventVertexShapeFn[vertexData_] := Function[{pos, v, size},
  With[{data = vertexData[v]},
    If[AssociationQ[data] && KeyExistsQ[data, "InputStateEdges"],
      Inset[Framed[Row[{
        ResourceFunction["WolframModelPlot"][
          Rest /@ data["InputStateEdges"],
          GraphHighlight -> Rest /@ Select[data["InputStateEdges"], MemberQ[data["ConsumedEdges"], First[#]] &],
          GraphHighlightStyle -> Dashed, ImageSize -> 32],
        Graphics[{LightGray, Polygon[{{-0.5, 0.3}, {0.5, 0}, {-0.5, -0.3}}]}, ImageSize -> 8],
        ResourceFunction["WolframModelPlot"][
          Rest /@ data["OutputStateEdges"],
          GraphHighlight -> Rest /@ Select[data["OutputStateEdges"], MemberQ[data["ProducedEdges"], First[#]] &],
          ImageSize -> 32]
      }], Background -> LightYellow, RoundingRadius -> 3], pos, {0, 0}],
      (* Fallback for missing data *)
      Inset[Framed[v, Background -> LightYellow], pos, {0, 0}]
    ]
  ]
];

(* Compute dimension-based color for a state vertex *)
getDimensionColor[stateId_, dimensionData_, palette_, colorBy_, dimRange_] := Module[
  {perState, dimStats, value, t, color},

  (* No dimension data -> use default *)
  If[!AssociationQ[dimensionData] || !KeyExistsQ[dimensionData, "PerState"],
    Return[Missing[]]];

  perState = dimensionData["PerState"];
  dimStats = Lookup[perState, stateId, Missing[]];
  If[MissingQ[dimStats], Return[Missing[]]];

  (* Get value based on colorBy mode *)
  value = Switch[colorBy,
    "Mean", Lookup[dimStats, "Mean", Missing[]],
    "Variance", Lookup[dimStats, "Variance", Missing[]],
    "Min", Lookup[dimStats, "Min", Missing[]],
    "Max", Lookup[dimStats, "Max", Missing[]],
    _, Lookup[dimStats, "Mean", Missing[]]
  ];
  If[MissingQ[value] || !NumericQ[value], Return[Missing[]]];

  (* Normalize to [0, 1] *)
  t = Clip[(value - dimRange[[1]]) / Max[dimRange[[2]] - dimRange[[1]], 0.001], {0, 1}];

  (* Get color from palette *)
  color = ColorData[palette][t];
  color
];

(* Create graph from FFI GraphData - main entry point *)
(* graphData: <|"Vertices" -> {...}, "Edges" -> {...}, "VertexData" -> <|...|>|> *)
(* styled: True for full hypergraph rendering, False for structure only *)
(* dimensionData: optional dimension data for coloring states *)
createGraphFromData[graphData_Association, aspectRatio_, styled_:False, dimensionData_:<||>, dimPalette_:"TemperatureMap", dimColorBy_:"Mean", dimRange_:{0, 3}] := Module[
  {vertices, edgeList, vertexData, vertexLabels, vertexStyles, vertexShapes, edgeStyles, edgeLabels, hasDimData, epilogLegend, g, addLegend},

  vertices = graphData["Vertices"];
  vertexData = graphData["VertexData"];
  hasDimData = AssociationQ[dimensionData] && KeyExistsQ[dimensionData, "PerState"] && Length[dimensionData["PerState"]] > 0;

  (* Helper to wrap graph with legend if dimension data exists *)
  addLegend = If[hasDimData && dimColorBy =!= None,
    Function[graph, Legended[graph,
      BarLegend[{dimPalette, dimRange},
        LegendLabel -> "Hausdorff Dimension",
        LegendMarkerSize -> {15, 150}
      ]
    ]],
    Identity
  ];

  (* Build edges with appropriate constructors based on Type *)
  edgeList = Map[
    Switch[#["Type"],
      "StateEvent", UndirectedEdge[#["From"], #["To"], Lookup[#, "Data", <||>]],
      "Branchial", UndirectedEdge[#["From"], #["To"], Lookup[#, "Data", <||>]],
      _, DirectedEdge[#["From"], #["To"], Lookup[#, "Data", #]]
    ] &,
    graphData["Edges"]
  ];

  (* Vertex labels (tooltips) - include dimension info if available *)
  vertexLabels = Map[
    Function[v,
      With[{data = vertexData[v]},
        v -> Placed[
          If[AssociationQ[data],
            If[isStateVertexData[data],
              (* Add dimension info to state tooltip if available *)
              If[hasDimData && KeyExistsQ[dimensionData["PerState"], data["Id"]],
                Column[{formatStateTooltip[data],
                  Row[{Style["Dimension: ", Bold], dimensionData["PerState"][data["Id"]]}]}],
                formatStateTooltip[data]
              ],
              formatEventTooltip[data]
            ],
            ToString[v]  (* Fallback for missing data *)
          ], Tooltip]
      ]
    ],
    vertices
  ];

  (* Edge styles by type *)
  edgeStyles = Map[
    With[{e = #},
      Switch[e["Type"],
        "StateEvent", UndirectedEdge[e["From"], e["To"], _] -> graphDataEdgeStyles["StateEvent"],
        "Branchial", UndirectedEdge[e["From"], e["To"], _] -> graphDataEdgeStyles["Branchial"],
        _, DirectedEdge[e["From"], e["To"], _] -> graphDataEdgeStyles[e["Type"]]
      ]
    ] &,
    graphData["Edges"]
  ];

  (* Edge labels (tooltips for edges with Data) *)
  edgeLabels = {
    DirectedEdge[_, _, tag_?AssociationQ] :> Placed[
      Which[
        KeyExistsQ[tag, "RuleIndex"], formatEventTooltip[tag],
        KeyExistsQ[tag, "ProducerEvent"], formatCausalEdgeTooltip[tag],
        KeyExistsQ[tag, "EventId"], Row[{"Event ", tag["EventId"]}],
        True, ""
      ], Tooltip],
    UndirectedEdge[_, _, tag_?AssociationQ] :> Placed[
      Which[
        KeyExistsQ[tag, "State1"] || KeyExistsQ[tag, "Event1"], formatBranchialEdgeTooltip[tag],
        KeyExistsQ[tag, "EventId"], Row[{"Event ", tag["EventId"]}],
        True, ""
      ], Tooltip]
  };

  (* No legend in the graph itself - dimension is shown via vertex colors *)
  epilogLegend = {};

  If[styled,
    (* Styled mode: use shape functions for hypergraph rendering *)
    (* When dimension data available, color state backgrounds *)
    vertexShapes = Map[
      Function[v,
        With[{data = vertexData[v]},
          v -> If[AssociationQ[data] && isStateVertexData[data],
            If[hasDimData,
              makeStyledStateVertexWithDimensionFn[vertexData, dimensionData, dimPalette, dimColorBy, dimRange],
              makeStyledStateVertexShapeFn[vertexData]
            ],
            makeStyledEventVertexShapeFn[vertexData]
          ]
        ]
      ],
      vertices
    ];
    addLegend[Graph[vertices, edgeList,
      VertexSize -> 1/2, VertexLabels -> vertexLabels, VertexShapeFunction -> vertexShapes,
      EdgeLabels -> edgeLabels, EdgeStyle -> edgeStyles,
      GraphLayout -> "LayeredDigraphEmbedding", AspectRatio -> aspectRatio,
      Epilog -> epilogLegend]]
    ,
    (* Structure mode: simple styles, with dimension coloring if available *)
    vertexStyles = Map[
      Function[v,
        With[{data = vertexData[v]},
          If[AssociationQ[data] && isStateVertexData[data],
            (* State vertex: use dimension color if available *)
            With[{dimColor = getDimensionColor[data["Id"], dimensionData, dimPalette, dimColorBy, dimRange]},
              v -> If[MissingQ[dimColor],
                stateVertexStyle,
                Directive[dimColor, EdgeForm[Darker[dimColor]]]
              ]
            ],
            (* Event vertex: use default style *)
            v -> eventVertexStyle
          ]
        ]
      ],
      vertices
    ];
    addLegend[Graph[vertices, edgeList,
      VertexLabels -> vertexLabels, VertexStyle -> vertexStyles,
      EdgeLabels -> edgeLabels, EdgeStyle -> edgeStyles,
      GraphLayout -> "LayeredDigraphEmbedding", AspectRatio -> aspectRatio,
      Epilog -> epilogLegend]]
  ]
];

(* State vertex shape function with dimension coloring *)
makeStyledStateVertexWithDimensionFn[vertexData_, dimensionData_, palette_, colorBy_, dimRange_] := Function[{pos, v, size},
  With[{data = vertexData[v]},
    If[AssociationQ[data] && KeyExistsQ[data, "Edges"],
      With[{dimColor = getDimensionColor[data["Id"], dimensionData, palette, colorBy, dimRange]},
        With[{bgColor = If[MissingQ[dimColor], LightBlue, Lighter[dimColor, 0.3]]},
          Inset[Framed[
            ResourceFunction["WolframModelPlot"][Rest /@ data["Edges"], ImageSize -> {32, 32}],
            Background -> bgColor, RoundingRadius -> 3,
            FrameStyle -> If[MissingQ[dimColor], Automatic, Darker[dimColor]]
          ], pos, {0, 0}]
        ]
      ],
      (* Fallback for missing data *)
      Inset[Framed[v, Background -> LightBlue], pos, {0, 0}]
    ]
  ]
];

(* ============================================================================ *)
(* Rule Normalization: Symbolic to Numeric Vertices *)
(* ============================================================================ *)

(* Normalize a single rule: convert symbolic vertices to consecutive integers *)
(* Example: {{a, b}, {b, c}} -> {{a, c}} becomes {{0, 1}, {1, 2}} -> {{0, 2}} *)
normalizeRule[rule_Rule] := Module[
  {lhs, rhs, allVertices, vertexMap, mapVertex},

  lhs = rule[[1]];
  rhs = rule[[2]];

  (* Collect all unique vertices from LHS first, then RHS *)
  (* LHS vertices get lower indices, ensuring pattern matching works correctly *)
  allVertices = DeleteDuplicates[Join[
    Flatten[lhs],
    Flatten[rhs]
  ]];

  (* Create mapping: vertex -> integer index (0-based) *)
  vertexMap = Association[MapIndexed[#1 -> #2[[1]] - 1 &, allVertices]];

  (* Map vertices to integers *)
  mapVertex[v_] := vertexMap[v];

  (* Apply mapping to LHS and RHS *)
  Map[mapVertex, lhs, {2}] -> Map[mapVertex, rhs, {2}]
];

(* Normalize a list of rules *)
normalizeRules[rules_List] := normalizeRule /@ rules;

(* Check if a rule already uses numeric vertices *)
ruleIsNumeric[rule_Rule] := AllTrue[
  Flatten[{rule[[1]], rule[[2]]}],
  IntegerQ
];

(* ============================================================================ *)
(* Main Function: HGEvolve *)
(* ============================================================================ *)

HGEvolve[rules_List, initialEdges_List, steps_Integer,
         property : (_String | {__String}) : "EvolutionCausalBranchialGraph",
         OptionsPattern[]] := Module[
  {inputData, wxfBytes, resultBytes, wxfData, requiredData, options,
   states, events, causalEdges, branchialEdges, aspectRatio, props,
   includeStateContents, includeEventContents, canonicalizeStates, canonicalizeEvents, graphProperties,
   normalizedRules, rulesAssoc, initialStatesData},

  If[Head[performRewriting] =!= LibraryFunction,
    Return[$Failed]
  ];

  (* Normalize property to list and deduplicate *)
  props = DeleteDuplicates[Flatten[{property}]];

  (* Get content options *)
  includeStateContents = OptionValue["IncludeStateContents"];
  includeEventContents = OptionValue["IncludeEventContents"];

  (* Get canonicalization options - used for per-graph-type ID selection *)
  canonicalizeStates = OptionValue["CanonicalizeStates"];
  canonicalizeEvents = OptionValue["CanonicalizeEvents"];

  (* Dimension analysis requires states with edges *)
  dimensionAnalysis = OptionValue["DimensionAnalysis"];
  If[dimensionAnalysis,
    includeStateContents = True  (* Force state contents for dimension computation *)
  ];

  (* Compute required data components - fail explicitly on unknown properties *)
  (* Pass canonicalizeStates to conditionally add States when state canonicalization is needed *)
  requiredData = computeRequiredData[props, includeStateContents, includeEventContents, canonicalizeStates];
  If[requiredData === $Failed, Return[$Failed]];

  (* Add States to required data if dimension analysis enabled *)
  If[dimensionAnalysis && !MemberQ[requiredData, "States"],
    requiredData = Append[requiredData, "States"]
  ];

  (* Collect all graph properties for FFI *)
  graphProperties = Select[props, StringMatchQ[#, "*Graph*"]&];

  (* Debug: print what data we're requesting from FFI *)
  If[OptionValue["DebugFFI"],
    Print["HGEvolve FFI Debug:"];
    Print["  Requested properties: ", props];
    Print["  Required data components: ", requiredData];
    Print["  Graph properties: ", graphProperties];
  ];

  (* Build options *)
  aspectRatio = OptionValue["AspectRatio"];
  (* Convert BranchialStep: All -> 0, positive for 1-based step, negative for from-end *)
  (* EvolutionCausalBranchialGraph defaults to All, BranchialGraph defaults to -1 (final step) *)
  (* Use first graph property for branchial step default, or empty string if none *)
  branchialStepValue = Replace[OptionValue["BranchialStep"], {
    Automatic :> If[Length[graphProperties] > 0 && StringMatchQ[First[graphProperties], "*Evolution*Branchial*"], 0, -1],
    All -> 0
  }];

  (* Get dimension radius for config *)
  dimensionRadius = OptionValue["DimensionRadius"];

  options = <|
    "HashStrategy" -> OptionValue["HashStrategy"],
    "CanonicalizeStates" -> OptionValue["CanonicalizeStates"],
    "CanonicalizeEvents" -> OptionValue["CanonicalizeEvents"],
    "CausalTransitiveReduction" -> OptionValue["CausalTransitiveReduction"],
    "MaxSuccessorStatesPerParent" -> OptionValue["MaxSuccessorStatesPerParent"],
    "MaxStatesPerStep" -> OptionValue["MaxStatesPerStep"],
    "ExplorationProbability" -> OptionValue["ExplorationProbability"],
    "ExploreFromCanonicalStatesOnly" -> OptionValue["ExploreFromCanonicalStatesOnly"],
    "ShowProgress" -> OptionValue["ShowProgress"],
    "ShowGenesisEvents" -> OptionValue["ShowGenesisEvents"],
    "BranchialStep" -> branchialStepValue,  (* 0=All, positive=1-based step, negative=from end *)
    "EdgeDeduplication" -> OptionValue["EdgeDeduplication"],
    "RequestedData" -> requiredData,
    "GraphProperties" -> graphProperties,  (* List of graph properties for FFI to generate *)
    (* Dimension analysis - compute in C++ instead of WL round-trips *)
    "DimensionAnalysis" -> dimensionAnalysis,
    "DimensionMinRadius" -> dimensionRadius[[1]],
    "DimensionMaxRadius" -> dimensionRadius[[2]],
    (* Geodesic analysis - trace test particles through graph *)
    "GeodesicAnalysis" -> OptionValue["GeodesicAnalysis"],
    "GeodesicSources" -> Replace[OptionValue["GeodesicSources"], Automatic -> {-1}],  (* -1 = auto-select *)
    "GeodesicMaxSteps" -> OptionValue["GeodesicMaxSteps"],
    "GeodesicBundleWidth" -> OptionValue["GeodesicBundleWidth"],
    "GeodesicFollowGradient" -> OptionValue["GeodesicFollowGradient"],
    "GeodesicDimensionPercentile" -> OptionValue["GeodesicDimensionPercentile"],
    (* Topological/Particle analysis - Robertson-Seymour defect detection *)
    "TopologicalAnalysis" -> OptionValue["TopologicalAnalysis"],
    "TopologicalCharge" -> OptionValue["TopologicalCharge"],
    "DetectK5Minors" -> OptionValue["DetectK5Minors"],
    "DetectK33Minors" -> OptionValue["DetectK33Minors"],
    "DetectDimensionSpikes" -> OptionValue["DetectDimensionSpikes"],
    "DetectHighDegree" -> OptionValue["DetectHighDegree"],
    "DimensionSpikeThreshold" -> OptionValue["DimensionSpikeThreshold"],
    "DegreePercentile" -> OptionValue["DegreePercentile"],
    "ChargeRadius" -> OptionValue["ChargeRadius"],
    (* Uniform random evolution mode *)
    "UniformRandom" -> OptionValue["UniformRandom"],
    "MatchesPerStep" -> OptionValue["MatchesPerStep"],
    (* Curvature analysis *)
    "CurvatureAnalysis" -> OptionValue["CurvatureAnalysis"],
    "CurvatureMethod" -> OptionValue["CurvatureMethod"],
    (* Entropy analysis *)
    "EntropyAnalysis" -> OptionValue["EntropyAnalysis"],
    (* Rotation curve analysis *)
    "RotationCurveAnalysis" -> OptionValue["RotationCurveAnalysis"],
    "RotationNumRadii" -> OptionValue["RotationNumRadii"],
    "RotationSamplesPerRadius" -> OptionValue["RotationSamplesPerRadius"],
    (* Hilbert space analysis *)
    "HilbertSpaceAnalysis" -> OptionValue["HilbertSpaceAnalysis"],
    "HilbertStep" -> OptionValue["HilbertStep"],
    (* Branchial analysis *)
    "BranchialAnalysis" -> OptionValue["BranchialAnalysis"],
    (* Multispace analysis *)
    "MultispaceAnalysis" -> OptionValue["MultispaceAnalysis"],
    (* Initial condition (sprinkling) *)
    "InitialCondition" -> OptionValue["InitialCondition"],
    "SprinklingDensity" -> OptionValue["SprinklingDensity"],
    "SprinklingTimeExtent" -> OptionValue["SprinklingTimeExtent"],
    "SprinklingSpatialExtent" -> OptionValue["SprinklingSpatialExtent"]
  |>;

  (* Normalize rules: convert symbolic vertices to integers if needed *)
  normalizedRules = normalizeRules[rules];

  (* Convert rules to Association *)
  rulesAssoc = Association[Table["Rule" <> ToString[i] -> normalizedRules[[i]], {i, Length[normalizedRules]}]];

  (* Handle single vs multiple initial states *)
  initialStatesData = If[Depth[initialEdges] == 3, {initialEdges}, initialEdges];

  (* Build input *)
  inputData = <|
    "InitialStates" -> initialStatesData,
    "Rules" -> rulesAssoc,
    "Steps" -> steps,
    "Options" -> options
  |>;

  (* Call FFI *)
  wxfBytes = BinarySerialize[inputData];
  resultBytes = performRewriting[wxfBytes];

  If[!ByteArrayQ[resultBytes] || Length[resultBytes] == 0, Return[$Failed]];

  wxfData = BinaryDeserialize[resultBytes];
  If[!AssociationQ[wxfData], Return[$Failed]];

  (* Extract data - validate that requested data was returned *)
  (* Only use defaults for data we didn't request *)
  states = If[MemberQ[requiredData, "States"],
    If[KeyExistsQ[wxfData, "States"], wxfData["States"],
      Message[HGEvolve::missingdata, "States"]; Return[$Failed]],
    <||>
  ];
  events = If[MemberQ[requiredData, "Events"] || MemberQ[requiredData, "EventsMinimal"],
    If[KeyExistsQ[wxfData, "Events"], wxfData["Events"],
      Message[HGEvolve::missingdata, "Events"]; Return[$Failed]],
    <||>
  ];
  causalEdges = If[MemberQ[requiredData, "CausalEdges"],
    If[KeyExistsQ[wxfData, "CausalEdges"], wxfData["CausalEdges"],
      Message[HGEvolve::missingdata, "CausalEdges"]; Return[$Failed]],
    {}
  ];
  branchialEdges = If[MemberQ[requiredData, "BranchialEdges"],
    If[KeyExistsQ[wxfData, "BranchialEdges"], wxfData["BranchialEdges"],
      Message[HGEvolve::missingdata, "BranchialEdges"]; Return[$Failed]],
    {}
  ];
  branchialStateEdges = If[MemberQ[requiredData, "BranchialStateEdges"] || MemberQ[requiredData, "BranchialStateEdgesAllSiblings"],
    If[KeyExistsQ[wxfData, "BranchialStateEdges"], wxfData["BranchialStateEdges"],
      Message[HGEvolve::missingdata, "BranchialStateEdges"]; Return[$Failed]],
    {}
  ];
  branchialStateVertices = If[MemberQ[requiredData, "BranchialStateEdges"] || MemberQ[requiredData, "BranchialStateEdgesAllSiblings"],
    If[KeyExistsQ[wxfData, "BranchialStateVertices"], wxfData["BranchialStateVertices"], {}],
    {}
  ];

  (* Debug: print what was returned from FFI *)
  If[OptionValue["DebugFFI"],
    Print["  FFI response keys: ", Keys[wxfData]];
    Print["  FFI response size: ", ByteCount[wxfData], " bytes"];
    Print["  States count: ", Length[states]];
    Print["  Events count: ", Length[events]];
    Print["  CausalEdges count: ", Length[causalEdges]];
    Print["  BranchialEdges count: ", Length[branchialEdges]];
  ];

  (* Dimension data is now computed in C++ by the FFI - no WL round-trips *)
  (* FFI returns DimensionData with PerState stats and GlobalRange *)
  dimensionData = If[dimensionAnalysis && KeyExistsQ[wxfData, "DimensionData"],
    wxfData["DimensionData"],
    <||>
  ];

  (* Geodesic data - test particle paths through graph *)
  geodesicData = If[OptionValue["GeodesicAnalysis"] && KeyExistsQ[wxfData, "GeodesicData"],
    wxfData["GeodesicData"],
    <||>
  ];

  (* Topological data - particle detection via Robertson-Seymour *)
  topologicalData = If[OptionValue["TopologicalAnalysis"] && KeyExistsQ[wxfData, "TopologicalData"],
    wxfData["TopologicalData"],
    <||>
  ];

  (* Curvature data - Ollivier-Ricci and dimension gradient *)
  curvatureData = If[OptionValue["CurvatureAnalysis"] && KeyExistsQ[wxfData, "CurvatureData"],
    wxfData["CurvatureData"],
    <||>
  ];

  (* Entropy data - graph entropy and information measures *)
  entropyData = If[OptionValue["EntropyAnalysis"] && KeyExistsQ[wxfData, "EntropyData"],
    wxfData["EntropyData"],
    <||>
  ];

  (* Rotation curve data - orbital velocity vs radius *)
  rotationData = If[OptionValue["RotationCurveAnalysis"] && KeyExistsQ[wxfData, "RotationData"],
    wxfData["RotationData"],
    <||>
  ];

  (* Hilbert space data - state bitvector inner products *)
  hilbertData = If[OptionValue["HilbertSpaceAnalysis"] && KeyExistsQ[wxfData, "HilbertSpaceData"],
    wxfData["HilbertSpaceData"],
    <||>
  ];

  (* Branchial data - distribution sharpness and branch entropy *)
  branchialData = If[OptionValue["BranchialAnalysis"] && KeyExistsQ[wxfData, "BranchialData"],
    wxfData["BranchialData"],
    <||>
  ];

  (* Multispace data - vertex/edge probabilities across branches *)
  multispaceData = If[OptionValue["MultispaceAnalysis"] && KeyExistsQ[wxfData, "MultispaceData"],
    wxfData["MultispaceData"],
    <||>
  ];

  (* Dimension coloring options *)
  dimPalette = OptionValue["DimensionPalette"];
  dimColorBy = OptionValue["DimensionColorBy"];
  dimRange = Replace[OptionValue["DimensionRange"], Automatic :>
    If[KeyExistsQ[dimensionData, "GlobalRange"], dimensionData["GlobalRange"], {0, 3}]];

  (* Return requested properties *)
  If[Length[props] == 1,
    (* Single property: return directly *)
    getProperty[First[props], states, events, causalEdges, branchialEdges, branchialStateEdges, branchialStateVertices, wxfData, aspectRatio, includeStateContents, includeEventContents, canonicalizeStates, canonicalizeEvents, dimensionData, dimPalette, dimColorBy, dimRange, geodesicData, topologicalData, curvatureData, entropyData, rotationData, hilbertData, branchialData, multispaceData],
    (* Multiple properties: return association *)
    Association[# -> getProperty[#, states, events, causalEdges, branchialEdges, branchialStateEdges, branchialStateVertices, wxfData, aspectRatio, includeStateContents, includeEventContents, canonicalizeStates, canonicalizeEvents, dimensionData, dimPalette, dimColorBy, dimRange, geodesicData, topologicalData, curvatureData, entropyData, rotationData, hilbertData, branchialData, multispaceData] & /@ props]
  ]
]

(* Property getter *)
(* Graph properties are handled via FFI GraphData - keyed by property name *)
getProperty[prop_, states_, events_, causalEdges_, branchialEdges_, branchialStateEdges_, branchialStateVertices_, wxfData_, aspectRatio_, includeStateContents_, includeEventContents_, canonicalizeStates_, canonicalizeEvents_, dimensionData_:<||>, dimPalette_:"TemperatureMap", dimColorBy_:"Mean", dimRange_:{0, 3}, geodesicData_:<||>, topologicalData_:<||>, curvatureData_:<||>, entropyData_:<||>, rotationData_:<||>, hilbertData_:<||>, branchialData_:<||>, multispaceData_:<||>] := Module[
  {isGraphProperty, isStyled, graphData},

  (* Graph properties: use FFI-provided GraphData keyed by property name *)
  isGraphProperty = StringMatchQ[prop, "*Graph*"];
  If[isGraphProperty,
    If[KeyExistsQ[wxfData, "GraphData"] && KeyExistsQ[wxfData["GraphData"], prop],
      graphData = wxfData["GraphData"][prop];
      isStyled = !StringMatchQ[prop, "*Structure"];
      Return[createGraphFromData[graphData, aspectRatio, isStyled, dimensionData, dimPalette, dimColorBy, dimRange]],
      (* GraphData for this property not available *)
      Return[$Failed]
    ]
  ];

  (* Non-graph properties: return raw data or counts *)
  Switch[prop,
    "States", states,
    "Events", events,
    "CausalEdges", causalEdges,
    "BranchialEdges", branchialEdges,
    "BranchialStateEdges", branchialStateEdges,
    "NumStates", wxfData["NumStates"],
    "NumEvents", wxfData["NumEvents"],
    "NumCausalEdges", wxfData["NumCausalEdges"],
    "NumBranchialEdges", wxfData["NumBranchialEdges"],
    "GlobalEdges", wxfData["GlobalEdges"],
    "StateBitvectors", wxfData["StateBitvectors"],
    "Debug", <|
      "NumStates" -> wxfData["NumStates"],
      "NumEvents" -> wxfData["NumEvents"],
      "NumCausalEdges" -> wxfData["NumCausalEdges"],
      "NumBranchialEdges" -> wxfData["NumBranchialEdges"]
    |>,
    "All", Module[{result = wxfData},
      If[Length[dimensionData] > 0, result = Append[result, "DimensionData" -> dimensionData]];
      If[Length[geodesicData] > 0, result = Append[result, "GeodesicData" -> geodesicData]];
      If[Length[topologicalData] > 0, result = Append[result, "TopologicalData" -> topologicalData]];
      If[Length[curvatureData] > 0, result = Append[result, "CurvatureData" -> curvatureData]];
      If[Length[entropyData] > 0, result = Append[result, "EntropyData" -> entropyData]];
      If[Length[rotationData] > 0, result = Append[result, "RotationData" -> rotationData]];
      If[Length[hilbertData] > 0, result = Append[result, "HilbertSpaceData" -> hilbertData]];
      If[Length[branchialData] > 0, result = Append[result, "BranchialData" -> branchialData]];
      If[Length[multispaceData] > 0, result = Append[result, "MultispaceData" -> multispaceData]];
      result
    ],
    "DimensionData", dimensionData,
    "GeodesicData", geodesicData,
    "TopologicalData", topologicalData,
    "CurvatureData", curvatureData,
    "EntropyData", entropyData,
    "RotationData", rotationData,
    "HilbertSpaceData", hilbertData,
    "BranchialData", branchialData,
    "MultispaceData", multispaceData,
    _, $Failed
  ]
]

(* ============================================================================ *)
(* HGHausdorffAnalysis - Local dimension estimation via FFI *)
(* ============================================================================ *)

Options[HGHausdorffAnalysis] = {
  "Formula" -> "LinearRegression",  (* or "DiscreteDerivative" *)
  "SaturationThreshold" -> 0.5,
  "MinRadius" -> 1,
  "MaxRadius" -> 5,
  "NumAnchors" -> 6,
  "AnchorSeparation" -> 3,
  "Aggregation" -> "Mean",  (* or "Min", "Max" - for discrete derivative *)
  "Directed" -> False  (* True: use directed edges for BFS *)
};

HGHausdorffAnalysis::nolib = "HypergraphRewriting library not loaded.";

HGHausdorffAnalysis[edges_List, opts:OptionsPattern[]] := Module[
  {formula, satThreshold, minR, maxR, numAnchors, anchorSep, aggregation, directed,
   edgesNormalized, inputAssoc, wxfInput, wxfOutput, result},

  If[performHausdorffAnalysis === $Failed || !ValueQ[performHausdorffAnalysis],
    Message[HGHausdorffAnalysis::nolib];
    Return[$Failed]
  ];

  (* Get options *)
  formula = OptionValue["Formula"];
  satThreshold = OptionValue["SaturationThreshold"];
  minR = OptionValue["MinRadius"];
  maxR = OptionValue["MaxRadius"];
  numAnchors = OptionValue["NumAnchors"];
  anchorSep = OptionValue["AnchorSeparation"];
  aggregation = OptionValue["Aggregation"];
  directed = OptionValue["Directed"];

  (* Normalize edges: accept both {v1, v2} and DirectedEdge[v1, v2] *)
  edgesNormalized = Replace[edges, {
    DirectedEdge[v1_, v2_] :> {v1, v2},
    UndirectedEdge[v1_, v2_] :> {v1, v2},
    Rule[v1_, v2_] :> {v1, v2}
  }, {1}];

  (* Build input association *)
  inputAssoc = <|
    "Edges" -> edgesNormalized,
    "Options" -> <|
      "Formula" -> formula,
      "SaturationThreshold" -> N[satThreshold],
      "MinRadius" -> minR,
      "MaxRadius" -> maxR,
      "NumAnchors" -> numAnchors,
      "AnchorSeparation" -> anchorSep,
      "Aggregation" -> aggregation,
      "Directed" -> directed
    |>
  |>;

  (* Convert to WXF and call FFI *)
  wxfInput = BinarySerialize[inputAssoc];
  wxfOutput = performHausdorffAnalysis[wxfInput];

  If[wxfOutput === $Failed,
    Return[$Failed]
  ];

  (* Parse result *)
  result = BinaryDeserialize[wxfOutput];
  result
]

(* ============================================================================ *)
(* HGStateDimensionPlot - Single graph with dimension-colored vertices *)
(* ============================================================================ *)

Options[HGStateDimensionPlot] = {
  "Palette" -> "TemperatureMap",
  "DimensionRange" -> Automatic,
  "Formula" -> "LinearRegression",
  "MinRadius" -> 1,
  "MaxRadius" -> 5,
  "Directed" -> False,
  ImageSize -> 400
};

HGStateDimensionPlot[edges_List, opts:OptionsPattern[]] := Module[
  {palette, dimRange, formula, minR, maxR, directed, imgSize,
   analysis, perVertex, vertices, dimMin, dimMax, colorFunc, vertexColors,
   vertexCoords, graphOpts, g},

  palette = OptionValue["Palette"];
  dimRange = OptionValue["DimensionRange"];
  formula = OptionValue["Formula"];
  minR = OptionValue["MinRadius"];
  maxR = OptionValue["MaxRadius"];
  directed = OptionValue["Directed"];
  imgSize = OptionValue[ImageSize];

  (* Compute dimensions *)
  analysis = HGHausdorffAnalysis[edges,
    "Formula" -> formula, "MinRadius" -> minR, "MaxRadius" -> maxR,
    "Directed" -> directed];

  If[!AssociationQ[analysis] || !KeyExistsQ[analysis, "PerVertex"],
    Return[$Failed]];

  perVertex = analysis["PerVertex"];
  vertices = Union[Flatten[edges, 1]];

  (* Compute range *)
  {dimMin, dimMax} = If[dimRange === Automatic,
    {Min[Values[perVertex]], Max[Values[perVertex]]},
    dimRange];

  (* Color function *)
  colorFunc = ColorData[palette];

  (* Vertex colors - use Replace to handle list-based vertex IDs *)
  vertexColors = Association[Table[
    v -> With[{d = Replace[perVertex[v], _Missing -> -1]},
      If[d > 0,
        colorFunc[Clip[(d - dimMin)/(dimMax - dimMin + 0.001), {0, 1}]],
        Gray
      ]
    ],
    {v, vertices}
  ]];

  (* If vertices are 2D tuples {i,j}, use them as coordinates *)
  vertexCoords = If[AllTrue[vertices, ListQ[#] && Length[#] == 2 &],
    Table[v -> v, {v, vertices}],
    {}
  ];

  (* Build graph options *)
  graphOpts = {
    VertexStyle -> Normal[vertexColors],
    VertexSize -> Medium,
    EdgeStyle -> Directive[Gray, Opacity[0.5]],
    VertexLabels -> Table[v -> Placed[
      Column[{v, Row[{"d=", Round[Replace[perVertex[v], _Missing -> -1], 0.01]}]}],
      Tooltip], {v, vertices}],
    ImageSize -> imgSize
  };

  (* Add coordinates or spring layout *)
  If[Length[vertexCoords] > 0,
    AppendTo[graphOpts, VertexCoordinates -> vertexCoords],
    AppendTo[graphOpts, GraphLayout -> "SpringElectricalEmbedding"]
  ];

  g = Graph[vertices, DirectedEdge @@@ edges, Sequence @@ graphOpts];

  Legended[g, BarLegend[{palette, {dimMin, dimMax}}, LegendLabel -> "Dimension"]]
]

(* ============================================================================ *)
(* HGTimestepUnionPlot - Union graph at a timestep with dimension coloring *)
(* ============================================================================ *)

Options[HGTimestepUnionPlot] = {
  "Palette" -> "TemperatureMap",
  "DimensionRange" -> Automatic,
  "Formula" -> "LinearRegression",
  "MinRadius" -> 1,
  "MaxRadius" -> 5,
  "Directed" -> False,
  "EdgeFilter" -> None,  (* None or {"MinStates" -> n} *)
  "VertexCoordinates" -> None,  (* List of {x,y} positions indexed by vertex ID, or Association *)
  "Layout" -> "MDS",  (* "MDS", "Spring", "TSNE", "Given" *)
  "LayoutDimension" -> 2,  (* 2 or 3 for 3D plots *)
  ImageSize -> 500
};

HGTimestepUnionPlot[evolutionResult_Association, step_Integer, opts:OptionsPattern[]] := Module[
  {palette, dimRange, formula, minR, maxR, directed, edgeFilter, positionsOpt, layoutOpt, imgSize,
   states, statesList, availableSteps, statesAtStep, allEdges, edgeTally, filteredTally, filteredEdges,
   unionEdges, vertices, analysis, perVertex, validDims, meanDim, dimMin, dimMax, colorFunc, vertexColors,
   edgeOpacity, gradientEdge, vertexCoords, graphOpts, g},

  palette = OptionValue["Palette"];
  dimRange = OptionValue["DimensionRange"];
  formula = OptionValue["Formula"];
  minR = OptionValue["MinRadius"];
  maxR = OptionValue["MaxRadius"];
  directed = OptionValue["Directed"];
  edgeFilter = OptionValue["EdgeFilter"];
  positionsOpt = OptionValue["VertexCoordinates"];
  layoutOpt = OptionValue["Layout"];
  imgSize = OptionValue[ImageSize];

  (* Get states - handle both <|"States" -> ...|> and direct states association *)
  states = If[KeyExistsQ[evolutionResult, "States"],
    evolutionResult["States"],
    (* Check if this IS the states directly (keys are integers = state ids) *)
    If[AllTrue[Keys[evolutionResult], IntegerQ],
      evolutionResult,
      Return[Failure["NoStates", <|"Message" -> "No States key in input"|>]]]];

  (* Convert to list if association *)
  statesList = If[AssociationQ[states], Values[states], states];

  (* Check format *)
  If[Length[statesList] == 0,
    Return[Failure["EmptyStates", <|"Message" -> "States list is empty"|>]]];

  (* Get available steps *)
  availableSteps = Union[#["Step"] & /@ statesList];

  (* Filter to requested step *)
  statesAtStep = Select[statesList, #["Step"] == step &];

  If[Length[statesAtStep] == 0,
    Return[Failure["NoStatesAtStep", <|
      "Message" -> StringJoin["No states at step ", ToString[step]],
      "AvailableSteps" -> availableSteps
    |>]]];

  (* Build union of edges and count occurrences *)
  (* State edges have format {edgeId, v1, v2} - extract {v1, v2} *)
  allEdges = Flatten[Table[
    Table[
      With[{rawEdge = If[Length[e] == 3, Rest[e], e]},
        Sort[rawEdge]
      ],
      {e, s["Edges"]}
    ],
    {s, statesAtStep}
  ], 1];

  (* Tally edges: {{edge, count}, ...} *)
  edgeTally = Tally[allEdges];

  (* Filter edges if requested *)
  filteredTally = If[AssociationQ[edgeFilter] && KeyExistsQ[edgeFilter, "MinStates"],
    Select[edgeTally, #[[2]] >= edgeFilter["MinStates"] &],
    edgeTally
  ];

  (* Extract just the edges *)
  filteredEdges = filteredTally[[All, 1]];

  unionEdges = filteredEdges;
  vertices = Union[Flatten[unionEdges, 1]];

  (* Compute dimensions on union graph *)
  analysis = HGHausdorffAnalysis[unionEdges,
    "Formula" -> formula, "MinRadius" -> minR, "MaxRadius" -> maxR,
    "Directed" -> directed];

  If[!AssociationQ[analysis] || !KeyExistsQ[analysis, "PerVertex"],
    perVertex = <||>,
    perVertex = analysis["PerVertex"]
  ];

  (* Compute range and mean for fallback *)
  validDims = Select[Values[perVertex], # > 0 &];
  meanDim = If[Length[validDims] > 0, Mean[validDims], 2.0];
  {dimMin, dimMax} = If[dimRange === Automatic,
    If[Length[validDims] > 0,
      {Min[validDims], Max[validDims]},
      {1, 3}],
    dimRange];

  colorFunc = ColorData[palette];

  (* Use mean dimension as fallback for missing/invalid vertices *)
  (* Use Replace instead of Lookup to handle list-based vertex IDs *)
  vertexColors = Association[Table[
    v -> With[{d = Replace[perVertex[v], _Missing -> meanDim]},
      With[{dVal = If[d > 0, d, meanDim]},
        colorFunc[Clip[(dVal - dimMin)/(dimMax - dimMin + 0.001), {0, 1}]]
      ]
    ],
    {v, vertices}
  ]];

  (* Edge opacity by frequency - use string keys to avoid Lookup list issue *)
  edgeOpacity = Association[Table[
    With[{e = ec[[1]], count = ec[[2]], numStates = Length[statesAtStep]},
      ToString[Sort[e]] -> N[0.3 + 0.7 * count/numStates]
    ],
    {ec, filteredTally}
  ]];

  (* Build graph - handle empty case *)
  If[Length[vertices] == 0 || Length[unionEdges] == 0,
    Return[Failure["EmptyGraph", <|"Message" -> "No vertices or edges"|>]]];

  (* Custom edge shape function with gradient line and colored arrowhead *)
  gradientEdge = Function[{pts, edge},
    With[{c1 = vertexColors[edge[[1]]], c2 = vertexColors[edge[[2]]],
          opKey = ToString[Sort[{edge[[1]], edge[[2]]}]]},
      With[{op = Replace[edgeOpacity[opKey], _Missing -> 0.5]},
        {Opacity[op], c2, Arrowheads[0.025],
         Arrow[Line[pts, VertexColors -> {c1, c2}]]}
      ]
    ]
  ];

  (* Build graph options based on layout *)
  graphOpts = {
    VertexStyle -> Normal[vertexColors],
    VertexSize -> 0.3,
    EdgeShapeFunction -> gradientEdge,
    ImageSize -> imgSize,
    ImagePadding -> 20,
    PlotLabel -> Row[{"Step ", step, " (", Length[statesAtStep], " states)"}]
  };

  (* Add layout option *)
  Switch[layoutOpt,
    "MDS",
      AppendTo[graphOpts, GraphLayout -> "SpectralEmbedding"],
    "Spring",
      AppendTo[graphOpts, GraphLayout -> "SpringElectricalEmbedding"],
    "Given",
      (* Use provided coordinates - handle 0-based vs 1-based indexing *)
      If[ListQ[positionsOpt] && Length[positionsOpt] > 0,
        With[{minV = Min[vertices], maxV = Max[vertices]},
          vertexCoords = If[minV == 0,
            (* 0-based: vertex v -> positions[[v+1]] *)
            Table[v -> positionsOpt[[v + 1]], {v, Select[vertices, # >= 0 && # < Length[positionsOpt] &]}],
            (* 1-based: vertex v -> positions[[v]] *)
            Table[v -> positionsOpt[[v]], {v, Select[vertices, # >= 1 && # <= Length[positionsOpt] &]}]
          ];
          If[Length[vertexCoords] > 0,
            AppendTo[graphOpts, VertexCoordinates -> vertexCoords]
          ]
        ]
      ],
    _,
      (* Default: spectral embedding *)
      AppendTo[graphOpts, GraphLayout -> "SpectralEmbedding"]
  ];

  g = Graph[vertices, DirectedEdge @@@ unionEdges, Sequence @@ graphOpts];

  Legended[g, BarLegend[{palette, {dimMin, dimMax}}, LegendLabel -> "Dimension"]]
]

(* ============================================================================ *)
(* HGDimensionFilmstrip - Grid of timestep union graphs *)
(* ============================================================================ *)

Options[HGDimensionFilmstrip] = {
  "Palette" -> "TemperatureMap",
  "DimensionRange" -> Automatic,
  "Formula" -> "LinearRegression",
  "MinRadius" -> 1,
  "MaxRadius" -> 5,
  "Directed" -> False,
  "Steps" -> All,
  "EdgeFilter" -> None,
  "VertexCoordinates" -> None,
  "Layout" -> "MDS",
  ImageSize -> 400
};

HGDimensionFilmstrip[evolutionResult_Association, opts:OptionsPattern[]] := Module[
  {palette, dimRange, formula, minR, maxR, directed, stepsOpt, edgeFilter, positionsOpt, layoutOpt, imgSize,
   states, statesList, allSteps, selectedSteps, plots},

  palette = OptionValue["Palette"];
  dimRange = OptionValue["DimensionRange"];
  formula = OptionValue["Formula"];
  minR = OptionValue["MinRadius"];
  maxR = OptionValue["MaxRadius"];
  directed = OptionValue["Directed"];
  stepsOpt = OptionValue["Steps"];
  edgeFilter = OptionValue["EdgeFilter"];
  positionsOpt = OptionValue["VertexCoordinates"];
  layoutOpt = OptionValue["Layout"];
  imgSize = OptionValue[ImageSize];

  (* Get states - handle both <|"States" -> ...|> and direct states association *)
  states = If[KeyExistsQ[evolutionResult, "States"],
    evolutionResult["States"],
    If[AllTrue[Keys[evolutionResult], IntegerQ],
      evolutionResult,
      Return[$Failed]]];

  statesList = If[AssociationQ[states], Values[states], states];
  allSteps = Union[#["Step"] & /@ statesList];

  selectedSteps = If[stepsOpt === All,
    allSteps,
    Intersection[Flatten[{stepsOpt}], allSteps]
  ];

  (* Generate plots for each step *)
  plots = Table[
    Quiet[HGTimestepUnionPlot[evolutionResult, step,
      "Palette" -> palette,
      "DimensionRange" -> dimRange,
      "Formula" -> formula,
      "MinRadius" -> minR,
      "MaxRadius" -> maxR,
      "Directed" -> directed,
      "EdgeFilter" -> edgeFilter,
      "VertexCoordinates" -> positionsOpt,
      "Layout" -> layoutOpt,
      ImageSize -> imgSize
    ]],
    {step, selectedSteps}
  ];

  (* Filter out failures *)
  plots = Select[plots, !FailureQ[#] && # =!= $Failed &];

  If[Length[plots] == 0, Return[$Failed]];

  (* Arrange in column *)
  Column[plots, Spacings -> 2]
]

End[]
EndPackage[]
