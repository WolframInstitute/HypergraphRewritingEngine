(* ::Package:: *)

BeginPackage["HypergraphRewriting`"]

PackageExport["HGEvolve"]
PackageExport["HGEvolveV2"]
PackageExport["EdgeId"]

(* Public symbols - exposed API functions *)
HGEvolve::usage = "HGEvolve[rules, initialEdges, steps, property] performs multiway rewriting evolution and returns the specified property."
HGEvolveV2::usage = "HGEvolveV2[rules, initialEdges, steps, property] performs multiway rewriting using the unified V2 engine with selectable hash strategy."
EdgeId::usage = "EdgeId[id] wraps an edge identifier in state visualizations."

(* Options for HGEvolve *)
Options[HGEvolve] = {
  "CanonicalizeStates" -> True,
  "CanonicalizeEvents" -> False,
  "DeduplicateEvents" -> False,
  "CausalTransitiveReduction" -> True,
  "EarlyTermination" -> False,
  "PatchBasedMatching" -> False,
  "FullCapture" -> True,
  "AspectRatio" -> None,
  "MaxSuccessorStatesPerParent" -> 0,  (* 0 = unlimited *)
  "MaxStatesPerStep" -> 0,              (* 0 = unlimited *)
  "ExplorationProbability" -> 1.0       (* 1.0 = always explore *)
};

(* Options for HGEvolveV2 - includes hash strategy selection *)
Options[HGEvolveV2] = {
  "CanonicalizeStates" -> True,
  "CanonicalizeEvents" -> True,         (* V2 defaults to True *)
  "CausalTransitiveReduction" -> True,
  "AspectRatio" -> None,
  "MaxSuccessorStatesPerParent" -> 0,   (* 0 = unlimited *)
  "MaxStatesPerStep" -> 0,              (* 0 = unlimited *)
  "ExplorationProbability" -> 1.0,      (* 1.0 = always explore *)
  "HashStrategy" -> "iUT"               (* "WL" | "UT" | "iUT" *)
};

Begin["`Private`"]

(* Load the compiled library *)
$HypergraphLibrary = Quiet[FindLibrary["HypergraphRewriting"]]

If[$HypergraphLibrary === $Failed,
  (* Manual library path construction with platform-specific naming *)
  Module[{libraryName, libraryPath, pacletRoot, attemptedPath},
    (* Windows uses no prefix, Unix-like systems use "lib" prefix *)
    libraryName = If[StringMatchQ[$SystemID, "Windows*"],
      "HypergraphRewriting",
      "libHypergraphRewriting"
    ];

    (* Get paclet root directory *)
    pacletRoot = DirectoryName[$InputFileName, 2];

    (* Construct full library path *)
    libraryPath = FileNameJoin[{
      pacletRoot,
      "LibraryResources",
      $SystemID,
      libraryName <> "." <> Internal`DynamicLibraryExtension[]
    }];

    (* Try to find the library *)
    $HypergraphLibrary = Quiet[FindFile[libraryPath]];

    (* Debug output if still failed *)
    If[$HypergraphLibrary === $Failed,
      Print["HypergraphRewriting: Failed to find library"];
      Print["  SystemID: ", $SystemID];
      Print["  Paclet root: ", pacletRoot];
      Print["  Attempted path: ", libraryPath];
      Print["  File exists: ", FileExistsQ[libraryPath]];

      (* List what's actually in the LibraryResources directory *)
      If[DirectoryQ[FileNameJoin[{pacletRoot, "LibraryResources"}]],
        Print["  Available platforms: ", FileNames[All, FileNameJoin[{pacletRoot, "LibraryResources"}]]];
        If[DirectoryQ[FileNameJoin[{pacletRoot, "LibraryResources", $SystemID}]],
          Print["  Files in ", $SystemID, ": ", FileNames[All, FileNameJoin[{pacletRoot, "LibraryResources", $SystemID}]]];
        ];
      ];
    ];
  ]
]

If[$HypergraphLibrary === $Failed,
  Message[HypergraphRewriting::nolib, "Could not find HypergraphRewriting library. Please ensure it is compiled and installed."];
  $Failed
]

(* LibraryLink function declarations *)
If[$HypergraphLibrary =!= $Failed,
  performRewriting = LibraryFunctionLoad[$HypergraphLibrary, "performRewriting", {LibraryDataType[ByteArray]}, LibraryDataType[ByteArray]];
  performRewritingV2 = LibraryFunctionLoad[$HypergraphLibrary, "performRewritingV2", {LibraryDataType[ByteArray]}, LibraryDataType[ByteArray]];

  If[Head[performRewriting] === LibraryFunction && Head[performRewritingV2] === LibraryFunction,
    Print["HypergraphRewriting: Library functions (v1 + v2) loaded successfully from ", $HypergraphLibrary],
    Print["HypergraphRewriting: Failed to load library functions from ", $HypergraphLibrary]
  ],
  Print["HypergraphRewriting: Library not found - functions will not work"]
];

(* Error messages *)
HypergraphRewriting::nolib = "Could not load HypergraphRewriting library: `1`"
HypergraphRewriting::invarg = "Invalid argument: `1`"
HypergraphRewriting::libcall = "Library function call failed: `1`"
HypergraphRewriting::notimpl = "`1`"

(* ============================================================================ *)
(* Vertex Shape Functions - Factored out to avoid repetition *)
(* ============================================================================ *)

(* State vertex shape function - renders hypergraph states *)
stateVertexShapeFunction := Function[
  Inset[
    Framed[
      ResourceFunction["WolframModelPlot"][Rest /@ #2, ImageSize -> {32, 32}],
      Background -> LightBlue, RoundingRadius -> 3
    ], #1, {0, 0}
  ]
]

(* Event vertex shape function - renders from->to transitions *)
eventVertexShapeFunction := Function[
  With[{from = #2[[1]], to = #2[[2]], tag = #2[[3]]},
    Inset[
      Framed[
        Row[{
          ResourceFunction["WolframModelPlot"][
            Rest /@ from,
            GraphHighlight -> Rest /@ Select[from, MemberQ[tag["ConsumedEdges"], First[#]] &],
            GraphHighlightStyle -> Dashed,
            ImageSize -> 32
          ],
          Graphics[{LightGray, FilledCurve[
            {{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}},
            {{{-1., 0.1848}, {0.2991, 0.1848}, {-0.1531, 0.6363}, {0.109, 0.8982}, {1., 0.0034},
            {0.109, -0.8982}, {-0.1531, -0.6363}, {0.2991, -0.1848}, {-1., -0.1848}, {-1., 0.1848}}}
          ]}, ImageSize -> 8],
          ResourceFunction["WolframModelPlot"][
            Rest /@ to,
            GraphHighlight -> Rest /@ Select[to, MemberQ[tag["ProducedEdges"], First[#]] &],
            ImageSize -> 32
          ]
        }],
        Background -> LightYellow, RoundingRadius -> 3
      ], #1, {0, 0}
    ]
  ]
]

HGEvolve[rules_List, initialEdges_List, steps_Integer, property_String : "EvolutionCausalBranchialGraph", OptionsPattern[]] := Module[{inputData, wxfByteArray, resultByteArray, rulesAssoc, result, options},
  If[Head[performRewriting] =!= LibraryFunction,
    Return["Library function performRewriting not loaded"]
  ];

  (* Parse options with defaults *)
  options = Association[
    "CanonicalizeStates" -> OptionValue["CanonicalizeStates"],
    "CanonicalizeEvents" -> OptionValue["CanonicalizeEvents"],
    "DeduplicateEvents" -> OptionValue["DeduplicateEvents"],
    "CausalTransitiveReduction" -> OptionValue["CausalTransitiveReduction"],
    "EarlyTermination" -> OptionValue["EarlyTermination"],
    "PatchBasedMatching" -> OptionValue["PatchBasedMatching"],
    "FullCapture" -> OptionValue["FullCapture"],
    "AspectRatio" -> OptionValue["AspectRatio"],
    "MaxSuccessorStatesPerParent" -> OptionValue["MaxSuccessorStatesPerParent"],
    "MaxStatesPerStep" -> OptionValue["MaxStatesPerStep"],
    "ExplorationProbability" -> OptionValue["ExplorationProbability"]
  ];

  (* Convert rules to Association format for WXF *)
  rulesAssoc = Association[
    Table[
      "Rule" <> ToString[i] -> rules[[i]],
      {i, Length[rules]}
    ]
  ];

  (* Detect if we have single state or multiple states using Depth *)
  (* Depth 3: {{1,2},{2,3}} = single state (list of edges) *)
  (* Depth 4: {{{1,2}},{{3,4}}} = multiple states (list of states) *)
  initialStatesData = If[Depth[initialEdges] == 3,
    (* Single state: wrap in outer list to make uniform *)
    {initialEdges},
    (* Multiple states: already in correct format *)
    initialEdges
  ];

  (* Create input association with options *)
  inputData = Association[
    "InitialStates" -> initialStatesData,
    "Rules" -> rulesAssoc,
    "Steps" -> steps,
    "Options" -> options
  ];

  (* Serialize to WXF and call library function using ByteArray directly *)
  wxfByteArray = BinarySerialize[inputData];
  resultByteArray = performRewriting[wxfByteArray];

  If[ByteArrayQ[resultByteArray] && Length[resultByteArray] > 0,
    Module[{wxfData, states, events, causalEdges, branchialEdges, numStates, numEvents, numCausalEdges, numBranchialEdges},
      (* Deserialize WXF result *)
      wxfData = BinaryDeserialize[resultByteArray];

      If[AssociationQ[wxfData],
        (* Extract data *)
        states = Lookup[wxfData, "States", Association[]];
        events = Lookup[wxfData, "Events", {}];
        causalEdges = Lookup[wxfData, "CausalEdges", {}];
        branchialEdges = Lookup[wxfData, "BranchialEdges", {}];
        numStates = Lookup[wxfData, "NumStates", Length[states]];
        numEvents = Lookup[wxfData, "NumEvents", Length[events]];
        numCausalEdges = Lookup[wxfData, "NumCausalEdges", Length[causalEdges]];
        numBranchialEdges = Lookup[wxfData, "NumBranchialEdges", Length[branchialEdges]];

        (* Return requested property *)
        Switch[property,
          "States", states,
          "Events", events,
          "CausalEdges", causalEdges,
          "BranchialEdges", branchialEdges,
          "NumStates", numStates,
          "NumEvents", numEvents,
          "NumCausalEdges", numCausalEdges,
          "NumBranchialEdges", numBranchialEdges,
          "Debug", Association[
            "NumStates" -> numStates,
            "NumEvents" -> numEvents,
            "NumCausalEdges" -> numCausalEdges,
            "NumBranchialEdges" -> numBranchialEdges,
            "StatesLength" -> Length[states],
            "EventsLength" -> Length[events],
            "CausalEdgesLength" -> Length[causalEdges],
            "BranchialEdgesLength" -> Length[branchialEdges]
          ],
          "StatesGraph", HGCreateStatesGraph[states, events, True, options["AspectRatio"]],
          "StatesGraphStructure", HGCreateStatesGraph[states, events, False, options["AspectRatio"]],
          "CausalGraph", HGCreateCausalGraph[states, events, causalEdges, True, options["AspectRatio"]],
          "CausalGraphStructure", HGCreateCausalGraph[states, events, causalEdges, False, options["AspectRatio"]],
          "BranchialGraph", HGCreateBranchialGraph[states, events, branchialEdges, True, options["AspectRatio"]],
          "BranchialGraphStructure", HGCreateBranchialGraph[states, events, branchialEdges, False, options["AspectRatio"]],
          "EvolutionGraph", HGCreateEvolutionGraph[states, events, {}, {}, True, options["AspectRatio"]],
          "EvolutionGraphStructure", HGCreateEvolutionGraph[states, events, {}, {}, False, options["AspectRatio"]],
          "EvolutionCausalGraph", HGCreateEvolutionGraph[states, events, causalEdges, {}, True, options["AspectRatio"]],
          "EvolutionCausalGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, {}, False, options["AspectRatio"]],
          "EvolutionBranchialGraph", HGCreateEvolutionGraph[states, events, {}, branchialEdges, True, options["AspectRatio"]],
          "EvolutionBranchialGraphStructure", HGCreateEvolutionGraph[states, events, {}, branchialEdges, False, options["AspectRatio"]],
          "EvolutionCausalBranchialGraph", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges, True, options["AspectRatio"]],
          "EvolutionCausalBranchialGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges, False, options["AspectRatio"]],
          _, $Failed
        ],
        (* WXF parsing failed *)
        $Failed
      ]
    ],
    $Failed
  ]
]

HGCreateStatesGraph[states_, events_, enableVertexStyles_ : True, aspectRatio_ : None] := Module[{stateEdges, eventList},
  (* Events is now an Association, get values and expand by multiplicity *)
  eventList = Flatten[Map[
    ConstantArray[#, Lookup[#, "Multiplicity", 1]] &,
    Values[events]
  ], 1];

  (* Use canonical state IDs for graph edges to link isomorphic states *)
  (* Extract state edges (now states are Associations with "Edges" and "IsInitialState") *)
  stateEdges = Map[
    DirectedEdge[
      Lookup[states[Lookup[#, "CanonicalInputStateId", #["InputStateId"]]], "Edges"],
      Lookup[states[Lookup[#, "CanonicalOutputStateId", #["OutputStateId"]]], "Edges"]
    ] &,
    eventList
  ];

  (* Include states that are either connected by events OR are initial states *)
  connectedOrInitialStates = If[Length[stateEdges] == 0,
    (* No events: include all state edges *)
    Map[Lookup[#, "Edges"] &, Values[states]],
    (* Has events: include connected states plus any initial states *)
    DeleteDuplicates[Join[
      Flatten[List @@@ stateEdges, 1],
      Map[Lookup[#, "Edges"] &, Select[Values[states], Lookup[#, "IsInitialState", False] == "True" &]]
    ]]
  ];

  Graph[
    connectedOrInitialStates,
    stateEdges,
    VertexSize -> If[enableVertexStyles, 1/2, Automatic],
    VertexShapeFunction -> If[enableVertexStyles, stateVertexShapeFunction, Automatic],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]]
    ],
    EdgeStyle -> Hue[0.75, 0, 0.35],
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> aspectRatio
  ]
]

HGCreateCausalGraph[states_, events_, causalEdges_, enableVertexStyles_ : True, aspectRatio_ : None] := Module[{eventVertices, causalEventEdges, connectedEvents},
  (* Events is now an Association keyed by event ID *)
  eventVertices = Association[Map[
    #["EventId"] -> DirectedEdge[
      Lookup[states[#["InputStateId"]], "Edges"],
      Lookup[states[#["OutputStateId"]], "Edges"],
      #
    ] &,
    Values[events]
  ]];

  (* Causal edges now have From/To/Multiplicity structure *)
  causalEventEdges = If[Length[causalEdges] > 0 && Length[eventVertices] > 0,
    Flatten[Map[
      ConstantArray[
        DirectedEdge[eventVertices[#["From"]], eventVertices[#["To"]]],
        #["Multiplicity"]
      ] &,
      causalEdges
    ], 1],
    {}
  ];

  (* Only include events that have causal edges *)
  connectedEvents = DeleteDuplicates[Flatten[List @@@ causalEventEdges]];

  Graph[
    connectedEvents,
    causalEventEdges,
    VertexSize -> If[enableVertexStyles, 1/2, Automatic],
    VertexShapeFunction -> If[enableVertexStyles, {_DirectedEdge -> eventVertexShapeFunction}, Automatic],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      {_DirectedEdge -> Directive[LightYellow, EdgeForm[RGBColor[0.8, 0.8, 0.4]]]}
    ],
    EdgeStyle -> ResourceFunction["WolframPhysicsProjectStyleData"]["CausalGraph"]["EdgeStyle"],
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> aspectRatio
  ]
]

HGCreateBranchialGraph[states_, events_, branchialEdges_, enableVertexStyles_ : True, aspectRatio_ : None] := Module[{branchialStateEdges},
  (* Use canonical state IDs for graph edges to link isomorphic states *)
  (* Branchial edges now have From/To/Multiplicity structure *)
  branchialStateEdges = If[Length[branchialEdges] > 0 && Length[events] > 0,
    Flatten[Map[
      Module[{fromEvent, toEvent},
        fromEvent = events[#["From"]];
        toEvent = events[#["To"]];
        ConstantArray[
          UndirectedEdge[
            Lookup[states[Lookup[fromEvent, "CanonicalOutputStateId", fromEvent["OutputStateId"]]], "Edges"],
            Lookup[states[Lookup[toEvent, "CanonicalOutputStateId", toEvent["OutputStateId"]]], "Edges"]
          ],
          #["Multiplicity"]
        ]
      ] &,
      branchialEdges
    ], 1],
    {}
  ];

  Graph[
    branchialStateEdges,
    VertexSize -> If[enableVertexStyles, 1/2, Automatic],
    VertexShapeFunction -> If[enableVertexStyles, stateVertexShapeFunction, Automatic],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]]
    ],
    EdgeStyle -> ResourceFunction["WolframPhysicsProjectStyleData"]["BranchialGraph"]["EdgeStyle"],
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> aspectRatio
  ]
]

HGCreateEvolutionGraph[states_, events_, causalEdges_ : {}, branchialEdges_ : {}, enableVertexStyles_ : True, aspectRatio_ : None] := Module[{
  stateVertices, eventVertices, eventList, allVertices, stateToEventEdges,
  eventToStateEdges, causalGraphEdges, branchialGraphEdges, allEdges, baseGraph, baseEdges,
  formatStateForTooltip
},
  (* Helper function to format state for tooltip: wrap EdgeID with EdgeId[...] *)
  formatStateForTooltip[stateEdges_List] := Map[
    Prepend[Rest[#], EdgeId[First[#]]] &,
    stateEdges
  ];

  (* Extract just the edges from state data *)
  stateVertices = Map[Lookup[#, "Edges"] &, Values[states]];

  (* Events is now an Association, expand by multiplicity for visualization *)
  eventList = Flatten[Map[
    ConstantArray[#, Lookup[#, "Multiplicity", 1]] &,
    Values[events]
  ], 1];

  (* Event vertices contain raw state data for visualization, but carry event metadata *)
  eventVertices = Map[
    DirectedEdge[
      Lookup[states[#["InputStateId"]], "Edges"],
      Lookup[states[#["OutputStateId"]], "Edges"],
      #
    ] &,
    eventList
  ];

  (* Build association for edge lookup *)
  eventVertexAssoc = Association[Map[
    #["EventId"] -> DirectedEdge[
      Lookup[states[#["InputStateId"]], "Edges"],
      Lookup[states[#["OutputStateId"]], "Edges"],
      #
    ] &,
    Values[events]
  ]];

  (* Graph edges use canonical state IDs to link isomorphic states *)
  stateToEventEdges = MapThread[
    UndirectedEdge[Lookup[states[Lookup[#1, "CanonicalInputStateId", #1["InputStateId"]]], "Edges"], #2] &,
    {eventList, eventVertices}
  ];

  eventToStateEdges = MapThread[
    DirectedEdge[#2, Lookup[states[Lookup[#1, "CanonicalOutputStateId", #1["OutputStateId"]]], "Edges"]] &,
    {eventList, eventVertices}
  ];

  causalGraphEdges = If[Length[causalEdges] > 0 && Length[eventVertexAssoc] > 0,
    Flatten[Map[
      ConstantArray[
        DirectedEdge[eventVertexAssoc[#["From"]], eventVertexAssoc[#["To"]]],
        #["Multiplicity"]
      ] &,
      causalEdges
    ], 1],
    {}
  ];

  branchialGraphEdges = If[Length[branchialEdges] > 0 && Length[eventVertexAssoc] > 0,
    Flatten[Map[
      ConstantArray[
        UndirectedEdge[eventVertexAssoc[#["From"]], eventVertexAssoc[#["To"]]],
        #["Multiplicity"]
      ] &,
      branchialEdges
    ], 1],
    {}
  ];

  allEdges = Join[stateToEventEdges, eventToStateEdges, causalGraphEdges, branchialGraphEdges];

  (* Include states that are either connected by events OR are initial states *)
  connectedOrInitialStateVertices = If[Length[stateToEventEdges] == 0 && Length[eventToStateEdges] == 0,
    (* No state-event edges: include all states *)
    stateVertices,
    (* Has edges: include connected states plus any initial states *)
    DeleteDuplicates[Join[
      Cases[
        Join[stateToEventEdges, eventToStateEdges],
        (UndirectedEdge | DirectedEdge)[s_List, _] | (UndirectedEdge | DirectedEdge)[_, s_List] :> s,
        {1}
      ],
      Map[Lookup[#, "Edges"] &, Select[Values[states], Lookup[#, "IsInitialState", False] == "True" &]]
    ]]
  ];

  allVertices = Join[connectedOrInitialStateVertices, eventVertices];

  Graph[
    allVertices,
    allEdges,
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    PerformanceGoal -> "Quality",
    VertexLabels -> {
      v_List :> With[{formatted = formatStateForTooltip[v]}, Placed[HoldForm[formatted], Tooltip]],
      DirectedEdge[_, _, tag_] :> Placed[tag, Tooltip]
    },
    VertexSize -> If[enableVertexStyles, 1/2, Automatic],
    VertexShapeFunction -> If[enableVertexStyles,
      {
        _DirectedEdge -> eventVertexShapeFunction,
        Except[_DirectedEdge] -> stateVertexShapeFunction
      },
      Automatic
    ],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      {
        _DirectedEdge -> Directive[LightYellow, EdgeForm[RGBColor[0.8, 0.8, 0.4]]],
        Except[_DirectedEdge] -> Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]]
      }
    ],
    EdgeStyle -> {
      UndirectedEdge[Except[_DirectedEdge], _DirectedEdge] -> Hue[0.75, 0, 0.35],
      DirectedEdge[_DirectedEdge, Except[_DirectedEdge]] -> Hue[0.75, 0, 0.35],
      DirectedEdge[_DirectedEdge, _DirectedEdge] -> ResourceFunction["WolframPhysicsProjectStyleData"]["CausalGraph"]["EdgeStyle"],
      UndirectedEdge[_DirectedEdge, _DirectedEdge] -> ResourceFunction["WolframPhysicsProjectStyleData"]["BranchialGraph"]["EdgeStyle"]
    },
    AspectRatio -> aspectRatio,
    PlotLabel -> None
  ]
]

(* HGEvolveV2 - Unified V2 engine with hash strategy selection *)
HGEvolveV2[rules_List, initialEdges_List, steps_Integer, property_String : "EvolutionCausalBranchialGraph", OptionsPattern[]] := Module[{inputData, wxfByteArray, resultByteArray, rulesAssoc, result, options},
  If[Head[performRewritingV2] =!= LibraryFunction,
    Return["Library function performRewritingV2 not loaded"]
  ];

  (* Parse options with defaults *)
  options = Association[
    "CanonicalizeStates" -> OptionValue["CanonicalizeStates"],
    "CanonicalizeEvents" -> OptionValue["CanonicalizeEvents"],
    "CausalTransitiveReduction" -> OptionValue["CausalTransitiveReduction"],
    "AspectRatio" -> OptionValue["AspectRatio"],
    "MaxSuccessorStatesPerParent" -> OptionValue["MaxSuccessorStatesPerParent"],
    "MaxStatesPerStep" -> OptionValue["MaxStatesPerStep"],
    "ExplorationProbability" -> OptionValue["ExplorationProbability"],
    "HashStrategy" -> OptionValue["HashStrategy"]
  ];

  (* Convert rules to Association format for WXF *)
  rulesAssoc = Association[
    Table[
      "Rule" <> ToString[i] -> rules[[i]],
      {i, Length[rules]}
    ]
  ];

  (* Detect if we have single state or multiple states using Depth *)
  (* Depth 3: {{1,2},{2,3}} = single state (list of edges) *)
  (* Depth 4: {{{1,2}},{{3,4}}} = multiple states (list of states) *)
  initialStatesData = If[Depth[initialEdges] == 3,
    (* Single state: wrap in outer list to make uniform *)
    {initialEdges},
    (* Multiple states: already in correct format *)
    initialEdges
  ];

  (* Create input association with options *)
  inputData = Association[
    "InitialStates" -> initialStatesData,
    "Rules" -> rulesAssoc,
    "Steps" -> steps,
    "Options" -> options
  ];

  (* Serialize to WXF and call library function using ByteArray directly *)
  wxfByteArray = BinarySerialize[inputData];
  resultByteArray = performRewritingV2[wxfByteArray];

  If[ByteArrayQ[resultByteArray] && Length[resultByteArray] > 0,
    Module[{wxfData, states, events, causalEdges, branchialEdges, numStates, numEvents, numCausalEdges, numBranchialEdges},
      (* Deserialize WXF result *)
      wxfData = BinaryDeserialize[resultByteArray];

      If[AssociationQ[wxfData],
        (* Extract data *)
        states = Lookup[wxfData, "States", Association[]];
        events = Lookup[wxfData, "Events", {}];
        causalEdges = Lookup[wxfData, "CausalEdges", {}];
        branchialEdges = Lookup[wxfData, "BranchialEdges", {}];
        numStates = Lookup[wxfData, "NumStates", Length[states]];
        numEvents = Lookup[wxfData, "NumEvents", Length[events]];
        numCausalEdges = Lookup[wxfData, "NumCausalEdges", Length[causalEdges]];
        numBranchialEdges = Lookup[wxfData, "NumBranchialEdges", Length[branchialEdges]];

        (* Return requested property - same as HGEvolve *)
        Switch[property,
          "States", states,
          "Events", events,
          "CausalEdges", causalEdges,
          "BranchialEdges", branchialEdges,
          "NumStates", numStates,
          "NumEvents", numEvents,
          "NumCausalEdges", numCausalEdges,
          "NumBranchialEdges", numBranchialEdges,
          "Debug", Association[
            "NumStates" -> numStates,
            "NumEvents" -> numEvents,
            "NumCausalEdges" -> numCausalEdges,
            "NumBranchialEdges" -> numBranchialEdges,
            "StatesLength" -> Length[states],
            "EventsLength" -> Length[events],
            "CausalEdgesLength" -> Length[causalEdges],
            "BranchialEdgesLength" -> Length[branchialEdges]
          ],
          "StatesGraph", HGCreateStatesGraph[states, events, True, options["AspectRatio"]],
          "StatesGraphStructure", HGCreateStatesGraph[states, events, False, options["AspectRatio"]],
          "CausalGraph", HGCreateCausalGraph[states, events, causalEdges, True, options["AspectRatio"]],
          "CausalGraphStructure", HGCreateCausalGraph[states, events, causalEdges, False, options["AspectRatio"]],
          "BranchialGraph", HGCreateBranchialGraph[states, events, branchialEdges, True, options["AspectRatio"]],
          "BranchialGraphStructure", HGCreateBranchialGraph[states, events, branchialEdges, False, options["AspectRatio"]],
          "EvolutionGraph", HGCreateEvolutionGraph[states, events, {}, {}, True, options["AspectRatio"]],
          "EvolutionGraphStructure", HGCreateEvolutionGraph[states, events, {}, {}, False, options["AspectRatio"]],
          "EvolutionCausalGraph", HGCreateEvolutionGraph[states, events, causalEdges, {}, True, options["AspectRatio"]],
          "EvolutionCausalGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, {}, False, options["AspectRatio"]],
          "EvolutionBranchialGraph", HGCreateEvolutionGraph[states, events, {}, branchialEdges, True, options["AspectRatio"]],
          "EvolutionBranchialGraphStructure", HGCreateEvolutionGraph[states, events, {}, branchialEdges, False, options["AspectRatio"]],
          "EvolutionCausalBranchialGraph", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges, True, options["AspectRatio"]],
          "EvolutionCausalBranchialGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges, False, options["AspectRatio"]],
          _, $Failed
        ],
        (* WXF parsing failed *)
        $Failed
      ]
    ],
    $Failed
  ]
]

End[]  (* `Private` *)

EndPackage[]  (* `HypergraphRewriting` *)