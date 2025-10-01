(* ::Package:: *)

BeginPackage["HypergraphRewriting`"]

PackageImport["WolframInstitute`Hypergraph`"]

(* Public symbols - exposed API functions *)
HGEvolve::usage = "HGEvolve[rules, initialEdges, steps, property] performs multiway rewriting evolution and returns the specified property."

(* Options for HGEvolve *)
Options[HGEvolve] = {
  "CanonicalizeStates" -> True,
  "CanonicalizeEvents" -> False,
  "CausalTransitiveReduction" -> True,
  "EarlyTermination" -> False,
  "PatchBasedMatching" -> False,
  "FullCapture" -> True,
  "AspectRatio" -> 1/2
};


Begin["`Private`"]

(* Load the compiled library *)
$HypergraphLibrary = Quiet[FindLibrary["HypergraphRewriting"]]

If[$HypergraphLibrary === $Failed,
  $HypergraphLibrary = Quiet[FindFile[FileNameJoin[{DirectoryName[$InputFileName, 2], "LibraryResources", $SystemID, "libHypergraphRewriting" <> Internal`DynamicLibraryExtension[]}]]]
]

If[$HypergraphLibrary === $Failed,
  Message[HypergraphRewriting::nolib, "Could not find HypergraphRewriting library. Please ensure it is compiled and installed."];
  $Failed
]

(* LibraryLink function declarations *)
If[$HypergraphLibrary =!= $Failed,
  performRewriting = LibraryFunctionLoad[$HypergraphLibrary, "performRewriting", {LibraryDataType[ByteArray]}, LibraryDataType[ByteArray]];

  If[Head[performRewriting] === LibraryFunction,
    Print["HypergraphRewriting: Library functions loaded successfully from ", $HypergraphLibrary],
    Print["HypergraphRewriting: Failed to load performRewriting from ", $HypergraphLibrary]
  ],
  Print["HypergraphRewriting: Library not found - functions will not work"]
];

(* Error messages *)
HypergraphRewriting::nolib = "Could not load HypergraphRewriting library: `1`"
HypergraphRewriting::invarg = "Invalid argument: `1`"
HypergraphRewriting::libcall = "Library function call failed: `1`"
HypergraphRewriting::notimpl = "`1`"

HGEvolve[rules_List, initialEdges_List, steps_Integer, property_String : "EvolutionCausalBranchialGraph", OptionsPattern[]] := Module[{inputData, rulesAssoc, result, options},

  If[Head[performRewriting] =!= LibraryFunction,
    Return["Library function performRewriting not loaded"]
  ];

  (* Parse options with defaults *)
  options = Association[
    "CanonicalizeStates" -> OptionValue["CanonicalizeStates"],
    "CanonicalizeEvents" -> OptionValue["CanonicalizeEvents"],
    "CausalTransitiveReduction" -> OptionValue["CausalTransitiveReduction"],
    "EarlyTermination" -> OptionValue["EarlyTermination"],
    "PatchBasedMatching" -> OptionValue["PatchBasedMatching"],
    "FullCapture" -> OptionValue["FullCapture"],
    "AspectRatio" -> OptionValue["AspectRatio"]
  ];

  (* Convert rules to Association format for WXF *)
  rulesAssoc = Association[
    Table[
      "Rule" <> ToString[i] -> rules[[i]],
      {i, Length[rules]}
    ]
  ];

  (* Create input association with options *)
  inputData = Association[
    "InitialEdges" -> initialEdges,
    "Rules" -> rulesAssoc,
    "Steps" -> steps,
    "Options" -> options
  ];

  (* Serialize to WXF and call library function using ByteArray directly *)
  wxfByteArray = BinarySerialize[inputData];
  resultByteArray = performRewriting[wxfByteArray];

  If[ByteArrayQ[resultByteArray] && Length[resultByteArray] > 0,
    Module[{wxfData, states, events, causalEdges, branchialEdges},
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
          "EvolutionGraph", HGCreateEvolutionGraph[states, events],
          "EvolutionGraphStructure", HGCreateEvolutionGraph[states, events, {}, {}, False],
          "EvolutionCausalGraph", HGCreateEvolutionGraph[states, events, causalEdges],
          "EvolutionCausalGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, {}, False],
          "EvolutionBranchialGraph", HGCreateEvolutionGraph[states, events, {}, branchialEdges],
          "EvolutionBranchialGraphStructure", HGCreateEvolutionGraph[states, events, {}, branchialEdges, False],
          "EvolutionCausalBranchialGraph", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges],
          "EvolutionCausalBranchialGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges, False],
          _, $Failed
        ],
        (* WXF parsing failed *)
        $Failed
      ]
    ],
    $Failed
  ]
]

HGCreateStatesGraph[states_, events_, enableVertexStyles_ : True, aspectRatio_ : 1/2] := Module[{stateEdges},
  stateEdges = Map[DirectedEdge[states[#["InputStateId"]], states[#["OutputStateId"]]] &, events];
  Graph[
    stateEdges,
    VertexShapeFunction -> If[enableVertexStyles,
      Function[
        Inset[
          Framed[
            ResourceFunction["WolframModelPlot"][Rest /@ #2, ImageSize -> 64],
            Background -> LightBlue, RoundingRadius -> 3
          ], #1
        ]
      ],
      Automatic
    ],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]]
    ],
    EdgeStyle -> Hue[0.75, 0, 0.35],
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> aspectRatio
  ]
]

HGCreateCausalGraph[states_, events_, causalEdges_, enableVertexStyles_ : True, aspectRatio_ : 1/2] := Module[{eventVertices, causalEventEdges, connectedEvents},
  eventVertices = Map[
    DirectedEdge[
      states[#["InputStateId"]],
      states[#["OutputStateId"]],
      #
    ] &,
    events
  ];

  causalEventEdges = If[Length[causalEdges] > 0 && Length[eventVertices] > 0,
    Map[DirectedEdge[eventVertices[[#[[1]] + 1]], eventVertices[[#[[2]] + 1]]] &, causalEdges],
    {}
  ];

  (* Only include events that have causal edges *)
  connectedEvents = DeleteDuplicates[Flatten[List @@@ causalEventEdges]];

  Graph[
    connectedEvents,
    causalEventEdges,
    VertexShapeFunction -> If[enableVertexStyles,
      {
        _DirectedEdge -> Function[
          With[{from = #2[[1]], to = #2[[2]], tag = #2[[3]]},
            Inset[
              Framed[
                Row[{
                  ResourceFunction["WolframModelPlot"][
                    Rest /@ from,
                    GraphHighlight -> Rest /@ Extract[from, Position[from, {Alternatives @@ tag["ConsumedEdges"], ___}]],
                    GraphHighlightStyle -> Dashed,
                    ImageSize -> 64
                  ],
                  Graphics[{LightGray, FilledCurve[
                    {{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}},
                    {{{-1., 0.1848}, {0.2991, 0.1848}, {-0.1531, 0.6363}, {0.109, 0.8982}, {1., 0.0034},
                    {0.109, -0.8982}, {-0.1531, -0.6363}, {0.2991, -0.1848}, {-1., -0.1848}, {-1., 0.1848}}}
                  ]}, ImageSize -> 12],
                  ResourceFunction["WolframModelPlot"][
                    Rest /@ to,
                    GraphHighlight -> Rest /@ Extract[to, Position[to, {Alternatives @@ tag["ProducedEdges"], ___}]],
                    ImageSize -> 64
                  ]
                }],
                Background -> LightYellow, RoundingRadius -> 3
              ], #1
            ]
          ]
        ]
      },
      Automatic
    ],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      {_DirectedEdge -> Directive[LightYellow, EdgeForm[RGBColor[0.8, 0.8, 0.4]]]}
    ],
    EdgeStyle -> ResourceFunction["WolframPhysicsProjectStyleData"]["CausalGraph"]["EdgeStyle"],
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> aspectRatio
  ]
]

HGCreateBranchialGraph[states_, events_, branchialEdges_, enableVertexStyles_ : True, aspectRatio_ : 1/2] := Module[{branchialStateEdges},
  branchialStateEdges = If[Length[branchialEdges] > 0 && Length[events] > 0,
    Map[UndirectedEdge[states[events[[#[[1]] + 1]]["OutputStateId"]], states[events[[#[[2]] + 1]]["OutputStateId"]]] &, branchialEdges],
    {}
  ];

  Graph[
    branchialStateEdges,
    VertexShapeFunction -> If[enableVertexStyles,
      Function[
        Inset[
          Framed[
            ResourceFunction["WolframModelPlot"][Rest /@ #2, ImageSize -> 64],
            Background -> LightBlue, RoundingRadius -> 3
          ], #1
        ]
      ],
      Automatic
    ],
    VertexStyle -> If[enableVertexStyles,
      Automatic,
      Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]]
    ],
    EdgeStyle -> ResourceFunction["WolframPhysicsProjectStyleData"]["BranchialGraph"]["EdgeStyle"],
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> aspectRatio
  ]
]

HGCreateEvolutionGraph[states_, events_, causalEdges_ : {}, branchialEdges_ : {}, enableVertexStyles_ : True] := Module[{
  stateVertices, eventVertices, allVertices, stateToEventEdges,
  eventToStateEdges, causalGraphEdges, branchialGraphEdges, allEdges
},
  stateVertices = Values[states];

  eventVertices = Map[
    DirectedEdge[
      states[#["InputStateId"]],
      states[#["OutputStateId"]],
      #
    ] &,
    events
  ];

  allVertices = Join[stateVertices, eventVertices];

  stateToEventEdges = MapThread[
    UndirectedEdge[states[#1["InputStateId"]], #2] &,
    {events, eventVertices}
  ];

  eventToStateEdges = MapThread[
    DirectedEdge[#2, states[#1["OutputStateId"]]] &,
    {events, eventVertices}
  ];

  causalGraphEdges = If[Length[causalEdges] > 0 && Length[eventVertices] > 0,
    Map[DirectedEdge[eventVertices[[#[[1]] + 1]], eventVertices[[#[[2]] + 1]]] &, causalEdges],
    {}
  ];

  branchialGraphEdges = If[Length[branchialEdges] > 0 && Length[eventVertices] > 0,
    Map[UndirectedEdge[eventVertices[[#[[1]] + 1]], eventVertices[[#[[2]] + 1]]] &, branchialEdges],
    {}
  ];

  allEdges = Join[stateToEventEdges, eventToStateEdges, causalGraphEdges, branchialGraphEdges];

  Graph[
    allVertices,
    allEdges,
    VertexLabels -> {
      v_ :> Placed[HoldForm[v], Tooltip],
      DirectedEdge[_, _, tag_] :> Placed[tag, Tooltip]
    },
    VertexShapeFunction -> If[enableVertexStyles,
      {
        _DirectedEdge -> Function[
          With[{from = #2[[1]], to = #2[[2]], tag = #2[[3]]},
            Inset[
              Framed[
                Row[{
                  ResourceFunction["WolframModelPlot"][
                    Rest /@ from, (* Remove edge IDs for plotting *)
                    GraphHighlight -> Rest /@ Select[from, MemberQ[tag["ConsumedEdges"], First[#]] &],
                    GraphHighlightStyle -> Dashed,
                    ImageSize -> 64
                  ],
                  Graphics[{LightGray, FilledCurve[
                    {{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}}},
                    {{{-1., 0.1848}, {0.2991, 0.1848}, {-0.1531, 0.6363}, {0.109, 0.8982}, {1., 0.0034},
                    {0.109, -0.8982}, {-0.1531, -0.6363}, {0.2991, -0.1848}, {-1., -0.1848}, {-1., 0.1848}}}
                  ]}, ImageSize -> 12],
                  ResourceFunction["WolframModelPlot"][
                    Rest /@ to, (* Remove edge IDs for plotting *)
                    GraphHighlight -> Rest /@ Select[to, MemberQ[tag["ProducedEdges"], First[#]] &],
                    ImageSize -> 64
                  ]
                }],
                Background -> LightYellow, RoundingRadius -> 3
              ], #1
            ]
          ]
        ],
        Except[_DirectedEdge] -> Function[
          Inset[
            Framed[
              ResourceFunction["WolframModelPlot"][Rest /@ #2, ImageSize -> 64], (* Remove edge IDs *)
              Background -> LightBlue, RoundingRadius -> 3
            ], #1
          ]
        ]
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
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
    AspectRatio -> 1/2,
    PlotLabel -> None
  ]
]

End[] (* `Private` *)

EndPackage[]