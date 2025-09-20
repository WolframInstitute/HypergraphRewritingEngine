(* ::Package:: *)

BeginPackage["HypergraphRewriting`"]

PackageImport["WolframInstitute`Hypergraph`"]

(* Public symbols - exposed API functions *)
HGCreate::usage = "HGCreate[edges] creates a new hypergraph from a list of hyperedges."
HGApplyRule::usage = "HGApplyRule[hypergraph, rule] applies a rewriting rule to a hypergraph."
HGApplyRules::usage = "HGApplyRules[hypergraph, rules, steps] applies multiple rewriting rules for specified steps."
HGCanonical::usage = "HGCanonical[hypergraph] returns the canonical form of a hypergraph."
HGPatternMatch::usage = "HGPatternMatch[hypergraph, pattern] finds all matches of a pattern in a hypergraph."
HGMultiwayEvolution::usage = "HGMultiwayEvolution[hypergraph, rules, steps] evolves a hypergraph using multiway rewriting."
HGWolframModel::usage = "HGWolframModel[init, rules, steps] creates and evolves a Wolfram model."

HGSetParallel::usage = "HGSetParallel[enabled] enables or disables parallel processing."
HGGetStats::usage = "HGGetStats[] returns performance statistics from the rewriting engine."
HGClearCache::usage = "HGClearCache[] clears internal caches."

(* Test functions *)
HGGetVersion::usage = "HGGetVersion[] returns the version of the hypergraph library."
HGDebugInfo::usage = "HGDebugInfo[] shows debug information about the library loading."

(* Rewriting functions *)
HGEvolve::usage = "HGEvolve[rules, initialEdges, steps, property] performs multiway rewriting evolution and returns the specified property."

(* Data type symbols *)
HypergraphObject::usage = "HypergraphObject[...] represents a hypergraph object."
RewritingRule::usage = "RewritingRule[lhs, rhs] represents a hypergraph rewriting rule."
MultiwayState::usage = "MultiwayState[...] represents a state in multiway evolution."
WolframModel::usage = "WolframModel[...] represents a Wolfram physics model."

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
  performRewriting = LibraryFunctionLoad[$HypergraphLibrary, "performRewriting", {{Integer, 1}}, {Integer, 1}];

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

(* Helper function to convert edges to proper format *)
edgesToIntegerArray[edges_List] := Module[{maxVertices, edgeArray},
  If[edges === {}, Return[{{}}]];

  (* Find maximum vertex ID to determine array dimensions *)
  maxVertices = Max[Flatten[edges, 1]];

  (* Convert each edge to a padded integer array *)
  edgeArray = Table[
    If[i <= Length[edges],
      PadRight[edges[[i]], maxVertices, -1], (* Pad with -1 for unused slots *)
      Table[-1, maxVertices]
    ],
    {i, Length[edges]}
  ];

  edgeArray
]

(* Convert integer array back to edge list *)
integerArrayToEdges[array_List] := Module[{edges},
  edges = Select[array, Length[#] > 0 &];
  Map[DeleteCases[#, -1] &, edges]
]

(* Public API Implementation *)

HGCreate[edges_List] := Module[{edgeArray},
  edgeArray = edgesToIntegerArray[edges];
  HypergraphObject[edgeArray]
]


HGEvolve[rules_List, initialEdges_List, steps_Integer, property_String : "EvolutionCausalBranchialGraph"] := Module[{inputData, rulesAssoc, result},

  If[Head[performRewriting] =!= LibraryFunction,
    Return["Library function performRewriting not loaded"]
  ];

  (* Convert rules to Association format for WXF *)
  rulesAssoc = Association[
    Table[
      "Rule" <> ToString[i] -> rules[[i]],
      {i, Length[rules]}
    ]
  ];

  (* Create input association *)
  inputData = Association[
    "InitialEdges" -> initialEdges,
    "Rules" -> rulesAssoc,
    "Steps" -> steps
  ];

  (* Serialize to WXF and call library function *)
  result = performRewriting[Normal[BinarySerialize[inputData]]];

  If[ListQ[result] && Length[result] > 0,
    Module[{wxfData, states, events, causalEdges, branchialEdges},
      (* Convert integer list to ByteArray and deserialize *)
      wxfData = BinaryDeserialize[ByteArray[result]];

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
          "StatesGraph", HGCreateStatesGraph[states, events],
          "StatesGraphStructure", HGCreateStatesGraph[states, events, False],
          "CausalGraph", HGCreateCausalGraph[states, events, causalEdges],
          "CausalGraphStructure", HGCreateCausalGraph[states, events, causalEdges, False],
          "BranchialGraph", HGCreateBranchialGraph[states, events, branchialEdges],
          "BranchialGraphStructure", HGCreateBranchialGraph[states, events, branchialEdges, False],
          "EvolutionGraph", HGCreateEvolutionGraph[states, events],
          "EvolutionCausalGraph", HGCreateEvolutionGraph[states, events, causalEdges],
          "EvolutionBranchialGraph", HGCreateEvolutionGraph[states, events, {}, branchialEdges],
          "EvolutionCausalBranchialGraph", HGCreateEvolutionGraph[states, events, causalEdges, branchialEdges],
          "EvolutionGraphStructure", HGCreateEvolutionGraph[states, events, {}, {}, False],
          "EvolutionCausalGraphStructure", HGCreateEvolutionGraph[states, events, causalEdges, {}, False],
          "EvolutionBranchialGraphStructure", HGCreateEvolutionGraph[states, events, {}, branchialEdges, False],
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

HGApplyRule[hypergraph_, rule_, opts___] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_apply_rule",
    {{_?NumericQ, 1, "Constant"}, {_?NumericQ, 2, "Constant"}} -> {_?NumericQ, 1, "Shared"}];

  If[Head[result] === LibraryFunction,
    result[Flatten[hypergraph], Flatten[rule]],
    $Failed
  ]
]

HGApplyRules[hypergraph_, rules_List, opts___] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_apply_rules",
    {{_?NumericQ, 1, "Constant"}, {_?NumericQ, 2, "Constant"}} -> {_?NumericQ, 1, "Shared"}];

  If[Head[result] === LibraryFunction,
    result[Flatten[hypergraph], Flatten[rules]],
    $Failed
  ]
]

HGPatternMatch[hypergraph_, pattern_, opts___] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_pattern_match",
    {{_?NumericQ, 1, "Constant"}, {_?NumericQ, 1, "Constant"}} -> {_?NumericQ, 1, "Shared"}];

  If[Head[result] === LibraryFunction,
    result[Flatten[hypergraph], Flatten[pattern]],
    $Failed
  ]
]

HGMultiwayEvolution[hypergraph_, rules_List, steps_, opts___] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_multiway_evolution",
    {{_?NumericQ, 1, "Constant"}, {_?NumericQ, 2, "Constant"}, Integer} -> {_?NumericQ, 1, "Shared"}];

  If[Head[result] === LibraryFunction,
    result[Flatten[hypergraph], Flatten[rules], steps],
    $Failed
  ]
]

HGWolframModel[rule_, initial_, steps_, opts___] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_wolfram_model",
    {{_?NumericQ, 2, "Constant"}, {_?NumericQ, 1, "Constant"}, Integer} -> {_?NumericQ, 1, "Shared"}];

  If[Head[result] === LibraryFunction,
    result[Flatten[rule], Flatten[initial], steps],
    $Failed
  ]
]

(* Debug function to show raw data *)
HGCreateMultiwayGraphDebug[result_, requestedSteps_] := Module[{},
  Graph[{}, {},
    PlotLabel -> StringForm["Debug: Raw result = ``, Requested steps = ``", result, requestedSteps]
  ]
]

HGCreateStatesGraph[states_, events_, enableVertexStyles_ : True] := Module[{stateVertices, stateEdges},
  stateVertices = Values[states];
  stateEdges = Map[DirectedEdge[states[#["InputStateId"]], states[#["OutputStateId"]]] &, events];
  Graph[
    stateVertices,
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
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top}
  ]
]

HGCreateCausalGraph[states_, events_, causalEdges_, enableVertexStyles_ : True] := Module[{eventVertices, causalEventEdges, connectedEvents},
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
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top}
  ]
]

HGCreateBranchialGraph[states_, events_, branchialEdges_, enableVertexStyles_ : True] := Module[{branchialStateEdges},
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
    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top}
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

(* Utility functions *)

HGSetParallel[enabled_] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_set_parallel", {True | False} -> "Void"];

  If[Head[result] === LibraryFunction,
    result[TrueQ[enabled]],
    $Failed
  ]
]

HGGetStats[] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_get_stats", {} -> {_?NumericQ, 1, "Shared"}];

  If[Head[result] === LibraryFunction,
    result[],
    $Failed
  ]
]

HGClearCache[] := Module[{lib, result},
  lib = loadHypergraphLibrary[];
  If[lib === $Failed, Return[$Failed]];

  result = LibraryFunctionLoad[lib, "hg_clear_cache", {} -> "Void"];

  If[Head[result] === LibraryFunction,
    result[],
    $Failed
  ]
]

(* Pretty printing *)
HypergraphObject /: Format[HypergraphObject[_]] := "HypergraphObject[<>]"
RewritingRule /: Format[RewritingRule[lhs_, rhs_]] := Row[{lhs, " \[Rule] ", rhs}]
MultiwayState /: Format[MultiwayState[data_]] := "MultiwayState[<>]"
WolframModel /: Format[WolframModel[init_, rules_, evolution_]] :=
  Column[{
    Row[{"Initial: ", init}],
    Row[{"Rules: ", Length[rules]}],
    Row[{"Evolution steps: ", Length[evolution]}]
  }]

End[] (* `Private` *)

EndPackage[]