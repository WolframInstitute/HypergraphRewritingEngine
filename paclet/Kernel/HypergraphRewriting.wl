(* ::Package:: *)

BeginPackage["HypergraphRewriting`"]

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
HGEvolve::usage = "HGEvolve[initialEdges, rules, steps] performs multiway rewriting evolution and returns events, causal and branchial edges."

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
  getVersion = LibraryFunctionLoad[$HypergraphLibrary, "getVersion", {}, "UTF8String"];
  testHypergraphCanonical = LibraryFunctionLoad[$HypergraphLibrary, "testHypergraphCanonical", {{Integer, 2, "Constant"}}, {Integer, 2}];
  performRewriting = LibraryFunctionLoad[$HypergraphLibrary, "performRewriting", {{Integer, 2, "Constant"}, {Integer, 4, "Constant"}, Integer}, {Integer, 1}];
  
  (* Check if functions loaded successfully *)
  If[Head[getVersion] === LibraryFunction && Head[testHypergraphCanonical] === LibraryFunction && Head[performRewriting] === LibraryFunction,
    Print["HypergraphRewriting: Library functions loaded successfully from ", $HypergraphLibrary],
    Print["HypergraphRewriting: Failed to load library functions from ", $HypergraphLibrary]
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

HGCanonical[HypergraphObject[edgeArray_]] := Module[{result},
  If[Head[testHypergraphCanonical] === LibraryFunction,
    result = testHypergraphCanonical[edgeArray];
    If[ListQ[result] && Length[result] > 0,
      integerArrayToEdges[result],
      $Failed
    ],
    "Library function not loaded"
  ]
]

HGGetVersion[] := If[Head[getVersion] === LibraryFunction, getVersion[], "Library function not loaded"]

HGDebugInfo[] := Module[{},
  Print["$HypergraphLibrary = ", $HypergraphLibrary];
  Print["$SystemID = ", $SystemID];
  Print["getVersion = ", Head[getVersion]];
  Print["testHypergraphCanonical = ", Head[testHypergraphCanonical]];
  Print["performRewriting = ", Head[performRewriting]];
  Print["DirectoryName[$InputFileName, 2] = ", DirectoryName[$InputFileName, 2]];
]

HGEvolve[initialEdges_List, rules_List, steps_Integer] := Module[{performRewritingWXF, inputData, rulesAssoc, result},
  Print["[HGEVOLVE] Starting HGEvolve with ", Length[initialEdges], " initial edges, ", Length[rules], " rules, ", steps, " steps"];
  
  (* Try to load the WXF-based function *)
  performRewritingWXF = LibraryFunctionLoad[$HypergraphLibrary, "performRewritingWXF", {{Integer, 1}}, {Integer, 1}];
  
  If[Head[performRewritingWXF] =!= LibraryFunction,
    Print["[HGEVOLVE] ERROR: Library function performRewritingWXF not loaded"];
    Return["Library function performRewritingWXF not loaded"]
  ];
  
  Print["[HGEVOLVE] Library function loaded successfully"];
  
  (* Convert rules to Association format for WXF *)
  rulesAssoc = Association[
    Table[
      "Rule" <> ToString[i] -> rules[[i]],
      {i, Length[rules]}
    ]
  ];
  
  Print["[HGEVOLVE] Created rules association: ", rulesAssoc];
  
  (* Create input association *)
  inputData = Association[
    "InitialEdges" -> initialEdges,
    "Rules" -> rulesAssoc,
    "Steps" -> steps
  ];
  
  Print["[HGEVOLVE] About to serialize input data: ", inputData];
  
  (* Serialize to WXF and call library function *)
  Print["[HGEVOLVE] Calling C++ function now..."];
  result = performRewritingWXF[Normal[BinarySerialize[inputData]]];
  Print["[HGEVOLVE] C++ function returned! Result length: ", If[ListQ[result], Length[result], "Not a list"]];
  
  If[ListQ[result] && Length[result] > 0,
    Module[{wxfBytes, wxfData, states, numStates, numEvents, stepsCompleted, numCausal, numBranchial, statesFormatted, edges},
      (* Debug: show what we got *)
      Print["Raw bytes from C++: ", result];
      Print["Byte count: ", Length[result]];
      
      (* Compare with a known good WXF *)
      testAssoc = <|"NumStates" -> 5, "NumEvents" -> 3, "StepsCompleted" -> 1|>;
      testWXF = BinarySerialize[testAssoc];
      Print["Known good WXF bytes: ", Normal[testWXF]];
      Print["Known good WXF length: ", Length[testWXF]];
      
      (* Convert integer list to ByteArray and deserialize *)
      wxfData = BinaryDeserialize[ByteArray[result]];
      
      If[AssociationQ[wxfData],
        (* Just return the clean multiway evolution data *)
        Print["Multiway Evolution Data:"];
        Print["States: ", Length[Lookup[wxfData, "States", {}]], " states"];
        Print["Events: ", Length[Lookup[wxfData, "Events", {}]], " events"];
        Print["CausalEdges: ", Length[Lookup[wxfData, "CausalEdges", {}]], " edges"];
        Print["BranchialEdges: ", Length[Lookup[wxfData, "BranchialEdges", {}]], " edges"];
        
        (* Create and return multiway graph visualization along with data *)
        Module[{states, events, causalEdges, branchialEdges, multiwayGraphViz},
          states = Lookup[wxfData, "States", {}];
          events = Lookup[wxfData, "Events", {}]; 
          causalEdges = Lookup[wxfData, "CausalEdges", {}];
          branchialEdges = Lookup[wxfData, "BranchialEdges", {}];
          
          (* Create multiway graph visualization *)
          multiwayGraphViz = HGCreateMultiwayGraph[
            Length[states], 
            Length[events], 
            causalEdges, 
            branchialEdges, 
            steps
          ];
          
          (* Return both data and visualization *)
          <|
            "States" -> states,
            "Events" -> events,
            "CausalEdges" -> causalEdges,
            "BranchialEdges" -> branchialEdges,
            "MultiwayGraph" -> multiwayGraphViz
          |>
        ],
        (* WXF parsing failed - return raw data for debugging *)
        wxfData
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

(* Create multiway graph visualization *)
HGCreateMultiwayGraph[numStates_, numEvents_, causalEdges_, branchialEdges_, steps_] := Module[{
  states, vertices, edges, stateVertices, eventVertices, causalEdgeList, branchialEdgeList
},
  (* Generate mock states for visualization *)
  states = Table[
    Table[{i, None, Random[Integer, {1, 10}], Random[Integer, {1, 10}]}, {i, Random[Integer, {2, 5}]}],
    {numStates}
  ];
  
  (* Create vertices for the graph - states are the actual vertices *)
  stateVertices = states;
  
  (* Create edges between states based on evolution events *)
  edges = {};
  If[numStates > 1,
    (* Create some sample transitions between states *)
    Do[
      If[i < numStates,
        AppendTo[edges, DirectedEdge[states[[i]], states[[i + 1]]]]
      ],
      {i, numStates - 1}
    ];
    
    (* Add some branching for demonstration *)
    If[numStates > 2,
      AppendTo[edges, DirectedEdge[states[[1]], states[[3]]]]
    ]
  ];
  
  (* Create the Graph with Wolfram multicomputation styling *)
  Graph[
    stateVertices,
    edges,
    EdgeStyle -> {Hue[0.75, 0, 0.35]},
    FormatType -> TraditionalForm,
    GraphLayout -> {"Dimension" -> 2, "VertexLayout" -> "LayeredDigraphEmbedding"},
    PerformanceGoal -> "Quality",
    VertexLabels -> {Placed[Automatic, Tooltip]},
    VertexSize -> {64},
    PlotLabel -> StringForm["Multiway Evolution: `` states, `` events, `` steps", numStates, numEvents, steps]
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