(* ::Package:: *)

BeginPackage["HypergraphRewriting`"]

PackageExport["HGEvolve"]

(* Sessions: an evolution a caller can CONTINUE. The engine has served these four verbs
   since #121; these are what makes them reachable. See the Sessions section below. *)
PackageExport["HGSessionObject"]
PackageExport["HGSessionOpen"]
PackageExport["HGSessionStep"]
PackageExport["HGSessionQuery"]
PackageExport["HGSessionClose"]


(* Public symbols *)
HGEvolve::usage = "HGEvolve[rules, initialEdges, steps, property] performs multiway rewriting evolution.
HGEvolve[rules, \"Grid\", steps, property] evolves from a grid initial condition.
HGEvolve[rules, <|\"Type\"->\"Grid\", \"Width\"->w, \"Height\"->h|>, steps, property] evolves from a custom grid.
HGEvolve[rules, \"Sprinkling\", steps, property] evolves from a Minkowski sprinkling."

Options[HGEvolve] = {
  "CanonicalizeStates" -> None,  (* None, Automatic, Full *)
  "CanonicalizeEvents" -> None,  (* None, Full, Automatic, or {keys...} *)
  "CausalTransitiveReduction" -> True,
  "MaxSuccessorStatesPerParent" -> 0,
  "MaxStatesPerStep" -> 0,
  "ExplorationProbability" -> 1.0,
  "TransitionRate" -> 1.0,  (* Keep each transition with this probability, drawn independently from the transition's own isomorphism-invariant identity and RandomSeed. Reproducible at any thread count and on either device, and it carries the spine guarantee: a state whose every draw failed still keeps its minimum-key transition, so a sparse sample reaches full depth instead of going extinct. ExplorationProbability thins STATES and has no spine. CPU only; the GPU warns and runs unsampled. *)
  "RuleWeights" -> {},  (* Per-rule multipliers on "TransitionRate", in rule order. {} weights every rule equally. A short list is a partial override: rules past its end take 1. Composes with the rate rather than replacing it, so "TransitionRate" -> 1 with weights {1, 0} still samples \[LongDash] rule 2 is dropped and rule 1 is untouched. CPU only; the GPU warns and weights every rule equally. *)
  "ExploreFromCanonicalStatesOnly" -> False,  (* Only explore from canonical state representatives *)
  "QuotientInitialStates" -> False,  (* True: isomorphic initial states collapse to one canonical root (needs ExploreFromCanonicalStatesOnly). False (default): each provided initial state is a distinct entry point, matching MultiwaySystem. *)
  "TargetDevice" -> "CPU",  (* "CPU" | "GPU" (like NetTrain[]). "GPU" runs the bundled hg_evolve_gpu binary when present, else falls back to CPU with a message. The GPU engine honors CanonicalizeStates (None | Automatic | Full) and its state counts match the CPU's in every mode. *)
  "ShowProgress" -> False,
  "ShowGenesisEvents" -> False,
  "AspectRatio" -> None,
  "DebugFFI" -> False,
  "IncludeStateContents" -> False,
  "IncludeCanonicalHashes" -> False,  (* True: include per-state IR canonical hash ("CanonicalHash"); stable across runs, for fusing pruned runs by isomorphism class *)
  "IncludeEventContents" -> False,
  "BranchialStep" -> Automatic,  (* Automatic: BranchialGraph->-1 (final), Evolution*Branchial*->All; or explicit: -1, All, 1-based step *)
  "EdgeDeduplication" -> True,  (* True: one edge per event pair; False: N edges for N shared hypergraph edges *)
  "UniformRandom" -> False,  (* True: with "MatchesPerStep", stop keeping new states once that many exist for the step. A cap by ARRIVAL ORDER, which depends on the schedule, not a uniform draw. "TransitionRate" is the uniform, reproducible sampler. *)
  "MatchesPerStep" -> 0,  (* How many matches to apply per step in uniform random mode (0 = all) *)
  (* Rulial-space plot: color each transition edge by the rule that fired it (the
     fiber of the rule -> multiway functor). Applies to the styled graph
     properties, whose edge payloads carry RuleIndex; Structure variants ship
     topology only and are unaffected. *)
  "ColorByRule" -> False,
  (* Initial Condition Options - alternative to InitialEdges *)
  "InitialCondition" -> "Edges",  (* "Edges", "Grid", "Sprinkling", "BrillLindquist", "Poisson", "Uniform" *)
  (* Topology options *)
  "Topology" -> "Flat",  (* "Flat", "Cylinder", "Torus", "Sphere", "Klein", "Mobius" *)
  "MajorRadius" -> 10.0,  (* Major radius for curved topologies *)
  "MinorRadius" -> 3.0,  (* Minor radius for torus *)
  (* Grid options *)
  "GridWidth" -> 10,  (* Grid width for "Grid" initial condition *)
  "GridHeight" -> 10,  (* Grid height for "Grid" initial condition *)
  "GridHoles" -> {},  (* List of {x, y, radius} for holes in grid *)
  (* Sprinkling/Minkowski options *)
  "SprinklingDensity" -> 500,  (* Number of spacetime points for sprinkling *)
  "SprinklingTimeExtent" -> 10.0,  (* Time dimension extent *)
  "SprinklingSpatialExtent" -> 10.0,  (* Spatial dimension extent *)
  "SprinklingSpatialDim" -> 2,  (* 1, 2, or 3 spatial dimensions *)
  "SprinklingLightconeAngle" -> 1.0,  (* Speed of light (c = 1 default) *)
  "SprinklingAlexandrovCutoff" -> 5.0,  (* Max proper time separation *)
  "SprinklingTransitivityReduction" -> True,  (* Remove redundant causal edges *)
  "SprinklingMaxEdgesPerVertex" -> 50,  (* Limit connectivity *)
  (* Brill-Lindquist options *)
  "BrillLindquistMass1" -> 3.0,  (* Mass of first black hole *)
  "BrillLindquistMass2" -> 3.0,  (* Mass of second black hole *)
  "BrillLindquistSeparation" -> 10.0,  (* Distance between black holes *)
  "BrillLindquistBoxX" -> {-15.0, 15.0},  (* X domain *)
  "BrillLindquistBoxY" -> {-15.0, 15.0},  (* Y domain *)
  (* Sampling options *)
  "EdgeThreshold" -> Automatic,  (* Max distance for edge creation *)
  "PoissonMinDistance" -> 1.0,  (* Minimum separation for Poisson disk *)
  "RandomSeed" -> Automatic  (* Random seed for reproducibility *)
};

HGSessionObject::usage =
  "HGSessionObject[data] is a live exploration held by an engine worker. Produced by " <>
  "HGSessionOpen, advanced by HGSessionStep, read by HGSessionQuery, released by HGSessionClose.";

HGSessionOpen::usage =
  "HGSessionOpen[rules, initialEdges, property] opens a continuable evolution and returns an " <>
  "HGSessionObject. The rules, the identity convention and the set of artifacts the run records " <>
  "are fixed here; HGSessionStep carries the SAME exploration further rather than re-running it.";
HGSessionStep::usage =
  "HGSessionStep[session, n] carries the session's exploration n steps further from the frontier " <>
  "it stopped at, and returns the property the session was opened for. HGSessionStep[session, n, " <>
  "property] returns a different property of the same accumulated graph, provided the session " <>
  "was opened recording what that property needs.";
HGSessionQuery::usage =
  "HGSessionQuery[session] re-reads the session's accumulated graph without exploring further. " <>
  "HGSessionQuery[session, property] reads a different property of it.";
HGSessionClose::usage =
  "HGSessionClose[session] releases the engine holding the session. The handle is not reused, " <>
  "so a later verb on a closed session reports that rather than answering from another one.";

HGSessionOpen::noworker =
  "No persistent engine worker is available for TargetDevice -> `1`, so no session can be " <>
  "opened. A session's handle names a live process; the one-shot fallback HGEvolve uses would " <>
  "mint a handle in a process that exits with the reply. HGEvolve itself still works.";
HGSessionOpen::nohandle =
  "The engine answered the Open but returned no session handle, so there is nothing to continue.";
HGSessionOpen::live =
  "A session is already live on this worker; close it before opening another. This build serves " <>
  "one session at a time so that opening a second cannot silently discard the first.";
HGSessionStep::badsession =
  "`1` is not an HGSessionObject.";
HGSessionStep::negsteps =
  "HGSessionStep needs a non-negative number of steps, not `1`. Use HGSessionQuery to re-read " <>
  "the session without exploring.";

Options[HGSessionOpen] = Options[HGEvolve];

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
];

(* Standalone process-isolation engine binary, shipped beside the DLL. It reads
   the WXF job on stdin and writes the WXF result on stdout (progress on stderr).
   When present it supersedes the LibraryLink call: an abort is a process kill
   and a crash cannot take down the kernel. See docs/ARCHITECTURE.md. *)
hgFindEngineBinary[base_String] := Module[{exeName, pacletRoot, exePath},
  exeName = If[StringMatchQ[$SystemID, "Windows*"], base <> ".exe", base];
  pacletRoot = DirectoryName[$InputFileName, 2];
  exePath = FileNameJoin[{pacletRoot, "LibraryResources", $SystemID, exeName}];
  Quiet[FindFile[exePath]]
];
$HypergraphEngineBinary = hgFindEngineBinary["hg_evolve"];
(* GPU variant, built with the CUDA backend. Selected by TargetDevice -> "GPU":
   it always runs hg_gpu::evolve and marshals through the same WXF path as the
   CPU binary. *)
$HypergraphEngineBinaryGPU = hgFindEngineBinary["hg_evolve_gpu"];
hgGpuBinaryAvailableQ[] := StringQ[$HypergraphEngineBinaryGPU] && FileExistsQ[$HypergraphEngineBinaryGPU];

(* Run an engine binary on the WXF job fed to its stdin, and collect the WXF
   result from its stdout. Progress/diagnostics arrive on stderr. stdout carries
   raw bytes; RunProcess returns them as a string decoded one byte per character,
   recovered with ToCharacterCode[..., "ISO8859-1"]. *)
(* The engine runs with a clean environment except for HG_*-prefixed variables, which are the
   engine's own instrument/debug channel (e.g. HG_FFI_PAYLOAD_FILE writes a per-key wire-size
   breakdown, HG_GPU_DBG_TIME prints phase attribution). *)
hgEngineEnvironment[] := Association @ Select[
  Normal @ GetEnvironment[],
  StringQ[First[#]] && StringStartsQ[First[#], "HG_"] &];

hgRunEngineBinary[exe_String, wxfBytes_ByteArray] := Module[{proc, outStr, stderr},
  proc = RunProcess[{exe}, All, wxfBytes, ProcessEnvironment -> hgEngineEnvironment[]];
  If[!AssociationQ[proc],
    Message[HGEvolve::enginefail, "spawn"]; Return[$Failed]];
  stderr = proc["StandardError"];
  If[StringQ[stderr] && StringTrim[stderr] =!= "",
    Message[HGEvolve::enginemsg, StringTrim[stderr]]];
  outStr = proc["StandardOutput"];
  If[proc["ExitCode"] =!= 0 || !StringQ[outStr] || outStr === "",
    Message[HGEvolve::enginefail, proc["ExitCode"]]; Return[$Failed]];
  ByteArray[ToCharacterCode[outStr, "ISO8859-1"]]
];

(* Persistent worker: keep an engine binary alive in --serve-socket mode and
   stream length-prefixed WXF jobs to it over a loopback TCP socket, so expensive
   per-process setup -- the GPU CUDA context (~0.7 s) above all -- is paid once
   and amortised across calls. A socket rather than the process pipe because
   Wolfram's StartProcess drops BinaryWrite to stdin, truncates WriteString at
   NUL, and does not surface a running child's stdout; SocketConnect +
   BinaryWrite/SocketReadMessage are NUL-safe. The worker publishes its
   OS-assigned port to a temp file (race-free); the WL side polls it, connects,
   then frames jobs ([8-byte little-endian length][payload] both directions; a
   zero-length reply means that job errored). Any transport failure marks the
   device broken and every call falls back to the one-shot RunProcess path --
   correct everywhere, amortised where the socket worker is available. When the
   kernel exits its socket closes and the worker's serve loop ends, so no process
   is left behind. *)
$hgWorkerProc = <||>;     (* device -> ProcessObject *)
$hgWorkerSock = <||>;     (* device -> SocketObject *)
$hgWorkerBroken = <||>;   (* device -> True once found unavailable *)

hgFrame[bytes_ByteArray] := Join[ByteArray[Reverse[IntegerDigits[Length[bytes], 256, 8]]], bytes];
hgWorkerExe[device_] := If[device === "GPU", $HypergraphEngineBinaryGPU, $HypergraphEngineBinary];

hgWorkerKill[device_] := (
  Quiet[If[Head[$hgWorkerSock[device]] === SocketObject, Close[$hgWorkerSock[device]]]];
  Quiet[If[MatchQ[$hgWorkerProc[device], _ProcessObject], KillProcess[$hgWorkerProc[device]]]];
  $hgWorkerProc = KeyDrop[$hgWorkerProc, device];
  $hgWorkerSock = KeyDrop[$hgWorkerSock, device];
);

hgWorkerStart[device_] := Module[{exe, portfile, proc, port, sock},
  exe = hgWorkerExe[device];
  If[!(StringQ[exe] && FileExistsQ[exe]), Return[$Failed]];
  portfile = FileNameJoin[{$TemporaryDirectory, "hgport-" <> ToString[$ProcessID] <>
    "-" <> device <> "-" <> IntegerString[RandomInteger[10^12]] <> ".txt"}];
  Quiet[If[FileExistsQ[portfile], DeleteFile[portfile]]];
  proc = StartProcess[{exe, "--serve-socket", portfile}];
  port = Null;
  Do[
    If[FileExistsQ[portfile],
      port = Quiet[ToExpression[StringTrim[Import[portfile, "String"]]]];
      If[IntegerQ[port], Break[]]];
    Pause[0.02],
    250];  (* poll up to ~5 s for the OS-assigned port *)
  Quiet[If[FileExistsQ[portfile], DeleteFile[portfile]]];
  If[!IntegerQ[port], Quiet[KillProcess[proc]]; Return[$Failed]];
  sock = TimeConstrained[SocketConnect["127.0.0.1:" <> ToString[port]], 10, $Failed];
  If[Head[sock] =!= SocketObject, Quiet[KillProcess[proc]]; Return[$Failed]];
  $hgWorkerProc[device] = proc; $hgWorkerSock[device] = sock;
  True
];

(* Read one length-prefixed response frame ([8-byte little-endian length][payload])
   and return the payload ByteArray straight for BinaryDeserialize -- no encoding
   or per-byte transform. Reassembly is O(n): the arbitrary-sized SocketReadMessage
   chunks are appended to an O(1)-append DynamicArray and Join'd once. Strict
   request/response (one full response read before the next request is sent) means
   no bytes spill past the frame, so no leftover buffer is needed. Returns $Failed
   on a dead socket, or an empty ByteArray if the engine flagged the job as errored
   (a zero-length reply). *)
hgReadFrame[sock_] := Module[{ds = CreateDataStructure["DynamicArray"], got = 0, len = -1, chunk},
  While[len < 0 || got < 8 + len,
    chunk = SocketReadMessage[sock];
    If[!ByteArrayQ[chunk], Return[$Failed]];
    ds["Append", chunk]; got += Length[chunk];
    If[len < 0 && got >= 8,
      len = FromDigits[Reverse[Normal[Take[Join @@ Normal[ds], 8]]], 256]]];
  If[len == 0, ByteArray[{}], Take[Join @@ Normal[ds], {9, 8 + len}]]
];

hgWorkerTry[device_, wxfBytes_ByteArray] := Module[{payload},
  If[TrueQ[$hgWorkerBroken[device]], Return[$Failed]];
  If[Head[$hgWorkerSock[device]] =!= SocketObject
      || Quiet[ProcessStatus[$hgWorkerProc[device]]] =!= "Running",
    hgWorkerKill[device];
    If[hgWorkerStart[device] =!= True, $hgWorkerBroken[device] = True; Return[$Failed]]
  ];
  If[Quiet[BinaryWrite[$hgWorkerSock[device], hgFrame[wxfBytes]]] === $Failed,
    hgWorkerKill[device]; $hgWorkerBroken[device] = True; Return[$Failed]];
  payload = hgReadFrame[$hgWorkerSock[device]];
  Which[
    payload === $Failed, hgWorkerKill[device]; $hgWorkerBroken[device] = True; $Failed,
    Length[payload] == 0, $Failed,   (* engine flagged this job; worker still alive *)
    True, payload]
];

(* Route a serialized job to the engine: the persistent socket worker (GPU binary
   for TargetDevice -> "GPU", else CPU binary) when available, otherwise a
   one-shot RunProcess of the same binary, otherwise the in-process LibraryLink. *)
hgCallEngine[wxfBytes_, targetDevice_:"CPU"] := Module[{dev, r},
  dev = If[targetDevice === "GPU" && hgGpuBinaryAvailableQ[], "GPU", "CPU"];
  r = hgWorkerTry[dev, wxfBytes];
  If[ByteArrayQ[r], Return[r]];
  Which[
    dev === "GPU", hgRunEngineBinary[$HypergraphEngineBinaryGPU, wxfBytes],
    StringQ[$HypergraphEngineBinary] && FileExistsQ[$HypergraphEngineBinary],
      hgRunEngineBinary[$HypergraphEngineBinary, wxfBytes],
    True, performRewriting[wxfBytes]
  ]
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
  "All" -> {"States", "Events", "CausalEdges", "BranchialEdges", "NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}
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

HGEvolve::unknownic = "Unknown initial condition type `1`.";
HGEvolve::unknownprop = "Unknown property(s): `1`. Valid properties are: States, Events, CausalEdges, BranchialEdges, StatesGraph, CausalGraph, BranchialGraph, EvolutionGraph, their Structure variants, GlobalEdges, StateBitvectors, All.";
HGEvolve::missingdata = "FFI did not return requested data: `1`. This indicates a bug in the FFI layer.";
HGEvolve::gpudev = "TargetDevice -> \"GPU\" requested but no GPU engine binary (hg_evolve_gpu) is present for `1`; evaluating on the CPU. Build the paclet with BUILD_GPU to include it.";
HGEvolve::baddev = "TargetDevice -> `1` is not valid; use \"CPU\" or \"GPU\". Using CPU.";
HGEvolve::enginemsg = "Engine binary reported: `1`";
HGEvolve::enginefail = "Engine binary exited with code `1` and produced no result.";
HGEvolve::overflow = "The GPU engine reached a capacity limit (`1`; `2` overflow event(s)) and returned a PARTIAL result. Raise the device-memory cap / reduce the workload, or evaluate on the CPU.";
HGEvolve::warn = "The engine returned warnings (`1`): `2`";

(* ============================================================================ *)
(* Graph Creation Helpers *)
(* ============================================================================ *)

(* Vertex styles for Structure variants *)
stateVertexStyle = Directive[RGBColor[0.368417, 0.506779, 0.709798], EdgeForm[RGBColor[0.2, 0.3, 0.5]]];
eventVertexStyle = Directive[LightYellow, EdgeForm[RGBColor[0.8, 0.8, 0.4]]];

(* Get edges for plotting: strip edge IDs from Edges *)
(* When CanonicalizeStates -> Full, Edges contains IR-canonical edges from the FFI *)
stateDisplayEdges[data_Association] := Rest /@ data["Edges"];

(* Helper function to format edges for display: bold edge ID, truncate if over limit *)
formatEdgesForDisplay[stateEdges_List, maxEdges_Integer:5] := Module[
  {displayed = Take[stateEdges, UpTo[maxEdges]], formatted},
  formatted = Map[Prepend[Rest[#], Style[First[#], Bold]] &, displayed];
  If[Length[stateEdges] > maxEdges, Append[formatted, "..."], formatted]
];

(* Format state tooltip with full info *)
(* Rows appear only for fields the payload carries: Structure variants ship lean payloads
   (a state carries Id and Step; an event carries Id and its endpoint states), so every
   field except Id is conditional. *)
formatStateTooltip[stateData_Association] := Module[
  {rows = {{"Id:", stateData["Id"]}}},
  If[!MissingQ[stateData["CanonicalId"]], AppendTo[rows, {"CanonicalId:", stateData["CanonicalId"]}]];
  If[!MissingQ[stateData["Step"]], AppendTo[rows, {"Step:", stateData["Step"]}]];
  If[!MissingQ[stateData["IsInitial"]], AppendTo[rows, {"IsInitial:", stateData["IsInitial"]}]];
  If[!MissingQ[stateData["Edges"]],
    AppendTo[rows, {Row[{"Edges (", Length[stateData["Edges"]], "):"}], formatEdgesForDisplay[stateData["Edges"]]}]];
  Column[{
    Row[{Style["State", Bold]}],
    Grid[rows, Alignment -> Left, Spacings -> {1, 0.5}]
  }, Spacings -> 0.5]
];

formatEventTooltip[eventData_Association] := Module[
  {rows = {{"Id:", eventData["Id"]}}},
  If[!MissingQ[eventData["CanonicalId"]], AppendTo[rows, {"CanonicalId:", eventData["CanonicalId"]}]];
  If[!MissingQ[eventData["RuleIndex"]], AppendTo[rows, {"RuleIndex:", eventData["RuleIndex"]}]];
  If[!MissingQ[eventData["InputState"]], AppendTo[rows, {"InputState:", eventData["InputState"]}]];
  If[!MissingQ[eventData["OutputState"]], AppendTo[rows, {"OutputState:", eventData["OutputState"]}]];
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

(* Edge styles by type for GraphData-based graphs.

   MEMOIZED ON FIRST USE, NOT EVALUATED AT LOAD. The branchial style comes from a
   ResourceFunction, and an immediate assignment here put a resource-system lookup on the
   package-load path: every Needs["HypergraphRewriting`"] reached for it, including in
   kernels that never draw a graph, and on a machine without that resource cached it
   reaches for the network. The lookup belongs where the style is used, which is the
   styled-graph path inside HGEvolve.

   The memoization is the whole point of SetDelayed plus the self-assignment: the resource
   is fetched at most once per kernel, on the first styled graph, and never again. If it
   cannot be resolved the branchial edges fall back to a plain directive rather than
   leaving an unevaluated ResourceFunction in the graph. *)
branchialEdgeStyle[] := branchialEdgeStyle[] = Module[{s},
  s = Quiet@Check[ResourceFunction["WolframPhysicsProjectStyleData"]["BranchialGraph"]["EdgeStyle"], $Failed];
  If[Head[s] === Directive || Head[s] === RGBColor, s, Directive[Pink, Arrowheads[0.02]]]];

graphDataEdgeStyles[] := <|
  "Directed" -> Directive[Gray, Arrowheads[0.02]],
  "Causal" -> Directive[Orange, Arrowheads[0.02]],
  "Branchial" -> branchialEdgeStyle[],
  "StateEvent" -> Directive[Gray],  (* Same gray as EventState for consistency *)
  "EventState" -> Directive[Gray, Arrowheads[0.02]]
|>;

(* Check if vertex data represents a state (has Edges but no RuleIndex) *)
(* Events carry "InputState" in both their full and lean (Structure-variant) payloads; states
   never do. Keying on content fields would misclassify lean state payloads, which carry only
   Id and Step. *)
isStateVertexData[data_Association] := !KeyExistsQ[data, "InputState"];

(* State vertex shape function for styled mode using GraphData *)
makeStyledStateVertexShapeFn[vertexData_] := Function[{pos, v, size},
  With[{data = vertexData[v]},
    If[AssociationQ[data] && KeyExistsQ[data, "Edges"],
      Inset[Framed[
        ResourceFunction["WolframModelPlot"][stateDisplayEdges[data], ImageSize -> {32, 32}],
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
createGraphFromData[graphData_Association, aspectRatio_, styled_:False, dimensionData_:<||>, dimPalette_:"TemperatureMap", dimColorBy_:"Mean", dimRange_:{0, 3}, colorByRule_:False] := Module[
  {vertices, edgeList, vertexData, vertexLabels, vertexStyles, vertexShapes, edgeStyles, edgeStyleTable, edgeLabels, hasDimData, epilogLegend, g, addLegend},

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

  (* Build edges with appropriate constructors based on Type.

     THE TAG DROPS THE ENDPOINT STATE CONTENTS, and that is what keeps the graph from
     growing as edges x state size. An event's payload carries InputStateEdges and
     OutputStateEdges -- the complete edge lists of the states it runs between -- and
     tagging every edge with them stores each state's contents once per incident edge.
     Nothing reads them there: formatEventTooltip uses Id, CanonicalId, RuleIndex,
     InputState, OutputState, ConsumedEdges and ProducedEdges, and the only reader of the
     two heavy keys is makeStyledEventVertexShapeFn, which takes them from vertexData --
     where they are held once per vertex.

     MEASURED on {{1,2},{1,3}} -> {{1,2},{1,3},{2,3}} from {{1,2},{1,3}}, the rule the
     tutorial uses, at 5 steps / 475 states: the default EvolutionCausalBranchialGraph was
     2,482,714,476 bytes. *)
  edgeTag[a_] := If[AssociationQ[a], KeyDrop[a, {"InputStateEdges", "OutputStateEdges"}], a];
  edgeList = Map[
    Switch[#["Type"],
      "StateEvent", UndirectedEdge[#["From"], #["To"], edgeTag[Lookup[#, "Data", <||>]]],
      "Branchial", UndirectedEdge[#["From"], #["To"], edgeTag[Lookup[#, "Data", <||>]]],
      _, DirectedEdge[#["From"], #["To"], edgeTag[Lookup[#, "Data", #]]]
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
  edgeStyleTable = graphDataEdgeStyles[];
  edgeStyles = Map[
    With[{e = #},
      Switch[e["Type"],
        "StateEvent", UndirectedEdge[e["From"], e["To"], _] -> edgeStyleTable["StateEvent"],
        "Branchial", UndirectedEdge[e["From"], e["To"], _] -> edgeStyleTable["Branchial"],
        _, DirectedEdge[e["From"], e["To"], _] -> edgeStyleTable[e["Type"]]
      ]
    ] &,
    graphData["Edges"]
  ];

  (* Rulial-space coloring: each transition edge takes the color of the rule that
     fired it (edge payloads carry RuleIndex), with a swatch legend over the rules
     that actually fired. The style rule matches on the edge's own payload tag, so
     parallel transitions between the same states keep their own rules' colors.
     Edges without a RuleIndex (causal, branchial, Structure variants) keep their
     type styles from above. *)
  If[TrueQ[colorByRule],
    Module[{ruleEdges, firedRules, ruleColor},
      ruleEdges = Select[graphData["Edges"],
        AssociationQ[Lookup[#, "Data", <||>]] &&
          KeyExistsQ[Lookup[#, "Data", <||>], "RuleIndex"] &];
      firedRules = Sort @ DeleteDuplicates[#["Data"]["RuleIndex"] & /@ ruleEdges];
      ruleColor = Association[# -> ColorData[97][# + 1] & /@ firedRules];
      If[Length[firedRules] > 0,
        edgeStyles = Join[
          edgeStyles,
          Map[With[{e = #},
            DirectedEdge[e["From"], e["To"], edgeTag[e["Data"]]] ->
              Directive[ruleColor[e["Data"]["RuleIndex"]], Thickness[Medium]]] &,
            ruleEdges]];
        addLegend = Composition[
          Function[graph, Legended[graph,
            SwatchLegend[ruleColor /@ firedRules,
              ("Rule " <> ToString[# + 1]) & /@ firedRules]]],
          addLegend];
      ]
    ]
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
    (* ONE shape function for every vertex, not one per vertex.
       Each of these functions closes over the whole vertexData, so binding one PER VERTEX
       stored the entire vertex set once for each vertex -- quadratic in the vertex count,
       and the dominant term in the result by a wide margin. MEASURED at 4 steps / 75
       states on the tutorial's rule: VertexShapeFunction was 9,393,744 bytes of a
       10,205,036-byte graph (92%), 75 entries of 125,240 each.
       The two shape functions already dispatch on vertexData[v] internally, so a single
       function that picks between them is behaviour-preserving and captures vertexData a
       fixed number of times. *)
    vertexShapes = With[{
        stateFn = If[hasDimData,
          makeStyledStateVertexWithDimensionFn[vertexData, dimensionData, dimPalette, dimColorBy, dimRange],
          makeStyledStateVertexShapeFn[vertexData]],
        eventFn = makeStyledEventVertexShapeFn[vertexData]},
      (* Slots, not named parameters: stateFn and eventFn are themselves
         Function[{pos, v, size}, ...], and a named dispatcher binds the same three names,
         so the inner application does not reduce. #1 is the position, #2 the vertex,
         #3 the size; ## forwards all three unchanged. *)
      Function[If[AssociationQ[vertexData[#2]] && isStateVertexData[vertexData[#2]],
        stateFn[##], eventFn[##]]]];
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
            ResourceFunction["WolframModelPlot"][stateDisplayEdges[data], ImageSize -> {32, 32}],
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

(* Wrapper: single rule -> list of rules *)
HGEvolve[rule_Rule, initial_, steps_Integer, rest___] :=
  HGEvolve[{rule}, initial, steps, rest]

(* Wrapper: Graph input -> extract edge list *)
HGEvolve[rules_, g_Graph, steps_Integer, rest___] :=
  HGEvolve[rules, List @@@ EdgeList[g], steps, rest]

(* Wrapper: string initial condition -> association *)
HGEvolve[rules_, "Grid", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Grid"|>, steps, rest]

HGEvolve[rules_, "Sprinkling", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Sprinkling"|>, steps, rest]

HGEvolve[rules_, "Minkowski", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Sprinkling"|>, steps, rest]

HGEvolve[rules_, "BrillLindquist", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "BrillLindquist"|>, steps, rest]

HGEvolve[rules_, "Cylinder", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Cylinder"|>, steps, rest]

HGEvolve[rules_, "Torus", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Torus"|>, steps, rest]

HGEvolve[rules_, "Sphere", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Sphere"|>, steps, rest]

HGEvolve[rules_, "Klein", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Klein"|>, steps, rest]

HGEvolve[rules_, "Mobius", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Mobius"|>, steps, rest]

HGEvolve[rules_, "Poisson", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Poisson"|>, steps, rest]

HGEvolve[rules_, "Uniform", steps_Integer, rest___] :=
  HGEvolve[rules, <|"Type" -> "Uniform"|>, steps, rest]

(* Wrapper: IC generator result -> extract edges and pass to main *)
HGEvolve[rules_, icResult_Association, steps_Integer, rest___] /;
  KeyExistsQ[icResult, "Edges"] && !KeyExistsQ[icResult, "Type"] :=
  HGEvolve[rules, icResult["Edges"], steps, rest]

(* Wrapper: association initial condition -> generate edges in WL or pass to C++ *)
HGEvolve[rules_List, initialSpec_Association, steps_Integer,
         property : (_String | {__String}) : "EvolutionCausalBranchialGraph",
         opts:OptionsPattern[]] := Module[
  {icType, icResult, edges, newOpts,
   gridWidth, gridHeight, gridHoles, resolution,
   sprinklingDensity, sprinklingTime, sprinklingSpatial, spatialDim,
   lightcone, alexandrov, transitivity, maxEdgesPerVertex,
   mass1, mass2, separation, boxX, boxY, edgeThreshold,
   majorRadius, minorRadius, poissonMinDistance, seed},

  icType = Lookup[initialSpec, "Type", "Grid"];

  (* Extract common options *)
  seed = Lookup[initialSpec, "Seed", OptionValue[HGEvolve, {opts}, "RandomSeed"]];
  edgeThreshold = Lookup[initialSpec, "EdgeThreshold", OptionValue[HGEvolve, {opts}, "EdgeThreshold"]];

  (* Generate initial condition based on type *)
  Switch[icType,

    (* ===== FLAT TOPOLOGIES ===== *)

    "Grid",
    gridWidth = Lookup[initialSpec, "Width", OptionValue[HGEvolve, {opts}, "GridWidth"]];
    gridHeight = Lookup[initialSpec, "Height", OptionValue[HGEvolve, {opts}, "GridHeight"]];
    gridHoles = Lookup[initialSpec, "Holes", OptionValue[HGEvolve, {opts}, "GridHoles"]];
    If[gridHoles === {} || gridHoles === None,
      icResult = HGGrid[gridWidth, gridHeight, "RandomSeed" -> seed],
      icResult = HGGridWithHoles[gridWidth, gridHeight, gridHoles, "RandomSeed" -> seed]
    ];
    edges = icResult["Edges"],

    (* ===== CURVED TOPOLOGIES ===== *)

    "Cylinder",
    resolution = Lookup[initialSpec, "Resolution", OptionValue[HGEvolve, {opts}, "GridWidth"]];
    gridHeight = Lookup[initialSpec, "Height", OptionValue[HGEvolve, {opts}, "GridHeight"]];
    majorRadius = Lookup[initialSpec, "Radius", OptionValue[HGEvolve, {opts}, "MajorRadius"]];
    icResult = HGCylinder[resolution, gridHeight, "Radius" -> majorRadius];
    edges = icResult["Edges"],

    "Torus",
    resolution = Lookup[initialSpec, "Resolution", OptionValue[HGEvolve, {opts}, "GridWidth"]];
    majorRadius = Lookup[initialSpec, "MajorRadius", OptionValue[HGEvolve, {opts}, "MajorRadius"]];
    minorRadius = Lookup[initialSpec, "MinorRadius", OptionValue[HGEvolve, {opts}, "MinorRadius"]];
    icResult = HGTorus[resolution, "MajorRadius" -> majorRadius, "MinorRadius" -> minorRadius];
    edges = icResult["Edges"],

    "Sphere",
    resolution = Lookup[initialSpec, "Resolution", OptionValue[HGEvolve, {opts}, "GridWidth"]];
    majorRadius = Lookup[initialSpec, "Radius", OptionValue[HGEvolve, {opts}, "MajorRadius"]];
    icResult = HGSphere[resolution, "Radius" -> majorRadius];
    edges = icResult["Edges"],

    "Klein" | "KleinBottle",
    resolution = Lookup[initialSpec, "Resolution", OptionValue[HGEvolve, {opts}, "GridWidth"]];
    gridHeight = Lookup[initialSpec, "Height", OptionValue[HGEvolve, {opts}, "GridHeight"]];
    majorRadius = Lookup[initialSpec, "Radius", OptionValue[HGEvolve, {opts}, "MajorRadius"]];
    icResult = HGKleinBottle[resolution, gridHeight, "Radius" -> majorRadius];
    edges = icResult["Edges"],

    "Mobius" | "MobiusStrip",
    resolution = Lookup[initialSpec, "Resolution", OptionValue[HGEvolve, {opts}, "GridWidth"]];
    gridWidth = Lookup[initialSpec, "Width", 5];
    majorRadius = Lookup[initialSpec, "Radius", OptionValue[HGEvolve, {opts}, "MajorRadius"]];
    icResult = HGMobiusStrip[resolution, gridWidth, "Radius" -> majorRadius];
    edges = icResult["Edges"],

    (* ===== SPACETIME GEOMETRIES ===== *)

    "Sprinkling" | "Minkowski",
    sprinklingDensity = Lookup[initialSpec, "Density", OptionValue[HGEvolve, {opts}, "SprinklingDensity"]];
    sprinklingTime = Lookup[initialSpec, "TimeExtent", OptionValue[HGEvolve, {opts}, "SprinklingTimeExtent"]];
    sprinklingSpatial = Lookup[initialSpec, "SpatialExtent", OptionValue[HGEvolve, {opts}, "SprinklingSpatialExtent"]];
    spatialDim = Lookup[initialSpec, "SpatialDim", OptionValue[HGEvolve, {opts}, "SprinklingSpatialDim"]];
    lightcone = Lookup[initialSpec, "LightconeAngle", OptionValue[HGEvolve, {opts}, "SprinklingLightconeAngle"]];
    alexandrov = Lookup[initialSpec, "AlexandrovCutoff", OptionValue[HGEvolve, {opts}, "SprinklingAlexandrovCutoff"]];
    transitivity = Lookup[initialSpec, "TransitivityReduction", OptionValue[HGEvolve, {opts}, "SprinklingTransitivityReduction"]];
    maxEdgesPerVertex = Lookup[initialSpec, "MaxEdgesPerVertex", OptionValue[HGEvolve, {opts}, "SprinklingMaxEdgesPerVertex"]];
    icResult = HGMinkowskiSprinkling[sprinklingDensity,
      "SpatialDim" -> spatialDim,
      "TimeExtent" -> sprinklingTime,
      "SpatialExtent" -> sprinklingSpatial,
      "LightconeAngle" -> lightcone,
      "AlexandrovCutoff" -> alexandrov,
      "TransitivityReduction" -> transitivity,
      "MaxEdgesPerVertex" -> maxEdgesPerVertex,
      "RandomSeed" -> seed
    ];
    edges = icResult["Edges"],

    "BrillLindquist",
    sprinklingDensity = Lookup[initialSpec, "Density", OptionValue[HGEvolve, {opts}, "SprinklingDensity"]];
    mass1 = Lookup[initialSpec, "Mass1", OptionValue[HGEvolve, {opts}, "BrillLindquistMass1"]];
    mass2 = Lookup[initialSpec, "Mass2", OptionValue[HGEvolve, {opts}, "BrillLindquistMass2"]];
    separation = Lookup[initialSpec, "Separation", OptionValue[HGEvolve, {opts}, "BrillLindquistSeparation"]];
    boxX = Lookup[initialSpec, "BoxX", OptionValue[HGEvolve, {opts}, "BrillLindquistBoxX"]];
    boxY = Lookup[initialSpec, "BoxY", OptionValue[HGEvolve, {opts}, "BrillLindquistBoxY"]];
    edgeThreshold = Lookup[initialSpec, "EdgeThreshold", OptionValue[HGEvolve, {opts}, "EdgeThreshold"]];
    If[edgeThreshold === Automatic, edgeThreshold = 2.0];
    icResult = HGBrillLindquist[sprinklingDensity, {mass1, mass2}, separation,
      "BoxX" -> boxX,
      "BoxY" -> boxY,
      "EdgeThreshold" -> edgeThreshold,
      "RandomSeed" -> seed
    ];
    edges = icResult["Edges"],

    (* ===== SAMPLING METHODS ===== *)

    "Poisson" | "PoissonDisk",
    sprinklingDensity = Lookup[initialSpec, "Density", OptionValue[HGEvolve, {opts}, "SprinklingDensity"]];
    poissonMinDistance = Lookup[initialSpec, "MinDistance", OptionValue[HGEvolve, {opts}, "PoissonMinDistance"]];
    boxX = Lookup[initialSpec, "BoxX", {0, 10}];
    boxY = Lookup[initialSpec, "BoxY", {0, 10}];
    icResult = HGPoissonDisk[sprinklingDensity, poissonMinDistance,
      "BoxX" -> boxX,
      "BoxY" -> boxY,
      "EdgeThreshold" -> edgeThreshold,
      "RandomSeed" -> seed
    ];
    edges = icResult["Edges"],

    "Uniform" | "UniformRandom",
    sprinklingDensity = Lookup[initialSpec, "Density", OptionValue[HGEvolve, {opts}, "SprinklingDensity"]];
    boxX = Lookup[initialSpec, "BoxX", {0, 10}];
    boxY = Lookup[initialSpec, "BoxY", {0, 10}];
    icResult = HGUniformRandom[sprinklingDensity,
      "BoxX" -> boxX,
      "BoxY" -> boxY,
      "EdgeThreshold" -> edgeThreshold,
      "RandomSeed" -> seed
    ];
    edges = icResult["Edges"],

    (* Every initial-condition type is generated here in WL; an unknown type is
       a caller error, not something the engine can synthesize. *)
    _,
    Message[HGEvolve::unknownic, icType];
    Return[$Failed]
  ];

  (* Call main HGEvolve with generated edges *)
  HGEvolve[rules, edges, steps, property, opts]
]

(* The job's Options envelope: every option the engine parses, and how each is spelled on the
   wire. HGEvolve and HGSessionOpen both build it HERE. Two copies would be two chances for a
   session to send an option HGEvolve does not, or to spell one differently -- and the FFI
   answers an option it cannot parse by SKIPPING it and continuing, so a disagreement would
   surface as behaviour rather than as an error.

   `ov` is a function from option name to value: HGEvolve reads through its own OptionsPattern
   and HGSessionOpen through its, and the envelope does not care which. *)
hgJobOptions[ov_, requiredData_, graphProperties_] := Module[{branchialStepValue},
  (* Convert BranchialStep: All -> 0, positive for 1-based step, negative for from-end *)
  (* EvolutionCausalBranchialGraph defaults to All, BranchialGraph defaults to -1 (final step) *)
  (* Use first graph property for branchial step default, or empty string if none *)
  branchialStepValue = Replace[ov["BranchialStep"], {
    Automatic :> If[Length[graphProperties] > 0 && StringMatchQ[First[graphProperties], "*Evolution*Branchial*"], 0, -1],
    All -> 0
  }];

  <|
    "CanonicalizeStates" -> ov["CanonicalizeStates"],
    "CanonicalizeEvents" -> ov["CanonicalizeEvents"],
    "CausalTransitiveReduction" -> ov["CausalTransitiveReduction"],
    "MaxSuccessorStatesPerParent" -> ov["MaxSuccessorStatesPerParent"],
    "MaxStatesPerStep" -> ov["MaxStatesPerStep"],
    "ExplorationProbability" -> ov["ExplorationProbability"],
    "TransitionRate" -> ov["TransitionRate"],
    "RuleWeights" -> N[ov["RuleWeights"]],
    (* The seed the sampling draws use. Automatic means "a fresh one each run", which the
       engine spells as 0; anything else fixes the sample. Without this the option reached the
       initial-condition generators only, and a sampled evolution was irreproducible however it
       was set. *)
    "RandomSeed" -> Replace[ov["RandomSeed"], Automatic -> 0],
    "ExploreFromCanonicalStatesOnly" -> ov["ExploreFromCanonicalStatesOnly"],
    "QuotientInitialStates" -> ov["QuotientInitialStates"],
    "ShowProgress" -> ov["ShowProgress"],
    "ShowGenesisEvents" -> ov["ShowGenesisEvents"],
    "BranchialStep" -> branchialStepValue,  (* 0=All, positive=1-based step, negative=from end *)
    "EdgeDeduplication" -> ov["EdgeDeduplication"],
    "IncludeCanonicalHashes" -> ov["IncludeCanonicalHashes"],
    "RequestedData" -> requiredData,
    "GraphProperties" -> graphProperties,  (* List of graph properties for FFI to generate *)
    (* Uniform random evolution mode *)
    "UniformRandom" -> ov["UniformRandom"],
    "MatchesPerStep" -> ov["MatchesPerStep"]
  |>
]

(* Serialize a job, run it, deserialize the reply, and surface the engine's warning trail.
   Returns the reply association, or $Failed.

   `sessionQ` selects the TRANSPORT, and it is not a preference. hgCallEngine falls back to a
   one-shot RunProcess when the persistent worker is unavailable, which is right for an `Evolve`
   -- the job is self-contained and the process may exit as soon as it answers. It is WRONG for
   every session verb: an `Open` served that way mints a handle inside a process that exits with
   the reply, so the caller holds a handle naming nothing and the next `Step` reports an unknown
   session. So a session verb takes the worker or it takes nothing. *)
hgSendJob[inputData_Association, device_, sessionQ_] := Module[{wxfBytes, resultBytes, wxfData},
  wxfBytes = BinarySerialize[inputData];
  resultBytes = If[TrueQ[sessionQ],
    Module[{r = hgWorkerTry[If[device === "GPU" && hgGpuBinaryAvailableQ[], "GPU", "CPU"],
                            wxfBytes]},
      If[!ByteArrayQ[r], Message[HGSessionOpen::noworker, device]; Return[$Failed]];
      r],
    hgCallEngine[wxfBytes, device]];

  If[!ByteArrayQ[resultBytes] || Length[resultBytes] == 0, Return[$Failed]];
  wxfData = BinaryDeserialize[resultBytes];
  If[!AssociationQ[wxfData], Return[$Failed]];

  (* Surface the engine's warning trail. Both backends serve it under "Warnings"
     (Kind/Count/Context): GPU capacity overflows flag a PARTIAL result; engine
     option conflicts and analysis refusals report why an output is absent or
     reduced. Overflow kinds keep their dedicated message. *)
  If[KeyExistsQ[wxfData, "Warnings"] && Length[wxfData["Warnings"]] > 0,
    Module[{warns = wxfData["Warnings"], advisoryKinds, advisories, overflows},
      (* Advisory kinds are minted by the CPU FFI (engine option conflicts, analysis
         refusals); every other kind is a GPU capacity flag on a PARTIAL result. *)
      advisoryKinds = {"Engine", "OptionSkipped"};
      advisories = Select[warns, MemberQ[advisoryKinds, Lookup[#, "Kind", ""]] &];
      overflows = Complement[warns, advisories];
      If[Length[overflows] > 0,
        Message[HGEvolve::overflow,
          DeleteDuplicates[Lookup[overflows, "Kind", "?"]],
          Total[Lookup[overflows, "Count", 0]]]];
      If[Length[advisories] > 0,
        Message[HGEvolve::warn,
          DeleteDuplicates[Lookup[advisories, "Kind", "?"]],
          StringRiffle[DeleteDuplicates[Lookup[advisories, "Context", ""]], " | "]]];
    ]];
  wxfData
];

(* ONE job, run and interpreted -- HGEvolve and every session verb share this.
   The two phases of a call are BUILD an envelope and RUN it; only the first differs between
   `Evolve` and a session's `Step`/`Query`, so only the first is written twice. A second copy of
   the interpretation is how the verbs would come to disagree about what a reply means, which is
   the same defect the engine spent this project removing from its canonicalizer and matcher.

   `view` carries the interpretation settings the caller already resolved: RequestedData,
   AspectRatio, IncludeStateContents, IncludeEventContents, CanonicalizeStates,
   CanonicalizeEvents, ColorByRule, DebugFFI. They are values here rather than OptionValue[]
   reads because a session's Step is not an HGEvolve call and has no OptionsPattern to read. *)
hgRunJob[inputData_Association, device_, props_List, propertyWasList_, view_Association] :=
  Module[
  {wxfBytes, resultBytes, wxfData, requiredData, states, events, causalEdges, branchialEdges,
   branchialStateEdges, branchialStateVertices, aspectRatio, includeStateContents,
   includeEventContents, canonicalizeStates, canonicalizeEvents, colorByRule,
   dimensionData, geodesicData, topologicalData, curvatureData, alignmentData, entropyData,
   hilbertData, branchialData, multispaceData, dimPalette, dimColorBy, dimRange},

  requiredData            = view["RequestedData"];
  aspectRatio             = view["AspectRatio"];
  includeStateContents    = view["IncludeStateContents"];
  includeEventContents    = view["IncludeEventContents"];
  canonicalizeStates      = view["CanonicalizeStates"];
  canonicalizeEvents      = view["CanonicalizeEvents"];

  wxfData = hgSendJob[inputData, device, TrueQ[view["SessionQ"]]];
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
  If[TrueQ[view["DebugFFI"]],
    Print["  FFI response keys: ", Keys[wxfData]];
    Print["  FFI response size: ", ByteCount[wxfData], " bytes"];
    Print["  States count: ", Length[states]];
    Print["  Events count: ", Length[events]];
    Print["  CausalEdges count: ", Length[causalEdges]];
    Print["  BranchialEdges count: ", Length[branchialEdges]];
  ];

  (* The physics analyses live in the hypergraph_viz repo; these locals feed
     getProperty's plain-rendering path as empty. *)
  {dimensionData, geodesicData, topologicalData, curvatureData, alignmentData,
   entropyData, hilbertData, branchialData, multispaceData} = Table[<||>, 9];

  colorByRule = TrueQ[view["ColorByRule"]];
  {dimPalette, dimColorBy, dimRange} = {"TemperatureMap", "Mean", {0, 3}};

  (* Return requested properties *)
  (* String input returns data directly; list input always returns association *)
  If[Length[props] == 1 && !propertyWasList,
    (* Single string property: return directly *)
    getProperty[First[props], states, events, causalEdges, branchialEdges, branchialStateEdges, branchialStateVertices, wxfData, aspectRatio, includeStateContents, includeEventContents, canonicalizeStates, canonicalizeEvents, dimensionData, dimPalette, dimColorBy, dimRange, geodesicData, topologicalData, curvatureData, entropyData, hilbertData, branchialData, multispaceData, colorByRule],
    (* List input: return association keyed by property names *)
    Association[# -> getProperty[#, states, events, causalEdges, branchialEdges, branchialStateEdges, branchialStateVertices, wxfData, aspectRatio, includeStateContents, includeEventContents, canonicalizeStates, canonicalizeEvents, dimensionData, dimPalette, dimColorBy, dimRange, geodesicData, topologicalData, curvatureData, entropyData, hilbertData, branchialData, multispaceData, colorByRule] & /@ props]
  ]
]

(* Main implementation *)
HGEvolve[rules_List, initialEdges_List, steps_Integer,
         property : (_String | {__String}) : "EvolutionCausalBranchialGraph",
         OptionsPattern[]] := Module[
  {inputData, wxfBytes, resultBytes, wxfData, requiredData, options,
   states, events, causalEdges, branchialEdges, aspectRatio, props,
   includeStateContents, includeEventContents, canonicalizeStates, canonicalizeEvents, graphProperties, colorByRule,
   normalizedRules, rulesAssoc, initialStatesData},

  If[Head[performRewriting] =!= LibraryFunction,
    Return[$Failed]
  ];

  (* Track if original input was a list (for return format) *)
  propertyWasList = ListQ[property];

  (* Normalize property to list and deduplicate *)
  props = DeleteDuplicates[Flatten[{property}]];

  (* Get content options *)
  includeStateContents = OptionValue["IncludeStateContents"];
  includeEventContents = OptionValue["IncludeEventContents"];

  (* Get canonicalization options - used for per-graph-type ID selection *)
  canonicalizeStates = OptionValue["CanonicalizeStates"];
  canonicalizeEvents = OptionValue["CanonicalizeEvents"];

  (* Compute required data components - fail explicitly on unknown properties *)
  (* Pass canonicalizeStates to conditionally add States when state canonicalization is needed *)
  requiredData = computeRequiredData[props, includeStateContents, includeEventContents, canonicalizeStates];
  If[requiredData === $Failed, Return[$Failed]];

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
  (* TargetDevice: "CPU" runs the CPU binary; "GPU" runs the GPU binary when it is
     present (built with the CUDA backend), else falls back to CPU with a message. *)
  Switch[OptionValue["TargetDevice"],
    "CPU", Null,
    "GPU", If[!hgGpuBinaryAvailableQ[], Message[HGEvolve::gpudev, $SystemID]],
    _, Message[HGEvolve::baddev, OptionValue["TargetDevice"]]
  ];
  aspectRatio = OptionValue["AspectRatio"];
  options = hgJobOptions[OptionValue[#] &, requiredData, graphProperties];

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

  (* Run and interpret. Everything from here is shared with the session verbs (hgRunJob). *)
  hgRunJob[inputData, OptionValue["TargetDevice"], props, propertyWasList, <|
    "RequestedData"        -> requiredData,
    "AspectRatio"          -> aspectRatio,
    "IncludeStateContents" -> includeStateContents,
    "IncludeEventContents" -> includeEventContents,
    "CanonicalizeStates"   -> canonicalizeStates,
    "CanonicalizeEvents"   -> canonicalizeEvents,
    "ColorByRule"          -> OptionValue["ColorByRule"],
    "DebugFFI"             -> OptionValue["DebugFFI"]|>]
]

(* ============================================================================ *)
(* Sessions: an evolution a caller can CONTINUE                                 *)
(* ============================================================================ *)

(* WHAT A SESSION IS. HGEvolve answers one question and discards the exploration that answered
   it, so asking for three steps and then five re-runs the first three. A session keeps the
   engine, its graph and the frontier the budget stopped at, so `Step` carries the SAME
   exploration further and every state, event and relation already built keeps its identity.

   The four verbs are served by the engine binary and have been since #121; what did not exist
   until now was any way for a user to reach them, which is what this section is.

     s = HGSessionOpen[rules, init, property]      an exploration, opened at depth 0
     HGSessionStep[s, n]                           carry it n steps further; returns the property
     HGSessionQuery[s]                             re-read it without exploring
     HGSessionClose[s]                             release the engine

   WHAT IS FIXED AT OPEN, and why the verbs refuse to change it:
     - THE RULES. A session's rule set was fixed when it opened; applying different ones would
       answer about a system the session is not exploring. Step and Query carry no rules and the
       engine rejects a job that does.
     - THE IDENTITY CONVENTION (CanonicalizeStates, CanonicalizeEvents, ShowGenesisEvents).
       These decide what a state and an event ARE. The engine reads them back from its own graph
       rather than from a Step's envelope, so a Query cannot report exact canonical forms as
       tree-mode ones in fields that all still parse.
     - WHAT THE RUN RECORDS. An artifact a session was not opened for cannot be produced later:
       the evolution that would have built it has already run. Open with the property you intend
       to ask for. Asking for another returns an empty relation and the engine says so on the
       warning trail rather than serving the emptiness silently.

   ONE SESSION AT A TIME per worker (D7). A second HGSessionOpen while one is live is an error,
   not an eviction: evicting would discard a caller's exploration without being asked.

   THE HANDLE NAMES A PROCESS, which is why the session verbs do not use hgCallEngine's one-shot
   fallback. See hgSendJob. *)


HGSessionOpen[rules_List, initialEdges_List,
              property : (_String | {__String}) : "EvolutionCausalBranchialGraph",
              opts : OptionsPattern[]] := Module[
  {props, propertyWasListLocal, requiredData, graphProperties, view, options, device,
   normalizedRules, rulesAssoc, initialStatesData, inputData, reply, handle},

  propertyWasListLocal = ListQ[property];
  props = DeleteDuplicates[Flatten[{property}]];
  requiredData = computeRequiredData[props, OptionValue["IncludeStateContents"],
                                     OptionValue["IncludeEventContents"],
                                     OptionValue["CanonicalizeStates"]];
  If[requiredData === $Failed, Return[$Failed]];
  graphProperties = Select[props, StringMatchQ[#, "*Graph*"] &];
  device = OptionValue["TargetDevice"];

  (* The session's own record of how to interpret every later reply. A Step is not an HGEvolve
     call and has no OptionsPattern to read, so these are resolved once, here, and carried by the
     object -- which is also what makes a Step's answer use the convention the session was opened
     under rather than whatever the Step's own call happened to say. *)
  view = <|
    "RequestedData"        -> requiredData,
    "AspectRatio"          -> OptionValue["AspectRatio"],
    "IncludeStateContents" -> OptionValue["IncludeStateContents"],
    "IncludeEventContents" -> OptionValue["IncludeEventContents"],
    "CanonicalizeStates"   -> OptionValue["CanonicalizeStates"],
    "CanonicalizeEvents"   -> OptionValue["CanonicalizeEvents"],
    "ColorByRule"          -> OptionValue["ColorByRule"],
    "DebugFFI"             -> OptionValue["DebugFFI"],
    "SessionQ"             -> True|>;

  options = hgJobOptions[OptionValue[HGSessionOpen, {opts}, #] &, requiredData,
                         graphProperties];

  normalizedRules = normalizeRules[rules];
  rulesAssoc = Association[
    Table["Rule" <> ToString[i] -> normalizedRules[[i]], {i, Length[normalizedRules]}]];
  initialStatesData = If[Depth[initialEdges] == 3, {initialEdges}, initialEdges];

  inputData = <|"InitialStates" -> initialStatesData, "Rules" -> rulesAssoc,
                "Steps" -> 0, "Options" -> options, "Op" -> "Open"|>;

  reply = hgSendJob[inputData, device, True];
  If[!AssociationQ[reply], Return[$Failed]];
  handle = Lookup[reply, "Session", Missing[]];
  If[!IntegerQ[handle] || handle === 0, Message[HGSessionOpen::nohandle]; Return[$Failed]];

  HGSessionObject[<|"Handle" -> handle, "Device" -> device, "View" -> view,
                    "Properties" -> props, "PropertyWasList" -> propertyWasListLocal,
                    "GraphProperties" -> graphProperties, "Options" -> options|>]
];

(* Step and Query differ in ONE field. Writing them as two bodies would be two chances to
   disagree about what a held verb's envelope contains. *)
hgSessionVerb[HGSessionObject[d_Association], op_String, steps_Integer, property_] := Module[
  {props, wasList, requiredData, graphProperties, view, options, inputData},

  props = If[property === Automatic, d["Properties"], DeleteDuplicates[Flatten[{property}]]];
  wasList = If[property === Automatic, d["PropertyWasList"], ListQ[property]];
  requiredData = computeRequiredData[props, d["View"]["IncludeStateContents"],
                                     d["View"]["IncludeEventContents"],
                                     d["View"]["CanonicalizeStates"]];
  If[requiredData === $Failed, Return[$Failed]];
  graphProperties = Select[props, StringMatchQ[#, "*Graph*"] &];

  (* The session's options, with only what this call may vary. A held verb carries NO rules: the
     engine rejects a job that does, because applying them would answer about a different system. *)
  options = d["Options"];
  options["Op"] = op;
  options["RequestedData"] = requiredData;
  options["GraphProperties"] = graphProperties;

  view = d["View"];
  view["RequestedData"] = requiredData;

  inputData = <|"Steps" -> steps, "Options" -> options, "Op" -> op,
                "Session" -> d["Handle"]|>;

  hgRunJob[inputData, d["Device"], props, wasList, view]
];

HGSessionStep[s_HGSessionObject, steps_Integer, property_ : Automatic] :=
  If[steps < 0,
    Message[HGSessionStep::negsteps, steps]; $Failed,
    hgSessionVerb[s, "Step", steps, property]];
HGSessionStep[other_, ___] := (Message[HGSessionStep::badsession, other]; $Failed);

HGSessionQuery[s_HGSessionObject, property_ : Automatic] :=
  hgSessionVerb[s, "Query", 0, property];
HGSessionQuery[other_, ___] := (Message[HGSessionStep::badsession, other]; $Failed);

HGSessionClose[HGSessionObject[d_Association]] := Module[{reply},
  reply = hgSendJob[<|"Op" -> "Close", "Session" -> d["Handle"],
                      "Options" -> <|"Op" -> "Close"|>|>, d["Device"], True];
  If[AssociationQ[reply], Null, $Failed]
];
HGSessionClose[other_] := (Message[HGSessionStep::badsession, other]; $Failed);

(* A session shows what it is and what it holds, never its internals: a handle a user can read
   invites constructing one, and a constructed handle addresses whatever session that number
   happens to name. *)
HGSessionObject /: MakeBoxes[obj : HGSessionObject[d_Association], fmt_] :=
  BoxForm`ArrangeSummaryBox[HGSessionObject, obj, None,
    {BoxForm`SummaryItem[{"device: ", d["Device"]}],
     BoxForm`SummaryItem[{"records: ", Row[d["Properties"], ", "]}]},
    {BoxForm`SummaryItem[{"canonicalization: ", d["View"]["CanonicalizeStates"]}]},
    fmt];

(* Property getter *)
(* Graph properties are handled via FFI GraphData - keyed by property name *)
getProperty[prop_, states_, events_, causalEdges_, branchialEdges_, branchialStateEdges_, branchialStateVertices_, wxfData_, aspectRatio_, includeStateContents_, includeEventContents_, canonicalizeStates_, canonicalizeEvents_, dimensionData_:<||>, dimPalette_:"TemperatureMap", dimColorBy_:"Mean", dimRange_:{0, 3}, geodesicData_:<||>, topologicalData_:<||>, curvatureData_:<||>, entropyData_:<||>, hilbertData_:<||>, branchialData_:<||>, multispaceData_:<||>, colorByRule_:False] := Module[
  {isGraphProperty, isStyled, graphData},

  (* Graph properties: use FFI-provided GraphData keyed by property name *)
  isGraphProperty = StringMatchQ[prop, "*Graph*"];
  If[isGraphProperty,
    If[KeyExistsQ[wxfData, "GraphData"] && KeyExistsQ[wxfData["GraphData"], prop],
      graphData = wxfData["GraphData"][prop];
      isStyled = !StringMatchQ[prop, "*Structure"];
      Return[createGraphFromData[graphData, aspectRatio, isStyled, dimensionData, dimPalette, dimColorBy, dimRange, colorByRule]],
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
      result
    ],
    _, $Failed
  ]
]

(* ============================================================================ *)
HGToGraph[icResult_Association] := Module[{edges, coords, graphEdges, vertices},
  edges = icResult["Edges"];
  coords = Lookup[icResult, "VertexCoordinates", None];
  HGToGraph[edges, coords]
]

HGToGraph[edges_List] := HGToGraph[edges, None]

HGToGraph[edges_List, None] := Module[{graphEdges, vertices},
  graphEdges = DirectedEdge @@@ edges;
  vertices = Union[Flatten[edges]];
  Graph[vertices, graphEdges]
]

HGToGraph[edges_List, coords_Association] := Module[{graphEdges, vertices, coordList},
  graphEdges = DirectedEdge @@@ edges;
  vertices = Union[Flatten[edges]];
  coordList = Table[v -> coords[v], {v, vertices}];
  Graph[vertices, graphEdges, VertexCoordinates -> coordList]
]

(* Shared helper: finalize IC result, optionally returning Graph *)
icFinalizeResult[result_Association, returnGraph_] :=
  If[returnGraph, HGToGraph[result], result]

(* Common options for all IC generators *)
$ICCommonOptions = {
  "Graph" -> False,
  "RandomSeed" -> Automatic
};

(* ============================================================================ *)
(* Initial Condition Generators - Pure Wolfram Language Implementations *)
(* ============================================================================ *)
(* These generate edge lists and vertex coordinates that can be passed to HGEvolve
   or used directly for analysis/visualization. *)

(* ---------------------------------------------------------------------------- *)
(* HGGrid - Regular rectangular grid *)
(* ---------------------------------------------------------------------------- *)

Options[HGGrid] = Join[$ICCommonOptions, {
  "Diagonals" -> False,
  "RandomizeDirections" -> True
}];

HGGrid[width_Integer, height_Integer, opts:OptionsPattern[]] := Module[
  {vertices, coords, edges, vertexIndex, addDiagonals, randomize,
   i, j, v1, v2, idx, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  addDiagonals = OptionValue["Diagonals"];
  randomize = OptionValue["RandomizeDirections"];

  (* Create vertex positions *)
  vertexIndex = Association[];
  coords = Association[];
  idx = 1;
  Do[
    vertexIndex[{i, j}] = idx;
    coords[idx] = {i - 1, j - 1};
    idx++,
    {i, width}, {j, height}
  ];

  vertices = Range[width * height];

  (* Create edges *)
  edges = {};
  Do[
    v1 = vertexIndex[{i, j}];
    If[i < width,
      v2 = vertexIndex[{i + 1, j}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ];
    If[j < height,
      v2 = vertexIndex[{i, j + 1}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ];
    If[addDiagonals,
      If[i < width && j < height,
        v2 = vertexIndex[{i + 1, j + 1}];
        AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
      ];
      If[i < width && j > 1,
        v2 = vertexIndex[{i + 1, j - 1}];
        AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
      ]
    ],
    {i, width}, {j, height}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords,
    "Topology" -> "Grid",
    "Width" -> width,
    "Height" -> height,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGGridWithHoles - Grid with circular exclusion zones *)
(* ---------------------------------------------------------------------------- *)

Options[HGGridWithHoles] = Join[$ICCommonOptions, {
  "Diagonals" -> False,
  "RandomizeDirections" -> True
}];

HGGridWithHoles[width_Integer, height_Integer, holes_List, opts:OptionsPattern[]] := Module[
  {vertices, coords, edges, vertexIndex, addDiagonals, randomize,
   i, j, v1, v2, idx, pos, insideHole, mid, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  addDiagonals = OptionValue["Diagonals"];
  randomize = OptionValue["RandomizeDirections"];

  insideHole[{x_, y_}] := AnyTrue[holes,
    With[{cx = #[[1]], cy = #[[2]], r = #[[3]]},
      (x - cx)^2 + (y - cy)^2 < r^2
    ] &
  ];

  vertexIndex = Association[];
  coords = Association[];
  idx = 1;
  Do[
    pos = {i - 1, j - 1};
    If[!insideHole[pos],
      vertexIndex[{i, j}] = idx;
      coords[idx] = pos;
      idx++
    ],
    {i, width}, {j, height}
  ];

  vertices = Range[idx - 1];

  edges = {};
  Do[
    If[KeyExistsQ[vertexIndex, {i, j}],
      v1 = vertexIndex[{i, j}];
      If[i < width && KeyExistsQ[vertexIndex, {i + 1, j}],
        v2 = vertexIndex[{i + 1, j}];
        mid = (coords[v1] + coords[v2]) / 2;
        If[!insideHole[mid],
          AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
        ]
      ];
      If[j < height && KeyExistsQ[vertexIndex, {i, j + 1}],
        v2 = vertexIndex[{i, j + 1}];
        mid = (coords[v1] + coords[v2]) / 2;
        If[!insideHole[mid],
          AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
        ]
      ];
      If[addDiagonals,
        If[i < width && j < height && KeyExistsQ[vertexIndex, {i + 1, j + 1}],
          v2 = vertexIndex[{i + 1, j + 1}];
          mid = (coords[v1] + coords[v2]) / 2;
          If[!insideHole[mid],
            AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
          ]
        ];
        If[i < width && j > 1 && KeyExistsQ[vertexIndex, {i + 1, j - 1}],
          v2 = vertexIndex[{i + 1, j - 1}];
          mid = (coords[v1] + coords[v2]) / 2;
          If[!insideHole[mid],
            AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
          ]
        ]
      ]
    ],
    {i, width}, {j, height}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords,
    "Topology" -> "GridWithHoles",
    "Width" -> width,
    "Height" -> height,
    "Holes" -> holes,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGCylinder - Cylindrical topology (theta wraps, z open) *)
(* ---------------------------------------------------------------------------- *)

Options[HGCylinder] = Join[$ICCommonOptions, {
  "Radius" -> 1.0,
  "RandomizeDirections" -> True
}];

HGCylinder[resolution_Integer, height_Integer, opts:OptionsPattern[]] := Module[
  {radius, randomize, vertices, coords2D, coords3D, edges,
   vertexIndex, i, j, idx, theta, z, v1, v2, dtheta, dz, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  radius = OptionValue["Radius"];
  randomize = OptionValue["RandomizeDirections"];

  dtheta = 2 Pi / resolution;
  dz = 1.0;

  vertexIndex = Association[];
  coords2D = Association[];
  coords3D = Association[];
  idx = 1;
  Do[
    theta = (i - 1) * dtheta;
    z = (j - 1) * dz;
    vertexIndex[{i, j}] = idx;
    coords2D[idx] = {theta, z};
    coords3D[idx] = {radius * Cos[theta], radius * Sin[theta], z};
    idx++,
    {i, resolution}, {j, height}
  ];

  vertices = Range[resolution * height];

  edges = {};
  Do[
    v1 = vertexIndex[{i, j}];
    v2 = vertexIndex[{Mod[i, resolution] + 1, j}];
    AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]];
    If[j < height,
      v2 = vertexIndex[{i, j + 1}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ],
    {i, resolution}, {j, height}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords2D,
    "VertexCoordinates3D" -> coords3D,
    "Topology" -> "Cylinder",
    "Resolution" -> resolution,
    "Height" -> height,
    "Radius" -> radius,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGTorus - Toroidal topology (both directions wrap) *)
(* ---------------------------------------------------------------------------- *)

Options[HGTorus] = Join[$ICCommonOptions, {
  "MajorRadius" -> 3.0,
  "MinorRadius" -> 1.0,
  "RandomizeDirections" -> True
}];

HGTorus[resolution_Integer, opts:OptionsPattern[]] := Module[
  {majorR, minorR, randomize, vertices, coords2D, coords3D, edges,
   vertexIndex, i, j, idx, theta, phi, v1, v2, rho, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  majorR = OptionValue["MajorRadius"];
  minorR = OptionValue["MinorRadius"];
  randomize = OptionValue["RandomizeDirections"];

  vertexIndex = Association[];
  coords2D = Association[];
  coords3D = Association[];
  idx = 1;
  Do[
    theta = (i - 1) * 2 Pi / resolution;
    phi = (j - 1) * 2 Pi / resolution;
    vertexIndex[{i, j}] = idx;
    coords2D[idx] = {theta, phi};
    rho = majorR + minorR * Cos[phi];
    coords3D[idx] = {rho * Cos[theta], rho * Sin[theta], minorR * Sin[phi]};
    idx++,
    {i, resolution}, {j, resolution}
  ];

  vertices = Range[resolution^2];

  edges = {};
  Do[
    v1 = vertexIndex[{i, j}];
    v2 = vertexIndex[{Mod[i, resolution] + 1, j}];
    AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]];
    v2 = vertexIndex[{i, Mod[j, resolution] + 1}];
    AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]],
    {i, resolution}, {j, resolution}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords2D,
    "VertexCoordinates3D" -> coords3D,
    "Topology" -> "Torus",
    "Resolution" -> resolution,
    "MajorRadius" -> majorR,
    "MinorRadius" -> minorR,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGSphere - Spherical topology using UV grid with pole handling *)
(* ---------------------------------------------------------------------------- *)

Options[HGSphere] = Join[$ICCommonOptions, {
  "Radius" -> 1.0,
  "RandomizeDirections" -> True
}];

HGSphere[resolution_Integer, opts:OptionsPattern[]] := Module[
  {radius, randomize, vertices, coords2D, coords3D, edges,
   vertexIndex, idx, nLat, nLon, theta, phi, lonCount, sinTheta,
   i, j, v1, v2, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  radius = OptionValue["Radius"];
  randomize = OptionValue["RandomizeDirections"];

  nLat = resolution;
  nLon = 2 * resolution;

  vertexIndex = Association[];
  coords2D = Association[];
  coords3D = Association[];
  idx = 1;

  Do[
    theta = Pi * (i - 0.5) / nLat;
    sinTheta = Sin[theta];
    lonCount = Max[1, Round[nLon * sinTheta]];
    Do[
      phi = 2 Pi * (j - 1) / lonCount;
      vertexIndex[{i, j, lonCount}] = idx;
      coords2D[idx] = {theta, phi};
      coords3D[idx] = {radius * sinTheta * Cos[phi], radius * sinTheta * Sin[phi], radius * Cos[theta]};
      idx++,
      {j, lonCount}
    ],
    {i, nLat}
  ];

  vertices = Range[idx - 1];
  edges = {};

  Module[{lonCounts, latOffsets, latIdx, lonIdx, currentLat, nextLat, v1Lon, closestJ},
    lonCounts = Table[Max[1, Round[nLon * Sin[Pi * (i - 0.5) / nLat]]], {i, nLat}];
    latOffsets = Prepend[Accumulate[Most[lonCounts]], 0];
    Do[
      currentLat = lonCounts[[latIdx]];
      Do[
        v1 = latOffsets[[latIdx]] + lonIdx;
        v2 = latOffsets[[latIdx]] + Mod[lonIdx, currentLat] + 1;
        AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]];
        If[latIdx < nLat,
          nextLat = lonCounts[[latIdx + 1]];
          v1Lon = (lonIdx - 1) / currentLat;
          closestJ = Clip[Round[v1Lon * nextLat] + 1, {1, nextLat}];
          v2 = latOffsets[[latIdx + 1]] + closestJ;
          AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
        ],
        {lonIdx, currentLat}
      ],
      {latIdx, nLat}
    ]
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords2D,
    "VertexCoordinates3D" -> coords3D,
    "Topology" -> "Sphere",
    "Resolution" -> resolution,
    "Radius" -> radius,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGKleinBottle - Klein bottle topology (theta wraps with z-flip) *)
(* ---------------------------------------------------------------------------- *)

Options[HGKleinBottle] = Join[$ICCommonOptions, {
  "Radius" -> 1.0,
  "RandomizeDirections" -> True
}];

HGKleinBottle[resolution_Integer, height_Integer, opts:OptionsPattern[]] := Module[
  {radius, randomize, vertices, coords2D, edges,
   vertexIndex, i, j, idx, theta, z, v1, v2, zFlipped, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  radius = OptionValue["Radius"];
  randomize = OptionValue["RandomizeDirections"];

  vertexIndex = Association[];
  coords2D = Association[];
  idx = 1;
  Do[
    theta = (i - 1) * 2 Pi / resolution;
    z = (j - 1) * 1.0;
    vertexIndex[{i, j}] = idx;
    coords2D[idx] = {radius * theta, z};
    idx++,
    {i, resolution}, {j, height}
  ];

  vertices = Range[resolution * height];

  edges = {};
  Do[
    v1 = vertexIndex[{i, j}];
    If[i < resolution,
      v2 = vertexIndex[{i + 1, j}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ];
    If[i == resolution,
      zFlipped = height - j + 1;
      v2 = vertexIndex[{1, zFlipped}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ];
    If[j < height,
      v2 = vertexIndex[{i, j + 1}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ],
    {i, resolution}, {j, height}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords2D,
    "Topology" -> "KleinBottle",
    "Resolution" -> resolution,
    "Height" -> height,
    "Radius" -> radius,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGMobiusStrip - Mobius strip (theta wraps with z-flip, finite width) *)
(* ---------------------------------------------------------------------------- *)

Options[HGMobiusStrip] = Join[$ICCommonOptions, {
  "Radius" -> 2.0,
  "RandomizeDirections" -> True
}];

HGMobiusStrip[resolution_Integer, width_Integer, opts:OptionsPattern[]] := Module[
  {radius, randomize, vertices, coords2D, coords3D, edges,
   vertexIndex, i, j, idx, theta, w, v1, v2, wFlipped, halfTwist, seed, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  radius = OptionValue["Radius"];
  randomize = OptionValue["RandomizeDirections"];

  vertexIndex = Association[];
  coords2D = Association[];
  coords3D = Association[];
  idx = 1;
  Do[
    theta = (i - 1) * 2 Pi / resolution;
    w = (j - 1) / (width - 1) - 0.5;
    vertexIndex[{i, j}] = idx;
    coords2D[idx] = {radius * theta, w * radius};
    halfTwist = theta / 2;
    coords3D[idx] = {
      (radius + w * Cos[halfTwist]) * Cos[theta],
      (radius + w * Cos[halfTwist]) * Sin[theta],
      w * Sin[halfTwist]
    };
    idx++,
    {i, resolution}, {j, width}
  ];

  vertices = Range[resolution * width];

  edges = {};
  Do[
    v1 = vertexIndex[{i, j}];
    If[i < resolution,
      v2 = vertexIndex[{i + 1, j}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ];
    If[i == resolution,
      wFlipped = width - j + 1;
      v2 = vertexIndex[{1, wFlipped}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ];
    If[j < width,
      v2 = vertexIndex[{i, j + 1}];
      AppendTo[edges, If[randomize && RandomReal[] < 0.5, {v2, v1}, {v1, v2}]]
    ],
    {i, resolution}, {j, width}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords2D,
    "VertexCoordinates3D" -> coords3D,
    "Topology" -> "MobiusStrip",
    "Resolution" -> resolution,
    "Width" -> width,
    "Radius" -> radius,
    "VertexCount" -> Length[vertices],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGMinkowskiSprinkling - Causal set by Minkowski sprinkling *)
(* ---------------------------------------------------------------------------- *)

Options[HGMinkowskiSprinkling] = Join[$ICCommonOptions, {
  "SpatialDim" -> 2,
  "TimeExtent" -> 10.0,
  "SpatialExtent" -> 10.0,
  "LightconeAngle" -> 1.0,
  "AlexandrovCutoff" -> 5.0,
  "TransitivityReduction" -> True,
  "MaxEdgesPerVertex" -> 50
}];

HGMinkowskiSprinkling[n_Integer, opts:OptionsPattern[]] := Module[
  {spatialDim, timeExtent, spatialExtent, lightcone, alexandrov,
   transitivity, maxEdges, seed, points, edges, coords2D,
   i, j, dt, dx2, tau2, causalPairs, directLinks, reduced,
   hasIntermediate, k, m, dimensionEstimate, nPairs, nRelated, r, result},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  spatialDim = OptionValue["SpatialDim"];
  timeExtent = OptionValue["TimeExtent"];
  spatialExtent = OptionValue["SpatialExtent"];
  lightcone = OptionValue["LightconeAngle"];
  alexandrov = OptionValue["AlexandrovCutoff"];
  transitivity = OptionValue["TransitivityReduction"];
  maxEdges = OptionValue["MaxEdgesPerVertex"];

  (* Generate random spacetime points *)
  points = Table[
    Prepend[
      RandomReal[{-spatialExtent/2, spatialExtent/2}, spatialDim],
      RandomReal[{0, timeExtent}]  (* Time coordinate first *)
    ],
    {n}
  ];

  (* Sort by time *)
  points = SortBy[points, First];

  (* Build causal edges *)
  (* Point a precedes b if: t_b > t_a AND |x_b - x_a| < c * (t_b - t_a) *)
  causalPairs = {};
  Do[
    Do[
      dt = points[[j, 1]] - points[[i, 1]];
      If[dt > 0,  (* j is in future of i *)
        dx2 = Total[(points[[j, 2 ;; spatialDim + 1]] - points[[i, 2 ;; spatialDim + 1]])^2];
        If[dx2 < lightcone^2 * dt^2,  (* Inside lightcone *)
          tau2 = dt^2 - dx2 / lightcone^2;  (* Proper time squared *)
          If[tau2 <= alexandrov^2,  (* Within Alexandrov interval *)
            AppendTo[causalPairs, {i, j}]
          ]
        ]
      ],
      {j, i + 1, n}
    ],
    {i, n - 1}
  ];

  (* Transitivity reduction *)
  If[transitivity && Length[causalPairs] > 0,
    (* Build adjacency list *)
    directLinks = Table[{}, {n}];
    Do[
      AppendTo[directLinks[[pair[[1]]]], pair[[2]]],
      {pair, causalPairs}
    ];

    (* Remove redundant edges *)
    Do[
      reduced = {};
      Do[
        hasIntermediate = False;
        Do[
          If[k != j && MemberQ[directLinks[[k]], j],
            hasIntermediate = True;
            Break[]
          ],
          {k, directLinks[[i]]}
        ];
        If[!hasIntermediate,
          AppendTo[reduced, j]
        ],
        {j, directLinks[[i]]}
      ];
      directLinks[[i]] = reduced,
      {i, n}
    ];

    (* Rebuild edge list *)
    causalPairs = Flatten[Table[{i, #} & /@ directLinks[[i]], {i, n}], 1]
  ];

  (* Limit edges per vertex *)
  If[maxEdges > 0,
    directLinks = Table[{}, {n}];
    Do[
      AppendTo[directLinks[[pair[[1]]]], pair[[2]]],
      {pair, causalPairs}
    ];
    causalPairs = Flatten[Table[
      {i, #} & /@ Take[directLinks[[i]], UpTo[maxEdges]],
      {i, n}
    ], 1]
  ];

  edges = causalPairs;

  (* 2D coordinates: use (x, t) for visualization *)
  coords2D = Association[Table[
    i -> {points[[i, 2]], points[[i, 1]]},  (* x, t *)
    {i, n}
  ]];

  (* Dimension estimate (Myrheim-Meyer) *)
  nPairs = 0;
  nRelated = 0;
  Do[
    Do[
      If[i != j,
        nPairs++;
        If[MemberQ[edges, {Min[i, j], Max[i, j]}] ||
           MemberQ[edges, {i, j}] || MemberQ[edges, {j, i}],
          nRelated++
        ]
      ],
      {j, i + 1, Min[i + 100, n]}  (* Sample nearby pairs *)
    ],
    {i, Min[n - 1, 100]}
  ];
  r = If[nPairs > 0, N[nRelated / nPairs], 0];
  (* Approximate: d/2^d = r, solve for d *)
  dimensionEstimate = If[r > 0.001 && r < 0.9,
    2.0,  (* Default for typical sprinkling *)
    spatialDim + 1  (* Fallback to expected dimension *)
  ];

  result = <|
    "Edges" -> edges,
    "SpacetimePoints" -> points,
    "VertexCoordinates" -> coords2D,
    "Topology" -> "Minkowski",
    "SpatialDim" -> spatialDim,
    "TimeExtent" -> timeExtent,
    "SpatialExtent" -> spatialExtent,
    "DimensionEstimate" -> dimensionEstimate,
    "VertexCount" -> n,
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGBrillLindquist - Brill-Lindquist curved spacetime around black holes *)
(* ---------------------------------------------------------------------------- *)

Options[HGBrillLindquist] = Join[$ICCommonOptions, {
  "BoxX" -> {-15.0, 15.0},
  "BoxY" -> {-15.0, 15.0},
  "EdgeThreshold" -> 2.0,
  "RandomizeDirections" -> True
}];

HGBrillLindquist[n_Integer, {mass1_, mass2_}, separation_, opts:OptionsPattern[]] := Module[
  {boxX, boxY, edgeThreshold, seed, randomize, result,
   bh1Center, bh2Center, bh1Radius, bh2Radius,
   insideHorizon, conformalFactor, volumeElement,
   points, coords, edges, maxVol, attempts, maxAttempts,
   pt, vol, i, j, dist, mid, v1, v2},

  boxX = OptionValue["BoxX"];
  boxY = OptionValue["BoxY"];
  edgeThreshold = OptionValue["EdgeThreshold"];
  seed = OptionValue["RandomSeed"];
  randomize = OptionValue["RandomizeDirections"];

  If[seed =!= Automatic, SeedRandom[seed]];

  (* Black hole positions and horizon radii *)
  bh1Center = {separation / 2, 0};
  bh2Center = {-separation / 2, 0};
  bh1Radius = mass1 / 2;
  bh2Radius = mass2 / 2;

  (* Check if point is inside either horizon *)
  insideHorizon[{x_, y_}] :=
    Norm[{x, y} - bh1Center] < bh1Radius || Norm[{x, y} - bh2Center] < bh2Radius;

  (* Brill-Lindquist conformal factor: psi = 1 + m1/(2*r1) + m2/(2*r2) *)
  conformalFactor[{x_, y_}] := Module[{r1, r2},
    r1 = Max[Norm[{x, y} - bh1Center], 0.001];
    r2 = Max[Norm[{x, y} - bh2Center], 0.001];
    1 + mass1 / (2 r1) + mass2 / (2 r2)
  ];

  (* Volume element is psi^4 *)
  volumeElement[pt_] := conformalFactor[pt]^4;

  (* Maximum volume element for rejection sampling *)
  maxVol = 100.0;

  (* Rejection sampling *)
  points = {};
  attempts = 0;
  maxAttempts = n * 1000;

  While[Length[points] < n && attempts < maxAttempts,
    attempts++;
    pt = {
      RandomReal[boxX],
      RandomReal[boxY]
    };

    (* Reject if inside horizon *)
    If[insideHorizon[pt], Continue[]];

    (* Accept with probability proportional to volume element *)
    vol = volumeElement[pt];
    If[RandomReal[] < vol / maxVol,
      AppendTo[points, pt]
    ]
  ];

  (* Build coordinate association *)
  coords = Association[Table[i -> points[[i]], {i, Length[points]}]];

  (* Build edges *)
  edges = {};
  Do[
    Do[
      dist = Norm[points[[i]] - points[[j]]];
      If[dist < edgeThreshold,
        mid = (points[[i]] + points[[j]]) / 2;
        If[!insideHorizon[mid],
          AppendTo[edges,
            If[randomize && RandomReal[] < 0.5, {j, i}, {i, j}]
          ]
        ]
      ],
      {j, i + 1, Length[points]}
    ],
    {i, Length[points] - 1}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords,
    "Topology" -> "BrillLindquist",
    "Mass1" -> mass1,
    "Mass2" -> mass2,
    "Separation" -> separation,
    "HorizonCenters" -> {bh1Center, bh2Center},
    "HorizonRadii" -> {bh1Radius, bh2Radius},
    "VertexCount" -> Length[points],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGPoissonDisk - Poisson disk sampling with minimum separation *)
(* ---------------------------------------------------------------------------- *)

Options[HGPoissonDisk] = Join[$ICCommonOptions, {
  "BoxX" -> {0, 10},
  "BoxY" -> {0, 10},
  "EdgeThreshold" -> Automatic,
  "RandomizeDirections" -> True
}];

HGPoissonDisk[n_Integer, minDistance_, opts:OptionsPattern[]] := Module[
  {boxX, boxY, edgeThreshold, seed, randomize, result,
   points, coords, edges, attempts, maxAttempts,
   candidate, valid, i, j, dist},

  boxX = OptionValue["BoxX"];
  boxY = OptionValue["BoxY"];
  edgeThreshold = OptionValue["EdgeThreshold"];
  seed = OptionValue["RandomSeed"];
  randomize = OptionValue["RandomizeDirections"];

  If[seed =!= Automatic, SeedRandom[seed]];

  (* Auto edge threshold *)
  If[edgeThreshold === Automatic,
    edgeThreshold = minDistance * 2.0
  ];

  (* Dart-throwing Poisson disk sampling *)
  points = {};
  attempts = 0;
  maxAttempts = n * 100;

  While[Length[points] < n && attempts < maxAttempts,
    attempts++;
    candidate = {
      RandomReal[boxX],
      RandomReal[boxY]
    };

    (* Check minimum distance to all existing points *)
    valid = True;
    Do[
      If[Norm[candidate - points[[i]]] < minDistance,
        valid = False;
        Break[]
      ],
      {i, Length[points]}
    ];

    If[valid,
      AppendTo[points, candidate]
    ]
  ];

  (* Build coordinate association *)
  coords = Association[Table[i -> points[[i]], {i, Length[points]}]];

  (* Build edges *)
  edges = {};
  Do[
    Do[
      dist = Norm[points[[i]] - points[[j]]];
      If[dist < edgeThreshold,
        AppendTo[edges,
          If[randomize && RandomReal[] < 0.5, {j, i}, {i, j}]
        ]
      ],
      {j, i + 1, Length[points]}
    ],
    {i, Length[points] - 1}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords,
    "Sampling" -> "PoissonDisk",
    "MinDistance" -> minDistance,
    "EdgeThreshold" -> edgeThreshold,
    "VertexCount" -> Length[points],
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ---------------------------------------------------------------------------- *)
(* HGUniformRandom - Uniform random point cloud *)
(* ---------------------------------------------------------------------------- *)

Options[HGUniformRandom] = Join[$ICCommonOptions, {
  "BoxX" -> {0, 10},
  "BoxY" -> {0, 10},
  "EdgeThreshold" -> Automatic,
  "RandomizeDirections" -> True
}];

HGUniformRandom[n_Integer, opts:OptionsPattern[]] := Module[
  {boxX, boxY, edgeThreshold, seed, randomize, result,
   points, coords, edges, i, j, dist, width, height, area, spacing},

  seed = OptionValue["RandomSeed"];
  If[seed =!= Automatic, SeedRandom[seed]];

  boxX = OptionValue["BoxX"];
  boxY = OptionValue["BoxY"];
  edgeThreshold = OptionValue["EdgeThreshold"];
  randomize = OptionValue["RandomizeDirections"];

  (* Auto edge threshold based on expected spacing *)
  If[edgeThreshold === Automatic,
    width = boxX[[2]] - boxX[[1]];
    height = boxY[[2]] - boxY[[1]];
    area = width * height;
    spacing = Sqrt[area / n];
    edgeThreshold = spacing * 1.5
  ];

  (* Generate random points *)
  points = Table[{RandomReal[boxX], RandomReal[boxY]}, {n}];

  (* Build coordinate association *)
  coords = Association[Table[i -> points[[i]], {i, n}]];

  (* Build edges *)
  edges = {};
  Do[
    Do[
      dist = Norm[points[[i]] - points[[j]]];
      If[dist < edgeThreshold,
        AppendTo[edges,
          If[randomize && RandomReal[] < 0.5, {j, i}, {i, j}]
        ]
      ],
      {j, i + 1, n}
    ],
    {i, n - 1}
  ];

  result = <|
    "Edges" -> edges,
    "VertexCoordinates" -> coords,
    "Sampling" -> "Uniform",
    "EdgeThreshold" -> edgeThreshold,
    "VertexCount" -> n,
    "EdgeCount" -> Length[edges]
  |>;

  icFinalizeResult[result, OptionValue["Graph"]]
]

(* ============================================================================ *)
(* Syntax information and front-end autocompletion                              *)
(* ============================================================================ *)
(* SyntaxInformation drives the editor: ArgumentsPattern colours argument count *)
(* and OptionNames supplies the option-name dropdown when typing a call.        *)

SyntaxInformation[HGEvolve] = {
  "ArgumentsPattern" -> {_, _, _, _., OptionsPattern[]},
  "OptionNames" -> Keys[Options[HGEvolve]]};
SyntaxInformation[HGGrid] = {
  "ArgumentsPattern" -> {_, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGGrid]]};
SyntaxInformation[HGGridWithHoles] = {
  "ArgumentsPattern" -> {_, _, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGGridWithHoles]]};
SyntaxInformation[HGCylinder] = {
  "ArgumentsPattern" -> {_, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGCylinder]]};
SyntaxInformation[HGTorus] = {
  "ArgumentsPattern" -> {_, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGTorus]]};
SyntaxInformation[HGSphere] = {
  "ArgumentsPattern" -> {_, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGSphere]]};
SyntaxInformation[HGKleinBottle] = {
  "ArgumentsPattern" -> {_, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGKleinBottle]]};
SyntaxInformation[HGMobiusStrip] = {
  "ArgumentsPattern" -> {_, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGMobiusStrip]]};
SyntaxInformation[HGMinkowskiSprinkling] = {
  "ArgumentsPattern" -> {_, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGMinkowskiSprinkling]]};
SyntaxInformation[HGBrillLindquist] = {
  "ArgumentsPattern" -> {_, _, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGBrillLindquist]]};
SyntaxInformation[HGPoissonDisk] = {
  "ArgumentsPattern" -> {_, _, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGPoissonDisk]]};
SyntaxInformation[HGUniformRandom] = {
  "ArgumentsPattern" -> {_, OptionsPattern[]}, "OptionNames" -> Keys[Options[HGUniformRandom]]};
SyntaxInformation[HGToGraph] = {"ArgumentsPattern" -> {_, OptionsPattern[]}};
End[]
EndPackage[]
