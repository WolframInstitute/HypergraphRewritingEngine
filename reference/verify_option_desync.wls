(* Gate: a malformed option value must not desynchronize the WXF option parser.
   CanonicalizeEvents -> {"InputState", 42} fails mid-list (the header parses, the
   integer element does not); recovery re-aligns to the value start and skips it,
   so options AFTER it still apply. Discriminator: CanonicalizeStates -> Full is
   parsed AFTER CanonicalizeEvents and changes NumStates on wolfram-2to4
   (canonical merge vs raw tree); a desynced parser loses it. *)

copyDir = FileNameJoin[{$TemporaryDirectory,
  "hgpaclet_" <> IntegerString[Hash[CreateUUID[]], 16, 12]}];
CopyDirectory[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}], copyDir];
PacletDirectoryLoad[copyDir];
Needs["HypergraphRewriting`"];

rule = {{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}, {z, x}};
init = {{1, 2}, {2, 3}};

nFullClean = Quiet[HGEvolve[rule, init, 3, "NumStates", "CanonicalizeStates" -> Full], {LibraryFunction::load, URLSubmit::offline}];
nNone = Quiet[HGEvolve[rule, init, 3, "NumStates"], {LibraryFunction::load, URLSubmit::offline}];
nFullAfterBad = Quiet[HGEvolve[rule, init, 3, "NumStates",
  "CanonicalizeEvents" -> {"InputState", 42}, "CanonicalizeStates" -> Full],
  {LibraryFunction::load, URLSubmit::offline}];
warned = MemberQ[$MessageList, HoldForm[MessageName[HGEvolve, "warn"]]];

Print["full=", nFullClean, " none=", nNone, " fullAfterBad=", nFullAfterBad,
  " skip-warned=", warned];

ok = IntegerQ[nFullClean] && IntegerQ[nFullAfterBad] &&
  nFullClean =!= nNone &&            (* the discriminator discriminates *)
  nFullAfterBad === nFullClean &&    (* the option AFTER the bad one applied *)
  warned;                            (* and the skip is on the trail *)
Print[If[ok, "DESYNC GATE PASS", "DESYNC GATE FAIL"]];

(* An integer "TransitionRate" or "ExplorationProbability" is sent as a real: 1 is the unsampled
   run, with no OptionSkipped warning. *)
Module[{plain, asInt, msgs},
  plain = HGEvolve[rule, init, 3, "NumStates"];
  {asInt, msgs} = Block[{$MessageList = {}},
    {HGEvolve[rule, init, 3, "NumStates", "TransitionRate" -> 1,
       "ExplorationProbability" -> 1], $MessageList}];
  Print["plain=", plain, " integer options=", asInt, " messages=", msgs];
  Print[If[asInt === plain && msgs === {}, "INTEGER RATE PASS", "INTEGER RATE FAIL"]]];

(* "RandomSeed" reaches the surface initial conditions: the same seed gives the same initial
   state, and a different seed a different one. *)
Module[{chain = {{{1, 2}} -> {{1, 3}, {3, 2}}}, init, rows},
  init[type_, s_] := Lookup[Values[HGEvolve[chain, type, 0, "States", "RandomSeed" -> s]], "Edges"];
  rows = Table[{type, ListQ[init[type, 5]] && init[type, 5] === init[type, 5],
                 init[type, 5] =!= init[type, 6]},
    {type, {"Cylinder", "Torus", "Sphere", "Klein", "Mobius"}}];
  Print["surface seed rows {type, same seed equal, other seed differs}: ", rows];
  Print[If[AllTrue[rows, #[[2]] && #[[3]] &], "SURFACE SEED PASS", "SURFACE SEED FAIL"]]];

(* "ShowProgress" -> True prints progress through the persistent worker, and nothing is printed
   without it. *)
Module[{printed, run},
  run[show_] := (printed = {};
    Block[{Print = Function[Null, AppendTo[printed, StringJoin[ToString /@ {##}]]]},
      HGEvolve[rule, init, 3, "NumStates", "ShowProgress" -> show]];
    printed);
  With[{on = run[True], off = run[False]},
    Print["progress lines with ShowProgress: ", Length[on], " ", Take[on, UpTo[2]], "; without: ", Length[off]];
    Print[If[Length[on] >= 2 && off === {}, "SHOW PROGRESS PASS", "SHOW PROGRESS FAIL"]]]];

(* A warning is reported as HGEvolve::overflow only when the engine marks it "Partial" -> 1. A
   GPU capacity overflow is partial; EventSigRawFallback and CountSaturated describe a complete
   run and are reported as HGEvolve::warn. *)
Module[{names},
  names[w_] := Block[{$MessageList = {}},
    HypergraphRewriting`Private`hgReportWarnings[w]; ToString /@ $MessageList];
  With[{partial = names[{<|"Kind" -> "replay work stack", "Count" -> 1, "Context" -> "c", "Partial" -> 1|>}],
        fallback = names[{<|"Kind" -> "EventSigRawFallback", "Count" -> 2, "Context" -> "c", "Partial" -> 0|>}],
        saturated = names[{<|"Kind" -> "CountSaturated", "Count" -> 1, "Context" -> "c", "Partial" -> 0|>}]},
    Print["partial: ", partial, " fallback: ", fallback, " saturated: ", saturated];
    Print[If[partial === {"HGEvolve::overflow"} && fallback === {"HGEvolve::warn"} &&
             saturated === {"HGEvolve::warn"}, "WARNING KINDS PASS", "WARNING KINDS FAIL"]]]];

(* Every Graph option given to HGEvolve or HGSessionOpen reaches the returned graph, and replaces
   the kernel's default: ImageSize, AspectRatio (as a symbol or a string), and GraphLayout over
   the default layered layout. A session's Query uses the options it was opened with. *)
Module[{chain = {{{1, 2}} -> {{1, 3}, {3, 2}}}, g, h, t, s, q, ok, msgs},
  {ok, msgs} = Block[{$MessageList = {}},
    g = HGEvolve[chain, {{1, 2}}, 2, "StatesGraphStructure", ImageSize -> 123,
      AspectRatio -> 1/2, GraphLayout -> "SpringEmbedding"];
    h = HGEvolve[chain, {{1, 2}}, 2, "StatesGraph", "AspectRatio" -> 1/3];
    t = HGEvolve[chain, "Grid", 1, "StatesGraphStructure", ImageSize -> 55];
    s = HGSessionOpen[chain, {{1, 2}}, "StatesGraphStructure", ImageSize -> 77];
    HGSessionStep[s, 2];
    q = HGSessionQuery[s, "StatesGraphStructure"];
    HGSessionClose[s];
    {{OptionValue[Options[g], ImageSize], OptionValue[Options[g], AspectRatio],
      OptionValue[Options[g], GraphLayout], OptionValue[Options[h], AspectRatio],
      OptionValue[Options[t], ImageSize], OptionValue[Options[q], ImageSize]},
     ToString /@ $MessageList}];
  Print["graph options {ImageSize, AspectRatio, GraphLayout, string AspectRatio, grid ImageSize, session ImageSize}: ",
    ok, " messages: ", msgs];
  Print[If[ok === {123, 1/2, "SpringEmbedding", 1/3, 55, 77} && msgs === {},
    "GRAPH OPTIONS PASS", "GRAPH OPTIONS FAIL"]]];

(* The package context holds the public symbols and nothing else. *)
Module[{names = Sort[Names["HypergraphRewriting`*"]]},
  Print["public names: ", names];
  Print[If[names === Sort[{"HGEvolve", "HGSessionObject", "HGSessionOpen", "HGSessionStep",
      "HGSessionQuery", "HGSessionClose", "HGSessionFrontier"}],
    "PUBLIC NAMES PASS", "PUBLIC NAMES FAIL"]]];

(* Malformed input to HGEvolve and HGSessionOpen issues one message naming the argument and gives
   $Failed, before anything reaches the engine. Symbolic vertices in an initial state and a rule
   that deletes its edges are accepted. *)
Module[{chain = {{{1, 2}} -> {{1, 3}, {3, 2}}}, run, rows, valid},
  run[expr_] := Block[{$MessageList = {}}, {expr, ToString /@ $MessageList}];
  SetAttributes[run, HoldFirst];
  rows = {
    {run[HGEvolve[{{{1, 2}} -> 3}, {{1, 2}}, 1, "NumStates"]], {$Failed, {"HGEvolve::badrule"}}},
    {run[HGEvolve[{{1, 2} -> {{1, 3}}}, {{1, 2}}, 1, "NumStates"]], {$Failed, {"HGEvolve::badrule"}}},
    {run[HGEvolve[Append[chain, "x"], {{1, 2}}, 1, "NumStates"]], {$Failed, {"HGEvolve::badrule"}}},
    {run[HGEvolve[chain, {1, 2}, 1, "NumStates"]], {$Failed, {"HGEvolve::badinit"}}},
    {run[HGEvolve[chain, {}, 1, "NumStates"]], {$Failed, {"HGEvolve::badinit"}}},
    {run[HGEvolve[chain, {{1, 2}}, -1, "NumStates"]], {$Failed, {"HGEvolve::steps"}}},
    {run[HGEvolve[chain, {{1, 2}}, 2.5, "NumStates"]], {$Failed, {"HGEvolve::steps"}}},
    {run[HGEvolve[chain, "Grid", -1, "NumStates"]], {$Failed, {"HGEvolve::steps"}}},
    {run[HGSessionOpen[{{{1, 2}} -> 3}, {{1, 2}}, "NumStates"]], {$Failed, {"HGEvolve::badrule"}}},
    {run[HGSessionOpen[chain, {1, 2}, "NumStates"]], {$Failed, {"HGEvolve::badinit"}}}};
  valid = {
    {run[HGEvolve[{{x, y}} -> {{x, z}, {z, y}}, {{a, b}}, 2, "NumStates"]], {4, {}}},
    {run[HGEvolve[{{{1, 2}, {2, 3}} -> {}}, {{1, 2}, {2, 3}}, 1, "NumStates"]], {2, {}}}};
  Print["input rows {got, expected}: ", Join[rows, valid]];
  Print[If[AllTrue[Join[rows, valid], #[[1]] === #[[2]] &], "INPUT CHECKS PASS", "INPUT CHECKS FAIL"]]];
