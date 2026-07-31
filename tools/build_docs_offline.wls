#!/usr/bin/env wolframscript
(* Offline, front-end-free documentation build.
   1. MarkdownToNotebook converts paclet/Documentation/Source/*.md -> generated/*.nb with
      example evaluation SKIPPED ($convertDepth>0 makes MTN treat this as a nested convert,
      so example cells are emitted input-only). This avoids MTN's per-example resource-system
      cloud calls, which hang (online) or trigger a license error (offline) in a
      network-restricted environment.
   2. The local paclet is loaded and every Input cell is evaluated in the kernel (no front
      end), its result spliced back as an Output cell via ToBoxes. Graph/Graphics results
      become GraphicsBox/Graphics3DBox that render when the notebook is opened.
   Args: <MarkdownToNotebook.wl path> [name-filter regex, optional] *)

mtnPath = $ScriptCommandLine[[2]];
filter  = If[Length[$ScriptCommandLine] >= 3, $ScriptCommandLine[[3]], ""];
repo    = ParentDirectory[DirectoryName[$InputFileName]];
$AllowInternet = False;

Get[mtnPath];
$convertDepth = 1;   (* skip example evaluation during conversion (structure only) *)

srcDir  = FileNameJoin[{repo, "paclet", "Documentation", "Source"}];
docRoot = FileNameJoin[{repo, "paclet", "Documentation", "Source", "generated"}];
subdirFor[t_] := Switch[t, "Symbol", {"ReferencePages","Symbols"}, "Guide", {"Guides"},
   "TechNote", {"Tutorials"}, _, {"ReferencePages","Symbols"}];

mds = FileNames["*.md", srcDir];
If[filter =!= "", mds = Select[mds, StringContainsQ[FileBaseName[#], RegularExpression[filter]] &]];

(* --- 1. structure (input-only) --- *)
targets = {};
Do[Module[{meta, tmpl, name, target},
    meta = MarkdownToNotebook[f, "Association"]["Metadata"];
    tmpl = Lookup[meta, "Template", "Symbol"];
    name = Lookup[meta, "Name", FileBaseName[f]];
    target = FileNameJoin[Join[{docRoot}, subdirFor[tmpl], {name <> ".nb"}]];
    If[! DirectoryQ[DirectoryName[target]],
       CreateDirectory[DirectoryName[target], CreateIntermediateDirectories -> True]];
    MarkdownToNotebook[f, target];
    AppendTo[targets, target];
    Print["struct  ", name]],
  {f, mds}];

(* --- 2. headless evaluation --- *)
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

evalToBoxes[b_] := Module[{e, r},
    e = Quiet @ MakeExpression[BoxData[b], StandardForm];
    r = TimeConstrained[Quiet @ Check[ReleaseHold[e], $Failed], 60, $TimedOut];
    Quiet @ Check[ToBoxes[r, StandardForm], RowBox[{"\[LeftSkeleton]", ToString[Head[r]], "\[RightSkeleton]"}]]
];

Do[Module[{nb},
    nb = Get[t];
    nb = nb /. Cell[_, "Output", ___] :> Sequence[];                 (* strip stale outputs *)
    nb = nb /. c : Cell[BoxData[b_], "Input", o___] :>
                 Sequence[c, Cell[BoxData[evalToBoxes[b]], "Output"]];
    Export[t, nb, "NB"];
    Print["eval    ", FileNameTake[t]]],
  {t, targets}];

Print["DONE ", Length[targets], " docs"];
