#!/usr/bin/env wolframscript
(* Emit the figure of the computed object: the smallest corpus case worked through.

   The paper defines states, events, causal edges and branchial edges and then measures them; this
   figure shows one instance of each, taken from the engine rather than drawn by hand. The case is
   the `binary` rule {{x,y}} -> {{x,z},{z,y}} from the seed {{1,2}} to depth two. The engine is
   asked for the canonical states, the events, the causal edges, the branchial edges and the global
   edge store; the raw states are rebuilt from the events (an event's output is its input minus the
   consumed edges plus the produced ones), which is the same reconstruction the engine performs, so
   the figure shows the four raw states, the canonical state two of them collapse to, the events
   between them and the causal edges between the events. A change in the engine's answer changes
   the picture rather than leaving a hand-drawn one asserting the old relation.

   Output: paper/tables/f_object_binary.tex, a tikzpicture headed by the provenance line the
   paper's integrity check reads (commit, machine, source). Run with the Windows wolframscript from
   WSL, through a copy under /mnt/c/Temp with the repository named explicitly, because a Windows
   kernel reading a freshly edited file through the WSL share can be served a stale copy:
     cp tools/dev/object_figure.wls /mnt/c/Temp/object_figure_$(date +%s).wls
     "/mnt/c/Program Files/Wolfram Research/Wolfram/14.3/wolframscript.exe" \
         -file "$(wslpath -w /mnt/c/Temp/object_figure_<stamp>.wls)" "$(wslpath -w "$PWD")" *)

repo = If[Length[$ScriptCommandLine] >= 2, $ScriptCommandLine[[2]],
  ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

(* The commit, read from .git without a git executable: HEAD names a ref, the ref file holds it. *)
gitCommit[] := Module[{head, ref, path},
  head = StringTrim[Import[FileNameJoin[{repo, ".git", "HEAD"}], "Text"]];
  If[StringStartsQ[head, "ref: "],
    ref = StringDrop[head, 5];
    path = FileNameJoin[Prepend[FileNameSplit[ref], FileNameJoin[{repo, ".git"}]]];
    If[FileExistsQ[path], StringTake[StringTrim[Import[path, "Text"]], 8], "unknown"],
    StringTake[head, 8]]];

rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
init = {{1, 2}};
steps = 2;

res = HGEvolve[rules, init, steps,
  {"States", "Events", "CausalEdges", "BranchialEdges", "GlobalEdges"},
  "CanonicalizeStates" -> Full];
If[!AssociationQ[res], Print["HGEvolve failed: ", res]; Exit[1]];

states = Values[res["States"]];       (* canonical states: Id, CanonicalId, Step, Edges, IsInitial *)
events = Values[res["Events"]];       (* Id, InputState/OutputState (raw), Canonical*State,
                                         ConsumedEdges/ProducedEdges (global edge ids) *)
causal = res["CausalEdges"];          (* From, To (event ids) *)
branchial = res["BranchialEdges"];
globalEdges = Association[(#[[1]] -> Rest[#]) & /@ res["GlobalEdges"]];

(* --- raw states, rebuilt from the events ------------------------------------------------------ *)
initial = SelectFirst[states, #["IsInitial"] == 1 &];
rawEdges = <|initial["Id"] -> (#[[1]] & /@ initial["Edges"])|>;
rawStep = <|initial["Id"] -> 0|>;
rawCanon = <|initial["Id"] -> initial["CanonicalId"]|>;
pending = events;
While[pending =!= {},
  Module[{ready = Select[pending, KeyExistsQ[rawEdges, #["InputState"]] &]},
    If[ready === {}, Print["unresolvable events: ", pending]; Exit[1]];
    Do[
      rawEdges[ev["OutputState"]] =
        Join[Complement[rawEdges[ev["InputState"]], ev["ConsumedEdges"]], ev["ProducedEdges"]];
      rawStep[ev["OutputState"]] = rawStep[ev["InputState"]] + 1;
      rawCanon[ev["OutputState"]] = ev["CanonicalOutputState"],
      {ev, ready}];
    pending = Complement[pending, ready]]];
rawIds = Keys[rawEdges];
Print["raw states: ", Length[rawIds], "  canonical: ", Length[states], "  events: ", Length[events],
  "  causal: ", Length[causal], "  branchial: ", Length[branchial]];

edgeText[ids_] := StringRiffle[
  ("\\{" <> StringRiffle[ToString /@ globalEdges[#], ","] <> "\\}") & /@ Sort[ids], ",\\,"];

(* --- layout: one column per step, raw states stacked within a column ------------------------- *)
byStep = GroupBy[rawIds, rawStep];
stepList = Sort[Keys[byStep]];
colX = AssociationThread[stepList -> {0, 2.8, 6.1, 9.4, 12.7}[[;; Length[stepList]]]];
pos = Association[];
Do[With[{ids = Sort[byStep[s]]},
  Do[pos[ids[[i]]] = {colX[s], -1.5 * (i - (Length[ids] + 1)/2)}, {i, Length[ids]}]],
  {s, stepList}];
merged = Select[GroupBy[rawIds, rawCanon], Length[#] > 1 &];

lines = {};
add[s_] := AppendTo[lines, s];
add["\\begin{tikzpicture}["];
add["    st/.style={rectangle, draw, rounded corners, inner sep=3pt, font=\\scriptsize},"];
add["    ev/.style={->, >=latex, thin},"];
add["    cau/.style={->, >=latex, thick, densely dashed},"];
add["    lbl/.style={font=\\scriptsize\\itshape, inner sep=1pt}"];
add["]"];
Do[With[{p = pos[r]},
  add[StringTemplate["\\node[st] (s``) at (``,``) {$``$};"][r, p[[1]], p[[2]], edgeText[rawEdges[r]]]]],
  {r, rawIds}];
Do[With[{a = pos[ev["InputState"]], b = pos[ev["OutputState"]]},
  add[StringTemplate["\\draw[ev] (s``) -- node[lbl, ``, pos=0.6]{$e_{``}$} (s``);"][
    ev["InputState"], If[b[[2]] >= a[[2]], "above left", "below left"], ev["Id"], ev["OutputState"]]]],
  {ev, events}];
Do[With[{ids = Sort[cl]},
  Do[add[StringTemplate["\\draw[<->, gray, thin] (s``) -- node[lbl, right, text width=2.1cm]{one canonical state} (s``);"][
    ids[[i]], ids[[i + 1]]]], {i, Length[ids] - 1}]],
  {cl, Values[merged]}];
(* Causal row: the events again as labels, the causal edges dashed between them. *)
yc = -1.5 * (Max[Length /@ Values[byStep]] + 1)/2 - 0.7;
add[StringTemplate["\\node[lbl, anchor=west] at (-0.6,``) {causal:};"][yc]];
(* An event sits at the column of its output state (its step), stacked as that state is. *)
Do[With[{p = pos[ev["OutputState"]]},
  add[StringTemplate["\\node[font=\\scriptsize] (c``) at (``,``) {$e_{``}$};"][
    ev["Id"], 0.9 + 1.7 * (rawStep[ev["OutputState"]] - 1), yc + p[[2]] * 0.55, ev["Id"]]]],
  {ev, events}];
noteX = 0.9 + 1.7 * (Max[Values[rawStep]] - 1) + 1.1;
Do[add[StringTemplate["\\draw[cau] (c``) -- (c``);"][c["From"], c["To"]]], {c, causal}];
Do[add[StringTemplate["\\draw[<->, gray, densely dotted] (c``) -- node[lbl, above]{branchial} (c``);"][
  b["From"], b["To"]]], {b, branchial}];
(* When the branchial relation is empty and two events share an input state, say what separates
   sibling events from a branchial pair: they consume different edges. *)
If[branchial === {},
  Module[{siblings = Select[GatherBy[events, #["InputState"] &], Length[#] > 1 &]},
    If[siblings =!= {},
      With[{pair = First[siblings]},
        If[Intersection @@ (#["ConsumedEdges"] & /@ pair) === {},
          add[StringTemplate["\\node[lbl, anchor=west, text width=6.2cm] at (``,``) {no branchial pair: `` consume \\emph{different} edges of $``$};"][
            noteX, yc, StringRiffle[("$e_{" <> ToString[#["Id"]] <> "}$") & /@ pair, " and "],
            edgeText[rawEdges[pair[[1]]["InputState"]]]]]]]]]];
add["\\end{tikzpicture}"];

prov = StringTemplate["% GENERATED by tools/dev/object_figure.wls -- do not edit.\n% commit `` | measured on `` | raw states ``, canonical ``, events ``, causal ``, branchial `` | source: tools/dev/object_figure.wls\n"][
  gitCommit[], StringTrim[$MachineName], Length[rawIds], Length[states], Length[events],
  Length[causal], Length[branchial]];
out = FileNameJoin[{repo, "paper", "tables", "f_object_binary.tex"}];
Export[out, prov <> StringRiffle[lines, "\n"] <> "\n", "Text"];
Print["wrote ", out];
