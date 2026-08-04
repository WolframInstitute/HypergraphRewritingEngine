#!/usr/bin/env wolframscript
(* Drop documentation links to symbols the paclet does not export.

   The guide lists one entry per documented function, and a reference page carries a
   "See Also" row of them, both as Cells holding a PackageLink to `.../ref/<Symbol>`. The
   visualisation split removed the functions and their pages went with them, so every such
   link now points at a page that is not there.

   THE KEEP SET IS READ FROM THE KERNEL SOURCE, not written down here. PackageExport lines
   in paclet/Kernel/HypergraphRewriting.wl are what the paclet actually exports, so a
   symbol added or removed there changes this script's behaviour without editing it.

   THREE SHAPES ARE REMOVED, and they are different:
     - a guide entry: a whole "GuideText" Cell, since the cell IS the entry
     - a See Also item: the inline Cell alone, because its siblings may be live links
     - the separator that item left behind, and any section reduced to nothing

   Run once; tools/dev/doc_symbols_check.py is what keeps it true afterwards. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
docs = FileNameJoin[{repo, "paclet", "Documentation", "English"}];
kernel = FileNameJoin[{repo, "paclet", "Kernel", "HypergraphRewriting.wl"}];

exported = DeleteDuplicates@StringCases[Import[kernel, "Text"],
             "PackageExport[\"" ~~ s : LetterCharacter .. ~~ "\"]" :> s];
Print["exported by the Kernel: ", exported];
If[exported === {}, Print["FAIL: no PackageExport found"]; Exit[1]];

(* The ref a subexpression links to, or Missing. *)
refOf[e_] := Module[{refs},
  refs = Cases[e, s_String /; StringContainsQ[s, "/ref/"] :>
                 StringReplace[Last[StringSplit[s, "/ref/"]], WhitespaceCharacter -> ""],
               Infinity];
  If[refs === {}, Missing[], First[refs]]];

dead[e_] := With[{r = refOf[e]}, !MissingQ[r] && !MemberQ[exported, r]];

total = 0;
Do[
  Module[{nb, pruned, before, after},
    nb = Get[file];
    If[Head[nb] =!= Notebook, Print["skip (not a Notebook): ", file]; Continue[]];
    before = Length[Select[Cases[nb, _Cell, Infinity], dead]];

    pruned = nb //. {
      (* A guide entry cell. *)
      Cell[c_, "GuideText", r___] /; dead[Cell[c, "GuideText", r]] :> Sequence[],
      (* A See Also / inline item. *)
      Cell[c_, "InlineFormula", r___] /; dead[Cell[c, "InlineFormula", r]] :> Sequence[]
    };
    (* Separators stranded by a removed item: a leading one, a trailing one, or two
       adjacent. \[FilledVerySmallSquare] is the See Also separator. *)
    pruned = pruned //. {
      TextData[{sep_String /; StringContainsQ[sep, "\[FilledVerySmallSquare]"], rest___}] :>
        TextData[{rest}],
      TextData[{most___, sep_String /; StringContainsQ[sep, "\[FilledVerySmallSquare]"]}] :>
        TextData[{most}],
      TextData[{a___, s1_String /; StringContainsQ[s1, "\[FilledVerySmallSquare]"],
                s2_String /; StringContainsQ[s2, "\[FilledVerySmallSquare]"], b___}] :>
        TextData[{a, s1, b}]
    };
    (* A group whose heading is all that is left. *)
    pruned = pruned //. {
      Cell[CellGroupData[{Cell[_, "GuideFunctionsSubsection", ___]}, ___], ___] :> Sequence[],
      Cell[TextData[{}], _String, ___] :> Sequence[]
    };

    after = Length[Select[Cases[pruned, _Cell, Infinity], dead]];
    If[before =!= 0 || pruned =!= nb,
      Export[file, pruned, "NB"];
      Print[FileNameTake[file], ": ", before, " dead -> ", after]];
    total += after;
  ],
  {file, FileNames["*.nb", docs, Infinity]}];

Print["dead links remaining across all notebooks: ", total];
Exit[If[total === 0, 0, 1]]
