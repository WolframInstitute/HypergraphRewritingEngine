#!/usr/bin/env wolframscript
(* Convert the documentation sources (docs/en/**/*.md) into the paclet's documentation
   notebooks with WolframInstitute/MarkdownToNotebook (the tools/MarkdownToNotebook submodule).

   Each source's Template frontmatter selects the directory under paclet/Documentation/English,
   and the notebook is named after the page's Name, which must equal the source's file name:
     docs/en/ReferencePages/Symbols/<Name>.md -> English/ReferencePages/Symbols/<Name>.nb
     docs/en/Guides/<Name>.md                 -> English/Guides/<Name>.nb
     docs/en/Tutorials/<Name>.md              -> English/Tutorials/<Name>.nb
   The notebooks are tracked, because the paclet ships them; the markdown is the source.

   Usage (build_docs.sh finds wolframscript, stages the output and places it):
     ./build_docs.sh                  generate and evaluate into Documentation/English/
     ./build_docs.sh structure        input-only cells into docs/en/.generated/, never English/
     ./build_docs.sh only=<regex>     only the sources whose file name matches <regex>
   Arguments of this script, in any order:
     structure      do not evaluate examples (input-only cells)
     only=<regex>   build a subset; the Name check and the stale sweep still cover every source
     out=<dir>      write the notebooks under <dir> (build_docs.sh stages to a Windows-local
                    directory: the Windows kernel cannot reliably Export onto the WSL share)
     dest=<dir>     the directory the notebooks are finally placed in; the incremental manifest
                    is read from it

   Examples evaluate during conversion against the paclet in paclet/. A reference page or guide
   evaluates each section on its own; a tutorial evaluates as one document, so a symbol defined
   in one section is available in the next. *)

prefixArgQ[p_][a_] := StringMatchQ[ToString[a], p ~~ "=" ~~ ___];
stripPrefix[p_, a_] := StringDrop[a, StringLength[p] + 1];
scriptArgs    = Rest[$ScriptCommandLine];
structureMode = MemberQ[scriptArgs, "structure"];
outArg        = SelectFirst[scriptArgs, prefixArgQ["out"], None];
destArg       = SelectFirst[scriptArgs, prefixArgQ["dest"], None];
onlyArg       = SelectFirst[scriptArgs, prefixArgQ["only"], None];
onlyPattern   = If[onlyArg === None, None, RegularExpression[stripPrefix["only", onlyArg]]];

repo    = ParentDirectory[DirectoryName[$InputFileName]];
srcDir  = FileNameJoin[{repo, "docs", "en"}];
mtnPath = FileNameJoin[{repo, "tools", "MarkdownToNotebook", "MarkdownToNotebook.wl"}];
If[!FileExistsQ[mtnPath],
  Print["MarkdownToNotebook.wl not found: ", mtnPath,
        "\n  run: git submodule update --init tools/MarkdownToNotebook"]; Exit[1]];
Get[mtnPath];

(* Input-only notebooks never go where the paclet ships them. *)
docRoot = Which[
  outArg =!= None, stripPrefix["out", outArg],
  structureMode,   FileNameJoin[{srcDir, ".generated"}],
  True,            FileNameJoin[{repo, "paclet", "Documentation", "English"}]];
dest = If[destArg =!= None, stripPrefix["dest", destArg], docRoot];

(* The examples load HypergraphRewriting` from this checkout, not from an installed release. *)
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];

(* Everything a notebook's content depends on besides its markdown: the engine binaries and the
   kernel files the examples run through, the converter, and this script. *)
hashFiles[fs_List] := FileHash[#, "SHA256"] & /@ Sort[fs];
engineStamp = If[structureMode, "no-eval",
  Hash[{$SystemID,
    hashFiles[FileNames["*", FileNameJoin[{repo, "paclet", "LibraryResources", $SystemID}]]],
    hashFiles[FileNames["*.wl", FileNameJoin[{repo, "paclet", "Kernel"}]]]}]];
toolStamp = Hash[{FileHash[mtnPath, "SHA256"], FileHash[$InputFileName, "SHA256"]}];

If[structureMode, $convertDepth = 1];

subdirFor[tmpl_] := Switch[tmpl,
  "Symbol",   {"ReferencePages", "Symbols"},
  "Guide",    {"Guides"},
  "TechNote", {"Tutorials"},
  _,          {"ReferencePages", "Symbols"}];

files = Select[FileNames["*.md", srcDir, Infinity],
  ! StringContainsQ[#, FileNameJoin[{srcDir, ".generated"}]] &];
If[files === {}, Print["no source docs in ", srcDir]; Exit[0]];

(* Every source: its template, its notebook's path relative to English/, and whether it is built
   in this run. A Name that differs from the file name stops the build: the notebook and the
   page's URI are named after the Name, and a reader finds the source by the file name. *)
(* "Evaluate" -> False: the "Association" form builds the whole notebook, and with evaluation on it
   would run every page's examples just to read its frontmatter, under the default section
   resets, which leaves engine sessions open that a later page then cannot open. *)
pages = Map[
  Function[f, Module[{meta = MarkdownToNotebook[f, "Association", "Evaluate" -> False]["Metadata"],
                      tmpl, name},
    tmpl = Lookup[meta, "Template", "Symbol"];
    name = Lookup[meta, "Name", FileBaseName[f]];
    <|"File" -> f, "Template" -> tmpl, "Name" -> name,
      "Rel" -> FileNameJoin[Join[subdirFor[tmpl], {name <> ".nb"}]],
      "Build" -> (onlyPattern === None || StringContainsQ[FileBaseName[f], onlyPattern])|>]],
  files];
badNames = Select[pages, #Name =!= FileBaseName[#File] &];
If[badNames =!= {},
  Scan[Print["ERROR: ", #File, " has Name \"", #Name, "\"; it must be \"", FileBaseName[#File], "\""] &,
    badNames];
  Exit[1]];

(* Incremental build: the manifest in `dest` maps each notebook to the key that produced the copy
   there. A page is rebuilt when its key changed or its notebook is missing. *)
buildKey[f_] := Hash[{FileHash[f, "SHA256"], engineStamp, toolStamp, structureMode}];
manifestName = ".doc_manifest.wl";
oldManifest  = With[{p = FileNameJoin[{dest, manifestName}]},
  If[FileExistsQ[p], Replace[Quiet@Get[p], Except[_?AssociationQ] -> <||>], <||>]];
newManifest  = KeyTake[oldManifest, Lookup[pages, "Rel"]];

failed = {};
Do[
  Module[{f = page["File"], rel = page["Rel"], target, key},
    target = FileNameJoin[{docRoot, rel}];
    key    = buildKey[f];
    Which[
      ! page["Build"], Null,
      Lookup[oldManifest, rel, None] === key && FileExistsQ[FileNameJoin[{dest, rel}]],
        Print["cached ", page["Template"], " -> ", rel],
      True,
        If[! DirectoryQ[DirectoryName[target]],
          CreateDirectory[DirectoryName[target], CreateIntermediateDirectories -> True]];
        If[FileExistsQ[target], DeleteFile[target]];
        (* "UseCache" -> False: the converter's own output cache is keyed by the example code
           alone, so after an engine or kernel change it would serve the old outputs. Whether a
           page is rebuilt at all is decided above, by a key that covers both. *)
        MarkdownToNotebook[f, target,
          "EvaluateSeparator" -> If[page["Template"] === "TechNote", None, Automatic],
          "UseCache" -> False];
        (* Export can fail without aborting; a notebook that was not written is a failure. *)
        If[TrueQ[FileExistsQ[target]] && FileByteCount[target] > 0,
          Print["built ", page["Template"], " -> ", rel]; newManifest[rel] = key,
          Print["FAILED ", page["Template"], " -> ", rel, "  (Export did not write the notebook)"];
          AppendTo[failed, rel]]]],
  {page, pages}];

If[failed =!= {},
  Print[""];
  Print["ERROR: ", Length[failed], " notebook(s) failed to write:"];
  Scan[Print["  ", #] &, failed];
  Exit[1]];

Put[newManifest, FileNameJoin[{docRoot, manifestName}]];
(* The notebooks every source maps to, one relative path per line. build_docs.sh deletes any
   notebook under English/ that is not listed, so a renamed or deleted page leaves nothing
   behind. *)
Export[FileNameJoin[{docRoot, ".doc_pages"}], StringRiffle[Lookup[pages, "Rel"], "\n"], "Text"];
(* Success sentinel: build_docs.sh places the notebooks only if this file exists, because
   wolframscript can exit non-zero after a clean run ("The product exited because of a license
   error"). *)
Export[FileNameJoin[{docRoot, ".build_ok"}], "ok", "Text"];
Print["DONE ", Count[pages, _?(#Build &)], " of ", Length[pages], " docs"];
