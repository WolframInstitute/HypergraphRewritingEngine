#!/usr/bin/env wolframscript
(* Convert authored markdown docs (paclet/Documentation/Source/*.md) into paclet
   documentation notebooks with WolframInstitute/MarkdownToNotebook. Each source's
   Template frontmatter (Symbol / Guide / TechNote) selects the target directory
   under paclet/Documentation/English, so the notebooks land where the paclet
   documentation build and the paclet repository expect them.

   Usage:
     git clone https://github.com/WolframInstitute/MarkdownToNotebook
     wolframscript -file tools/build_docs.wls /path/to/MarkdownToNotebook/MarkdownToNotebook.wl

   (DocumentationBuild produces the fully-indexed Documentation-Center pages from
   these authoring notebooks; that step needs the front end and is left to the
   publish workflow.) *)

If[Length[$ScriptCommandLine] < 2,
  Print["usage: build_docs.wls <path-to-MarkdownToNotebook.wl>"]; Exit[1]];
mtnPath = $ScriptCommandLine[[2]];
If[!FileExistsQ[mtnPath], Print["MarkdownToNotebook.wl not found: ", mtnPath]; Exit[1]];
Get[mtnPath];

repo    = ParentDirectory[DirectoryName[$InputFileName]];
srcDir  = FileNameJoin[{repo, "paclet", "Documentation", "Source"}];
(* Staging output: mirrors the Documentation/English layout but never overwrites a
   shipped, fully-built page. The publish workflow runs DocumentationBuild on these
   authoring notebooks (front end, evaluates examples, indexes) and promotes the
   result into Documentation/English. *)
docRoot = FileNameJoin[{repo, "paclet", "Documentation", "Source", "generated"}];

subdirFor[tmpl_] := Switch[tmpl,
  "Symbol",   {"ReferencePages", "Symbols"},
  "Guide",    {"Guides"},
  "TechNote", {"Tutorials"},
  _,          {"ReferencePages", "Symbols"}];

files = FileNames["*.md", srcDir];
If[files === {}, Print["no source docs in ", srcDir]; Exit[0]];

Do[
  Module[{meta, tmpl, name, target},
    meta   = MarkdownToNotebook[f, "Association"]["Metadata"];
    tmpl   = Lookup[meta, "Template", "Symbol"];
    name   = Lookup[meta, "Name", FileBaseName[f]];
    target = FileNameJoin[Join[{docRoot}, subdirFor[tmpl], {name <> ".nb"}]];
    If[!DirectoryQ[DirectoryName[target]],
      CreateDirectory[DirectoryName[target], CreateIntermediateDirectories -> True]];
    MarkdownToNotebook[f, target];
    Print["built ", tmpl, " -> ", FileNameDrop[target, FileNameSplit[repo] // Length]]],
  {f, files}];

Print["DONE ", Length[files], " docs"];
