#!/usr/bin/env wolframscript
(* Convert authored markdown docs (paclet/Documentation/Source/*.md) into paclet
   documentation notebooks with WolframInstitute/MarkdownToNotebook. Each source's
   Template frontmatter (Symbol / Guide / TechNote) selects the target directory
   under paclet/Documentation/English, so the notebooks land where the paclet
   documentation build and the paclet repository expect them.

   Usage (prefer the wrapper, which finds wolframscript + inits the submodule):
     ./build_docs.sh              # generate + evaluate into Documentation/English/
     ./build_docs.sh structure    # input-only cells (evaluate later in the front end)
   Or directly (MarkdownToNotebook defaults to the tools/MarkdownToNotebook submodule):
     wolframscript -file tools/build_docs.wls [structure]

   The generated notebooks are build output (gitignored); the markdown is the source of
   truth (this mirrors WolframInstitute/MarkdownToNotebook's own paclets, e.g. AccessibleColors).
   create_paclet_archive's PacletBuild indexes them into the final Documentation-Center pages. *)

(* Args are optional and order-independent:
     structure     skip inline evaluation (input-only cells)
     out=<dir>     write the notebooks under <dir> instead of paclet/Documentation/English
     <path>        an explicit path to MarkdownToNotebook.wl (defaults to the vendored submodule)
   `out=` exists so the wrapper can stage the notebooks onto a Windows-local directory when the
   Wolfram kernel is the Windows install driving WSL: the Windows kernel cannot reliably Export
   onto the WSL 9P share (large notebooks fail Export::noopen), so build_docs.sh has it write to
   /mnt/c/Temp and copies the results into place from the Linux side. *)
outArgQ[a_]   := StringMatchQ[ToString[a], "out=" ~~ ___];
scriptArgs    = Rest[$ScriptCommandLine];
structureMode = MemberQ[scriptArgs, "structure"];
outArg        = SelectFirst[scriptArgs, outArgQ, None];
mtnArg        = SelectFirst[scriptArgs, (# =!= "structure" && ! outArgQ[#]) &, None];
mtnPath       = If[mtnArg =!= None, mtnArg,
  FileNameJoin[{DirectoryName[$InputFileName], "MarkdownToNotebook", "MarkdownToNotebook.wl"}]];
If[!FileExistsQ[mtnPath],
  Print["MarkdownToNotebook.wl not found: ", mtnPath,
        "\n  run: git submodule update --init tools/MarkdownToNotebook"]; Exit[1]];
Get[mtnPath];

repo    = ParentDirectory[DirectoryName[$InputFileName]];
srcDir  = FileNameJoin[{repo, "paclet", "Documentation", "Source"}];
(* Write the authoring notebooks straight into the standard paclet layout under
   Documentation/English. MarkdownToNotebook evaluates each page's example cells during
   conversion (against the paclet loaded below), so the notebooks carry the current engine's
   rendered output; create_paclet_archive's PacletBuild indexes them from here. The
   notebooks are build output (gitignored) — the markdown under Documentation/Source is the
   committed source of truth. `out=` (from build_docs.sh) redirects this to a staging directory. *)
docRoot = If[outArg =!= None,
  StringDrop[outArg, StringLength["out="]],
  FileNameJoin[{repo, "paclet", "Documentation", "English"}]];

(* Load the paclet being built so MarkdownToNotebook's per-page Needs["HypergraphRewriting`"]
   (from each source's `Context:` frontmatter) resolves to THIS build and the example cells
   render the current engine's output, not an installed release's. *)
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];

(* `structure` as the second argument generates INPUT-ONLY notebooks (MarkdownToNotebook's
   nested-convert mode): the cells are laid out but examples are NOT evaluated during
   conversion, so it makes no per-example resource-system/cloud calls. Use this on a machine
   that cannot reach the Wolfram resource system to evaluate inline; then open each notebook
   and Evaluate Notebook in the front end, which runs the examples against the local engine
   only (no cloud). Without this flag, examples are evaluated during conversion. *)
If[structureMode, $convertDepth = 1];

subdirFor[tmpl_] := Switch[tmpl,
  "Symbol",   {"ReferencePages", "Symbols"},
  "Guide",    {"Guides"},
  "TechNote", {"Tutorials"},
  _,          {"ReferencePages", "Symbols"}];

files = FileNames["*.md", srcDir];
If[files === {}, Print["no source docs in ", srcDir]; Exit[0]];

failed = {};
Do[
  Module[{meta, tmpl, name, target, rel},
    meta   = MarkdownToNotebook[f, "Association"]["Metadata"];
    tmpl   = Lookup[meta, "Template", "Symbol"];
    name   = Lookup[meta, "Name", FileBaseName[f]];
    target = FileNameJoin[Join[{docRoot}, subdirFor[tmpl], {name <> ".nb"}]];
    rel    = FileNameJoin[Join[subdirFor[tmpl], {name <> ".nb"}]];
    If[!DirectoryQ[DirectoryName[target]],
      CreateDirectory[DirectoryName[target], CreateIntermediateDirectories -> True]];
    (* Delete any prior copy so a post-write FileExistsQ is a true success signal, not a
       leftover from an earlier build. *)
    If[FileExistsQ[target], DeleteFile[target]];
    MarkdownToNotebook[f, target];
    (* MarkdownToNotebook's Export can fail (e.g. Export::noopen) without aborting, which would
       otherwise leave a missing/stale notebook while we falsely report success. Verify. *)
    If[TrueQ[FileExistsQ[target]] && FileByteCount[target] > 0,
      Print["built ", tmpl, " -> ", rel],
      Print["FAILED ", tmpl, " -> ", rel, "  (Export did not write the notebook)"];
      AppendTo[failed, rel]]],
  {f, files}];

If[failed =!= {},
  Print[""];
  Print["ERROR: ", Length[failed], " of ", Length[files], " notebook(s) failed to write:"];
  Scan[Print["  ", #] &, failed];
  Exit[1]];

(* Success sentinel. build_docs.sh gates notebook placement on THIS file, not on wolframscript's
   process exit code: the benign exit-time "The product exited because of a license error" message
   can make an otherwise-clean run return non-zero, which would abort the wrapper (set -e) before
   it ever moves the notebooks into English/. Reaching here means every notebook wrote and verified. *)
Export[FileNameJoin[{docRoot, ".build_ok"}], "ok", "Text"];
Print["DONE ", Length[files], " docs"];
