#!/usr/bin/env wolframscript
(* Assemble the distributable .paclet archive from the built paclet directory.

   IT STAGES FIRST, AND THAT IS THE POINT. CreatePacletArchive zips the directory it is
   given, so pointing it at `paclet/` ships whatever happens to be sitting there --
   including `Documentation/Source/generated/`, which is a doc-build INTERMEDIATE and is
   gitignored for that reason. Measured: archiving `paclet/` directly produced a 96 MB
   file from a 599 MB tree, 512 MB of which was that intermediate, against 6.4 MB of
   built documentation. So this copies out the four things PacletInfo.wl actually
   declares -- Kernel, LibraryResources, Documentation/English and PacletInfo.wl itself
   -- and archives that.

   THE STAGING ROOT IS $TemporaryDirectory, WHICH IS LOAD-BEARING UNDER WSL. Run through
   the Windows wolframscript against a repository on the Linux side, the source path is a
   \\wsl.localhost UNC path, and CreatePacletArchive fails there with a J/Link error. The
   staging root is local to whichever kernel runs this, so the archive is always built
   from a native path and copied back afterwards.

   IT VERIFIES RATHER THAN ONLY PRODUCING. The archive is read back and checked to carry
   a library for every SystemID PacletInfo.wl declares: an archive missing one installs
   without complaint and then fails on that platform alone. *)

root = DirectoryName[DirectoryName[$InputFileName]];
src  = FileNameJoin[{root, "paclet"}];
dist = FileNameJoin[{root, "dist"}];
If[!DirectoryQ[src], Print["FAIL: no paclet directory at ", src]; Exit[1]];
If[!DirectoryQ[dist], CreateDirectory[dist]];

info = Get[FileNameJoin[{src, "PacletInfo.wl"}]];
If[Head[info] =!= PacletObject, Print["FAIL: PacletInfo.wl did not give a PacletObject"]; Exit[1]];
platforms = info["SystemID"];
Print["source:     ", src];
Print["version:    ", info["Version"]];
Print["platforms:  ", platforms];

(* Stage exactly what the paclet declares. Anything not copied here does not ship. *)
stage = FileNameJoin[{$TemporaryDirectory, "hgre_paclet_stage"}];
If[DirectoryQ[stage], DeleteDirectory[stage, DeleteContents -> True]];
CreateDirectory[stage];
CopyFile[FileNameJoin[{src, "PacletInfo.wl"}], FileNameJoin[{stage, "PacletInfo.wl"}]];
CopyDirectory[FileNameJoin[{src, "Kernel"}], FileNameJoin[{stage, "Kernel"}]];
CopyDirectory[FileNameJoin[{src, "LibraryResources"}], FileNameJoin[{stage, "LibraryResources"}]];
CreateDirectory[FileNameJoin[{stage, "Documentation"}]];
CopyDirectory[FileNameJoin[{src, "Documentation", "English"}],
              FileNameJoin[{stage, "Documentation", "English"}]];

(* The doc build leaves its own markers inside English/; they are gitignored and are not
   documentation. *)
Do[If[FileExistsQ[f], DeleteFile[f]],
   {f, FileNameJoin[{stage, "Documentation", "English", #}] & /@ {".build_ok", ".doc_manifest.wl"}}];

(* FileNames returns directories too, and FileByteCount rejects them. *)
treeBytes[d_] := Total[FileByteCount /@ Select[FileNames["*", d, Infinity], !DirectoryQ[#] &]];
Print["staged:     ", Round[treeBytes[stage]/1024^2], " MB (source tree is ",
      Round[treeBytes[src]/1024^2], " MB)"];

archive = CreatePacletArchive[stage, dist];
If[FailureQ[archive] || !FileExistsQ[archive],
  Print["FAIL: CreatePacletArchive returned ", archive]; Exit[1]];
Print["archive:    ", archive];
Print["size:       ", Round[FileByteCount[archive]/1024^2], " MB"];

names = Import[archive, {"ZIP", "FileNames"}];
Print["entries:    ", Length[names]];
Do[Print["  ", p, ": ", Count[names, _?(StringContainsQ[#, p] &)], " file(s)"], {p, platforms}];
missing = Select[platforms, Function[p, !AnyTrue[names, StringContainsQ[#, p] &]]];
If[missing =!= {}, Print["FAIL: archive carries no library for: ", missing]; Exit[1]];

leaked = Select[names, StringContainsQ[#, "Documentation/Source"] &];
If[leaked =!= {},
  Print["FAIL: ", Length[leaked], " doc-build intermediate(s) leaked into the archive"];
  Exit[1]];

DeleteDirectory[stage, DeleteContents -> True];
Print["OK: archive carries all ", Length[platforms],
      " platforms and no doc-build intermediates"];
Exit[0]
