#!/usr/bin/env wolframscript
(* Build the installable .paclet archive from paclet/ and validate it.
   Usage: wolframscript -file tools/build_paclet.wls [outputDir]
   Prints the archive path + size and confirms it reads as a valid paclet. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
pacletDir = FileNameJoin[{repo, "paclet"}];
outDir = If[Length[$ScriptCommandLine] >= 2, Last[$ScriptCommandLine],
            FileNameJoin[{repo, "paclet_archive"}]];
If[!DirectoryQ[outDir], CreateDirectory[outDir, CreateIntermediateDirectories -> True]];

Print["paclet dir: ", pacletDir];
info = PacletObject[File[pacletDir]];
Print["name/version: ", info["Name"], " ", info["Version"]];

archive = CreatePacletArchive[pacletDir, outDir];
If[!StringQ[archive] || !FileExistsQ[archive],
  Print["BUILD_FAIL: CreatePacletArchive did not return a file"]; Exit[1]];
Print["archive: ", archive];
Print["size_MB: ", N[Round[FileByteCount[archive]/1024.^2, 0.01]]];

(* Validate the archive reads back as a well-formed paclet. *)
back = Quiet@PacletObject[File[archive]];
If[MatchQ[back, _PacletObject] && back["Name"] === info["Name"],
  Print["VALIDATE_OK ", back["Name"], " ", back["Version"]],
  Print["VALIDATE_FAIL: archive did not read back as the expected paclet"]; Exit[1]];
Print["DONE"];
