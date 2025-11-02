#!/usr/bin/env wolframscript
(* Build paclet with documentation using PacletBuild *)

If[$ScriptCommandLine === {},
    Print["Usage: build_paclet.wls <source_dir> <dest_dir>"];
    Exit[1]
];

sourceDir = $ScriptCommandLine[[2]];
destDir = $ScriptCommandLine[[3]];

Needs["PacletTools`"];

Print["Building paclet from: ", sourceDir];
Print["Output directory: ", destDir];

result = PacletTools`PacletBuild[sourceDir, destDir];

If[FailureQ[result],
    Print["Paclet build failed: ", result];
    Exit[1],
    Print["Paclet build succeeded!"];
    Print["Result: ", result];
    Exit[0]
];
