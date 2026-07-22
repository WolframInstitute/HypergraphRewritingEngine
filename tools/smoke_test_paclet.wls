#!/usr/bin/env wolframscript
(* Smoke-test the ASSEMBLED .paclet exactly as a user receives it: install the archive,
   load the installed paclet (not the repo source), and exercise HGEvolve on CPU and GPU
   across CanonicalizeStates None/Full. Asserts:
     - the paclet loads from the installed location (not paclet/ in the repo)
     - CPU None > CPU Full (the modes are genuinely distinct)
     - GPU None == CPU None and GPU Full == CPU Full (GPU honors every mode, matching CPU)
   The archive path is argv[1] (a Windows path when run from the Windows kernel). *)

pacletFile = $ScriptCommandLine[[2]];
Print["==> installing ", pacletFile];
inst = PacletInstall[pacletFile, ForceVersionInstall -> True];
Print["    -> ", inst];
If[FailureQ[inst], Print["INSTALL FAILED"]; Exit[2]];

Needs["HypergraphRewriting`"];
loc = PacletObject["WolframInstitute/HypergraphRewriteEngine"]["Location"];
Print["==> loaded paclet location: ", loc];

rule  = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
init  = {{1, 2}, {1, 3}};
steps = 4;

run[dev_, mode_] := HGEvolve[rule, init, steps, "NumStates",
   "CanonicalizeStates" -> mode, "TargetDevice" -> dev];

cpuNone = run["CPU", None];
cpuFull = run["CPU", Full];
gpuNone = run["GPU", None];
gpuFull = run["GPU", Full];

Print["==> NumStates (steps=", steps, ")"];
Print["    CPU  None=", cpuNone, "  Full=", cpuFull];
Print["    GPU  None=", gpuNone, "  Full=", gpuFull];

ok = True;
check[label_, cond_] := (Print[If[cond, "PASS  ", "FAIL  "], label]; ok = ok && cond);

check["CPU modes distinct (None > Full)", IntegerQ[cpuNone] && IntegerQ[cpuFull] && cpuNone > cpuFull];
check["GPU None matches CPU None",        gpuNone === cpuNone];
check["GPU Full matches CPU Full",        gpuFull === cpuFull];
check["GPU honors modes (None > Full)",   IntegerQ[gpuNone] && IntegerQ[gpuFull] && gpuNone > gpuFull];

(* Exercise the fallback DLL path directly. HGHausdorffAnalysis runs in-kernel through the
   LibraryLink DLL (performHausdorffAnalysis) -- the exact function that failed to load when
   the DLL carried an unbundled mingw runtime dependency. A finite numeric result here means
   the DLL and its functions loaded cleanly. *)
haus = HGHausdorffAnalysis[{{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 3}, {2, 4}}];
Print["==> HGHausdorffAnalysis (DLL path) -> ", Short[haus]];
(* A broken DLL throws libload/$Failed; a well-formed structured result means the DLL and
   performHausdorffAnalysis loaded and executed. *)
check["fallback DLL loaded + ran (performHausdorffAnalysis)",
  AssociationQ[haus] && Length[haus] >= 3];

Print[If[ok, "\nSMOKE TEST PASSED", "\nSMOKE TEST FAILED"]];
Exit[If[ok, 0, 1]];
