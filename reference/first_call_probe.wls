(* One fresh kernel's first engine call, for tools/dev/first_call_gate.sh.
   Usage: first_call_probe.wls PACLETDIR MARKERFILE DEVICE [CASE]
   Writes "start <kernel process id>" to MARKERFILE before anything else, runs CASE, and then
   writes "returned <result>" and "messages <names>". A marker with "start" and no "returned" is a
   kernel that ended during the case. CASE:
     first         (the default) load the paclet in PACLETDIR and make the kernel's first HGEvolve
                   call on DEVICE ("CPU" or "GPU": Wolfram's rule from {{0,0},{0,0}}, 3 steps,
                   "MemoryLimit" 3 GB)
     notebook      the notebook builds' order before the same call: definitions, a file written
                   and read back, a plot rasterized, then the paclet loaded and the call made
     needs         load the paclet and make no call
     print         Print[1]; no paclet
     startprocess  StartProcess of cmd /c echo (sh -c echo off Windows) and its output; no paclet
     runprocess    RunProcess of the same command; no paclet *)
{pacletDir, marker, device} = $ScriptCommandLine[[2 ;; 4]];
probeCase = If[Length[$ScriptCommandLine] >= 5, $ScriptCommandLine[[5]], "first"];
Export[marker, "start " <> ToString[$ProcessID], "Text"];
echoCommand = If[$OperatingSystem === "Windows", {"cmd", "/c", "echo", "x"}, {"sh", "-c", "echo x"}];
load[] := (PacletDirectoryLoad[pacletDir]; Needs["HypergraphRewriting`"]);
firstCall[] := Symbol["HypergraphRewriting`HGEvolve"][
  {{x, y}, {x, z}} -> {{x, y}, {x, w}, {y, w}, {z, w}}, {{0, 0}, {0, 0}}, 3, "NumStates",
  "TargetDevice" -> device, "Tuning" -> <|"MemoryLimit" -> 3 2^30|>];
d = EvaluationData[Switch[probeCase,
  "first", load[]; firstCall[],
  "notebook", Module[{f, tmp, data},
    f[k_] := k^2 + 1;
    tmp = FileNameJoin[{$TemporaryDirectory, "first_call_probe_" <> ToString[$ProcessID] <> ".m"}];
    Export[tmp, Table[{k, f[k]}, {k, 100}]];
    data = Import[tmp];
    DeleteFile[tmp];
    Rasterize[ListPlot[data], ImageSize -> 200];
    load[];
    firstCall[]],
  "needs", load[]; Length[Names["HypergraphRewriting`*"]],
  "print", Print[1]; 1,
  "startprocess", Module[{p = StartProcess[echoCommand], s}, s = ReadString[p]; StringLength[StringTrim[s]]],
  "runprocess", StringLength[StringTrim[RunProcess[echoCommand, "StandardOutput"]]],
  _, $Failed]];
Export[marker, "start " <> ToString[$ProcessID] <> "\nreturned " <> ToString[d["Result"]] <>
  "\nmessages " <> ToString[d["MessagesText"]],
  "Text"];
Exit[0];
