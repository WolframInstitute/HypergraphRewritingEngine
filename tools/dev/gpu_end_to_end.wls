#!/usr/bin/env wolframscript
(* The end-to-end CPU against GPU check of the release gate.

   Each workload runs through the paclet on "TargetDevice" -> "CPU" and on "TargetDevice" -> "GPU".
   Each device gets one warm call, then three timed calls, and the minimum of the three is kept.
   The script prints CPU time, GPU time and CPU/GPU per workload, and whether the two results are
   equal. A GPU result that carries a partial-result warning (HGEvolve::overflow) is reported as
   partial.

   tools/dev/quiet_check.sh runs first. On a busy machine the script prints the readings and exits
   with status 2 without timing anything.

   Exit status: 0 when every workload has equal results and CPU/GPU of at least 1, 1 otherwise,
   2 when the machine is not quiet.

   Usage: wolframscript -file tools/dev/gpu_end_to_end.wls [--agree-only] [pacletDir | installed] [filter]
     --agree-only  skip the quiet check and the timing: one call per device, and the exit status
                   reports only equal results and partial results
     pacletDir     the paclet to load (default: the repository's paclet/)
     filter        run only the workloads whose name contains this string *)

here = DirectoryName[$InputFileName];
repo = ParentDirectory[ParentDirectory[here]];
args = Rest[$ScriptCommandLine];
agreeOnly = MemberQ[args, "--agree-only"];
args = DeleteCases[args, "--agree-only"];
src = If[Length[args] >= 1, args[[1]], FileNameJoin[{repo, "paclet"}]];
filter = If[Length[args] >= 2, args[[2]], ""];

quietScript = FileNameJoin[{here, "quiet_check.sh"}];
quiet = If[agreeOnly, <|"ExitCode" -> 0, "StandardOutput" -> "agree-only: no timing\n"|>,
   If[$OperatingSystem === "Windows",
  RunProcess[{"wsl.exe", "-e", "bash", "-c", "bash \"$(wslpath -u '" <> quietScript <> "')\""}],
  RunProcess[{"bash", quietScript}]]];
WriteString[$Output, quiet["StandardOutput"]];
If[quiet["ExitCode"] =!= 0,
  Print["The machine is not quiet: no timing taken."];
  Exit[2]];

If[src =!= "installed", PacletDirectoryLoad[src]];
Needs["HypergraphRewriting`"];
Print["paclet: ", PacletFind["WolframInstitute/HypergraphRewriteEngine"][[1]]["Location"]];

ruleA = {{x, y}, {y, z}, {z, w}, {w, v}} -> {{y, u}, {u, v}, {w, x}, {x, u}};
wpp = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}};
wolfram = {{x, y}, {x, z}} -> {{x, y}, {x, w}, {y, w}, {z, w}};
SetAttributes[workload, HoldRest];
workloads = {
  workload["grid8 stats 10 steps", HGEvolve[ruleA, <|"Type" -> "Grid", "Width" -> 8, "Height" -> 8, "Seed" -> 1|>, 10, "StepStatistics", "MaxStatesPerStep" -> 400, "StepStatisticsBranchial" -> All, "TargetDevice" -> $device]],
  workload["grid16 stats 8 steps", HGEvolve[ruleA, <|"Type" -> "Grid", "Width" -> 16, "Height" -> 16, "Seed" -> 1|>, 8, "StepStatistics", "MaxStatesPerStep" -> 1000, "TargetDevice" -> $device]],
  workload["torus9 stats 40 steps", HGEvolve[ruleA, <|"Type" -> "Torus", "Resolution" -> 9, "Seed" -> 1|>, 40, "StepStatistics", "UniformRandom" -> True, "MatchesPerStep" -> 10, "RandomSeed" -> 1, "TargetDevice" -> $device]],
  workload["wpp 6 full stats", HGEvolve[wpp, {{1, 2}, {1, 3}}, 6, "StepStatistics", "TargetDevice" -> $device]],
  workload["wpp 7 full counts", HGEvolve[wpp, {{1, 2}, {1, 3}}, 7, {"NumStates", "NumEvents"}, "TargetDevice" -> $device]],
  workload["wpp 7 canon counts", HGEvolve[wpp, {{1, 2}, {1, 3}}, 7, {"NumStates", "NumEvents"}, "CanonicalizeStates" -> Full, "TargetDevice" -> $device]],
  workload["wolfram 6 canon counts", HGEvolve[wolfram, {{0, 0}, {0, 0}}, 6, {"NumStates", "NumEvents"}, "CanonicalizeStates" -> Full, "TargetDevice" -> $device]],
  workload["wolfram 5 full counts", HGEvolve[wolfram, {{0, 0}, {0, 0}}, 5, {"NumStates", "NumEvents"}, "TargetDevice" -> $device]],
  workload["wpp 8 quotient counts", HGEvolve[wpp, {{1, 2}, {1, 3}}, 8, {"NumStates", "NumEvents"}, "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True, "TargetDevice" -> $device]],
  workload["wpp 9 quotient counts", HGEvolve[wpp, {{1, 2}, {1, 3}}, 9, {"NumStates", "NumEvents"}, "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True, "TargetDevice" -> $device]],
  workload["wpp 8 quotient stats", HGEvolve[wpp, {{1, 2}, {1, 3}}, 8, "StepStatistics", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True, "TargetDevice" -> $device]],
  workload["wpp 5 canon graphs", HGEvolve[wpp, {{1, 2}, {1, 3}}, 5, {"NumStates", "CausalGraph", "BranchialGraph"}, "CanonicalizeStates" -> Full, "TargetDevice" -> $device]]
};
workloads = Select[workloads, StringContainsQ[#[[1]], filter] &];

(* One call: {seconds, result, partial} with partial True when HGEvolve::overflow was issued.
   Quiet suppresses the printing of the message, and Check still sets partial. *)
call[w_, d_] := Block[{$device = d},
  Module[{t, r, partial = False},
    t = First[AbsoluteTiming[
      Quiet[Check[r = ReleaseHold[Extract[w, 2, Hold]], partial = True, {HGEvolve::overflow}]]]];
    {t, r, partial}]];

fmt[x_, d_] := ToString[NumberForm[x, {7, d}]];

failed = False;
Print[StringPadRight["workload", 26], " | CPU s  | GPU s  | CPU/GPU | same | GPU partial"];
Do[
  Module[{name = w[[1]], best = <||>, res = <||>, partial = False, ratio, same},
    If[!agreeOnly, Do[call[w, d], {d, {"CPU", "GPU"}}]];
    Do[Module[{ts = {}, r},
        Do[Module[{c = call[w, d]}, AppendTo[ts, c[[1]]]; r = c[[2]];
            If[d === "GPU" && c[[3]], partial = True]], {If[agreeOnly, 1, 3]}];
        best[d] = Min[ts]; res[d] = r], {d, {"CPU", "GPU"}}];
    ratio = best["CPU"]/best["GPU"];
    same = res["CPU"] === res["GPU"];
    If[(ratio < 1 && !agreeOnly) || !same || partial, failed = True];
    Print[StringPadRight[name, 26], " | ", fmt[best["CPU"], 3], " | ", fmt[best["GPU"], 3],
      " | ", fmt[ratio, 2], " | ", same, " | ", partial]],
  {w, workloads}];
Exit[If[failed, 1, 0]];
