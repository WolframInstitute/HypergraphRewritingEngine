(* Checks of the paclet's worker transport (paclet/Kernel/HypergraphRewriting.wl: hgWorkerStart,
   hgReadFrame, hgWorkerTry) against fake workers. A fake worker is a PowerShell process that
   writes a prepared byte stream of frames to its standard output and then sleeps; it never reads
   its input. The checks:
     startlog   a log frame written before the handshake reply reaches HGEvolve::enginelog
     buffer     200 calls to a worker that writes a log frame after each reply return the 200
                replies in order (bytes read past a reply are kept for the next call)
     abort      an abort during the handshake leaves no worker running
     silent     a worker that never answers the handshake is killed and the call returns $Failed
     retry      the call after a missed handshake starts a worker again
     cap        after $hgWorkerStartTries missed handshakes in a row a call starts no worker
     files      the byte files written for the fake workers are deleted at the end
   Usage, from wolframscript on Windows:
     hgTestPacletDir = "<paclet dir>"; Get["<this file>"]
   Prints TRANSPORT_OK, or TRANSPORT_FAIL and the failed checks, TRANSPORT_LOAD_FAIL when the
   paclet did not load, or TRANSPORT_SKIP off Windows. *)
PacletDirectoryLoad[Global`hgTestPacletDir];
Needs["HypergraphRewriting`"];
If[!MemberQ[$Packages, "HypergraphRewriting`"], Print["TRANSPORT_LOAD_FAIL"]; Exit[0]];
If[$OperatingSystem =!= "Windows", Print["TRANSPORT_SKIP: the fake workers need PowerShell"]; Exit[0]];

Begin["HypergraphRewriting`Private`"];

tcPowerShell = FileNameJoin[{Environment["SystemRoot"], "System32", "WindowsPowerShell", "v1.0",
  "powershell.exe"}];

(* One frame as the worker writes it: kind 0 (reply), 2^61 (log) or 2^63 (progress). *)
tcFrame[kind_Integer, payload_ByteArray] :=
  Join[ByteArray[Reverse[IntegerDigits[kind + Length[payload], 256, 8]]], payload];
tcText[s_String] := StringToByteArray[s];
tcReply[i_Integer] := ByteArray[Join[{82}, IntegerDigits[i, 256, 4]]];

(* A command that writes `bytes` to standard output and sleeps for 600 s. Its byte file is in
   tcFiles, which the check deletes at the end. *)
tcFiles = {};
tcWriter[bytes_ByteArray] := Module[{file, str},
  file = FileNameJoin[{$TemporaryDirectory,
    "hg_fake_worker_" <> ToString[$ProcessID] <> "_" <> ToString[RandomInteger[10^9]] <> ".bin"}];
  str = OpenWrite[file, BinaryFormat -> True];
  BinaryWrite[str, bytes];
  Close[str];
  AppendTo[tcFiles, file];
  {tcPowerShell, "-NoProfile", "-NonInteractive", "-Command",
   "$b=[IO.File]::ReadAllBytes('" <> file <> "'); $o=[Console]::OpenStandardOutput(); " <>
   "$o.Write($b,0,$b.Length); $o.Flush(); Start-Sleep -Seconds 600"}];
tcSilent = {tcPowerShell, "-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 600"};

tcRunning[] := Select[Processes[], Quiet[ProcessStatus[#]] === "Running" &];
tcReset[] := (hgWorkerKill["CPU"]; $hgWorkerStartFailures = <||>;);
tcFailed = {};
tcCheck[name_String, ok_] := If[!TrueQ[ok], AppendTo[tcFailed, name]];
tcJob = ByteArray[{1, 2, 3}];

(* startlog *)
tcReset[];
Block[{hgWorkerCommand},
  hgWorkerCommand[_] = tcWriter[Join[tcFrame[2^61, tcText["startup warning 7"]], tcFrame[0, tcReply[0]],
    tcFrame[0, tcReply[1]]]];
  tcData = EvaluationData[hgWorkerTry["CPU", tcJob]]];
tcCheck["startlog", tcData["Result"] === tcReply[1] &&
  AnyTrue[tcData["MessagesText"], StringContainsQ[#, "startup warning 7"] &]];
tcReset[];

(* buffer *)
Block[{hgWorkerCommand},
  hgWorkerCommand[_] = tcWriter[Join @@ Join[{tcFrame[0, tcReply[0]]},
    Table[Join[tcFrame[2^63, tcText["progress " <> ToString[i]]], tcFrame[0, tcReply[i]],
      tcFrame[2^61, tcText["after " <> ToString[i]]]], {i, 200}]]];
  (* The calls stop at the first wrong reply. *)
  tcReplies = {};
  Quiet[Do[
    AppendTo[tcReplies, TimeConstrained[hgWorkerTry["CPU", tcJob], 5, "TIMEOUT"]];
    If[Last[tcReplies] =!= tcReply[i], Break[]], {i, 200}]]];
tcCheck["buffer", tcReplies === Table[tcReply[i], {i, 200}]];
If[tcReplies =!= Table[tcReply[i], {i, 200}],
  Print["buffer: call ", Length[tcReplies], " gave ", Last[tcReplies]]];
tcReset[];

(* abort *)
tcBefore = Length[tcRunning[]];
Block[{hgWorkerCommand, $hgHandshakeSeconds = 60},
  hgWorkerCommand[_] = tcSilent;
  tcAbort = TimeConstrained[hgWorkerTry["CPU", tcJob], 3, "ABORTED"]];
Pause[0.5];
tcCheck["abort", tcAbort === "ABORTED" && Length[tcRunning[]] === tcBefore];
tcReset[];

(* silent, retry *)
tcBefore = Length[tcRunning[]];
Block[{hgWorkerCommand, $hgHandshakeSeconds = 2},
  hgWorkerCommand[_] = tcSilent;
  tcSilentResult = hgWorkerTry["CPU", tcJob]];
Pause[0.5];
tcCheck["silent", tcSilentResult === $Failed && Length[tcRunning[]] === tcBefore &&
  $hgWorkerStartFailures["CPU"] === 1];
tcRetry = hgWorkerTry["CPU", $hgHandshakeJob];
tcCheck["retry", ByteArrayQ[tcRetry] && !KeyExistsQ[$hgWorkerStartFailures, "CPU"] &&
  Quiet[ProcessStatus[$hgWorkerProc["CPU"]]] === "Running"];
tcReset[];

(* cap *)
Block[{hgWorkerCommand, $hgHandshakeSeconds = 0.5},
  hgWorkerCommand[_] = tcSilent;
  tcCapResults = Table[hgWorkerTry["CPU", tcJob], {$hgWorkerStartTries}];
  tcBefore = Length[Processes[]];
  tcCapTime = First[AbsoluteTiming[tcCapLast = hgWorkerTry["CPU", tcJob]]]];
tcCheck["cap", tcCapResults === Table[$Failed, {$hgWorkerStartTries}] && tcCapLast === $Failed &&
  Length[Processes[]] === tcBefore && $hgWorkerStartFailures["CPU"] === $hgWorkerStartTries];
tcReset[];

Scan[Quiet[KillProcess[#]] &, tcRunning[]];
Quiet[DeleteFile[tcFiles]];
tcCheck["files", NoneTrue[tcFiles, FileExistsQ]];
If[tcFailed === {}, Print["TRANSPORT_OK"], Print["TRANSPORT_FAIL: ", tcFailed]];

End[];
