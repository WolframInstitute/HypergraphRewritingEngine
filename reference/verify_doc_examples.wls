#!/usr/bin/env wolframscript
(* Evaluate every example in the documentation SOURCE against the local engine.

   WHAT THIS ANSWERS. The release checklist asks that the HGEvolve example pages evaluate
   cleanly. The pages are generated from Documentation/Source/*.md, so the markdown fences
   are where an example is authored and where a wrong one is written down. Each fenced
   block is evaluated in order inside one shared context, because the blocks are written
   as a running session -- a later block reuses `rules` bound by an earlier one.

   WHAT COUNTS AS A FAILURE. Either the evaluation throws a message, or it returns a
   Failure / $Failed / an unevaluated HGEvolve[...]. An unevaluated head is the important
   one: a documented option name that HGEvolve does not accept comes back as an
   unevaluated expression rather than an error, so a check that only watched for messages
   would call it clean.

   Exit code is the number of failing blocks. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

(* WARM THE KERNEL BEFORE ANY BLOCK IS JUDGED.
   The FIRST call into the package in a fresh kernel drags in the paclet manager, the stream
   method table and the front-end probe, and offline they emit a pile of messages that have
   nothing to do with the documentation: URLFetch::offline, URLSubmit::offline,
   RemoveInputStreamMethod::name, MIMETypeToFormatList::fmterr, DeleteFile::fdnfnd,
   RenameFile::eexist, Extract::keyw, "A front end is not available". Measured: that call
   returns a Graph with failureQ False, and the IDENTICAL call immediately after it in the same
   kernel emits NOTHING. So the messages are startup, not the example.
   This is why two GettingStarted blocks were recorded as known failures. They were never
   failures -- the gate was reporting the kernel's own first-use noise, and a gate that fails on
   something the code did not do teaches its reader to ignore it.
   Warming here rather than quieting there keeps the check strict: after this line, any message a
   block emits is the block's. *)
Quiet@Check[HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 1], Null];
(* And the device path separately: its first call spawns a process, which touches the stream and
   MIME tables and emits MIMETypeToFormatList::fmterr the once. On a machine with no device this
   returns unevaluated or $Failed, which Quiet@Check absorbs -- warming a path that is not there
   costs nothing and judging a block on its absence would be wrong anyway. *)
Quiet@Check[HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 1, "NumStates",
                     "TargetDevice" -> "GPU"], Null];

(* Fenced blocks, in file order. A fence line opens or closes; text between an odd and an
   even fence is code. Blocks tagged with a non-Wolfram language are skipped. *)
blocksOf[path_] := Module[{lines, out = {}, cur = {}, inside = False},
  lines = StringSplit[Import[path, "Text"], "\n"];
  Do[
    If[StringStartsQ[StringTrim[l], "```"],
      If[inside, AppendTo[out, StringRiffle[cur, "\n"]]; cur = {}; inside = False,
                 inside = True],
      If[inside, AppendTo[cur, l]]],
    {l, lines}];
  Select[out, StringTrim[#] =!= "" &]];

sources = FileNames["*.md", FileNameJoin[{repo, "paclet", "Documentation", "Source"}]];
total = 0; failed = {};

Do[
  Module[{blocks = blocksOf[src]},
    Print["\n=== ", FileNameTake[src], ": ", Length[blocks], " block(s) ==="];
    Do[
      Module[{code = blocks[[i]], msgs = {}, res, bad},
        total++;
        (* The handler fires for EVERY message, including ones that are switched off --
           those arrive wrapped in $Off[...] and are by definition not shown to a user, so
           they are not what this is looking for. Nothing is wrapped in Quiet here for the
           same reason: quieting would make every message look switched-off and the check
           would pass on anything. *)
        res = Block[{$Context = "DocExample`", $ContextPath = {"DocExample`",
                     "HypergraphRewriting`", "System`"}},
          Internal`HandlerBlock[{"Message",
              With[{nm = ToString[#[[1, 1]]]},
                If[!StringStartsQ[nm, "$Off"], AppendTo[msgs, nm]]] &},
            Check[ToExpression[code], $Failed]]];
        (* An unevaluated HGEvolve head means the arguments did not match any definition. *)
        bad = FailureQ[res] || res === $Failed ||
              (Head[res] === HypergraphRewriting`HGEvolve) ||
              !FreeQ[res, _HypergraphRewriting`HGEvolve];
        If[bad || msgs =!= {},
          AppendTo[failed, {FileNameTake[src], i}];
          Print["FAIL  block ", i, "  ", StringTake[StringReplace[code, "\n" -> " "],
                                                    UpTo[70]]];
          If[msgs =!= {},
            Print["      ", Length[msgs], " message(s), distinct: ",
                  Take[Tally[msgs], UpTo[6]]]];
          If[bad, Print["      returned: ", ToString[Short[res, 2]]]],
          Print["PASS  block ", i]]],
      {i, Length[blocks]}]],
  {src, sources}];

Print["\n============================================================"];
Print["Blocks evaluated: ", total];
Print["Failed: ", If[failed === {}, "NONE", failed]];
Exit[Length[failed]]
