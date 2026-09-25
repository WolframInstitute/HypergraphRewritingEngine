#!/usr/bin/env wolframscript
(* Evaluate every example in the documentation SOURCE against the local engine.

   WHAT THIS ANSWERS. The release checklist asks that the HGEvolve example pages evaluate
   cleanly. The pages are generated from docs/en/**/*.md, so the markdown fences
   are where an example is authored and where a wrong one is written down. Each fenced
   block is evaluated in order inside one shared context, because the blocks are written
   as a running session -- a later block reuses `rules` bound by an earlier one.

   WHAT COUNTS AS A FAILURE. Either the evaluation throws a message, or it returns a
   Failure / $Failed / an unevaluated HGEvolve[...]. An unevaluated head is the important
   one: a documented option name that HGEvolve does not accept comes back as an
   unevaluated expression rather than an error, so a check that only watched for messages
   would call it clean.

   A HINT IS CHECKED. A block may be followed by a comment <!-- => ... --> recording what it
   returns, for the author. The converter drops these comments, so nothing else compares them
   with the output, and a hint that went stale when the engine changed would stay on the page's
   source. Here the block's result, in InputForm with runs of whitespace collapsed, must equal
   the hint, or the part of the hint before a "; " that starts a note.

   A MESSAGE THE PAGE STATES IS ALLOWED. A hint of the form "... the message Sym::tag is issued"
   documents that the example raises that message (a page's Possible Issues section shows
   them); the block then passes if it raises exactly the messages its hint names.

   A GPU EXAMPLE THAT RAN ON THE CPU FAILS: HGEvolve::gpudev says the device was not available,
   so the page would show a result the device did not produce. HG_ALLOW_GPU_FALLBACK=1 reports
   it and continues, for a machine without the GPU worker.

   Each page's blocks share one context of their own, so a definition on one page is not
   visible on the next.

   Exit code is the number of failing blocks. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];
(* The gate's own symbols live in a context of their own, so no name used here can collide with
   a name an example defines. *)
Begin["VerifyDocExamples`"];

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

(* Fenced blocks, in file order, each with the hint that follows it (None when there is none).
   A fence line opens or closes; text between an odd and an even fence is code. *)
blocksOf[path_] := Module[{lines, out = {}, cur = {}, inside = False, n, j, hint},
  lines = StringSplit[Import[path, "Text"], "\n"];
  n = Length[lines];
  Do[
    With[{l = lines[[k]]},
      If[StringStartsQ[StringTrim[l], "```"],
        If[inside,
          j = k + 1;
          While[j <= n && StringTrim[lines[[j]]] === "", j++];
          hint = If[j <= n,
            StringCases[StringTrim[lines[[j]]], StartOfString ~~ "<!-- =>" ~~ h__ ~~ "-->" ~~ EndOfString :> StringTrim[h]],
            {}];
          AppendTo[out, {StringRiffle[cur, "\n"], If[hint === {}, None, First[hint]]}];
          cur = {}; inside = False,
          inside = True],
        If[inside, AppendTo[cur, l]]]],
    {k, n}];
  Select[out, StringTrim[First[#]] =!= "" &]];

collapse[text_String] := StringReplace[StringTrim[text], Whitespace -> " "];
hintMatchQ[res_, hint_String] := With[{got = collapse[ToString[res, InputForm, PageWidth -> Infinity]]},
  got === collapse[hint] ||
  (StringContainsQ[hint, "; "] && got === collapse[First[StringSplit[hint, "; ", 2]]])];
gpuFallbackOK = Environment["HG_ALLOW_GPU_FALLBACK"] === "1";

sources = Select[FileNames["*.md", FileNameJoin[{repo, "docs", "en"}], Infinity],
  ! StringContainsQ[#, ".generated"] &];
total = 0; failed = {};

Do[
  Module[{blocks = blocksOf[src], ctx = "DocExample`" <> FileBaseName[src] <> "`"},
    Print["\n=== ", FileNameTake[src], ": ", Length[blocks], " block(s) ==="];
    Do[
      Module[{code = blocks[[i, 1]], hint = blocks[[i, 2]], msgs = {}, firstMsg = None, res, bad, hintBad},
        total++;
        (* The handler fires for EVERY message, as Hold[Message[name, args], on], including ones
           that are switched off (on is False), which a user is not shown and this ignores.
           Nothing is wrapped in Quiet here for the same reason: quieting would make every
           message look switched off and the check would pass on anything. The name is taken
           held: extracted, Sym::tag evaluates to the message's text. *)
        res = Block[{$Context = ctx, $ContextPath = {ctx, "HypergraphRewriting`", "System`"}},
          Internal`HandlerBlock[{"Message",
              With[{nm = ToString[Extract[#, {1, 1}, HoldForm]]},
                If[#[[2]] =!= False,
                  AppendTo[msgs, nm];
                  If[firstMsg === None, firstMsg = ToString[Short[#, 3], InputForm]]]] &},
            Check[ToExpression[code], $Failed]]];
        (* An unevaluated HGEvolve head means the arguments did not match any definition. *)
        (* A block with a hint is judged by its hint, which may document $Failed. *)
        bad = (hint === None && (FailureQ[res] || res === $Failed)) ||
              (Head[res] === HypergraphRewriting`HGEvolve) ||
              !FreeQ[res, _HypergraphRewriting`HGEvolve];
        If[gpuFallbackOK && MemberQ[msgs, "HGEvolve::gpudev"],
          Print["WARN  block ", i, "  ran on the CPU (HGEvolve::gpudev)"];
          msgs = DeleteCases[msgs, "HGEvolve::gpudev"]];
        hintBad = StringQ[hint] && ! hintMatchQ[res, hint];
        (* The messages the hint says the block issues. *)
        With[{stated = If[StringQ[hint],
            StringCases[hint, "the message " ~~ m : (WordCharacter .. ~~ "::" ~~ WordCharacter ..) ~~ " is issued" :> m],
            {}]},
          If[stated =!= {} && Union[msgs] === Union[stated], msgs = {}]];
        If[bad || msgs =!= {} || hintBad,
          AppendTo[failed, {FileNameTake[src], i}];
          Print["FAIL  block ", i, "  ", StringTake[StringReplace[code, "\n" -> " "],
                                                    UpTo[70]]];
          If[msgs =!= {},
            Print["      ", Length[msgs], " message(s), distinct: ",
                  Take[Tally[msgs], UpTo[6]], "\n      first: ", firstMsg]];
          If[bad, Print["      returned: ", ToString[Short[res, 2]]]];
          If[hintBad, Print["      hint:     ", hint, "\n      returned: ",
                            collapse[ToString[res, InputForm, PageWidth -> Infinity]]]],
          Print["PASS  block ", i]]],
      {i, Length[blocks]}]],
  {src, sources}];

Print["\n============================================================"];
Print["Blocks evaluated: ", total];
Print["Failed: ", If[failed === {}, "NONE", failed]];
Exit[Length[failed]]
