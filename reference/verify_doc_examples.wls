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
   so the page would show a result the device did not produce. --allow-gpu-fallback reports
   it and continues, for a machine without the GPU worker.

   Each page's blocks share one context of their own, so a definition on one page is not
   visible on the next.

   A GRAPHICAL RESULT HAS ITS IMAGE. A block whose result contains a Graph, Graphics,
   Graphics3D or Image is followed in the markdown by the image line tools/doc_images.wl
   defines, naming docs/en/images/<Page>/<Page>-<n>.png, and that file exists. A block without
   one fails, and so does an image line after a block whose result is not graphical.
   --render-images rasterizes each graphical result as a notebook shows it (Rasterize,
   ImageResolution 144, white background), writes the PNG, writes or replaces the image line
   (its alt text is the first sentence of the paragraph before the block), removes image lines
   from blocks whose result is not graphical, and deletes PNGs in images/<Page>/ that no line
   names.
   ./build_docs.sh images runs this mode.

   Exit code is the number of failing blocks. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];
(* The gate's own symbols live in a context of their own, so no name used here can collide with
   a name an example defines. *)
Begin["VerifyDocExamples`"];
(* The whole run is one evaluation, so after three messages of one name General::stop would
   suppress the rest; a notebook evaluates each cell on its own and never reaches it. *)
Off[General::stop];

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

Get[FileNameJoin[{repo, "tools", "doc_images.wl"}]];
renderImages = MemberQ[Rest[$ScriptCommandLine], "--render-images"];

(* The page's lines, read and written as UTF-8 bytes so a rewrite changes only the image lines. *)
pageLines[path_] := StringSplit[ByteArrayToString[ReadByteArray[path], "UTF-8"], "\n", All];
writePageLines[path_, lines_] := Export[path, StringToByteArray[StringRiffle[lines, "\n"], "UTF-8"], "Binary"];

(* Fenced blocks, in file order. Each records its code, the hint that follows it (None when there
   is none), the line numbers of its opening and closing fences, the line number of the image line
   after it (None when there is none) and the paragraph before it. A fence line opens or closes;
   text between an odd and an even fence is code. Between the closing fence and the hint come blank
   lines and at most one image line. *)
blocksOf[path_] := Module[{lines, out = {}, cur = {}, inside = False, n, j, hint, open = 0, img, p, para},
  lines = pageLines[path];
  n = Length[lines];
  Do[
    With[{l = lines[[k]]},
      If[StringStartsQ[StringTrim[l], "```"],
        If[inside,
          j = k + 1;
          While[j <= n && StringTrim[lines[[j]]] === "", j++];
          img = None;
          If[j <= n && docImageLineQ[lines[[j]]],
            img = j; j++;
            While[j <= n && StringTrim[lines[[j]]] === "", j++]];
          hint = If[j <= n,
            StringCases[StringTrim[lines[[j]]], StartOfString ~~ "<!-- =>" ~~ h__ ~~ "-->" ~~ EndOfString :> StringTrim[h]],
            {}];
          p = open - 1;
          While[p >= 1 && StringTrim[lines[[p]]] === "", p--];
          para = {};
          While[p >= 1 && StringTrim[lines[[p]]] =!= "" &&
                ! StringStartsQ[StringTrim[lines[[p]]], "```" | "#" | "---" | "<!--" | "|" | "!["],
            PrependTo[para, lines[[p]]]; p--];
          (* A block that follows another block's hint, image or fence shares its paragraph. *)
          If[para === {} && p >= 1 && out =!= {} &&
             StringStartsQ[StringTrim[lines[[p]]], "```" | "<!--" | "!["],
            para = {out[[-1]]["Para"]}];
          AppendTo[out, <|"Code" -> StringRiffle[cur, "\n"], "Hint" -> If[hint === {}, None, First[hint]],
            "Open" -> open, "Close" -> k, "Image" -> img, "Para" -> StringRiffle[para, " "]|>];
          cur = {}; inside = False,
          inside = True; open = k],
        If[inside, AppendTo[cur, l]]]],
    {k, n}];
  Select[out, StringTrim[#Code] =!= "" &]];

graphicalQ[res_] := ! FreeQ[res, _Graph | _Graphics | _Graphics3D | _Image | _Image3D];

(* Alt text: the first sentence of the paragraph before the block in plain text, without the colon
   that introduces the block. *)
altText[para_String, page_String, i_Integer] := Module[{t},
  t = StringReplace[para, {"[" ~~ s : Shortest[Except["]"] ...] ~~ "]()" :> s,
    "<code>" | "</code>" | "*" | "`" | "[" | "]" -> ""}];
  t = StringTrim[StringReplace[t, Whitespace -> " "]];
  t = StringTrim[First[StringSplit[t, RegularExpression["(?<=\\.)\\s+(?=[A-Z])"], 2], ""]];
  t = StringTrim[StringReplace[t, ":" ~~ EndOfString -> ""]];
  If[t === "", page <> " example " <> IntegerString[i], t]];

collapse[text_String] := StringReplace[StringTrim[text], Whitespace -> " "];
hintMatchQ[res_, hint_String] := With[{got = collapse[ToString[res, InputForm, PageWidth -> Infinity]]},
  got === collapse[hint] ||
  (StringContainsQ[hint, "; "] && got === collapse[First[StringSplit[hint, "; ", 2]]])];
gpuFallbackOK = MemberQ[Rest[$ScriptCommandLine], "--allow-gpu-fallback"];

(* Page names (file base names) given after the script limit the run to those pages:
   wolframscript -file verify_doc_examples.wls HGEvolve SamplingAndPruning. With none, every page
   is evaluated. Arguments, not environment variables: a variable set in WSL does not reach the
   Windows kernel. *)
pageFilter = Select[Rest[$ScriptCommandLine], ! StringStartsQ[#, "--"] &];
enDir = FileNameJoin[{repo, "docs", "en"}];
sources = Select[FileNames["*.md", enDir, Infinity],
  ! StringContainsQ[#, ".generated"] &&
    (pageFilter === {} || MemberQ[pageFilter, FileBaseName[#]]) &];
total = 0; failed = {};

Do[
  Module[{blocks = blocksOf[src], page = FileBaseName[src], ctx = "DocExample`" <> FileBaseName[src] <> "`",
          up, edits = {}, lines, imgDir},
    up = StringRepeat["../", Length[FileNameSplit[DirectoryName[src]]] - Length[FileNameSplit[enDir]]];
    imgDir = FileNameJoin[{enDir, "images", page}];
    Print["\n=== ", FileNameTake[src], ": ", Length[blocks], " block(s) ==="];
    Do[
      Module[{code = blocks[[i]]["Code"], hint = blocks[[i]]["Hint"], img = blocks[[i]]["Image"],
              msgs = {}, firstMsg = None, res, bad, hintBad, stated, msgBad, graphical, rel, imgBad = None,
              raster},
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
            ToExpression[code]]];
        (* An unevaluated HGEvolve head means the arguments did not match any definition. *)
        (* A block with a hint is judged by its hint, which may document $Failed. *)
        bad = (hint === None && (FailureQ[res] || res === $Failed)) ||
              (Head[res] === HypergraphRewriting`HGEvolve) ||
              !FreeQ[res, _HypergraphRewriting`HGEvolve];
        If[gpuFallbackOK && MemberQ[msgs, "HGEvolve::gpudev"],
          Print["WARN  block ", i, "  ran on the CPU (HGEvolve::gpudev)"];
          msgs = DeleteCases[msgs, "HGEvolve::gpudev"]];
        hintBad = StringQ[hint] && ! hintMatchQ[res, hint];
        (* The messages the hint says the block issues; the block must issue exactly those. *)
        stated = If[StringQ[hint],
            StringCases[hint, "the message " ~~ m : (WordCharacter .. ~~ "::" ~~ WordCharacter ..) ~~ " is issued" :> m],
            {}];
        msgBad = Union[msgs] =!= Union[stated];
        graphical = graphicalQ[res];
        rel = docImageRel[page, i];
        Which[
          renderImages && graphical,
            raster = Quiet@Check[Rasterize[res, "Image", ImageResolution -> 144, Background -> White], $Failed];
            If[ImageQ[raster],
              If[! DirectoryQ[imgDir], CreateDirectory[imgDir, CreateIntermediateDirectories -> True]];
              Export[FileNameJoin[{enDir, rel}], raster, "PNG"];
              AppendTo[edits, {blocks[[i]]["Close"], img,
                "![" <> altText[blocks[[i]]["Para"], page, i] <> "](" <> up <> rel <> ")"}],
              imgBad = "Rasterize did not give an image"],
          renderImages,
            If[img =!= None, AppendTo[edits, {blocks[[i]]["Close"], img, None}]],
          graphical && img === None,
            imgBad = "graphical result without an image line",
          graphical && ! StringEndsQ[StringTrim[pageLines[src][[img]]], "](" <> up <> rel <> ")"],
            imgBad = "image line does not name " <> up <> rel,
          graphical && ! FileExistsQ[FileNameJoin[{enDir, rel}]],
            imgBad = "missing image file " <> rel,
          ! graphical && img =!= None,
            imgBad = "image line after a result that is not graphical"];
        If[bad || msgBad || hintBad || StringQ[imgBad],
          AppendTo[failed, {FileNameTake[src], i}];
          Print["FAIL  block ", i, "  ", StringTake[StringReplace[code, "\n" -> " "],
                                                    UpTo[70]]];
          If[msgBad,
            Print["      ", Length[msgs], " message(s), distinct: ",
                  Take[Tally[msgs], UpTo[6]], "; stated: ", stated, "\n      first: ", firstMsg]];
          If[bad, Print["      returned: ", ToString[Short[res, 2]]]];
          If[hintBad, Print["      hint:     ", hint, "\n      returned: ",
                            collapse[ToString[res, InputForm, PageWidth -> Infinity]]]];
          If[StringQ[imgBad], Print["      image:    ", imgBad]],
          Print["PASS  block ", i, If[graphical, "  (image)", ""]]]],
      {i, Length[blocks]}];
    (* Apply the image-line edits from the last block up, so earlier line numbers stay valid. An
       edit is {closing fence, existing image line or None, new line or None}. *)
    If[renderImages,
      lines = pageLines[src];
      Scan[Function[e, Which[
          e[[2]] =!= None && e[[3]] =!= None, lines[[e[[2]]]] = e[[3]],
          e[[2]] =!= None, lines = Delete[lines, If[StringTrim[lines[[e[[2]] - 1]]] === "",
                                                     {{e[[2]] - 1}, {e[[2]]}}, {{e[[2]]}}]],
          True, lines = Insert[lines, e[[3]], e[[1]] + 1]; lines = Insert[lines, "", e[[1]] + 1]]],
        ReverseSortBy[edits, First]];
      If[lines =!= pageLines[src], writePageLines[src, lines]; Print["wrote ", FileNameTake[src]]];
      With[{named = StringCases[StringRiffle[lines, "\n"],
                      "](" ~~ ("../" ..) ~~ "images/" ~~ page ~~ "/" ~~ r : (Except[")" | "/"] .. ~~ ".png") ~~ ")" :> r]},
        Scan[If[! MemberQ[named, FileNameTake[#]], DeleteFile[#]; Print["deleted ", #]] &,
          FileNames["*.png", imgDir]]]]],
  {src, sources}];

Print["\n============================================================"];
Print["Blocks evaluated: ", total];
Print["Failed: ", If[failed === {}, "NONE", failed]];
Exit[Length[failed]]
