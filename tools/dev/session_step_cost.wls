#!/usr/bin/env wolframscript
(* WHERE A SESSION STEP'S TIME GOES, split three ways, because the first split was not enough.

   Board #134 proposed delivering only what a Step ADDED. The first measurement here timed a Step
   against a Query at the same depth -- Query does the same serialisation and no exploration --
   and found query/step ~1.0 at every depth past the first. I read that as "the cost is
   serialisation and transport". It is not: a Query ALSO builds the Wolfram Graph from the reply,
   so that ratio bounds evolution against everything-else, and everything-else has two parts.

   So this splits a Step into three measured pieces:

     engine_ms   the engine call itself: WXF out, evolve, marshal, WXF back, deserialise.
                 Timed on hgSendJob, below the property interpretation.
     build_ms    turning the reply into the requested property -- for a *Graph property, the
                 Graph construction, which is the rest of the Step.
     step_ms     the whole call as a caller sees it.

   A DELIVERY DELTA CAN ONLY ATTACK engine_ms. It sends fewer bytes; it cannot make the caller's
   Graph smaller, because the caller's graph is the whole accumulated one either way. So
   engine_ms/step_ms is the ceiling on what delta delivery buys, and reading it is what stops
   "the reply is 24.8 MB" from being mistaken for "the reply is the cost".

   Usage: wolframscript -file tools/dev/session_step_cost.wls [property] [maxDepth] [delivery]
     delivery  "Full" (default) or "Delta"
*)

repo = ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

args = Rest[$ScriptCommandLine];
prop     = If[Length[args] >= 1, args[[1]], "StatesGraphStructure"];
maxDepth = If[Length[args] >= 2, ToExpression[args[[2]]], 6];
delivery = If[Length[args] >= 3, args[[3]], "Full"];

rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
init  = {{1, 2}};

(* The engine call is timed WHERE IT HAPPENS: hgSendJob records $hgLastEngineTime and
   $hgLastReplyBytes on every job. Timing it from outside would have to subtract two noisy
   numbers, and a wrapper redefined from here silently never fires -- the package's own
   definition is more specific and matches first, which is exactly how the first version of this
   script reported 0.0 ms for every engine call while the totals were right. *)
engineTime := HypergraphRewriting`Private`$hgLastEngineTime;
replyBytes := HypergraphRewriting`Private`$hgLastReplyBytes;

Print["property: ", prop, "   delivery: ", delivery, "   depths 1..", maxDepth];
Print[StringRiffle[{"depth", "states", "step_ms", "engine_ms", "build_ms", "engine/step",
                    "reply_bytes"}, "\t"]];

s = HGSessionOpen[rules, init, prop];
If[Head[s] =!= HGSessionObject, Print["FAIL: no session"]; Exit[1]];

Do[
  Module[{tStep, g, n},
    tStep = First[AbsoluteTiming[
      g = HGSessionStep[s, 1, Automatic, "Delivery" -> delivery]]];
    n = If[GraphQ[g], VertexCount[g], g];
    Print[StringRiffle[{d, n, Round[tStep*1000, 0.1], Round[engineTime*1000, 0.1],
                        Round[(tStep - engineTime)*1000, 0.1],
                        Round[engineTime/tStep, 0.001], replyBytes}, "\t"]]],
  {d, 1, maxDepth}];

HGSessionClose[s];
Print["\nengine/step is the fraction a delivery delta can attack: it shrinks the reply and "];
Print["cannot shrink the caller's graph, which is the whole accumulated one either way."];
