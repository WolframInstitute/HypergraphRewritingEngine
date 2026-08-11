#!/usr/bin/env wolframscript
(* WHAT A SESSION STEP COSTS, split into the two things a delivery delta would and would not fix.

   Board #134 proposes delivering only what a Step ADDED, on the ground that a Step re-serialises
   the whole accumulated graph -- 285,894 B for Structure and 2,633,992 B styled, measured at
   214d1c6, re-sent every call. That measurement is of the PAYLOAD. It says nothing about what
   fraction of a Step's wall time the payload is, and a delta that removes 90% of the bytes buys
   nothing if the bytes are 3% of the call.

   So this measures both, per step, at growing depth:
     - the wall time of the Step as the caller experiences it
     - the wall time of a Query at the same depth, which does the SAME serialisation and NO
       exploration. Query is the instrument: it isolates marshal + transport + deserialise from
       the evolution, without needing a profiler inside the engine.
     - the bytes the reply carried (ByteCount of the deserialised result is a proxy; the wire
       size is reported by the engine's own HG_FFI_PAYLOAD_FILE where a caller wants exact).

   Query/Step is then the fraction of a step a delivery delta could attack. A delta cannot make
   a Query free -- it still walks the graph to decide what is new -- so this is an upper bound on
   the win, which is the number that decides whether to build it.

   Usage: wolframscript -file tools/dev/session_step_cost.wls [property] [maxDepth]
*)

repo = ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

args = Rest[$ScriptCommandLine];
prop = If[Length[args] >= 1, args[[1]], "StatesGraphStructure"];
maxDepth = If[Length[args] >= 2, ToExpression[args[[2]]], 6];

rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
init  = {{1, 2}};

Print["property: ", prop, "   depths 1..", maxDepth];
Print[StringRiffle[{"depth", "states", "step_ms", "query_ms", "query/step", "reply_bytes"}, "\t"]];

s = HGSessionOpen[rules, init, {prop, "NumStates"}];
If[Head[s] =!= HGSessionObject, Print["FAIL: no session"]; Exit[1]];

Do[
  Module[{tStep, tQuery, r, bytes, n},
    tStep  = First[AbsoluteTiming[r = HGSessionStep[s, 1]]];
    tQuery = First[AbsoluteTiming[HGSessionQuery[s]]];
    (* Counts come from the same reply, so the row describes one graph rather than two. *)
    n = HGSessionQuery[s, "NumStates"];
    bytes = ByteCount[r];
    Print[StringRiffle[{d, n, Round[tStep*1000, 0.1], Round[tQuery*1000, 0.1],
                        Round[tQuery/tStep, 0.001], bytes}, "\t"]]],
  {d, 1, maxDepth}];

HGSessionClose[s];
Print["\nquery/step is the fraction of a Step that is serialisation + transport, and therefore "];
Print["the UPPER BOUND on what a delivery delta can remove -- a delta still walks the graph."];
