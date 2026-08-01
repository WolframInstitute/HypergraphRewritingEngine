#!/usr/bin/env wolframscript
(* End-to-end gate for the event-identity surface, through the REAL paclet DLL.

   Checks, on wolfram-2to4 at 4 steps (full capture unless stated):
     CanonicalizeEvents -> Automatic      NumEvents == 86   (the authority's count)
     CanonicalizeEvents -> "Positional"   NumEvents == 87   (this engine's positional identity)
     Automatic + ExploreFromCanonicalStatesOnly            == 86   (same observable, cheaper)

   The paclet directory is expected to be passed as a fresh copy by the caller (WSL serves a
   stale DLL from the original path after rebuilds). *)

pacletDir = $ScriptCommandLine[[2]];
PacletDirectoryLoad[pacletDir];
Needs["HypergraphRewriting`"];

rule = {{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}, {z, x}};
init = {{1, 2}, {2, 3}};

run[opts___] := HypergraphRewriting`HGEvolve[rule, init, 4, "NumEvents",
  "CanonicalizeStates" -> Full, opts];

auto = run["CanonicalizeEvents" -> Automatic];
pos  = run["CanonicalizeEvents" -> "Positional"];
quot = run["CanonicalizeEvents" -> Automatic, "ExploreFromCanonicalStatesOnly" -> True];

Print["Automatic  NumEvents = ", auto, "  (expect 86)"];
Print["Positional NumEvents = ", pos,  "  (expect 87)"];
Print["Auto+quot  NumEvents = ", quot, "  (expect 86)"];

If[auto === 86 && pos === 87 && quot === 86,
  Print["IDENTITY_E2E_OK"],
  Print["IDENTITY_E2E_FAIL"]];
