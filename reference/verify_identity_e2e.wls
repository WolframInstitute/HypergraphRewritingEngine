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

(* The key list {"ConsumedEdges", "ProducedEdges", "Step"} is the reference oracle's
   eventSigAutomaticCanonical (reference/MultiwayReference.wl): the input and output states with
   the consumed and produced edges marked, and the event's own step. Equal counts at 3 steps on
   four rules, with and without ExploreFromCanonicalStatesOnly. *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];
markedCases = {
  {{{0, 1}, {1, 2}} -> {{0, 2}, {2, 1}}, {{0, 1}, {1, 2}}},
  {{{0, 1}, {2, 3}} -> {{0, 2}, {1, 3}}, {{0, 1}, {2, 3}}},
  {{{0, 1}, {1, 2}} -> {{0, 1}, {1, 3}, {3, 2}, {2, 0}}, {{0, 1}, {1, 2}}},
  {{{0, 1}} -> {{0, 2}, {2, 1}}, {{0, 1}}}};
markedRows = Table[
  With[{oracle = Last[MultiwayEvolve[{c[[1]]}, c[[2]], 3]["PerStep"]]["eventsAutomaticCanonical"],
        engine = Table[Quiet[HypergraphRewriting`HGEvolve[c[[1]], c[[2]], 3, "NumEvents",
          "CanonicalizeStates" -> Full,
          "CanonicalizeEvents" -> {"ConsumedEdges", "ProducedEdges", "Step"},
          "ExploreFromCanonicalStatesOnly" -> q]], {q, {False, True}}]},
    {oracle, engine}],
  {c, markedCases}];
Print["marked key list {oracle, {full capture, quotient}}: ", markedRows];
Print[If[AllTrue[markedRows, #[[2]] === {#[[1]], #[[1]]} &],
  "MARKED_KEYS_ORACLE_OK", "MARKED_KEYS_ORACLE_FAIL"]];
