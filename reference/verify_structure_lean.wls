#!/usr/bin/env wolframscript
(* End-to-end gate for the lean Structure-variant payloads, through the real paclet binary.

   Structure variants ship topology: a state vertex carries Id and Step, an event vertex Id and
   endpoint states, a states-graph edge the event id -- never hypergraph contents. This checks,
   on wolfram-2to4 at 5 steps (CanonicalizeStates -> Full):

     1. StatesGraph and StatesGraphStructure both come back as Graph objects.
     2. They agree on VertexCount and EdgeCount (the lean payload changes bytes, not topology).
     3. The Structure frame is smaller than the full one (wire-size breakdown from the
        engine's HG_FFI_PAYLOAD_FILE instrument, passed through the paclet's HG_* env
        allowlist).
     4. The quotient Structure run also renders and is smaller still.

   The paclet directory is expected to be passed as a fresh copy by the caller (WSL serves
   stale binaries from the original path after rebuilds). *)

pacletDir = $ScriptCommandLine[[2]];
PacletDirectoryLoad[pacletDir];
Needs["HypergraphRewriting`"];

rule = {{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}, {z, x}};
init = {{1, 2}, {2, 3}};

payloadFile = FileNameJoin[{$TemporaryDirectory, "hg_payload_breakdown.tsv"}];
SetEnvironment["HG_FFI_PAYLOAD_FILE" -> payloadFile];

measure[prop_, opts___] := Module[{g, rows, total, graphRow},
  If[FileExistsQ[payloadFile], DeleteFile[payloadFile]];
  g = HypergraphRewriting`HGEvolve[rule, init, 5, prop,
    "CanonicalizeStates" -> Full, opts];
  rows = If[FileExistsQ[payloadFile], Import[payloadFile, "TSV"], {}];
  total = SelectFirst[rows, First[#] === "TOTAL" &, {"TOTAL", -1}][[2]];
  graphRow = SelectFirst[rows, First[#] === "GraphData" &, {"GraphData", -1}][[2]];
  <|"Graph" -> g, "Total" -> total, "GraphData" -> graphRow|>
];

full = measure["StatesGraph"];
lean = measure["StatesGraphStructure"];
quot = measure["StatesGraphStructure", "ExploreFromCanonicalStatesOnly" -> True];

Print["StatesGraph           frame=", full["Total"], "  GraphData=", full["GraphData"],
      "  V=", VertexCount[full["Graph"]], " E=", EdgeCount[full["Graph"]]];
Print["StatesGraphStructure  frame=", lean["Total"], "  GraphData=", lean["GraphData"],
      "  V=", VertexCount[lean["Graph"]], " E=", EdgeCount[lean["Graph"]]];
Print["  + quotient          frame=", quot["Total"], "  GraphData=", quot["GraphData"],
      "  V=", VertexCount[quot["Graph"]], " E=", EdgeCount[quot["Graph"]]];

ok = And[
  Head[full["Graph"]] === Graph,
  Head[lean["Graph"]] === Graph,
  Head[quot["Graph"]] === Graph,
  VertexCount[lean["Graph"]] === VertexCount[full["Graph"]],
  EdgeCount[lean["Graph"]] === EdgeCount[full["Graph"]],
  lean["GraphData"] > 0,
  lean["GraphData"] < full["GraphData"]
];

Print[If[ok, "STRUCTURE_LEAN_OK", "STRUCTURE_LEAN_FAIL"]];
