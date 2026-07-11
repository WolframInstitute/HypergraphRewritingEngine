#!/usr/bin/env wolframscript
(* Verify the freshly-built local paclet (HGEvolve) against the golden corpus.
   Checks the engine's observable outputs at the final step, with CanonicalizeStates->Full:
     States      == golden canonical states
     Events      == golden raw events (the "Events" property is the full-capture raw list)
     CausalEdges == golden causal (transitive reduction on)
     BranchialEdges == golden branchial
   Golden numbers are the Full-canon corpus from validate.wls / MultiwaySystem. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

(* Golden corpus (states, rawEvents, causal, branchial at the final step). *)
Get[FileNameJoin[{DirectoryName[$InputFileName], "golden_corpus.wl"}]];
cases = hgGoldenCases;

cnt[x_] := Which[ListQ[x], Length[x], AssociationQ[x], Length[x], True, Missing[]];

totalChecks = 0; passChecks = 0; failed = {};
Do[
  Module[{c = cases[[ci]], res, a, e, rulesHG},
    e = c["expected"];
    rulesHG = (Rule @@ # &) /@ c["rules"];  (* {lhs, rhs} -> (lhs -> rhs) *)
    res = HGEvolve[rulesHG, c["init"], c["steps"],
      {"States", "Events", "CausalEdges", "BranchialEdges"},
      "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full",
      "CausalTransitiveReduction" -> True];
    a = {cnt[res["States"]], cnt[res["Events"]], cnt[res["CausalEdges"]], cnt[res["BranchialEdges"]]};
    totalChecks++;
    If[a === e, passChecks++; Print["PASS  ", c["name"], "  ", a],
      AppendTo[failed, c["name"]];
      Print["FAIL  ", c["name"], "  expected ", e, "  actual ", a,
            "  {states,rawEvents,causal,branchial}"]
    ]
  ],
  {ci, Length[cases]}
];

Print["============================================================"];
Print["Cases passed: ", passChecks, " / ", totalChecks];
Print["Failed: ", If[failed === {}, "NONE", failed]];

(* TargetDevice consistency: "CPU" and "GPU" must produce the same
   States/Events/CausalEdges/BranchialEdges on a representative workload. The
   GPU device falls back to the CPU until the GPU binary is wired; once wired,
   this validates GPU == CPU end-to-end through the paclet. *)
Print["\n=== TargetDevice consistency ==="];
tdRules = (Rule @@ # &) /@ {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {2, 4}}}};
runTD[dev_] := Module[{r},
  r = HGEvolve[tdRules, {{1, 2}, {2, 3}}, 4,
    {"States", "Events", "CausalEdges", "BranchialEdges"},
    "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full",
    "CausalTransitiveReduction" -> True, "TargetDevice" -> dev];
  {cnt[r["States"]], cnt[r["Events"]], cnt[r["CausalEdges"]], cnt[r["BranchialEdges"]]}
];
tdCPU = runTD["CPU"]; tdGPU = Quiet[runTD["GPU"], HGEvolve::gpudev];
Print["CPU: ", tdCPU, "   GPU: ", tdGPU];
Print[If[tdCPU === tdGPU === {5, 33, 32, 43}, "PASS", "FAIL"],
      "  TargetDevice CPU == GPU == golden {5,33,32,43}"];
