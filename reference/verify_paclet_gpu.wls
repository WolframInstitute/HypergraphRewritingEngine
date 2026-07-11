#!/usr/bin/env wolframscript
(* Verify the GPU engine binary (hg_evolve_gpu) against the reference golden
   corpus, end to end through HGEvolve[..., TargetDevice -> "GPU"]. GPU counts
   must equal the CPU golden. Skips with a clear message when no GPU binary is
   present for this $SystemID (a CPU-only paclet build) -- run it on a box whose
   paclet was built with BUILD_GPU. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

gpuExe = HypergraphRewriting`Private`$HypergraphEngineBinaryGPU;
If[!(StringQ[gpuExe] && FileExistsQ[gpuExe]),
  Print["SKIP: no GPU engine binary (hg_evolve_gpu) for ", $SystemID,
        "; build the paclet with BUILD_GPU. (CPU golden covers correctness.)"];
  Exit[0]];
Print["GPU binary: ", gpuExe];

Get[FileNameJoin[{DirectoryName[$InputFileName], "golden_corpus.wl"}]];
cnt[x_] := Which[ListQ[x], Length[x], AssociationQ[x], Length[x], True, Missing[]];

totalChecks = 0; passChecks = 0; failed = {};
Do[
  Module[{c = hgGoldenCases[[ci]], res, a, e, rulesHG},
    e = c["expected"];
    rulesHG = (Rule @@ # &) /@ c["rules"];
    res = HGEvolve[rulesHG, c["init"], c["steps"],
      {"States", "Events", "CausalEdges", "BranchialEdges"},
      "CanonicalizeStates" -> "Full", "CanonicalizeEvents" -> "Full",
      "CausalTransitiveReduction" -> True, "TargetDevice" -> "GPU"];
    a = {cnt[res["States"]], cnt[res["Events"]], cnt[res["CausalEdges"]], cnt[res["BranchialEdges"]]};
    totalChecks++;
    If[a === e, passChecks++; Print["PASS  ", c["name"], "  ", a],
      AppendTo[failed, c["name"]];
      Print["FAIL  ", c["name"], "  expected ", e, "  actual ", a,
            "  {states,rawEvents,causal,branchial}"]
    ]
  ],
  {ci, Length[hgGoldenCases]}
];

Print["============================================================"];
Print["GPU cases passed: ", passChecks, " / ", totalChecks];
Print["Failed: ", If[failed === {}, "NONE", failed]];
