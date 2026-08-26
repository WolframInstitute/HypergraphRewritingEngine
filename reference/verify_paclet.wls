#!/usr/bin/env wolframscript
(* Verify the freshly-built local paclet (HGEvolve) against the golden corpus.
   Checks the engine's observable outputs at the final step, with CanonicalizeStates->Full:
     States      == golden canonical states
     Events      == golden raw events (the "Events" property is the full-capture raw list)
     CausalEdges == golden causal (transitive reduction on)
     BranchialEdges == golden branchial
   Golden numbers are the Full-canon corpus from validate.wls / MultiwaySystem. *)

(* WHERE THE PACLET COMES FROM IS AN ARGUMENT, because the two sources answer different
   questions. The default, `paclet/` on disk, checks the ENGINE. Passing --archive
   installs the assembled dist/*.paclet and checks the thing a user actually receives:
   whether staging dropped a file the engine needs, whether the declared platform library
   is found by SystemID, whether the Kernel context loads from an installed location. A
   directory that works proves nothing about an archive built from a subset of it.

   --archive UNINSTALLS AFTERWARDS. It writes into the user's paclet directory, which is
   outside this repository, so it puts it back the way it found it. *)
repo = ParentDirectory[DirectoryName[$InputFileName]];
useArchive = MemberQ[Rest[$ScriptCommandLine], "--archive"];

If[useArchive,
  Module[{files},
    files = FileNames["*.paclet", FileNameJoin[{repo, "dist"}]];
    If[files === {},
      Print["FAIL: no .paclet in dist/. Run reference/build_paclet_archive.wls first."];
      Exit[1]];
    Print["source: ARCHIVE ", First[files]];
    (* An earlier install of the same version would shadow this one. *)
    Quiet[PacletUninstall["WolframInstitute/HypergraphRewriteEngine"]];
    installed = PacletInstall[First[files], ForceVersionInstall -> True];
    If[FailureQ[installed], Print["FAIL: PacletInstall gave ", installed]; Exit[1]];
    Print["installed: ", installed["Location"]];
  ],
  Print["source: DIRECTORY ", FileNameJoin[{repo, "paclet"}]];
  PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
  PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
];
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

If[useArchive,
  PacletUninstall["WolframInstitute/HypergraphRewriteEngine"];
  Print["\nuninstalled: the user's paclet directory is as it was found"]];

(* THE GATE MUST BE ABLE TO FAIL. Every case printed PASS or FAIL and appended to
   `failed`, and the script then exited 0 regardless -- its only Exit[1]s are for a
   missing archive and a failed PacletInstall. The CI step runs this file and reads
   its status, so a golden corpus disagreeing on every case was a green run, and
   "verify_paclet 12/12" was recorded as closure evidence from a script that could
   not report otherwise. *)
Exit[If[Length[failed] == 0, 0, 1]]
