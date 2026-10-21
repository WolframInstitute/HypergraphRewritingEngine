#!/usr/bin/env wolframscript
(* Direct cross-check of our MultiwayReference against the authoritative
   Wolfram/Multicomputation MultiwaySystem paclet (the "Murzin" implementation),
   the same oracle the C++ engine was validated against.

   Per case and step we compare:
     states     : reference vs MultiwaySystem VertexCount[StatesGraph]
     eventsFull : reference vs MultiwaySystem events with CanonicalEventFunction->Full
     eventsAuto : reference vs MultiwaySystem events with CanonicalEventFunction->Automatic
     causal     : reference vs MultiwaySystem EdgeCount[CausalGraph]
     branchial  : reference vs MultiwaySystem undirected edges of EvolutionCausalBranchialGraph
*)

Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

(* --engine [pacletDir]: also compare HGEvolve's Automatic event counts (CanonicalizeStates -> Full)
   on the CPU and the GPU with MultiwaySystem's, per case and step. pacletDir defaults to the
   repository's paclet/. *)
engineArgs = Rest[$ScriptCommandLine];
useEngine = MemberQ[engineArgs, "--engine"];
If[useEngine,
  Module[{at = First[FirstPosition[engineArgs, "--engine"]], dir},
    dir = If[at < Length[engineArgs] && !StringStartsQ[engineArgs[[at + 1]], "--"], engineArgs[[at + 1]],
      FileNameJoin[{ParentDirectory[DirectoryName[$InputFileName]], "paclet"}]];
    PacletDirectoryLoad[dir];
    Needs["HypergraphRewriting`"]]];
engineAuto[rules_, init_, steps_, dev_, sm_ : Full] := Quiet[HGEvolve[rules, init, steps, "NumEvents",
  "CanonicalizeStates" -> sm, "CanonicalizeEvents" -> Automatic, "TargetDevice" -> dev]];

mwSystem =PacletSymbol["Wolfram/Multicomputation", "Wolfram`Multicomputation`MultiwaySystem"];

(* MultiwaySystem cumulative stats at `steps`, for a given event-canon function. *)
mwStats[rules_, init_, steps_, evCanon_] := Module[{multi, sg, cg, ecbg},
  multi = mwSystem[rules, {init}];
  sg = multi["StatesGraph", steps, "CanonicalStateFunction" -> "CanonicalHypergraph", "CanonicalEventFunction" -> evCanon];
  cg = multi["CausalGraph", steps, "CanonicalStateFunction" -> "CanonicalHypergraph", "CanonicalEventFunction" -> evCanon];
  ecbg = multi["EvolutionCausalBranchialGraph", steps, "CanonicalStateFunction" -> "CanonicalHypergraph", "CanonicalEventFunction" -> evCanon];
  <|
    "states" -> VertexCount[sg],
    "events" -> Length[Cases[VertexList[ecbg], _DirectedEdge, {1}]],
    "causal" -> EdgeCount[cg],
    "branchial" -> Length[Cases[EdgeList[ecbg], _UndirectedEdge]]
  |>
];

cases = {
  <|"name" -> "TestCase1", "rules" -> {{{{1,2}}, {{1,2},{2,3}}}}, "init" -> {{1,2}}, "steps" -> 4|>,
  <|"name" -> "TestCase2", "rules" -> {{{{1,2},{2,3}}, {{1,2},{2,3},{2,4}}}}, "init" -> {{1,2},{2,3},{3,1}}, "steps" -> 4|>,
  <|"name" -> "TestCase3", "rules" -> {{{{1,2,3}}, {{1,2},{1,3},{1,4}}}}, "init" -> {{1,2,3}}, "steps" -> 4|>,
  <|"name" -> "TestCase4", "rules" -> {{{{1,2,3}}, {{1,2},{1,3},{1,4}}}, {{{1,2}}, {{1,2},{1,3}}}}, "init" -> {{1,2,3}}, "steps" -> 4|>,
  <|"name" -> "TestCase5", "rules" -> {{{{1,2,3}}, {{1,2},{1,3},{1,4}}}, {{{1,2},{1,3}}, {{1,2},{1,3},{2,4}}}}, "init" -> {{1,2,3}}, "steps" -> 4|>,
  <|"name" -> "TestCase6", "rules" -> {{{{1,2},{2,3}}, {{1,2},{2,3},{2,4}}}}, "init" -> {{1,2},{2,3}}, "steps" -> 4|>,
  <|"name" -> "TestCase7", "rules" -> {{{{1,2},{2,3}}, {{1,2},{2,3},{2,4}}}}, "init" -> {{1,1},{1,1}}, "steps" -> 4|>,
  <|"name" -> "TestCase8", "rules" -> {{{{1,2},{2,3}}, {{4,1},{1,4},{2,3},{4,3}}}}, "init" -> {{1,2},{2,3}}, "steps" -> 3|>,
  <|"name" -> "TestCase9", "rules" -> {{{{1,2},{2,3}}, {{4,1},{1,4},{2,3},{4,3}}}}, "init" -> {{1,1},{1,1}}, "steps" -> 3|>,
  <|"name" -> "TestCase10", "rules" -> {{{{1,2},{2,3}}, {{1,3},{2,3},{3,4}}}}, "init" -> {{1,2},{2,3}}, "steps" -> 5|>,
  <|"name" -> "TestCase11", "rules" -> {{{{1,2},{2,3}}, {{1,3},{2,3},{3,4}}}}, "init" -> {{1,1},{1,1}}, "steps" -> 4|>,
  <|"name" -> "TestCase12", "rules" -> {{{{1,2,3},{5,1}}, {{1,5,6},{3,2},{3,5}}}}, "init" -> {{0,0,0},{0,0}}, "steps" -> 5|>,
  (* two rules on a state with a nontrivial automorphism group: the Automatic count depends on
     which automorphic labelling ranks the edges *)
  <|"name" -> "IdentityAndSwap", "rules" -> {{{{1},{2}}, {{1},{2}}}, {{{1},{2}}, {{2},{1}}}}, "init" -> {{1},{2}}, "steps" -> 3|>,
  <|"name" -> "SwapAndFresh", "rules" -> {{{{1},{2}}, {{2},{3}}}, {{{1},{2}}, {{2},{1}}}}, "init" -> {{3},{2}}, "steps" -> 2|>,
  (* MultiwaySystem gives the produced edge the consumed edge's token, and its two step-2 events
     here differ only in those tokens *)
  <|"name" -> "LoopToEdge", "rules" -> {{{{1,1}}, {{1,2}}}}, "init" -> {{1,1},{1,1}}, "steps" -> 2|>,
  (* reference/oracle_corpus.hpp workloads whose None-state Automatic rows are in golden_matrix.txt *)
  <|"name" -> "BinaryGrowth", "rules" -> {{{{1,2}}, {{1,3},{2,3}}}}, "init" -> {{1,2}}, "steps" -> 3|>,
  <|"name" -> "DisconnectedLhs", "rules" -> {{{{1,2},{3,4}}, {{1,3},{2,4}}}}, "init" -> {{1,2},{3,4}}, "steps" -> 3|>,
  <|"name" -> "Cycle4Automorphic", "rules" -> {{{{1,2},{2,3}}, {{1,2},{2,4},{4,3}}}}, "init" -> {{1,2},{2,3},{3,4},{4,1}}, "steps" -> 3|>,
  <|"name" -> "Star4Automorphic", "rules" -> {{{{1,2}}, {{1,2},{2,3}}}}, "init" -> {{1,2},{1,3},{1,4},{1,5}}, "steps" -> 2|>
};

mwRules[r_] := (#[[1]] -> #[[2]]) & /@ r;

agree = 0; disagree = 0; engineRows = 0; engineCpuAgree = 0; engineGpuAgree = 0;
engineRawCpuAgree = 0; engineRawGpuAgree = 0;
Do[
  Module[{c = cases[[ci]], ref, nsteps},
    nsteps = c["steps"];
    ref = MultiwayEvolve[c["rules"], c["init"], nsteps];
    Print["==== ", c["name"], " ===="];
    Do[
      Module[{rs = ref["PerStep"][[k]], full, auto, cmp},
        full = mwStats[mwRules[c["rules"]], c["init"], k, Full];
        auto = mwStats[mwRules[c["rules"]], c["init"], k, Automatic];
        cmp = {
          {"states", rs["states"], full["states"]},
          {"eventsFull", rs["eventsStates"], full["events"]},
          {"eventsAuto", rs["eventsAutomaticPositional"], auto["events"]},
          {"causal", rs["causal"], full["causal"]},
          {"branchial", rs["branchial"], full["branchial"]}
        };
        Do[
          With[{nm = row[[1]], a = row[[2]], b = row[[3]]},
            If[a === b, agree++,
              disagree++;
              Print["  step ", k, "  ", nm, "  ref=", a, "  MultiwaySystem=", b, "  MISMATCH"]
            ]
          ],
          {row, cmp}
        ];
        If[useEngine,
          (* Full states read ranks in the class frame, None states in each raw state *)
          Module[{cpu = engineAuto[mwRules[c["rules"]], c["init"], k, "CPU"],
                  gpu = engineAuto[mwRules[c["rules"]], c["init"], k, "GPU"],
                  rawCpu = engineAuto[mwRules[c["rules"]], c["init"], k, "CPU", None],
                  rawGpu = engineAuto[mwRules[c["rules"]], c["init"], k, "GPU", None]},
            engineRows++;
            If[cpu === auto["events"], engineCpuAgree++];
            If[gpu === auto["events"], engineGpuAgree++];
            If[rawCpu === auto["events"], engineRawCpuAgree++];
            If[rawGpu === auto["events"], engineRawGpuAgree++];
            If[Length[DeleteDuplicates[{auto["events"], cpu, gpu, rawCpu, rawGpu}]] > 1,
              Print["  step ", k, "  eventsAuto  MultiwaySystem=", auto["events"], "  Full states: CPU=",
                cpu, " GPU=", gpu, "  None states: CPU=", rawCpu, " GPU=", rawGpu]]]]
      ],
      {k, nsteps}
    ];
    Print["  (", c["name"], " checked)"]
  ],
  {ci, Length[cases]}
];

Print["============================================================"];
Print["Reference vs MultiwaySystem: ", agree, " agree, ", disagree, " disagree (across all dims/steps/cases)"];
If[useEngine,
  Print["HGEvolve Automatic events vs MultiwaySystem, Full states: CPU ", engineCpuAgree, " / ",
    engineRows, ", GPU ", engineGpuAgree, " / ", engineRows, "; None states: CPU ",
    engineRawCpuAgree, " / ", engineRows, ", GPU ", engineRawGpuAgree, " / ", engineRows,
    " (case steps)"]];
