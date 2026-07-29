#!/usr/bin/env wolframscript
(* Which Automatic event count is right?

   The CPU and the GPU disagree on EventCanonicalizationMode::Automatic for one workload:
   the directed 4-cycle under {{x,y},{y,z}} -> {{x,y},{y,w},{w,z}}, 3 steps, full capture,
   state identity Full. 144 applications; CPU reports 15 distinct events, GPU reports 19.
   None (144) and Full (3) agree on both.

   MultiwayReference computes BOTH Automatic conventions side by side, so it can say which
   engine matches which:

     eventsAutomaticPositional  ranks kept ORDERED by match/rhs position; does NOT quotient
                                state automorphisms, so symmetric edge-role assignments stay
                                distinct. This is what the MultiwaySystem paclet does.
     eventsAutomaticCanonical   consumed/produced edges marked as edge colors and the marked
                                hypergraph canonicalized; DOES quotient automorphisms.

   A 4-cycle has a rotation group of order 4, so this is exactly the workload on which the two
   conventions separate -- which is why the engines could disagree here and nowhere else.

   Also run on the rigid path, where the two conventions must coincide and both engines already
   agree, so a difference there would mean the discrepancy is not about automorphisms at all. *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

rule = {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 4}, {4, 3}}}};

cases = {
  <|"name" -> "cycle4 (rotation group, order 4)",
    "init" -> {{1, 2}, {2, 3}, {3, 4}, {4, 1}}|>,
  <|"name" -> "path3 (trivial automorphism group)",
    "init" -> {{1, 2}, {2, 3}, {3, 4}}|>
};

Print["engine measurement, cycle4: applications 144, None 144, Full 3, CPU-Automatic 15, GPU-Automatic 19"];
Print[""];

Do[
  Module[{res, row},
    res = MultiwayEvolve[rule, c["init"], 3, "FullCapture" -> True,
                         "StateCanonicalization" -> "Canonical"];
    row = Last[res["PerStep"]];
    Print[c["name"]];
    Print["  states                     ", row["states"]];
    Print["  eventsNone   (raw)         ", row["eventsNone"]];
    Print["  eventsStates (Full)        ", row["eventsStates"]];
    Print["  eventsAutomaticPositional  ", row["eventsAutomaticPositional"]];
    Print["  eventsAutomaticCanonical   ", row["eventsAutomaticCanonical"]];
    Print[""];
  ],
  {c, cases}
];
