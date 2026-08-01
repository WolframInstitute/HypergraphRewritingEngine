#!/usr/bin/env wolframscript
(* Gap 1 of the quotient reconstruction (#4): which Automatic event count is right?

   Full capture and the quotient replay agree exactly on event identity under Full
   (endpoints-only) on every probed workload, and disagree under Automatic:

     wolfram-2to4  steps=4   full capture 87   replay 86
     WPP           steps=4   full capture 54   replay 52
     binary-growth steps=4   both 10 (agrees here; disagreed 8 vs 6 at another depth)

   The replay is a strict subset in every case. The structural difference is WHERE the canonical
   edge ranks are read: full capture reads them from each RAW state's own canonical labelling --
   states differing by an automorphism can assign the same orbit different ranks -- while the
   replay reads them from ONE pinned frame per canonical class. That is precisely the split
   between the reference's two Automatic conventions:

     eventsAutomaticPositional  ranks ordered by match/rhs position, NO automorphism quotient
     eventsAutomaticCanonical   consumed/produced marked as colors, marked graph canonicalized,
                                automorphisms QUOTIENTED

   So: print both, per step, for the three workloads. The engine's Full-identity count anchors
   the depth convention (eventsStates must equal it at the matching row); then whichever
   convention reproduces 87/54 is what full capture implements, and whichever reproduces 86/52
   is what the replay implements. SPEC 4.2 chooses Canonical on rigor grounds
   (positional identity is Sort[]-tie-break-defined in the upstream canonicalizer, not an
   invariant), so the adjudication decides which SIDE must move, not which is "the bug". *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

cases = {
  <|"name" -> "wolfram-2to4",
    "rule" -> {{{{1, 2}, {2, 3}}, {{1, 2}, {2, 4}, {4, 3}, {3, 1}}}},
    "init" -> {{1, 2}, {2, 3}},
    "engine" -> "steps=4: Full-identity 73, fullcap-Auto 87, replay-Auto 86"|>,
  <|"name" -> "WPP",
    "rule" -> {{{{1, 2}, {1, 3}}, {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}},
    "init" -> {{1, 2}, {1, 3}},
    "engine" -> "steps=4: Full-identity 49, fullcap-Auto 54, replay-Auto 52"|>,
  <|"name" -> "binary-growth",
    "rule" -> {{{{1, 2}}, {{1, 3}, {3, 2}}}},
    "init" -> {{1, 2}},
    "engine" -> "steps=4: Full-identity 4 (Auto agrees 10=10 here)"|>
};

Do[
  Module[{res},
    Print[c["name"], "   [engine: ", c["engine"], "]"];
    res = MultiwayEvolve[c["rule"], c["init"], 5, "FullCapture" -> True,
                         "StateCanonicalization" -> "Canonical"];
    Do[
      Print["  step ", row["step"],
            "  states ", row["states"],
            "  raw ", row["eventsNone"],
            "  Full ", row["eventsStates"],
            "  AutoPositional ", row["eventsAutomaticPositional"],
            "  AutoCanonical ", row["eventsAutomaticCanonical"]],
      {row, res["PerStep"]}
    ];
    Print[""];
  ],
  {c, cases}
];
