(* Which event count is right for ByConsumedProducedEdges/Positional?

   The engine reports 8 for the binary-growth case under state mode Full and 6 under
   Automatic. Event identity is meant to be independent of the state-identity choice, so at
   most one of those is correct. The golden corpus pins only RAW events under Full, so it
   cannot settle it; the reference computes eventsAutomaticPositional directly, and that is
   the convention MultiwaySystem's CanonicalEventFunction -> Automatic defines.

   Reports the reference's counts under BOTH state canonicalization settings, since the
   reference's own axis independence is the property in question. *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

(* binary-growth, from reference/oracle_corpus.hpp: {{0,1}} -> {{0,2},{1,2}} *)
rules = {{{{0, 1}}, {{0, 2}, {1, 2}}}};
init  = {{0, 1}};
steps = 3;

report[label_, stateCanon_] := Module[{r},
  r = MultiwayEvolve[rules, init, steps,
        "StateCanonicalization" -> stateCanon,
        "EventCanonicalization" -> "None"];
  r = Last[r["PerStep"]];
  Print[label, ":"];
  Print["  states                    = ", r["states"]];
  Print["  eventsNone (raw)          = ", r["eventsNone"]];
  Print["  eventsStates              = ", r["eventsStates"]];
  Print["  eventsAutomaticPositional = ", r["eventsAutomaticPositional"]];
  Print["  eventsAutomaticCanonical  = ", r["eventsAutomaticCanonical"]];
  r
];

Print["reference/MultiwayReference.wl, binary-growth {{0,1}} -> {{0,2},{1,2}}, ", steps, " steps"];
Print[""];
a = report["StateCanonicalization -> Canonical (engine Full)", "Canonical"];
Print[""];
b = report["StateCanonicalization -> None    (engine None)", "None"];
Print[""];
Print["event counts independent of the state axis in the REFERENCE: ",
      If[a["eventsAutomaticPositional"] === b["eventsAutomaticPositional"]
         && a["eventsStates"] === b["eventsStates"], "YES", "NO"]];
