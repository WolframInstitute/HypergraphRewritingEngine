(* reference/verify_state_cost.wls

   "StateCost" against an independent computation (docs/SPEC.md section 5, StateCost). Each case
   evolves under "CanonicalizeStates" -> Full and "MaxStatesPerStep" -> N with a cost over
   "VertexCount", "EdgeCount", "MaxDegree" and "TwoSectionEdgeCount". For every step s, the
   successor classes of the classes the engine kept at step s come from MultiwayEvolve
   (reference/MultiwayReference.wl), one step from each, grouped by CanonicalForm; their costs
   come from the definitions of the statistics in Wolfram Language. The classes the engine kept at
   step s + 1 must be min(N, successor classes) of those successors, and each must cost no more
   than the N-th lowest successor cost. Rules that add a vertex at every rewrite, so no class
   recurs at a later step. Run on the CPU, and on the GPU when its engine is present.

   Run: wolframscript -file reference/verify_state_cost.wls [cpu]   ("cpu": the CPU only)
   Output: one PASS/FAIL line per case and device, then STATE COST VERIFIED or FAILED. *)

PacletDirectoryLoad[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}]];
Needs["HypergraphRewriting`"];
Get[FileNameJoin[{DirectoryName[$InputFileName], "MultiwayReference.wl"}]];

(* The statistics of a state given as its edges (vertex lists), by their definitions. *)
stats[vs_List] := Module[{flat = Flatten[vs], degs},
  degs = Values[Counts[flat]];
  <|"VertexCount" -> Length[Union[flat]], "EdgeCount" -> Length[vs],
    "MaxDegree" -> If[degs === {}, 0, Max[degs]],
    "TwoSectionEdgeCount" ->
      Length[DeleteDuplicates[Sort /@ Catenate[Subsets[DeleteDuplicates[#], {2}] & /@ vs]]]|>];

(* The successor classes of the states `parents` (vertex lists) under `rules`, as
   class -> vertex lists. *)
successors[rules_, parents_] := Module[{out = <||>},
  Do[
    With[{r = MultiwayEvolve[List @@@ rules, p, 1, "StateCanonicalization" -> "None"]},
      Do[If[ev["step"] == 1,
          With[{c = ev["outContents"]}, out[First[CanonicalForm[c]]] = c]],
        {ev, r["Events"]}]],
    {p, parents}];
  out];

cases = {
  {"growth", {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 4},
  {"two-rules", {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {2, 4}}},
    {{1, 2}, {2, 3}, {3, 1}}, 4},
  {"ternary", {{1, 2, 3}} -> {{1, 2, 4}, {4, 3, 5}}, {{1, 1, 1}, {1, 2, 3}}, 4}
};
costs = {
  {"vertices-degree", #VertexCount - 0.5 #MaxDegree &,
    Function[s, s["VertexCount"] - 0.5 s["MaxDegree"]]},
  {"pairs", Abs[#TwoSectionEdgeCount - 2 #EdgeCount] + 0.25 #MaxDegree &,
    Function[s, Abs[s["TwoSectionEdgeCount"] - 2 s["EdgeCount"]] + 0.25 s["MaxDegree"]]}
};
devices = If[! MemberQ[$ScriptCommandLine, "cpu"] &&
    MatchQ[Quiet@HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 1, "NumStates",
      "TargetDevice" -> {"GPU", "Strict" -> True}], _Integer],
  {"CPU", "GPU"}, {"CPU"}];
Print["devices: ", devices];

failures = 0;
Do[
  Module[{name, rules, init, steps, cname, cost, ref, states, byStep, ok = True, notes = {}},
    {name, rules, init, steps} = case;
    {cname, cost, ref} = c;
    rules = Flatten[{rules}];
    Do[
      states = HGEvolve[rules, init, steps, "States", "CanonicalizeStates" -> Full,
        "MaxStatesPerStep" -> n, "StateCost" -> cost, "RandomSeed" -> 11,
        "TargetDevice" -> dev];
      byStep = GroupBy[Values[states], #["Step"] &, Map[Rest, #["Edges"]] & /@ # &];
      Do[
        Module[{succ, kept, all, cut},
          succ = successors[rules, byStep[s]];
          kept = Lookup[byStep, s + 1, {}];
          all = Sort[ref[stats[#]] & /@ Values[succ]];
          If[Length[kept] =!= Min[n, Length[succ]],
            ok = False; AppendTo[notes, {s + 1, "kept", Length[kept], "of", Length[succ]}]];
          If[! SubsetQ[Keys[succ], First[CanonicalForm[#]] & /@ kept],
            ok = False; AppendTo[notes, {s + 1, "a kept class is not a successor"}]];
          If[kept =!= {},
            cut = all[[Min[n, Length[all]]]];
            If[Max[ref[stats[#]] & /@ kept] > cut,
              ok = False; AppendTo[notes, {s + 1, "cost", Max[ref[stats[#]] & /@ kept], cut}]]]],
        {s, 0, steps - 1}],
      {n, {1, 2, 3}}];
    Print[If[ok, "PASS ", "FAIL "], name, " ", cname, " ", dev, If[ok, "", {" ", notes}]];
    If[! ok, failures++]],
  {case, cases}, {c, costs}, {dev, devices}];

Print[If[failures == 0, "STATE COST VERIFIED", "STATE COST FAILED: " <> ToString[failures]]];
Exit[If[failures == 0, 0, 1]];
