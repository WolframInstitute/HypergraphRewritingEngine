#!/usr/bin/env wolframscript
(* Checks "EventOrderingFunction" against SetReplace's WolframModel.

   For each rule set, initial state and ordering of the corpus below, SetReplace's WolframModel
   (the installed SetReplace paclet) applies up to $maxEvents events in single history. The
   script compares, event by event, its "AllEventsList" (rule, input expressions -> output
   expressions, numbered from 1 in creation order) with:
     - ResourceFunction["WolframModel"],
     - SingleHistoryEvolve of reference/MultiwayReference.wl, on the first $referenceEvents events,
     - HGEvolve with "EventOrderingFunction" on the paclet in paclet/ (the CPU engine, and the GPU
       engine with --gpu).
   It also compares the expressions up to a relabelling of the vertices the evolution creates,
   and the causal graph (SetReplace's "CausalGraph" against "CausalEdges" without transitive
   reduction).

   --record writes SetReplace's outputs to reference/setreplace_order_cases.hpp, read by
   hypergraph/tests/test_event_ordering.cpp and gpu/tests/test_gpu_vs_cpu_differential.cpp.
   --json <file> writes them as JSON to <file>, for references in other languages.
   Exit status 0 when every comparison agrees. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
args = Rest[$ScriptCommandLine];
record = MemberQ[args, "--record"];
gpu = MemberQ[args, "--gpu"];
jsonFile = With[{p = FirstPosition[args, "--json"]}, If[MissingQ[p] || p[[1]] == Length[args], None, args[[p[[1]] + 1]]]];
Needs["SetReplace`"];
Get[FileNameJoin[{repo, "reference", "MultiwayReference.wl"}]];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];
Print["SetReplace ", PacletObject["SetReplace"]["Version"], ", WolframModel resource ",
  ResourceObject["WolframModel"]["Version"]];

$maxEvents = 200;
$referenceEvents = 40;

ruleSets = <|
  "gorard" -> {{{1, 2}, {2, 3}, {3, 4}, {4, 5}} -> {{2, 6}, {6, 5}, {4, 1}, {1, 6}}},
  "wolfram" -> {{{1, 2}, {1, 3}} -> {{1, 3}, {1, 4}, {2, 4}, {3, 4}}},
  "ternary" -> {{{1, 2, 3}, {3, 4}} -> {{4, 2, 5}, {5, 1}, {3, 5}}},
  "grow_contract" -> {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}, {2, 3}} -> {{1, 3}}},
  "subdivide_loop" -> {{{1, 2}, {2, 3}} -> {{1, 2}, {2, 4}, {4, 3}}, {{1, 1}} -> {{1, 2}, {2, 1}}},
  "pair_split" -> {{{1, 2}, {1, 3}} -> {{2, 3}}, {{1, 2}} -> {{1, 3}, {3, 2}}}
|>;
binaryInits = {
  {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 1}, {2, 4}},
  {{1, 1}, {1, 1}, {1, 2}},
  {{1, 2}, {2, 3}, {3, 1}, {1, 4}, {4, 5}, {5, 2}, {3, 5}, {5, 6}, {6, 1}}};
ternaryInits = {
  {{1, 2, 3}, {3, 4}, {4, 5, 6}, {6, 1}, {2, 2, 4}, {4, 3}},
  {{1, 1, 1}, {1, 1}, {1, 1, 1}, {1, 1}}};
orderings = {
  Automatic,
  {"OldestEdge", "RuleOrdering", "RuleIndex"},
  {"NewestEdge", "ReverseRuleOrdering", "ReverseRuleIndex"},
  {"LeastOldEdge", "RuleOrdering", "ReverseRuleIndex"},
  {"RuleIndex", "OldestEdge", "RuleOrdering"},
  {"ReverseRuleIndex", "LeastRecentEdge", "ReverseRuleOrdering"},
  {"RuleOrdering", "RuleIndex"}};
orderingNames[Automatic] = {"LeastRecentEdge", "RuleOrdering", "RuleIndex"};
orderingNames[l_List] := l;

(* Two expression lists agree when they are equal after a one-to-one renaming of vertices. *)
sameUpToRelabel[a_, b_] := Length[a] == Length[b] && (Length /@ a) === (Length /@ b) &&
  Module[{pairs = DeleteDuplicates[Transpose[{Flatten[a], Flatten[b]}]]},
    DuplicateFreeQ[pairs[[All, 1]]] && DuplicateFreeQ[pairs[[All, 2]]]];

causalPairs[g_Graph] := Union[List @@@ EdgeList[g]];

hgRun[rules_, init_, ordering_, device_] := Module[{r, ev, ids, edges},
  r = HGEvolve[rules, {init}, $maxEvents, {"Events", "GlobalEdges", "CausalEdges"},
    "EventOrderingFunction" -> ordering, "CausalTransitiveReduction" -> False,
    "CanonicalizeStates" -> None, "TargetDevice" -> device];
  If[!AssociationQ[r], Return[$Failed]];
  ev = SortBy[Values[r["Events"]], #["Id"] &];
  ids = AssociationThread[ev[[All, "Id"]] -> Range[Length[ev]]];
  edges = SortBy[r["GlobalEdges"], First];
  <|"Events" -> ({#["RuleIndex"] + 1, (#["ConsumedEdges"] + 1) -> (#["ProducedEdges"] + 1)} & /@ ev),
    "EdgeIds" -> edges[[All, 1]], "Expressions" -> edges[[All, 2 ;;]],
    "Causal" -> If[MatchQ[r["CausalEdges"], {__}],
      Union[Lookup[ids, #] & /@ Lookup[r["CausalEdges"], {"From", "To"}]], {}]|>];

cases = {};
failures = 0;
check[name_, ok_] := If[!TrueQ[ok], failures++; Print["  FAIL ", name]];
KeyValueMap[Function[{ruleName, rules},
  Do[
    Do[Module[{wm, sr, rf, ref, hg, label},
        label = ruleName <> " init " <> ToString[initIndex] <> " " <> ToString[orderingNames[ordering]];
        wm = WolframModel[rules, inits[[initIndex]], <|"MaxEvents" -> $maxEvents|>,
          "EventOrderingFunction" -> ordering];
        sr = <|"Events" -> wm["AllEventsList"], "Expressions" -> wm["AllEventsEdgesList"],
          "Causal" -> causalPairs[wm["CausalGraph"]]|>;
        rf = ResourceFunction["WolframModel"][rules, inits[[initIndex]], <|"MaxEvents" -> $maxEvents|>,
          "AllEventsList", "EventOrderingFunction" -> ordering];
        check[label <> ": ResourceFunction[\"WolframModel\"]", rf === sr["Events"]];
        ref = SingleHistoryEvolve[List @@@ rules, inits[[initIndex]],
          Min[$referenceEvents, Length[sr["Events"]]], orderingNames[ordering]];
        check[label <> ": SingleHistoryEvolve events",
          ref["Events"] === Take[sr["Events"], Length[ref["Events"]]]];
        check[label <> ": SingleHistoryEvolve expressions",
          sameUpToRelabel[ref["Expressions"], Take[sr["Expressions"], Length[ref["Expressions"]]]]];
        Do[
          hg = hgRun[rules, inits[[initIndex]], ordering, device];
          check[label <> ": HGEvolve " <> device <> " ran", AssociationQ[hg]];
          If[AssociationQ[hg],
            check[label <> ": HGEvolve " <> device <> " events", hg["Events"] === sr["Events"]];
            check[label <> ": HGEvolve " <> device <> " edge ids", hg["EdgeIds"] === Range[0, Length[hg["EdgeIds"]] - 1]];
            check[label <> ": HGEvolve " <> device <> " expressions", sameUpToRelabel[hg["Expressions"], sr["Expressions"]]];
            check[label <> ": HGEvolve " <> device <> " causal", hg["Causal"] === sr["Causal"]]],
          {device, If[gpu, {"CPU", "GPU"}, {"CPU"}]}];
        Print[label, ": ", Length[sr["Events"]], " events"];
        AppendTo[cases, <|"name" -> label, "rules" -> List @@@ rules, "init" -> inits[[initIndex]],
          "ordering" -> orderingNames[ordering], "max_events" -> $maxEvents,
          "events" -> ({#[[1]], #[[2, 1]], #[[2, 2]]} & /@ sr["Events"]),
          "expressions" -> sr["Expressions"], "causal" -> sr["Causal"]|>]],
      {ordering, orderings}],
    {initIndex, Length[inits = If[ruleName === "ternary", ternaryInits, binaryInits]]}]],
  ruleSets];

Print[Length[cases], " cases, ", failures, " failures"];

If[record,
  Module[{edges, stream, hpp, quote},
    edges[l_] := Join[{Length[l]}, Flatten[Join[{Length[#]}, #] & /@ l]];
    stream[c_] := Flatten[{c["max_events"], Length[c["rules"]], Join[edges[#[[1]]], edges[#[[2]]]] & /@ c["rules"],
      edges[c["init"]], Length[c["events"]],
      Join[{#[[1]], Length[#[[2]]]}, #[[2]], {Length[#[[3]]]}, #[[3]]] & /@ c["events"],
      edges[c["expressions"]], Length[c["causal"]], c["causal"]}];
    quote[x_String] := "\"" <> x <> "\"";
    hpp = "#pragma once\n// SetReplace's WolframModel (SetReplace " <> PacletObject["SetReplace"]["Version"] <>
      ") in single history under \"EventOrderingFunction\",\n" <>
      "// recorded by reference/verify_setreplace_order.wls --record and read by\n" <>
      "// hypergraph/tests/test_event_ordering.cpp. Events and expressions are numbered from 1 in\n" <>
      "// creation order, as SetReplace's \"AllEventsList\" and \"AllEventsEdgesList\" are.\n//\n" <>
      "// kStream holds the cases in order. A case is: the event limit, the rule count, each rule's\n" <>
      "// left-hand side and right-hand side, the initial state, the event count, each event (its rule\n" <>
      "// from 1, the input count, the inputs, the output count, the outputs), the expressions, the\n" <>
      "// causal pair count and the pairs (the event making an expression, then an event consuming\n" <>
      "// it). An edge list is its edge count, then each edge as its arity and its vertices.\n\n" <>
      "#include <cstdint>\n\nnamespace setreplace_order {\n\n" <>
      "inline constexpr uint32_t kCases = " <> ToString[Length[cases]] <> ";\n\n" <>
      "inline constexpr const char* kNames[kCases] = {\n" <>
      StringRiffle["    " <> quote[#["name"]] & /@ cases, ",\n"] <> "};\n\n" <>
      "// Each case's ordering names, ended by a null.\n" <>
      "inline constexpr const char* kOrderings[kCases][4] = {\n" <>
      StringRiffle["    {" <> StringRiffle[Append[quote /@ #["ordering"], "nullptr"], ", "] <> "}" & /@ cases, ",\n"] <> "};\n\n" <>
      "inline constexpr uint32_t kStream[] = {\n" <>
      StringRiffle["    " <> StringRiffle[ToString /@ #, ","] & /@ (stream /@ cases), ",\n"] <> "};\n\n" <>
      "}  // namespace setreplace_order\n";
    (* Line feeds only, on every platform. *)
    With[{stream = OpenWrite[FileNameJoin[{repo, "reference", "setreplace_order_cases.hpp"}], BinaryFormat -> True]},
      WriteString[stream, hpp]; Close[stream]];
    Print["recorded ", Length[cases], " cases"]]];

If[jsonFile =!= None, Export[jsonFile, cases, "RawJSON", "Compact" -> True]; Print["wrote ", jsonFile]];

Exit[If[failures == 0, 0, 1]];
