---
Template: TechNote
Name: AdvancedMultiwayEvolution
Title: Advanced Multiway Evolution
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/tutorial/AdvancedMultiwayEvolution
Keywords: [hypergraph, multiway, rewriting, canonicalization, isomorphism, causal graph, branchial graph, quotient, tutorial]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [GettingStarted, SamplingAndPruning]
---

The states, events and relations [HGEvolve]() reports depend on two settings: `"CanonicalizeStates"`, which selects when two states are one state, and `"CanonicalizeEvents"`, which selects when two applications of a rule are one event. This tutorial shows how each output follows from these settings, and covers the canonical hash, the edge store, the causal and branchial relations, quotient exploration and several rules. [Getting Started with Hypergraph Rewriting](paclet:WolframInstitute/HypergraphRewriteEngine/tutorial/GettingStarted) introduces the properties used here.

## The example

The rule matches two edges from a common vertex and adds the edge that closes the triangle:

```wl
rule = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}
```

<!-- => {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}} -->

The initial state is the rule's left side:

```wl
init = {{1, 2}, {1, 3}}
```

<!-- => {{1, 2}, {1, 3}} -->

Three steps with the default settings. Each vertex is a state reached by one history, so the graph is a tree:

```wl
HGEvolve[rule, init, 3, "StatesGraphStructure"]
```

## When two states are one state

`"CanonicalizeStates"` is applied while the evolution runs. Under `None` every history ends in its own state:

```wl
Table[HGEvolve[rule, init, k, "NumStates"], {k, 0, 4}]
```

<!-- => {1, 3, 7, 19, 75} -->

`Automatic` identifies states with the same edge list:

```wl
Table[HGEvolve[rule, init, k, "NumStates", "CanonicalizeStates" -> Automatic], {k, 0, 4}]
```

<!-- => {1, 3, 7, 17, 47} -->

`Full` identifies isomorphic states, states that are the same up to a renaming of vertices:

```wl
Table[HGEvolve[rule, init, k, "NumStates", "CanonicalizeStates" -> Full], {k, 0, 4}]
```

<!-- => {1, 2, 4, 7, 13} -->

The setting changes the number of states, not the number of rule applications. At three steps all 18 matches are applied under each setting:

```wl
Table[HGEvolve[rule, init, 3, "NumEvents", "CanonicalizeStates" -> mode], {mode, {None, Automatic, Full}}]
```

<!-- => {18, 18, 18} -->

`"ExploreFromCanonicalStatesOnly"`, below, changes which states are expanded.

---

The state records of two steps:

```wl
Dataset[HGEvolve[rule, init, 2, "States"]]
```

Each edge is `{id, v1, v2}`. An edge the rule did not touch keeps its id in the state that inherits it, and events, the edge store and the graphs use the same ids.

The 19 states of three steps, kept as `states`:

```wl
Keys[states = HGEvolve[rule, init, 3, "States"]]
```

<!-- => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18} -->

---

Under `Automatic`, two states are one state when their edge lists are equal in order, apart from the edge ids. Counting the distinct edge lists without ids gives the `Automatic` count:

```wl
Length @ DeleteDuplicates @ Map[Rest, Lookup[Values[states], "Edges"], {2}]
```

<!-- => 17 -->

Sorting each edge list first gives fewer, so the order of the edges is part of the comparison:

```wl
Length @ DeleteDuplicates[Sort /@ Map[Rest, Lookup[Values[states], "Edges"], {2}]]
```

<!-- => 12 -->

`"ContentStateId"` numbers the `Automatic` classes. The classes with more than one state are two pairs:

```wl
Sort @ Map[Function[record, Rest /@ record["Edges"]], Select[GatherBy[Values[states], Key["ContentStateId"]], Length[#] > 1 &], {2}]
```

<!-- => {{{{0, 1}, {0, 2}, {1, 2}, {1, 2}, {2, 2}}, {{0, 1}, {0, 2}, {1, 2}, {1, 2}, {2, 2}}}, {{{0, 2}, {0, 1}, {2, 1}, {2, 1}, {1, 1}}, {{0, 2}, {0, 1}, {2, 1}, {2, 1}, {1, 1}}}} -->

---

`"IncludeCanonicalHashes" -> True` adds a `"CanonicalHash"` to every record, which is equal for isomorphic states under any setting. The records, kept as `hashed`, and the first of them:

```wl
First[hashed = HGEvolve[rule, init, 3, "States", "IncludeCanonicalHashes" -> True]]
```

<!-- => <|"Id" -> 0, "CanonicalId" -> 0, "ContentStateId" -> 0, "Step" -> 0, "Edges" -> {{0, 0, 1}, {1, 0, 2}}, "IsInitial" -> 1, "CanonicalHash" -> 5070671360234372917|> -->

The 19 states have 7 distinct hashes, the number of states under `Full`:

```wl
Length @ DeleteDuplicates @ Lookup[Values[hashed], "CanonicalHash"]
```

<!-- => 7 -->

The number of states in each isomorphism class:

```wl
Sort @ Values @ Counts @ Lookup[Values[hashed], "CanonicalHash"]
```

<!-- => {1, 2, 2, 2, 2, 4, 6} -->

Each `Automatic` class lies inside one isomorphism class, so `Automatic` never identifies two states that `Full` keeps apart:

```wl
AllTrue[GatherBy[Values[hashed], Key["ContentStateId"]], SameQ @@ Lookup[#, "CanonicalHash"] &]
```

<!-- => True -->

---

The chain rule creates a new vertex at every event, so no two of its states have the same edge list. `Automatic` identifies none of its states, and `Full` reduces 154 states to 6:

```wl
Table[HGEvolve[{{1, 2}} -> {{1, 3}, {3, 2}}, {{1, 2}}, 5, "NumStates", "CanonicalizeStates" -> mode], {mode, {None, Automatic, Full}}]
```

<!-- => {154, 154, 6} -->

Use `Automatic` when vertex names matter and `Full` when only the shape of the hypergraph matters.

---

Under `Automatic` two pairs of states are merged, so the tree's 19 vertices become 17, two of them with two parents:

```wl
HGEvolve[rule, init, 3, "StatesGraphStructure", "CanonicalizeStates" -> Automatic, AspectRatio -> 1/3]
```

Under `Full` there are 7 states, and parallel edges are separate applications between the same two classes:

```wl
HGEvolve[rule, init, 3, "StatesGraphStructure", "CanonicalizeStates" -> Full, AspectRatio -> 1/2]
```

## When two applications are one event

Every match of the rule is applied, and `"CanonicalizeEvents"` selects which applications are one event. The event records of two steps, with isomorphic states identified:

```wl
Dataset[HGEvolve[rule, init, 2, "Events", "CanonicalizeStates" -> Full]]
```

An event's identity is built from components that do not change when vertices are renamed: the canonical input state, the canonical output state, the step, the rule, and the positions of the consumed and produced edges in the canonical labeling of their states.

- `None` uses no component, so every application is its own event.
- `Full` uses the input and output states.
- `Automatic` uses the states, the step and both edge position lists.
- `"Positional"` uses the same components as `Automatic`, with edge positions read from each state's own labeling, as the `MultiwaySystem` function of the Wolfram Multicomputation paclet does.

The number of events at three steps under each setting:

```wl
Table[HGEvolve[rule, init, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> mode], {mode, {None, Full, Automatic, "Positional"}}]
```

<!-- => {18, 7, 10, 11} -->

`Automatic` needs `"CanonicalizeStates" -> Full`, which computes the canonical labelings. The `"Positional"` identity depends on the labeling the canonicalizer picks for each state, so it is reproducible for fixed inputs but changes when the initial state is relabeled.

---

A list of components selects another identity. With the rule only, every application is one event, since there is one rule:

```wl
HGEvolve[rule, init, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> {"Rule"}]
```

<!-- => 1 -->

With the input state and the rule, there is one event for each class the rule is applied to:

```wl
HGEvolve[rule, init, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> {"InputState", "Rule"}]
```

<!-- => 4 -->

The components are `"InputState"`, `"OutputState"`, `"Step"`, `"Rule"`, `"ConsumedEdges"` and `"ProducedEdges"`.

---

The evolution graph shows states in blue and events in yellow. With events distinct, each class at two steps has one event vertex for each application reaching it:

```wl
HGEvolve[rule, init, 2, "EvolutionGraphStructure", "CanonicalizeStates" -> Full, AspectRatio -> 1/2]
```

With `"CanonicalizeEvents" -> Full`, the applications between the same two classes are one event vertex, drawn with one edge for each application:

```wl
HGEvolve[rule, init, 2, "EvolutionGraphStructure", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full, AspectRatio -> 1/2]
```

## The canonical hash across runs

State ids are assigned in the order states are found, so another run, or a run from a relabeled initial state, numbers its states differently. The `"CanonicalHash"` depends only on the isomorphism class, so it is the same in every run.

The hashes of the seven classes at three steps, kept as `classes`:

```wl
classes = Sort @ Lookup[Values[HGEvolve[rule, init, 3, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True]], "CanonicalHash"]
```

<!-- => {-3456209204954656657, -2510968229161682807, -2510405279208606450, -2052257726680550281, -1880196251657875627, -864756329166671601, 5070671360234372917} -->

The same evolution from the initial state with other vertex names reaches the same classes:

```wl
Sort @ Lookup["CanonicalHash"] @ Values @ HGEvolve[rule, {{7, 8}, {7, 9}}, 3, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True] === classes
```

<!-- => True -->

A run of two steps from the triangle, the state after the first step:

```wl
fromTriangle = Sort @ Lookup["CanonicalHash"] @ Values @ HGEvolve[rule, {{1, 2}, {1, 3}, {2, 3}}, 2, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True]
```

<!-- => {-3456209204954656657, -2510968229161682807, -2510405279208606450, -2052257726680550281, -1880196251657875627, -864756329166671601} -->

Every class it reaches is a class of the first run:

```wl
SubsetQ[classes, fromTriangle]
```

<!-- => True -->

The classes of two runs combined are the union of their hash sets.

## The edge store

Every state is a set of edges from one store of all edges the evolution created. Asking for the properties together gives ids from one run:

```wl
Keys[edgeData = HGEvolve[rule, init, 2, {"States", "GlobalEdges", "StateBitvectors"}]]
```

<!-- => {"States", "GlobalEdges", "StateBitvectors"} -->

`"GlobalEdges"` lists every edge once as `{id, v1, v2}`, in id order, starting with the two initial edges:

```wl
Take[edgeData["GlobalEdges"], 2]
```

<!-- => {{0, 0, 1}, {1, 0, 2}} -->

Two steps create 20 edges:

```wl
Length[edgeData["GlobalEdges"]]
```

<!-- => 20 -->

The vertex pairs in the store, with their multiplicities:

```wl
Sort @ Tally[Rest /@ edgeData["GlobalEdges"]]
```

<!-- => {{{0, 1}, 7}, {{0, 2}, 7}, {{1, 2}, 3}, {{2, 1}, 3}} -->

The ids after the initial edges are assigned in the order edges are created, so another run can give the same ids to other vertex pairs.

---

`"StateBitvectors"` gives the ids each state holds. The initial state holds the first two:

```wl
edgeData["StateBitvectors"][0]
```

<!-- => {0, 1} -->

The number of edges each state holds:

```wl
Length /@ edgeData["StateBitvectors"]
```

<!-- => <|0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4, 4 -> 4, 5 -> 4, 6 -> 4|> -->

Edge ids start at 0 and the store is in id order, so a state's edges are the store entries at its ids plus 1. They are the `"Edges"` of the state's record, for every state:

```wl
AllTrue[Keys[edgeData["States"]], Function[id, edgeData["GlobalEdges"][[edgeData["StateBitvectors"][id] + 1]] === edgeData["States"][id]["Edges"]]]
```

<!-- => True -->

---

A state and its successor hold the same id for an edge the event did not touch. The seven states hold 24 edges between them, from 20 distinct edges:

```wl
Total[Length /@ Values[edgeData["StateBitvectors"]]]
```

<!-- => 24 -->

Two edges are held by three states each, a state and its two successors:

```wl
Select[Values @ Counts @ Catenate @ Values @ edgeData["StateBitvectors"], GreaterThan[1]]
```

<!-- => {3, 3} -->

## The causal relation

An event causally precedes another when the other consumes an edge it produced. `"CausalEdges"` lists the pairs, with the producer as `"From"` and the consumer as `"To"`; `"RawFrom"` and `"RawTo"` are the ids of the individual applications, equal to `"From"` and `"To"` when events are not identified:

```wl
Dataset[HGEvolve[rule, init, 2, "CausalEdges"]]
```

By default the relation is transitively reduced: a pair joined by a longer path is dropped. The rule consumes two edges, so at step 3 an event can consume an edge produced one step earlier and an edge produced two steps earlier; the reduction drops the pair with the earlier event.

The reduced and unreduced counts at three steps:

```wl
{HGEvolve[rule, init, 3, "NumCausalEdges"], HGEvolve[rule, init, 3, "NumCausalEdges", "CausalTransitiveReduction" -> False]}
```

<!-- => {16, 20} -->

Reduced, the causal graph is two trees, one from each event of step 1:

```wl
HGEvolve[rule, init, 3, "CausalGraphStructure", AspectRatio -> 1/2]
```

Unreduced, it also has the four pairs from events to the events two steps later:

```wl
HGEvolve[rule, init, 3, "CausalGraphStructure", "CausalTransitiveReduction" -> False, AspectRatio -> 1/2]
```

---

When an event consumes two edges from the same producer, the pair is related through two edges. By default the graph has one edge per pair; with `"EdgeDeduplication" -> False` it has one edge per shared edge:

```wl
EdgeCount[HGEvolve[rule, init, 3, "CausalGraphStructure", "EdgeDeduplication" -> False]]
```

<!-- => 28 -->

The count of the relation is unchanged:

```wl
HGEvolve[rule, init, 3, "NumCausalEdges", "EdgeDeduplication" -> False]
```

<!-- => 16 -->

---

The initial edges are produced by a genesis event, which is left out by default. `"ShowGenesisEvents" -> True` includes it, with rule index 65535, no consumed edges and the initial edges produced:

```wl
SelectFirst[HGEvolve[rule, init, 1, "Events", "ShowGenesisEvents" -> True], #RuleIndex == 65535 &]
```

<!-- => <|"Id" -> 0, "CanonicalId" -> 0, "RuleIndex" -> 65535, "InputState" -> 1, "OutputState" -> 0, "CanonicalInputState" -> 1, "CanonicalOutputState" -> 0, "ConsumedEdges" -> {}, "ProducedEdges" -> {0, 1}|> -->

Its output state is the initial state, and its input state is an empty state added by the same option.

The four counts at two steps with and without the genesis event, which adds one event and two causal pairs:

```wl
BarChart[Transpose[Values /@ Table[HGEvolve[rule, init, 2, "Debug", "ShowGenesisEvents" -> shown], {shown, {True, False}}]], ChartLabels -> {{"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}, None}, ChartLegends -> {"genesis event shown", "default"}, LabelingFunction -> Above, ImageSize -> 480]
```

With the genesis event, the causal graph is one tree:

```wl
HGEvolve[rule, init, 2, "CausalGraphStructure", "ShowGenesisEvents" -> True, AspectRatio -> 1/2]
```

## The branchial relation

Two events are branchially related when they have the same input state and consumed a common edge, so no single history contains both. `"BranchialEdges"` lists the pairs. At two steps there are three, one for each expanded state, since the two orders of matching the same two edges are two applications that share both edges:

```wl
Dataset[HGEvolve[rule, init, 2, "BranchialEdges"]]
```

The relation is computed on the individual applications under any event identity, and is not reduced. Its size at three steps:

```wl
HGEvolve[rule, init, 3, "NumBranchialEdges"]
```

<!-- => 9 -->

---

The branchial graph joins two states when the events that produced them are branchially related. `"BranchialStep"` selects the step: `Automatic` is the final step for `"BranchialGraph"` and every step for the evolution graphs, `-1` is the final step, `All` is every step, and a positive integer is that step.

The final step of three, with six pairs among twelve states:

```wl
HGEvolve[rule, init, 3, "BranchialGraphStructure", AspectRatio -> 1/3]
```

The edge and vertex counts for `Automatic`, `All`, `-1`, `1`, `2` and `3`:

```wl
Table[With[{g = HGEvolve[rule, init, 3, "BranchialGraphStructure", "BranchialStep" -> step]}, {EdgeCount[g], VertexCount[g]}], {step, {Automatic, All, -1, 1, 2, 3}}]
```

<!-- => {{6, 12}, {9, 18}, {6, 12}, {1, 2}, {2, 4}, {6, 12}} -->

The pairs of steps 1, 2 and 3 add up to the 9 of `All`.

## Quotient exploration

Under `"CanonicalizeStates" -> Full` every state reached is still expanded: the 19 states of three steps are expanded and the states graph shows 7. `"ExploreFromCanonicalStatesOnly" -> True` expands each isomorphism class once, at its shortest depth, and reconstructs the events and relations of the states it did not expand. It needs `"CanonicalizeStates" -> Full`.

A rule from the Wolfram Physics Project, which reaches 45 classes in four steps:

```wl
wolframRule = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}
```

<!-- => {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}} -->

The four counts with every state expanded, kept as `capture`, and with each class expanded once, kept as `quotient`:

```wl
BarChart[Transpose[Values /@ {capture = HGEvolve[wolframRule, init, 4, "Debug", "CanonicalizeStates" -> Full], quotient = HGEvolve[wolframRule, init, 4, "Debug", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]}], ChartLabels -> {{"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}, None}, ChartLegends -> {"every state expanded", "each class expanded once"}, LabelingFunction -> Above, ImageSize -> 480]
```

```wl
capture === quotient
```

<!-- => True -->

The causal graph under quotient exploration has the 124 edges both runs count:

```wl
EdgeCount[HGEvolve[wolframRule, init, 4, "CausalGraphStructure", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]]
```

<!-- => 124 -->

Quotient exploration of the example reaches the same seven classes:

```wl
Sort @ Lookup["CanonicalHash"] @ Values @ HGEvolve[rule, init, 3, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True, "ExploreFromCanonicalStatesOnly" -> True] === classes
```

<!-- => True -->

---

The states graph under quotient exploration is also the one of full exploration, with one edge per application:

```wl
EdgeCount /@ {HGEvolve[wolframRule, init, 4, "StatesGraphStructure", "CanonicalizeStates" -> Full], HGEvolve[wolframRule, init, 4, "StatesGraphStructure", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]}
```

<!-- => {126, 126} -->

---

Each initial state starts its own evolution, even when initial states are isomorphic. Three copies of the example's initial state with different vertex names:

```wl
roots = {{{1, 2}, {1, 3}}, {{4, 5}, {4, 6}}, {{7, 8}, {7, 9}}}
```

<!-- => {{{1, 2}, {1, 3}}, {{4, 5}, {4, 6}}, {{7, 8}, {7, 9}}} -->

The number of states at two steps under `None` and under `Full`:

```wl
{HGEvolve[rule, roots, 2, "NumStates"], HGEvolve[rule, roots, 2, "NumStates", "CanonicalizeStates" -> Full]}
```

<!-- => {21, 4} -->

Under `Full` the states graph has three copies of every transition, 18 edges among 4 states, with or without quotient exploration:

```wl
HGEvolve[rule, roots, 2, "StatesGraphStructure", "CanonicalizeStates" -> Full, AspectRatio -> 1/2]
```

```wl
HGEvolve[rule, roots, 2, "Debug", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]
```

<!-- => <|"NumStates" -> 4, "NumEvents" -> 18, "NumCausalEdges" -> 12, "NumBranchialEdges" -> 9|> -->

## Several rules

With a list of rules, every rule is applied at every state, and each event records the index of its rule, counted from 0. Two rules that extend an edge by one edge and by two:

```wl
rules = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}
```

<!-- => {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}} -->

The four counts at two steps from two loops, up to isomorphism:

```wl
HGEvolve[rules, {{1, 1}, {1, 1}}, 2, "Debug", "CanonicalizeStates" -> Full]
```

<!-- => <|"NumStates" -> 10, "NumEvents" -> 32, "NumCausalEdges" -> 20, "NumBranchialEdges" -> 16|> -->

The number of events of each rule:

```wl
KeySort @ Counts @ Lookup["RuleIndex"] @ Values @ HGEvolve[rules, {{1, 1}, {1, 1}}, 2, "Events", "CanonicalizeStates" -> Full]
```

<!-- => <|0 -> 16, 1 -> 16|> -->

`"ColorByRule" -> True` colors each transition by its rule and adds a legend. The legend counts rules from 1, so its "Rule 1" is rule index 0:

```wl
HGEvolve[rules, {{1, 1}, {1, 1}}, 2, "StatesGraph", "CanonicalizeStates" -> Full, "ColorByRule" -> True, ImageSize -> 540]
```

`"RuleWeights"` thins the transitions of each rule separately; see [Sampling and Pruning the Multiway System](paclet:WolframInstitute/HypergraphRewriteEngine/tutorial/SamplingAndPruning).

## Reproducibility

The states, events and relations are determined by the rules, the initial states, the number of steps and the options, and do not depend on the number of threads or the order in which they run. The ids are assigned in the order states, events and edges are created, so two runs can give the same state different ids and list a relation in a different order. Compare runs by counts and canonical hashes.

Two runs give the same counts:

```wl
HGEvolve[rule, init, 3, "Debug"] === HGEvolve[rule, init, 3, "Debug"]
```

<!-- => True -->

Another run reaches the same classes:

```wl
Sort @ Lookup["CanonicalHash"] @ Values @ HGEvolve[rule, init, 3, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True] === classes
```

<!-- => True -->

The sampling options keep this property when a `"RandomSeed"` is set; the caps by arrival order do not. Both are described in [Sampling and Pruning the Multiway System](paclet:WolframInstitute/HypergraphRewriteEngine/tutorial/SamplingAndPruning).
