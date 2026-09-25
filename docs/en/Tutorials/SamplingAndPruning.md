---
Template: TechNote
Name: SamplingAndPruning
Title: Sampling and Pruning the Multiway System
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/tutorial/SamplingAndPruning
Keywords: [hypergraph, multiway, rewriting, sampling, pruning, transition rate, rule weights, random seed, canonical hash]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [GettingStarted]
---

The number of states of a multiway system grows quickly with the number of steps. [HGEvolve]() has two kinds of options that limit it:

- caps by arrival order, `"MaxSuccessorStatesPerParent"`, `"MaxStatesPerStep"` and `"UniformRandom"` with `"MatchesPerStep"`, which bound the work but keep states that depend on the thread schedule;
- reproducible selections, `"MatchesPerStateRule"`, `"TransitionRate"` with `"RuleWeights"`, and `"ExplorationProbability"`, which keep the same states on every run with the same `"RandomSeed"`.

This tutorial applies each to one rule, and combines two sampled runs using the canonical hashes of their states.

## The size of the full system

The chain rule splits one edge into two. Its number of states at steps 0 to 6:

```wl
chain = {{{1, 2}} -> {{1, 3}, {3, 2}}};
Table[HGEvolve[chain, {{1, 2}}, k, "NumStates"], {k, 0, 6}]
```

<!-- => {1, 2, 4, 10, 34, 154, 874} -->

---

The examples below use this rule, which consumes two edges meeting at a vertex and produces four:

```wl
rule = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}}
```

<!-- => {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}} -->

Its initial state is two loops at one vertex:

```wl
init = {{1, 1}, {1, 1}}
```

<!-- => {{1, 1}, {1, 1}} -->

The number of states at steps 0 to 5:

```wl
Table[HGEvolve[rule, init, k, "NumStates"], {k, 0, 5}]
```

<!-- => {1, 3, 9, 27, 117, 747} -->

The number of isomorphism classes among them:

```wl
Table[HGEvolve[rule, init, k, "NumStates", "CanonicalizeStates" -> Full], {k, 0, 5}]
```

<!-- => {1, 2, 3, 4, 6, 13} -->

`"CanonicalizeStates"` merges states but still applies every transition. The options in this tutorial skip transitions or states.

The full states graph at three steps, 27 states:

```wl
HGEvolve[rule, init, 3, "StatesGraphStructure", AspectRatio -> 1/2]
```

A run that sets no `"RandomSeed"` and is capped by arrival order or thinned by `"TransitionRate"` or `"ExplorationProbability"` can give a different result each time, and issues the message `HGEvolve::warn`. Every example below sets a seed.

## Caps by arrival order

`"MaxSuccessorStatesPerParent"`, `"MaxStatesPerStep"` and `"UniformRandom"` with `"MatchesPerStep"` count states as they are produced and keep no more once the cap is reached. The bound holds at any thread count. Which states are kept depends on the order the threads produce them in, so a capped run can keep different states on another run or thread count.

### Successors per parent

At most two successors per state, at three steps, gives a binary tree of 15 states:

```wl
capped = HGEvolve[rule, init, 3, "StatesGraphStructure", "MaxSuccessorStatesPerParent" -> 2, "RandomSeed" -> 3, AspectRatio -> 1/2]
```

Seven states have two successors and eight have none:

```wl
Sort @ Tally @ VertexOutDegree[capped]
```

<!-- => {{0, 8}, {2, 7}} -->

At four steps the tree has 31 states, against 117 in the full system:

```wl
HGEvolve[rule, init, 4, "NumStates", "MaxSuccessorStatesPerParent" -> 2, "RandomSeed" -> 3]
```

<!-- => 31 -->

### States per step

`"MaxStatesPerStep" -> 3` keeps at most three states per step. Step 1 has two states, so the cap applies from step 2:

```wl
HGEvolve[rule, init, 3, "StatesGraphStructure", "MaxStatesPerStep" -> 3, "RandomSeed" -> 3, AspectRatio -> 1/2]
```

At four steps:

```wl
HGEvolve[rule, init, 4, "NumStates", "MaxStatesPerStep" -> 3, "RandomSeed" -> 3]
```

<!-- => 12 -->

### Matches per step

`"UniformRandom" -> True` with `"MatchesPerStep" -> 3` also keeps at most three states per step, in arrival order, and gives the same count as `"MaxStatesPerStep" -> 3`:

```wl
{HGEvolve[rule, init, 4, "NumStates", "MaxStatesPerStep" -> 3, "RandomSeed" -> 3], HGEvolve[rule, init, 4, "NumStates", "UniformRandom" -> True, "MatchesPerStep" -> 3, "RandomSeed" -> 3]}
```

<!-- => {12, 12} -->

## Reproducible selections

These options decide each transition or state from its isomorphism-invariant identity and `"RandomSeed"`. The same seed keeps the same states at any thread count, on either device, and on every run.

### Transitions per state and rule

`"MatchesPerStateRule" -> k` keeps at most *k* transitions of each state for each rule. They are chosen after all matches of the state are found, by the transitions' identities and the seed. Under this option every state is matched in full, since a match passed down from the parent state would arrive too late to be counted.

Two transitions per state and rule, at three steps:

```wl
HGEvolve[rule, init, 3, "StatesGraphStructure", "MatchesPerStateRule" -> 2, "RandomSeed" -> 3, AspectRatio -> 1/2]
```

With one transition per state, the kept states form a path:

```wl
HGEvolve[rule, init, 4, "StatesGraphStructure", "MatchesPerStateRule" -> 1, "RandomSeed" -> 3, AspectRatio -> 1/4]
```

---

The full system has one isomorphism class at each step up to step 3, and two at step 4:

```wl
Sort @ Tally @ Lookup["Step"] @ Values @ HGEvolve[rule, init, 4, "States", "CanonicalizeStates" -> Full]
```

<!-- => {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 2}} -->

With `"IncludeCanonicalHashes" -> True`, each state record has a `"CanonicalHash"`, which is the same for isomorphic states on every run. The step and the class of each state on the path; the seed selects which of the two classes at step 4 it ends in:

```wl
kept = Lookup[{"Step", "CanonicalHash"}] @ Values @ HGEvolve[rule, init, 4, "States", "MatchesPerStateRule" -> 1, "RandomSeed" -> 3, "IncludeCanonicalHashes" -> True]
```

<!-- => {{0, -5669361067642602358}, {1, -5742333305762982118}, {2, 8839991810862101748}, {3, 8367264973557672493}, {4, -6704850690037478777}} -->

A second run with the same seed keeps the same states:

```wl
kept === Lookup[{"Step", "CanonicalHash"}] @ Values @ HGEvolve[rule, init, 4, "States", "MatchesPerStateRule" -> 1, "RandomSeed" -> 3, "IncludeCanonicalHashes" -> True]
```

<!-- => True -->

### Transition rate

`"TransitionRate" -> r` keeps each transition with probability *r*, drawn from the transition's identity and the seed.

At rate 0.5 and four steps, 14 of the 117 states are kept:

```wl
HGEvolve[rule, init, 4, "StatesGraphStructure", "TransitionRate" -> 0.5, "RandomSeed" -> 2, AspectRatio -> 1/2]
```

The number of states at several rates, with one seed:

```wl
Table[HGEvolve[rule, init, 4, "NumStates", "TransitionRate" -> r, "RandomSeed" -> 2], {r, {1., 0.75, 0.5, 0.25, 0.125}}]
```

<!-- => {117, 81, 14, 7, 5} -->

---

A state whose draws all fail keeps its transition with the lowest key, so the sample always reaches the requested depth. At rate 0.125 one path of five states remains:

```wl
HGEvolve[rule, init, 4, "StatesGraphStructure", "TransitionRate" -> 0.125, "RandomSeed" -> 2, AspectRatio -> 1/2]
```

The deepest step reached at that rate, for six seeds:

```wl
Table[Max @ Lookup["Step"] @ Values @ HGEvolve[rule, init, 4, "States", "TransitionRate" -> 0.125, "RandomSeed" -> s], {s, 1, 6}]
```

<!-- => {4, 4, 4, 4, 4, 4} -->

### Rule weights

`"RuleWeights"` gives a multiplier on `"TransitionRate"` for each rule, in rule order. A rule's transitions are kept with the product of the rate and its weight. Rules after the end of the list have weight 1, and the default `{}` gives every rule weight 1.

Two rules that each add a path to a matched edge:

```wl
two = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}
```

<!-- => {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}} -->

The transitions of each rule in their own color, at two steps with isomorphic states merged:

```wl
HGEvolve[two, init, 2, "StatesGraph", "ColorByRule" -> True, "CanonicalizeStates" -> Full, ImageSize -> 540]
```

The number of states at three steps with both rules, with the first only, and with the second only:

```wl
{HGEvolve[two, init, 3, "NumStates"], HGEvolve[{First[two]}, init, 3, "NumStates"], HGEvolve[{Last[two]}, init, 3, "NumStates"]}
```

<!-- => {317, 33, 59} -->

---

Weight 0 removes a rule. With weights `{1, 0}` the count is that of the first rule alone:

```wl
HGEvolve[two, init, 3, "NumStates", "RuleWeights" -> {1, 0}, "RandomSeed" -> 3]
```

<!-- => 33 -->

Weights `{1, 0.25}` at rate 0.5 keep the first rule's transitions with probability 0.5 and the second's with 0.125, as weights `{0.5, 0.125}` at rate 1 do:

```wl
{HGEvolve[two, init, 3, "NumStates", "TransitionRate" -> 0.5, "RuleWeights" -> {1, 0.25}, "RandomSeed" -> 3], HGEvolve[two, init, 3, "NumStates", "RuleWeights" -> {0.5, 0.125}, "RandomSeed" -> 3]}
```

<!-- => {11, 11} -->

`{0.25}` and `{0.25, 1}` give the same count:

```wl
{HGEvolve[two, init, 3, "NumStates", "RuleWeights" -> {0.25}, "RandomSeed" -> 3], HGEvolve[two, init, 3, "NumStates", "RuleWeights" -> {0.25, 1}, "RandomSeed" -> 3]}
```

<!-- => {97, 97} -->

### Exploration probability

`"ExplorationProbability" -> p` expands each state with probability *p*, drawn from the identity of the transition that created it and the seed. The initial state is always expanded. A state that is not expanded has no successors, so a run can stop before the requested depth.

At probability 0.5 and four steps, 24 of the 117 states:

```wl
HGEvolve[rule, init, 4, "StatesGraphStructure", "ExplorationProbability" -> 0.5, "RandomSeed" -> 14, AspectRatio -> 1/2]
```

The number of states for six seeds. A run with 3 states expanded neither state of step 1:

```wl
Table[HGEvolve[rule, init, 4, "NumStates", "ExplorationProbability" -> 0.5, "RandomSeed" -> s], {s, 11, 16}]
```

<!-- => {42, 35, 42, 24, 3, 12} -->

The deepest step each of those runs reached:

```wl
Table[Max @ Lookup["Step"] @ Values @ HGEvolve[rule, init, 4, "States", "ExplorationProbability" -> 0.5, "RandomSeed" -> s], {s, 11, 16}]
```

<!-- => {4, 4, 4, 4, 1, 3} -->

Since the initial state is always expanded, no run at two steps has fewer than the initial state and its two successors:

```wl
Min[Table[HGEvolve[rule, init, 2, "NumStates", "ExplorationProbability" -> 0.5, "RandomSeed" -> s], {s, 1, 20}]]
```

<!-- => 3 -->

---

With `"ExploreFromCanonicalStatesOnly" -> True` and `"CanonicalizeStates" -> Full`, each isomorphism class is expanded once, and the draw is made once per class. The number of classes kept for six seeds, of the 6 in the full system at four steps:

```wl
Table[HGEvolve[rule, init, 4, "NumStates", "ExplorationProbability" -> 0.5, "RandomSeed" -> s, "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True], {s, 1, 6}]
```

<!-- => {6, 2, 2, 2, 3, 3} -->

## Combining sampled runs

State ids are assigned separately in each run, so an `"Id"` from one run does not name a state of another. The `"CanonicalHash"` depends only on the state's isomorphism class, so the states of two runs can be matched by it. The two runs below are at rate 0.5 and five steps, with isomorphic states merged.

The class hashes of the first seed:

```wl
hashesA = Sort @ Lookup["CanonicalHash"] @ Values @ HGEvolve[rule, init, 5, "States", "CanonicalizeStates" -> Full, "TransitionRate" -> 0.5, "RandomSeed" -> 1, "IncludeCanonicalHashes" -> True]
```

<!-- => {-6704850690037478777, -6276439087999122262, -5742333305762982118, -5669361067642602358, -5239938230783464623, -1453811750045434078, 5293164704559906130, 7197693678543408660, 7499371674364276414, 8367264973557672493, 8839991810862101748} -->

The class hashes of the second seed:

```wl
hashesB = Sort @ Lookup["CanonicalHash"] @ Values @ HGEvolve[rule, init, 5, "States", "CanonicalizeStates" -> Full, "TransitionRate" -> 0.5, "RandomSeed" -> 2, "IncludeCanonicalHashes" -> True]
```

<!-- => {-6704850690037478777, -5769831544623873690, -5742333305762982118, -5669361067642602358, 4195566678071382530, 5293164704559906130, 7197693678543408660, 7499371674364276414, 8367264973557672493, 8839991810862101748} -->

The number of classes in each run, and in both:

```wl
{Length[hashesA], Length[hashesB], Length[Intersection[hashesA, hashesB]]}
```

<!-- => {11, 10, 8} -->

Together the two runs have the 13 classes of the full system at five steps:

```wl
{Length[Union[hashesA, hashesB]], HGEvolve[rule, init, 5, "NumStates", "CanonicalizeStates" -> Full]}
```

<!-- => {13, 13} -->

## All the options on one rule

The settings compared, the empty one being the full system:

```wl
#| collapse: input
controls = {{}, {"MaxSuccessorStatesPerParent" -> 2}, {"MaxStatesPerStep" -> 3}, {"UniformRandom" -> True, "MatchesPerStep" -> 3}, {"MatchesPerStateRule" -> 2}, {"TransitionRate" -> 0.5}, {"ExplorationProbability" -> 0.5}}
```

<!-- => {{}, {"MaxSuccessorStatesPerParent" -> 2}, {"MaxStatesPerStep" -> 3}, {"UniformRandom" -> True, "MatchesPerStep" -> 3}, {"MatchesPerStateRule" -> 2}, {"TransitionRate" -> 0.5}, {"ExplorationProbability" -> 0.5}} -->

The number of states at four steps under each setting, with one seed:

```wl
BarChart[Reverse @ Table[HGEvolve[rule, init, 4, "NumStates", "RandomSeed" -> 1, Sequence @@ control], {control, controls}], BarOrigin -> Left, ChartLabels -> Reverse @ Replace[controls, {{} -> "full system", setting_ :> Row[setting, ", "]}, {1}], LabelingFunction -> After, ImageSize -> 500]
```

`"MaxSuccessorStatesPerParent" -> 2` and `"MatchesPerStateRule" -> 2` both keep 31 states. The first keeps the first two successors of each state the threads produce; the second keeps the two transitions selected by the seed, and keeps the same ones on every run. The `"TransitionRate"` and `"ExplorationProbability"` counts are for this seed; other seeds give other counts.
