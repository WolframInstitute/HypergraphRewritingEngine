---
Template: Symbol
Name: HGEvolve
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGEvolve
Keywords: [hypergraph, multiway, rewriting, Wolfram physics, causal graph, branchial graph, evolution, canonicalization, isomorphism, sampling, initial condition]
SeeAlso: [HGSessionOpen, HGSessionStep, HGSessionQuery]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [GettingStarted, AdvancedMultiwayEvolution, SamplingAndPruning, InitialConditions, Sessions, GPUEvolution]
---

## Usage

<code>[HGEvolve]()[*rules*, *init*, *n*]</code> evolves the hypergraph *init* under *rules* for *n* steps in every possible way and gives the graph of the states and events reached, with the causal and branchial edges between them.

<code>[HGEvolve]()[*rules*, *init*, *n*, *prop*]</code> gives the property *prop* of the evolution.

<code>[HGEvolve]()[*rules*, *init*, *n*, {$prop_1$, $prop_2$, …}]</code> gives an association of the listed properties.

<code>[HGEvolve]()[*rules*, "*name*", *n*, …]</code> evolves a generated initial condition of the named family, shaped by the initial-condition options.

<code>[HGEvolve]()[*rules*, <|"Type" -> "*name*", …|>, *n*, …]</code> evolves a generated initial condition shaped by the keys of the association.

## Details & Options

- A hypergraph is a list of hyperedges, and a hyperedge is an ordered list of vertices of any length. The same hyperedge may occur more than once.
- A rule is written *lhs* `->` *rhs*, with a hypergraph on each side; *lhs* is not empty. *rules* is a single rule or a list of rules; with a list, every rule is applied at every step.
- The vertices of a rule are pattern variables, written as integers or as symbols. A variable on both sides is kept by every application, a variable on the left only is consumed with its edges, and a variable on the right only is a vertex the application creates.
- A match maps each hyperedge of *lhs* to a distinct hyperedge of the state with the same number of vertices, and each variable to one vertex. Distinct variables may map to the same vertex.
- An event removes the matched hyperedges and adds the hyperedges of *rhs*. Every match is applied.
- *init* is one of:
  - a hypergraph, whose vertices may be integers or symbols
  - a list of hypergraphs, each of which becomes a distinct initial state
  - a [Graph](), whose edge list is taken as the hypergraph
  - a string naming a generated initial condition
  - an association with a `"Type"` key naming a generated initial condition
  - an association with an `"Edges"` key and no `"Type"` key, whose `"Edges"` is the hypergraph
- *n* is a non-negative integer. The initial states are at step 0; step $k$ holds every state produced by an event applied to a state of step $k-1$.
- The evolution runs in a separate engine process, and the result reaches the kernel when it finishes.
- The possible properties are:

|   |   |
|---|---|
| `"StatesGraph"` | graph whose vertices are states and whose edges are events |
| `"CausalGraph"` | graph whose vertices are events, with a directed edge from an event to each event that consumed an edge it produced |
| `"BranchialGraph"` | graph whose vertices are states, with an undirected edge between the output states of two branchially related events |
| `"EvolutionGraph"` | graph with state and event vertices, each event joined to its input and output states |
| `"EvolutionCausalGraph"` | evolution graph with the causal edges |
| `"EvolutionBranchialGraph"` | evolution graph with the branchial edges |
| `"EvolutionCausalBranchialGraph"` | evolution graph with the causal and the branchial edges (the default) |
| `"StatesGraphStructure"`, `"CausalGraphStructure"`, `"BranchialGraphStructure"`, `"EvolutionGraphStructure"`, `"EvolutionCausalGraphStructure"`, `"EvolutionBranchialGraphStructure"`, `"EvolutionCausalBranchialGraphStructure"` | the same seven graphs with plain vertices and no styling |
| `"States"` | the state records, an association keyed by state id |
| `"Events"` | the event records, an association keyed by event id |
| `"EventClasses"` | for each event under `"CanonicalizeEvents"`, the ids of the rule applications it stands for |
| `"CausalEdges"` | the causal relation, a list of records with the producer and consumer event of each pair |
| `"BranchialEdges"` | the branchial relation, a list of records with the two events of each pair |
| `"NumStates"`, `"NumEvents"` | the number of states and of events |
| `"NumCausalEdges"`, `"NumBranchialEdges"` | the number of causal and of branchial pairs |
| `"GlobalEdges"` | every hyperedge the evolution created, each as its id followed by its vertices |
| `"StateBitvectors"` | an association from each state id to the ids of the hyperedges the state holds |
| `"StepStatistics"` | for each step, statistics of the states at that step |
| `"Debug"` | an association of the four counts |
| `"All"` | an association of the states, the events, the two relations and the four counts, with a `"Warnings"` key when the run warns |

- A list of properties gives an association keyed by the property names, computed from one evolution.
- Any graph property takes the suffix `Structure`. A styled graph draws each state as its hypergraph and each event as a box showing the edges it consumed and produced, with the record of each vertex and edge as a tooltip. In a `Structure` graph the state vertices are the state ids and the event vertices are `{"E", id}`; in the evolution graphs they are `{"S", id}` and `{"E", id}`. Every edge carries an association naming the event or the pair it stands for.
- A state record has the keys:

|   |   |
|---|---|
| `"Id"` | the state id |
| `"CanonicalId"` | the id of the state's class under `"CanonicalizeStates"`; its own id under `None` |
| `"ContentStateId"` | the lowest id of the states with the same edge list |
| `"Step"` | the step of the state |
| `"Edges"` | the hyperedges, each as its edge id followed by its vertices |
| `"IsInitial"` | 1 for an initial state, 0 otherwise |
| `"CanonicalHash"` | with `"IncludeCanonicalHashes" -> True`, a hash equal for isomorphic states |

- An event record has the keys `"Id"`, `"CanonicalId"` (the event's identity under `"CanonicalizeEvents"`; its own id under `None`), `"RuleIndex"` (the position of the rule that fired, counted from 0), `"InputState"`, `"OutputState"`, `"CanonicalInputState"`, `"CanonicalOutputState"`, `"ConsumedEdges"` and `"ProducedEdges"` (edge ids).
- A causal record has the keys `"From"` and `"To"` (the producer and consumer events under the event identity in use) and `"RawFrom"` and `"RawTo"` (the same pair as the ids of the individual applications). A branchial record has the keys `"From"` and `"To"`.
- The engine numbers the vertices of a state from 0 in order of first appearance, and hyperedges from 0 in order of creation. State and event ids are assigned in the order the engine creates them, which can differ between two evolutions of the same input, so ids are only meaningful within one result.
- Two events are causally related when one consumes a hyperedge the other produced. The causal relation is transitively reduced by default. `"NumCausalEdges"`, `"CausalEdges"` and the causal graphs give the same relation.
- Two events are branchially related when they are applied to the same state and consume a common hyperedge. `"BranchialEdges"` lists every such pair; the branchial graph joins the output states of the pairs at the step `"BranchialStep"` selects.
- A `"StepStatistics"` entry has the keys `"Step"`, `"RawStates"`, `"Classes"` (isomorphism classes), `"Redundancy"` (states per class), `"MaxMultiplicity"`, `"MultiplicityHistogram"`, `"ClassEntropyBits"` and `"ClassEntropyNormalized"` (the entropy of the states over the classes), `"Events"` and `"RuleCounts"` (the events whose output state is at the step), `"Invariants"` (for each of eleven per-state invariants, such as `"VertexCount"`, `"Components"` and `"IncidenceDiameter"`, its `"N"`, `"Mean"`, `"StandardDeviation"`, `"Min"`, `"Max"`, `"Median"` and `"Histogram"`), and histograms of hyperedge arity and vertex degree. Under `"ExploreFromCanonicalStatesOnly" -> True` they are computed from one state per class and the number of states in the class.
- [HGEvolve]() takes every option of [Graph]() and passes it to the graph a graph property returns, so [ImageSize](), [AspectRatio](), [VertexLabels]() or [GraphLayout]() change the picture. A styled graph needs about 55 points of width for each state in its widest layer. The following options can also be given:

|   |   |   |
|---|---|---|
| `"CanonicalizeStates"` | `None` | when two states are one state: `None` (never), `Automatic` (identical edge lists) or `Full` (isomorphic) |
| `"CanonicalizeEvents"` | `None` | when two applications are one event: `None`, `Full`, `Automatic`, `"Positional"` or a list of identity components |
| `"CausalTransitiveReduction"` | `True` | whether to drop a causal pair implied by a longer causal path |
| `"MaxSuccessorStatesPerParent"` | `0` | the most successors kept per state, in arrival order (`0` for no cap) |
| `"MaxStatesPerStep"` | `0` | the most states kept per step, in arrival order (`0` for no cap) |
| `"ExplorationProbability"` | `1.` | the probability of exploring each state |
| `"TransitionRate"` | `1.` | the probability of keeping each transition |
| `"RuleWeights"` | `{}` | per-rule multipliers on `"TransitionRate"`, in rule order |
| `"ExploreFromCanonicalStatesOnly"` | `False` | whether to expand each isomorphism class once, at its shortest depth |
| `"QuotientInitialStates"` | `False` | whether isomorphic initial states become one root under `"ExploreFromCanonicalStatesOnly"` |
| `"MatchesPerStateRule"` | `0` | the most transitions kept per state and rule, chosen by the transitions' identities (`0` for no cap) |
| `"UniformRandom"` | `False` | whether `"MatchesPerStep"` caps the states kept per step |
| `"MatchesPerStep"` | `0` | the per-step cap used with `"UniformRandom"` (`0` for no cap) |
| `"BranchialStep"` | `Automatic` | the step whose branchial pairs the branchial graph shows: `Automatic`, `All`, `-1` (the final step) or a step number from 1 |
| `"EdgeDeduplication"` | `True` | whether a causal or branchial graph has one edge per event pair, rather than one per shared hyperedge |
| `"TargetDevice"` | `"CPU"` | `"CPU"` or `"GPU"` |
| `"RandomSeed"` | `Automatic` | the seed of every sampling draw and of a generated initial condition; `Automatic` draws a new seed on each run |
| `"IncludeCanonicalHashes"` | `False` | whether each state record has a `"CanonicalHash"` |
| `"ShowGenesisEvents"` | `False` | whether the events that create the initial states are part of the result |
| `"ColorByRule"` | `False` | whether a styled graph colors each transition by the rule that fired it, with a legend |
| `"MultiedgeStyle"` | `Automatic` | `Automatic` draws one edge per event or pair; `"Merged"` draws edges with the same endpoints as one edge |
| `"ShowProgress"` | `False` | whether to print the engine's progress |
| `"DebugFFI"` | `False` | whether to print what is requested from the engine and what it returns |

- `"CanonicalizeStates"` is applied while the evolution runs, so it also determines which states are expanded. `Automatic` identifies states whose edge lists are identical. This is finer than isomorphism: two isomorphic states with different vertex names stay separate. The vertices of each initial state are numbered from 0 separately, so two initial states that differ only in vertex names are one state under `Automatic`.
- `"CanonicalizeEvents"` builds an event's identity from isomorphism-invariant components. `Full` uses the canonical input and output states. `Automatic` also uses the step and the canonical positions of the consumed and produced edges within the state's isomorphism class; it needs `"CanonicalizeStates" -> Full`. `"Positional"` uses the same components with positions read from each state's own vertex labels, as the Wolfram Multicomputation paclet does. A list of any of `"InputState"`, `"OutputState"`, `"Step"`, `"Rule"`, `"ConsumedEdges"` and `"ProducedEdges"` selects the components.
- With `"ExploreFromCanonicalStatesOnly" -> True`, every property is the same as under full exploration; only the cost of the evolution changes. It needs `"CanonicalizeStates" -> Full`.
- `"MaxSuccessorStatesPerParent"`, `"MaxStatesPerStep"` and `"UniformRandom"` with `"MatchesPerStep"` cap by arrival order. The bound holds at any thread count, and which states are kept depends on the thread schedule. `"TransitionRate"`, `"ExplorationProbability"` and `"MatchesPerStateRule"` draw from the identity of each transition or state together with `"RandomSeed"`, so the same seed keeps the same states at any thread count and on either device.
- `"TransitionRate"` keeps a state's lowest-keyed transition when every draw at that state failed, so a sparse sample reaches the requested depth. `"ExplorationProbability"` does not.
- A run that sets no `"RandomSeed"` and discards states through an arrival-order cap, `"ExplorationProbability"` or `"TransitionRate"` issues the message `HGEvolve::warn`, and its result can differ from run to run. `"MatchesPerStateRule"` keeps the same transitions with or without a seed and issues no warning.
- `"TargetDevice" -> "GPU"` runs the evolution on the GPU engine bundled for the platform, which gives the same states, events and relations as the CPU. Where no GPU engine is bundled, the message `HGEvolve::gpudev` is issued and the evolution runs on the CPU. A GPU evolution that reaches a capacity limit issues `HGEvolve::overflow` and gives a partial result.
- A generated initial condition is named by *init*: `"Grid"`, `"Cylinder"`, `"Torus"`, `"Sphere"`, `"Klein"`, `"Mobius"`, `"Sprinkling"` (also `"Minkowski"`), `"BrillLindquist"`, `"Poisson"` or `"Uniform"`. The same names go in the `"Type"` key of an association, where the other keys shape the instance and take precedence over the options below.
- The initial-condition options are:

|   |   |   |
|---|---|---|
| `"GridWidth"` | `10` | the width of a `"Grid"` (key `"Width"`) and the resolution of a curved surface (key `"Resolution"`) |
| `"GridHeight"` | `10` | the height of a `"Grid"`, `"Cylinder"` or `"Klein"` (key `"Height"`) |
| `"GridHoles"` | `{}` | holes cut from a `"Grid"`, each as `{x, y, radius}` (key `"Holes"`) |
| `"SprinklingDensity"` | `500` | the number of points of `"Sprinkling"`, `"BrillLindquist"`, `"Poisson"` and `"Uniform"` (key `"Density"`) |
| `"SprinklingTimeExtent"` | `10.` | the extent of the time dimension of a sprinkling (key `"TimeExtent"`) |
| `"SprinklingSpatialExtent"` | `10.` | the extent of each spatial dimension of a sprinkling (key `"SpatialExtent"`) |
| `"SprinklingSpatialDim"` | `2` | the number of spatial dimensions of a sprinkling: 1, 2 or 3 (key `"SpatialDim"`) |
| `"SprinklingLightconeAngle"` | `1.` | the speed of light of a sprinkling (key `"LightconeAngle"`) |
| `"SprinklingAlexandrovCutoff"` | `5.` | the largest proper-time separation joined by an edge of a sprinkling (key `"AlexandrovCutoff"`) |
| `"SprinklingTransitivityReduction"` | `True` | whether a sprinkling keeps only the causal edges not implied by longer paths (key `"TransitivityReduction"`) |
| `"SprinklingMaxEdgesPerVertex"` | `50` | the most edges at one vertex of a sprinkling (key `"MaxEdgesPerVertex"`) |
| `"BrillLindquistMass1"` | `3.` | the mass of the first black hole (key `"Mass1"`) |
| `"BrillLindquistMass2"` | `3.` | the mass of the second black hole (key `"Mass2"`) |
| `"BrillLindquistSeparation"` | `10.` | the distance between the black holes (key `"Separation"`) |
| `"BrillLindquistBoxX"` | `{-15., 15.}` | the x range of the `"BrillLindquist"` domain (key `"BoxX"`) |
| `"BrillLindquistBoxY"` | `{-15., 15.}` | the y range of the `"BrillLindquist"` domain (key `"BoxY"`) |
| `"EdgeThreshold"` | `Automatic` | the largest distance joined by an edge of `"BrillLindquist"`, `"Poisson"` and `"Uniform"`; `Automatic` is `2.` for `"BrillLindquist"`, twice `"PoissonMinDistance"` for `"Poisson"` and 1.5 times the expected point spacing for `"Uniform"` (key `"EdgeThreshold"`) |
| `"PoissonMinDistance"` | `1.` | the least separation of the points of `"Poisson"` (key `"MinDistance"`) |

- The seed of a generated initial condition is `"RandomSeed"` (key `"Seed"`).
- Malformed input issues one message and gives <code>[$Failed]()</code>: `HGEvolve::badrule` for a rule, `HGEvolve::badinit` for *init*, `HGEvolve::steps` for *n*, `HGEvolve::unknownprop` for a property and `HGEvolve::unknownic` for an initial-condition name. A platform whose paclet has no engine issues `HGEvolve::noengine`.
- [HGSessionOpen]() takes the same arguments and options and opens an evolution that [HGSessionStep]() continues without running it again.

## Basic Examples

A rule that replaces one binary edge by a path of two edges through a new vertex:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}}
```

<!-- => {{1, 2}} -> {{1, 3}, {3, 2}} -->

Two steps of the rule from a single edge, as the evolution graph with its causal edges. The chain rule has no branchial pairs, since no two of its matches share an edge:

```wl
HGEvolve[chain, {{1, 2}}, 2, ImageSize -> 600, AspectRatio -> 1/2]
```

The states graph after three steps, each state drawn as its hypergraph:

```wl
HGEvolve[chain, {{1, 2}}, 3, "StatesGraph", ImageSize -> 540, AspectRatio -> 1/2]
```

The number of states after three steps:

```wl
HGEvolve[chain, {{1, 2}}, 3, "NumStates"]
```

<!-- => 10 -->

Every state reached by a different history is a separate state, so the number of states grows quickly with the number of steps:

```wl
Table[HGEvolve[chain, {{1, 2}}, k, "NumStates"], {k, 0, 5}]
```

<!-- => {1, 2, 4, 10, 34, 154} -->

Identifying isomorphic states leaves one state per step:

```wl
Table[HGEvolve[chain, {{1, 2}}, k, "NumStates", "CanonicalizeStates" -> Full], {k, 0, 5}]
```

<!-- => {1, 2, 3, 4, 5, 6} -->

---

A rule that closes a triangle over two edges with the same first vertex:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}
```

<!-- => {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}} -->

Its causal graph after three steps:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "CausalGraph", ImageSize -> 640, AspectRatio -> 1/2]
```

Its branchial graph, joining the states of the final step produced by events that consumed a common edge:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "BranchialGraph", ImageSize -> 540, AspectRatio -> 1/2]
```

---

A rule that consumes two edges meeting at a vertex and produces four. The examples of the sampling options below use it:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}
```

<!-- => {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}} -->

Its states after two steps from two loops at one vertex:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 2, "StatesGraphStructure", AspectRatio -> 1/2]
```

---

With a list of rules, every rule is applied at every step. These two extend an edge by one or by two new edges:

```wl
extend = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}
```

<!-- => {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}} -->

One step from two loops, with each transition colored by the rule that fired it:

```wl
HGEvolve[extend, {{1, 1}, {1, 1}}, 1, "StatesGraph", "ColorByRule" -> True, ImageSize -> 420]
```

## Scope

### Rules

A single rule and a list holding that rule give the same evolution:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[{chain}, {{1, 2}}, 3, "NumStates"]
```

<!-- => 10 -->

The vertices of a rule may be symbols:

```wl
HGEvolve[{{x, y}} -> {{x, z}, {z, y}}, {{1, 2}}, 3, "NumStates"]
```

<!-- => 10 -->

---

The `"RuleIndex"` of an event is the position of its rule, counted from 0. The number of events of each rule in one step:

```wl
extend = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
KeySort @ Counts @ Lookup["RuleIndex"] @ Values @ HGEvolve[extend, {{1, 1}, {1, 1}}, 1, "Events"]
```

<!-- => <|0 -> 2, 1 -> 2|> -->

Over two steps, as pairs of rule index and count:

```wl
Sort @ Tally @ Lookup["RuleIndex"] @ Values @ HGEvolve[extend, {{1, 1}, {1, 1}}, 2, "Events"]
```

<!-- => {{0, 16}, {1, 16}} -->

---

Hyperedges may have any number of vertices. A rule that splits a ternary hyperedge into two that share the new vertex:

```wl
HGEvolve[{{1, 2, 3}} -> {{1, 2, 4}, {2, 4, 3}}, {{1, 1, 1}}, 2, "StatesGraph", ImageSize -> 420, AspectRatio -> 1/2]
```

---

A rule may delete the edges it matches:

```wl
HGEvolve[{{1, 2}, {2, 3}} -> {}, {{1, 2}, {2, 3}}, 1, "NumStates"]
```

<!-- => 2 -->

### Initial states

A list of hypergraphs makes each of them an initial state at step 0:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Dataset[HGEvolve[chain, {{{1, 2}}, {{1, 2}, {2, 3}}}, 0, "States"]]
```

Both are expanded:

```wl
HGEvolve[chain, {{{1, 2}}, {{1, 2}, {2, 3}}}, 2, "NumStates"]
```

<!-- => 13 -->

---

The vertices of an initial state may be symbols; they are numbered in order of first appearance:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{a, b}}, 0, "GlobalEdges"]
```

<!-- => {{0, 0, 1}} -->

---

The edge list of a [Graph]() is taken as the hypergraph:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, Graph[{1 -> 2, 2 -> 3}], 0, "GlobalEdges"]
```

<!-- => {{0, 0, 1}, {1, 1, 2}} -->

---

A string names a generated initial condition, shaped by the initial-condition options. A 2 by 2 grid has four edges:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, "Grid", 0, "GlobalEdges", "GridWidth" -> 2, "GridHeight" -> 2, "RandomSeed" -> 1]
```

<!-- => {{0, 0, 1}, {1, 2, 0}, {2, 2, 3}, {3, 3, 1}} -->

One step of the triangle rule on it:

```wl
HGEvolve[triangle, "Grid", 1, "StatesGraph", "GridWidth" -> 2, "GridHeight" -> 2, "RandomSeed" -> 1, ImageSize -> 420, AspectRatio -> 1/2]
```

---

An association with a `"Type"` key names the family, and its other keys shape the instance:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, <|"Type" -> "Grid", "Width" -> 2, "Height" -> 3, "Seed" -> 1|>, 0, "GlobalEdges"]
```

<!-- => {{0, 0, 1}, {1, 2, 0}, {2, 2, 3}, {3, 4, 2}, {4, 5, 4}, {5, 3, 1}, {6, 3, 5}} -->

---

An association with an `"Edges"` key and no `"Type"` key gives the hypergraph directly:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, <|"Edges" -> {{1, 2}}|>, 2, "NumStates"]
```

<!-- => 4 -->

### Steps

Zero steps gives the initial state alone:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
First @ HGEvolve[chain, {{1, 2}}, 0, "States"]
```

<!-- => <|"Id" -> 0, "CanonicalId" -> 0, "ContentStateId" -> 0, "Step" -> 0, "Edges" -> {{0, 0, 1}}, "IsInitial" -> 1|> -->

The initial edge `{1, 2}` is hyperedge 0, from vertex 0 to vertex 1:

```wl
HGEvolve[chain, {{1, 2}}, 0, "GlobalEdges"]
```

<!-- => {{0, 0, 1}} -->

### Properties

`"States"` gives the state records, keyed by state id:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Dataset[HGEvolve[chain, {{1, 2}}, 2, "States"]]
```

The keys of a state record:

```wl
Keys @ First @ HGEvolve[chain, {{1, 2}}, 2, "States"]
```

<!-- => {"Id", "CanonicalId", "ContentStateId", "Step", "Edges", "IsInitial"} -->

One record; each hyperedge is its id followed by its vertices:

```wl
HGEvolve[chain, {{1, 2}}, 2, "States"][3]
```

<!-- => <|"Id" -> 3, "CanonicalId" -> 3, "ContentStateId" -> 3, "Step" -> 2, "Edges" -> {{1, 0, 2}, {5, 2, 4}, {6, 4, 1}}, "IsInitial" -> 0|> -->

---

`"Events"` gives the event records, keyed by event id:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Dataset[HGEvolve[chain, {{1, 2}}, 2, "Events"]]
```

The keys of an event record:

```wl
Keys @ First @ HGEvolve[chain, {{1, 2}}, 2, "Events"]
```

<!-- => {"Id", "CanonicalId", "RuleIndex", "InputState", "OutputState", "CanonicalInputState", "CanonicalOutputState", "ConsumedEdges", "ProducedEdges"} -->

---

`"CausalEdges"` gives the causal relation, one record for each pair of a producer and a consumer event:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Dataset[HGEvolve[chain, {{1, 2}}, 2, "CausalEdges"]]
```

The keys of a causal record:

```wl
Keys @ First @ HGEvolve[chain, {{1, 2}}, 2, "CausalEdges"]
```

<!-- => {"From", "To", "RawFrom", "RawTo"} -->

---

`"BranchialEdges"` gives the branchial relation, one record for each pair of events applied to the same state that consumed a common edge:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
Dataset[HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "BranchialEdges"]]
```

The keys of a branchial record:

```wl
Keys @ First @ HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "BranchialEdges"]
```

<!-- => {"From", "To"} -->

---

`"StatesGraph"` gives the graph of states and the events between them:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "StatesGraph", ImageSize -> 420]
```

`"CausalGraph"` gives the graph of events with their causal edges:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "CausalGraph", ImageSize -> 600, AspectRatio -> 1/2]
```

`"BranchialGraph"` gives the graph of the states at the final step with their branchial edges:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "BranchialGraph", ImageSize -> 420, AspectRatio -> 1/2]
```

`"EvolutionGraph"` gives the graph of states and events, each event joined to its input and output states:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "EvolutionGraph", ImageSize -> 600, AspectRatio -> 1/2]
```

`"EvolutionCausalGraph"` adds the causal edges:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "EvolutionCausalGraph", ImageSize -> 600, AspectRatio -> 1/2]
```

`"EvolutionBranchialGraph"` adds the branchial edges:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "EvolutionBranchialGraph", ImageSize -> 600, AspectRatio -> 1/2]
```

`"EvolutionCausalBranchialGraph"`, the default, adds both:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "EvolutionCausalBranchialGraph", ImageSize -> 600, AspectRatio -> 1/2]
```

---

The suffix `Structure` gives the same graph with plain vertices:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "StatesGraphStructure", AspectRatio -> 1/2]
```

Its vertices are the state ids and its edges carry the event id:

```wl
EdgeList[HGEvolve[chain, {{1, 2}}, 2, "StatesGraphStructure"]]
```

<!-- => {DirectedEdge[0, 1, <|"EventId" -> 0|>], DirectedEdge[1, 2, <|"EventId" -> 1|>], DirectedEdge[1, 3, <|"EventId" -> 2|>]} -->

The vertices of a causal graph are events, named `{"E", id}`:

```wl
EdgeList[HGEvolve[chain, {{1, 2}}, 2, "CausalGraphStructure"]]
```

<!-- => {DirectedEdge[{"E", 0}, {"E", 1}, <|"ProducerEvent" -> 0, "ConsumerEvent" -> 1|>], DirectedEdge[{"E", 0}, {"E", 2}, <|"ProducerEvent" -> 0, "ConsumerEvent" -> 2|>]} -->

An evolution graph names states `{"S", id}` and events `{"E", id}`, with an undirected edge from a state to each event applied to it and a directed edge from the event to its output state:

```wl
EdgeList[HGEvolve[chain, {{1, 2}}, 1, "EvolutionGraphStructure"]]
```

<!-- => {UndirectedEdge[{"S", 0}, {"E", 0}, <|"EventId" -> 0|>], DirectedEdge[{"E", 0}, {"S", 1}, <|"EventId" -> 0|>]} -->

Two steps of the chain rule have six state and event edges and two causal pairs:

```wl
EdgeCount[HGEvolve[chain, {{1, 2}}, 2, "EvolutionCausalGraphStructure"]]
```

<!-- => 8 -->

One step of the triangle rule has four state and event edges and one branchial pair:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
EdgeCount[HGEvolve[triangle, {{1, 2}, {1, 3}}, 1, "EvolutionBranchialGraphStructure"]]
```

<!-- => 5 -->

The two events of that step match the two edges in the two orders and consume both, so the one branchial edge joins the two states they produced:

```wl
Map[Sort] @ Values @ EdgeTags @ HGEvolve[triangle, {{1, 2}, {1, 3}}, 1, "BranchialGraphStructure"]
```

<!-- => {{1, 2}} -->

---

`"NumStates"`, `"NumEvents"`, `"NumCausalEdges"` and `"NumBranchialEdges"` give the four counts:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, #] & /@ {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}
```

<!-- => {19, 18, 16, 9} -->

`"Debug"` gives the four counts as one association:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "Debug"]
```

<!-- => <|"NumStates" -> 19, "NumEvents" -> 18, "NumCausalEdges" -> 16, "NumBranchialEdges" -> 9|> -->

---

`"GlobalEdges"` gives every hyperedge the evolution created, each as its id followed by its vertices:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 2, "GlobalEdges"]
```

<!-- => {{0, 0, 1}, {1, 0, 2}, {2, 2, 1}, {3, 0, 3}, {4, 3, 2}, {5, 2, 4}, {6, 4, 1}} -->

`"StateBitvectors"` gives the ids of the hyperedges each state holds. Two states that hold the same id hold the same hyperedge:

```wl
HGEvolve[chain, {{1, 2}}, 2, "StateBitvectors"]
```

<!-- => <|0 -> {0}, 1 -> {1, 2}, 2 -> {2, 3, 4}, 3 -> {1, 5, 6}|> -->

---

`"StepStatistics"` gives one association per step. The number of states, of isomorphism classes, and the entropy of the states over the classes, for the first four steps of a rule of the Wolfram Physics Project:

```wl
stats = HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 3, "StepStatistics"];
Lookup[stats, {"Step", "RawStates", "Classes", "ClassEntropyBits"}]
```

The keys of one step:

```wl
Keys[Last[stats]]
```

The summary of one invariant over the states of the last step:

```wl
Last[stats]["Invariants", "VertexCount"]
```

---

`"All"` gives the states, the events, both relations and the four counts in one association:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Keys[HGEvolve[chain, {{1, 2}}, 2, "All"]]
```

<!-- => {"States", "Events", "CausalEdges", "BranchialEdges", "NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"} -->

---

A list of properties gives an association keyed by them, all from one evolution:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, {"NumStates", "NumEvents"}]
```

<!-- => <|"NumStates" -> 10, "NumEvents" -> 9|> -->

## Options

### "CanonicalizeStates"

The setting selects when two states are one state. Under `None`, the default, a hypergraph reached by two histories is two states. Five steps of the chain rule from one edge reach 154 states:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 5, "NumStates", "CanonicalizeStates" -> None]
```

<!-- => 154 -->

`Full` identifies isomorphic states. Every state at one step of the chain rule is a path of the same length, so five steps reach six states:

```wl
HGEvolve[chain, {{1, 2}}, 5, "NumStates", "CanonicalizeStates" -> Full]
```

<!-- => 6 -->

`Automatic` identifies states whose edge lists are identical, vertex names included. The chain rule creates a new vertex at every event, so no two of its states have identical edge lists:

```wl
HGEvolve[chain, {{1, 2}}, 5, "NumStates", "CanonicalizeStates" -> Automatic]
```

<!-- => 154 -->

---

Two identical initial states are one state under `Automatic`:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{{1, 2}}, {{1, 2}}}, 1, "NumStates", "CanonicalizeStates" -> Automatic]
```

<!-- => 3 -->

So are two initial states that differ only in their vertex names, since the vertices of each initial state are numbered from 0 separately:

```wl
HGEvolve[chain, {{{1, 2}}, {{5, 6}}}, 1, "NumStates", "CanonicalizeStates" -> Automatic]
```

<!-- => 3 -->

---

The setting is applied while the evolution runs, so it also determines which states are expanded. Two steps under `None`:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 2, "StatesGraphStructure", "CanonicalizeStates" -> None, AspectRatio -> 1/2]
```

The same two steps under `Full`, where the two paths of three edges are one state:

```wl
HGEvolve[chain, {{1, 2}}, 2, "StatesGraphStructure", "CanonicalizeStates" -> Full, AspectRatio -> 1/2]
```

---

Under `Full`, the `"Edges"` of a state record are written in the canonical labeling of its class:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Dataset[HGEvolve[chain, {{1, 2}}, 2, "States", "CanonicalizeStates" -> Full]]
```

### "CanonicalizeEvents"

The setting selects when two applications are one event. Under `None`, the default, every application is its own event; three steps of the chain rule have nine:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> None]
```

<!-- => 9 -->

`Full` identifies applications with the same canonical input state and the same canonical output state, one event per step here:

```wl
HGEvolve[chain, {{1, 2}}, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full]
```

<!-- => 3 -->

`Automatic` also uses the step and the canonical positions of the consumed and produced edges, so splitting an end edge of a path and splitting a middle edge are different events:

```wl
HGEvolve[chain, {{1, 2}}, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Automatic]
```

<!-- => 6 -->

`"Positional"` reads the edge positions from each state's own vertex labels, as the Wolfram Multicomputation paclet does. The result changes when the states are relabeled:

```wl
HGEvolve[chain, {{1, 2}}, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> "Positional"]
```

<!-- => 6 -->

A list of components selects the identity; here the input state, the output state and the rule:

```wl
HGEvolve[chain, {{1, 2}}, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> {"InputState", "OutputState", "Rule"}]
```

<!-- => 3 -->

---

Under `Full`, the two applications at step 2 have the same `"CanonicalId"` and `"CanonicalOutputState"`:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Dataset[HGEvolve[chain, {{1, 2}}, 2, "Events", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full]]
```

---

`"EventClasses"` gives, for each event, the rule applications it stands for, keyed by the event's `"CanonicalId"`. Under `Full` the nine applications of three steps are three events, standing for one, two and six applications:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "EventClasses", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full]
```

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Sort[Length /@ Values[HGEvolve[chain, {{1, 2}}, 3, "EventClasses", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full]]]
```

<!-- => {1, 2, 6} -->

---

The causal graph has one vertex per event under the identity in use:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "CausalGraphStructure", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Automatic, AspectRatio -> 1/2]
```

### "CausalTransitiveReduction"

By default a causal pair implied by a longer causal path is dropped. The reduced causal relation of the loops rule after three steps:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumCausalEdges"]
```

<!-- => 24 -->

With `False`, every pair is kept:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumCausalEdges", "CausalTransitiveReduction" -> False]
```

<!-- => 36 -->

[TransitiveReductionGraph]() of the unreduced causal graph has as many edges as the reduced relation:

```wl
EdgeCount @ TransitiveReductionGraph @ HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "CausalGraphStructure", "CausalTransitiveReduction" -> False]
```

<!-- => 24 -->

### "MaxSuccessorStatesPerParent"

A positive value caps the successors kept per state, in the order they arrive. Three steps of the loops rule without a cap:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates"]
```

<!-- => 27 -->

With at most two successors per state:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "MaxSuccessorStatesPerParent" -> 2, "RandomSeed" -> 7]
```

<!-- => 15 -->

No state of the capped states graph has more than two successors:

```wl
Max @ VertexOutDegree @ HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "StatesGraphStructure", "MaxSuccessorStatesPerParent" -> 2, "RandomSeed" -> 7]
```

<!-- => 2 -->

The bound holds at any thread count. Which successors are kept depends on the thread schedule, so a capped run can keep different states on another thread count.

### "MaxStatesPerStep"

A positive value caps the states kept per step, in the order they arrive:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "MaxStatesPerStep" -> 2, "RandomSeed" -> 7]
```

<!-- => 7 -->

The number of states at each step, as pairs of step and count:

```wl
Sort @ Tally @ Lookup["Step"] @ Values @ HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "States", "MaxStatesPerStep" -> 2, "RandomSeed" -> 7]
```

<!-- => {{0, 1}, {1, 2}, {2, 2}, {3, 2}} -->

Without the cap there are 1, 2, 6 and 18 states at the four steps:

```wl
Sort @ Tally @ Lookup["Step"] @ Values @ HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "States"]
```

<!-- => {{0, 1}, {1, 2}, {2, 6}, {3, 18}} -->

### "ExplorationProbability"

Below 1, each state is explored with the given probability:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "ExplorationProbability" -> 0.5, "RandomSeed" -> 7]
```

<!-- => 3 -->

Another seed explores other states:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "ExplorationProbability" -> 0.5, "RandomSeed" -> 8]
```

<!-- => 12 -->

---

A small probability can end the evolution before the requested depth; with 0.01 no state after step 1 is explored:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
Sort @ Tally @ Lookup["Step"] @ Values @ HGEvolve[loops, {{1, 1}, {1, 1}}, 6, "States", "ExplorationProbability" -> 0.01, "RandomSeed" -> 7]
```

<!-- => {{0, 1}, {1, 2}} -->

### "TransitionRate"

Below 1, each transition is kept with the given probability. The draw depends on the transition's isomorphism-invariant identity and the seed, so the same seed keeps the same transitions at any thread count and on either device:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 4, "NumStates", "TransitionRate" -> 0.25, "RandomSeed" -> 7]
```

<!-- => 7 -->

The four counts of that sample:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 4, "Debug", "TransitionRate" -> 0.25, "RandomSeed" -> 7]
```

<!-- => <|"NumStates" -> 7, "NumEvents" -> 6, "NumCausalEdges" -> 5, "NumBranchialEdges" -> 1|> -->

Without sampling, four steps reach 117 states:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 4, "NumStates"]
```

<!-- => 117 -->

---

A state whose every draw failed keeps its lowest-keyed transition, so the sample reaches the requested depth at any rate. At 0.01 a single line of states reaches step 6:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
Sort @ Tally @ Lookup["Step"] @ Values @ HGEvolve[loops, {{1, 1}, {1, 1}}, 6, "States", "TransitionRate" -> 0.01, "RandomSeed" -> 7]
```

<!-- => {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}} -->

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 6, "StatesGraphStructure", "TransitionRate" -> 0.01, "RandomSeed" -> 7, AspectRatio -> 1/2]
```

### "RuleWeights"

Per-rule multipliers on `"TransitionRate"`, in rule order; a rule's transitions are kept with the product of the two. Weights `{1, 0}` drop the second rule and keep every transition of the first:

```wl
extend = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
Sort @ Tally @ Lookup["RuleIndex"] @ Values @ HGEvolve[extend, {{1, 1}, {1, 1}}, 2, "Events", "RuleWeights" -> {1, 0}]
```

<!-- => {{0, 8}} -->

A fractional weight thins that rule's transitions:

```wl
Sort @ Tally @ Lookup["RuleIndex"] @ Values @ HGEvolve[extend, {{1, 1}, {1, 1}}, 2, "Events", "RuleWeights" -> {1., 0.25}, "RandomSeed" -> 7]
```

<!-- => {{0, 8}, {1, 4}} -->

---

Rules after the end of the list have weight 1, so `{0.25}` and `{0.25, 1}` give the same evolution:

```wl
extend = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
{HGEvolve[extend, {{1, 1}, {1, 1}}, 3, "NumEvents", "RuleWeights" -> {0.25}, "RandomSeed" -> 7], HGEvolve[extend, {{1, 1}, {1, 1}}, 3, "NumEvents", "RuleWeights" -> {0.25, 1}, "RandomSeed" -> 7]}
```

<!-- => {145, 145} -->

### "ExploreFromCanonicalStatesOnly"

With `True`, each isomorphism class is expanded once, at its shortest depth, instead of once for each of its states; the setting needs `"CanonicalizeStates" -> Full`. The four counts of the triangle rule after four steps with every state expanded:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 4, "Debug", "CanonicalizeStates" -> Full]
```

<!-- => <|"NumStates" -> 13, "NumEvents" -> 74, "NumCausalEdges" -> 72, "NumBranchialEdges" -> 61|> -->

The same counts with one state per class expanded. The events and relations of the other states are reconstructed:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 4, "Debug", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]
```

<!-- => <|"NumStates" -> 13, "NumEvents" -> 74, "NumCausalEdges" -> 72, "NumBranchialEdges" -> 61|> -->

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 4, "StatesGraphStructure", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True, AspectRatio -> 1/2]
```

---

When the events and branchial pairs are asked for only as `"NumEvents"` and `"NumBranchialEdges"`, they are computed from the number of states in each class, without building the individual events. Twenty steps of the chain rule:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 20, {"NumStates", "NumEvents"}, "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]
```

### "QuotientInitialStates"

With `True`, isomorphic initial states become one root under `"ExploreFromCanonicalStatesOnly"`. Two isomorphic paths as initial states:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{{1, 2}, {2, 3}}, {{1, 2}, {3, 1}}}, 2, "Debug", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True, "QuotientInitialStates" -> True]
```

<!-- => <|"NumStates" -> 3, "NumEvents" -> 16, "NumCausalEdges" -> 8, "NumBranchialEdges" -> 0|> -->

`Full` already identifies the two roots, so the counts are the same without the option:

```wl
HGEvolve[chain, {{{1, 2}, {2, 3}}, {{1, 2}, {3, 1}}}, 2, "Debug", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]
```

<!-- => <|"NumStates" -> 3, "NumEvents" -> 16, "NumCausalEdges" -> 8, "NumBranchialEdges" -> 0|> -->

Under `None` the two roots and their successors are separate:

```wl
HGEvolve[chain, {{{1, 2}, {2, 3}}, {{1, 2}, {3, 1}}}, 2, "NumStates"]
```

<!-- => 18 -->

### "MatchesPerStateRule"

A positive value keeps at most that many of a state's transitions for each rule. They are chosen after the state's matching is complete, by the transitions' identities and the seed, so the same transitions are kept at any thread count. Keeping one transition per state makes four steps of the chain rule a single path:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
EdgeList[HGEvolve[chain, {{1, 2}}, 4, "StatesGraphStructure", "MatchesPerStateRule" -> 1, "RandomSeed" -> 7]]
```

<!-- => {DirectedEdge[0, 1, <|"EventId" -> 0|>], DirectedEdge[1, 2, <|"EventId" -> 1|>], DirectedEdge[2, 3, <|"EventId" -> 2|>], DirectedEdge[3, 4, <|"EventId" -> 3|>]} -->

Keeping two:

```wl
HGEvolve[chain, {{1, 2}}, 4, "NumStates", "MatchesPerStateRule" -> 2, "RandomSeed" -> 7]
```

<!-- => 16 -->

---

The cap applies to each rule separately:

```wl
extend = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
Sort @ Tally @ Lookup["RuleIndex"] @ Values @ HGEvolve[extend, {{1, 1}, {1, 1}}, 3, "Events", "MatchesPerStateRule" -> 1, "RandomSeed" -> 7]
```

<!-- => {{0, 7}, {1, 7}} -->

### "UniformRandom"

With `True`, `"MatchesPerStep"` caps the states kept per step, in the order they arrive. With a cap of 1 the evolution follows one history:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
Sort @ Tally @ Lookup["Step"] @ Values @ HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "States", "UniformRandom" -> True, "MatchesPerStep" -> 1, "RandomSeed" -> 7]
```

<!-- => {{0, 1}, {1, 1}, {2, 1}, {3, 1}} -->

The cap keeps the states that arrive first, which depends on the thread schedule. `"TransitionRate"` draws uniformly and reproducibly.

### "MatchesPerStep"

The per-step cap used with `"UniformRandom" -> True`:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "UniformRandom" -> True, "MatchesPerStep" -> 2, "RandomSeed" -> 7]
```

<!-- => 7 -->

Without `"UniformRandom"` it has no effect:

```wl
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "MatchesPerStep" -> 2, "RandomSeed" -> 7]
```

<!-- => 27 -->

### "BranchialStep"

The branchial graph shows the pairs of one step. Under `Automatic`, `"BranchialGraph"` shows the final step and the evolution graphs with branchial edges show every step. Three steps of the triangle rule have nine branchial pairs:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "NumBranchialEdges"]
```

<!-- => 9 -->

The branchial graph of the final step has six of them:

```wl
EdgeCount[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "BranchialGraphStructure"]]
```

<!-- => 6 -->

A step number selects that step; the pairs at steps 1, 2 and 3:

```wl
Table[EdgeCount[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "BranchialGraphStructure", "BranchialStep" -> k]], {k, 3}]
```

<!-- => {1, 2, 6} -->

`All` shows every step:

```wl
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "BranchialGraphStructure", "BranchialStep" -> All]
```

### "EdgeDeduplication"

By default a causal graph has one edge per pair of events. With `False`, it has one edge for each hyperedge the consumer took from the producer, tagged with its `"EdgeIndex"`; the counts and `"CausalEdges"` are unchanged. Three steps of the chain rule under `Full` event identity have three causal pairs:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
EdgeCount[HGEvolve[chain, {{1, 2}}, 3, "CausalGraphStructure", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full]]
```

<!-- => 3 -->

With one edge per shared hyperedge:

```wl
EdgeCount[HGEvolve[chain, {{1, 2}}, 3, "CausalGraphStructure", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full, "EdgeDeduplication" -> False]]
```

<!-- => 8 -->

The keys of one edge of that graph:

```wl
Keys @ First @ EdgeTags @ HGEvolve[chain, {{1, 2}}, 3, "CausalGraphStructure", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> Full, "EdgeDeduplication" -> False]
```

<!-- => {"ProducerEvent", "ConsumerEvent", "EdgeIndex"} -->

### "TargetDevice"

`"GPU"` runs the evolution on the GPU engine, which gives the same counts as the CPU:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
{HGEvolve[chain, {{1, 2}}, 3, "Debug", "TargetDevice" -> "GPU"], HGEvolve[chain, {{1, 2}}, 3, "Debug", "TargetDevice" -> "CPU"]}
```

<!-- => {<|"NumStates" -> 10, "NumEvents" -> 9, "NumCausalEdges" -> 8, "NumBranchialEdges" -> 0|>, <|"NumStates" -> 10, "NumEvents" -> 9, "NumCausalEdges" -> 8, "NumBranchialEdges" -> 0|>} -->

### "RandomSeed"

The seed of every sampling draw, so a sampled evolution with a seed is reproducible:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 4, "NumStates", "TransitionRate" -> 0.25, "RandomSeed" -> 8]
```

<!-- => 9 -->

It is also the seed of a generated initial condition. Another seed places six sprinkled points elsewhere:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
{HGEvolve[triangle, "Sprinkling", 0, "GlobalEdges", "SprinklingDensity" -> 6, "RandomSeed" -> 1], HGEvolve[triangle, "Sprinkling", 0, "GlobalEdges", "SprinklingDensity" -> 6, "RandomSeed" -> 2]}
```

<!-- => {{{0, 0, 1}, {1, 0, 2}, {2, 3, 1}, {3, 3, 2}}, {{0, 0, 1}, {1, 0, 2}, {2, 3, 2}}} -->

### "IncludeCanonicalHashes"

With `True`, each state record has a `"CanonicalHash"`, equal for isomorphic states and the same on every run, so states from separate evolutions can be matched by isomorphism class. The two states of step 2 are both paths of three edges:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Lookup["CanonicalHash"] @ Values @ HGEvolve[chain, {{1, 2}}, 2, "States", "IncludeCanonicalHashes" -> True]
```

<!-- => {2342368696698430095, 7620563070704287110, 7597357389603944803, 7597357389603944803} -->

The ten states of three steps are in four classes, of one, one, two and six states:

```wl
Values @ Counts @ Lookup["CanonicalHash"] @ Values @ HGEvolve[chain, {{1, 2}}, 3, "States", "IncludeCanonicalHashes" -> True]
```

<!-- => {1, 1, 2, 6} -->

### "ShowGenesisEvents"

With `True`, the event that creates each initial state is part of the result, with `"RuleIndex"` 65535, and the causal pairs from it to the events that consumed the initial edges are counted:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 2, "Debug", "ShowGenesisEvents" -> True]
```

<!-- => <|"NumStates" -> 4, "NumEvents" -> 4, "NumCausalEdges" -> 3, "NumBranchialEdges" -> 0|> -->

Without it, the initial edges have no producer:

```wl
HGEvolve[chain, {{1, 2}}, 2, "Debug"]
```

<!-- => <|"NumStates" -> 4, "NumEvents" -> 3, "NumCausalEdges" -> 2, "NumBranchialEdges" -> 0|> -->

The genesis event is the first vertex of the causal graph:

```wl
HGEvolve[chain, {{1, 2}}, 2, "CausalGraphStructure", "ShowGenesisEvents" -> True, AspectRatio -> 1/2]
```

### "ColorByRule"

With `True`, a styled graph colors each transition by the rule that fired it and adds a legend of the rules:

```wl
extend = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
HGEvolve[extend, {{1, 1}, {1, 1}}, 1, "StatesGraph", "ColorByRule" -> True, ImageSize -> 420]
```

The result is a [Legended]() graph:

```wl
Head[HGEvolve[extend, {{1, 1}, {1, 1}}, 1, "StatesGraph", "ColorByRule" -> True]]
```

<!-- => Legended -->

### "MultiedgeStyle"

Under `Full`, many events join the same two states. Five steps of the chain rule have 153 events between six states:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
EdgeCount[HGEvolve[chain, {{1, 2}}, 5, "StatesGraph", "CanonicalizeStates" -> Full]]
```

<!-- => 153 -->

`"Merged"` draws the events between two states as one edge, whose tooltip lists them:

```wl
HGEvolve[chain, {{1, 2}}, 5, "StatesGraph", "CanonicalizeStates" -> Full, "MultiedgeStyle" -> "Merged", ImageSize -> 400, AspectRatio -> 1/2]
```

```wl
EdgeCount[HGEvolve[chain, {{1, 2}}, 5, "StatesGraph", "CanonicalizeStates" -> Full, "MultiedgeStyle" -> "Merged"]]
```

<!-- => 5 -->

### "ShowProgress"

With `True`, the engine's progress is printed; the result is unchanged:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "NumStates", "ShowProgress" -> True]
```

<!-- => 10 -->

### "DebugFFI"

With `True`, the properties requested from the engine, the data they need and the keys and size of the engine's reply are printed:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "NumStates", "DebugFFI" -> True]
```

<!-- => 10 -->

### Graph options

Every option of [Graph]() goes to the graph a graph property returns, and a value given in the call replaces the default:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {{1, 2}}, 3, "StatesGraphStructure", AspectRatio -> 1/3, ImageSize -> 300]
```

```wl
Options[HGEvolve[chain, {{1, 2}}, 3, "StatesGraphStructure", AspectRatio -> 1/3], AspectRatio]
```

<!-- => {AspectRatio -> 1/3} -->

[VertexLabels]() shows the state ids of a `Structure` graph:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 2, "StatesGraphStructure", VertexLabels -> Automatic, AspectRatio -> 1/2]
```

### "GridWidth"

The width of a grid, and the resolution of a curved surface. A grid 3 wide and 10 high, the default height, has 47 edges:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
Length[HGEvolve[triangle, "Grid", 0, "GlobalEdges", "GridWidth" -> 3]]
```

<!-- => 47 -->

A sphere of resolution 2 has 9 edges, and a torus of resolution 3 has 18:

```wl
{Length[HGEvolve[triangle, "Sphere", 0, "GlobalEdges", "GridWidth" -> 2]], Length[HGEvolve[triangle, "Torus", 0, "GlobalEdges", "GridWidth" -> 3]]}
```

<!-- => {9, 18} -->

### "GridHeight"

The height of a grid, a cylinder or a Klein bottle. A cylinder and a Klein bottle of resolution 3 and height 2:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
{Length[HGEvolve[triangle, "Cylinder", 0, "GlobalEdges", "GridWidth" -> 3, "GridHeight" -> 2]], Length[HGEvolve[triangle, "Klein", 0, "GlobalEdges", "GridWidth" -> 3, "GridHeight" -> 2]]}
```

<!-- => {9, 9} -->

### "GridHoles"

Holes cut from a grid, each as `{x, y, radius}`. A 4 by 4 grid has 24 edges, and 20 with a hole of radius 1 at its center:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
{Length[HGEvolve[triangle, "Grid", 0, "GlobalEdges", "GridWidth" -> 4, "GridHeight" -> 4]], Length[HGEvolve[triangle, "Grid", 0, "GlobalEdges", "GridWidth" -> 4, "GridHeight" -> 4, "GridHoles" -> {{2, 2, 1}}]]}
```

<!-- => {24, 20} -->

### "SprinklingDensity"

The number of points of a sprinkling, a Brill-Lindquist, a Poisson or a uniform initial condition. Six points sprinkled into Minkowski space give a small causal set:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, "Sprinkling", 0, "GlobalEdges", "SprinklingDensity" -> 6, "RandomSeed" -> 1]
```

<!-- => {{0, 0, 1}, {1, 0, 2}, {2, 3, 1}, {3, 3, 2}} -->

Twenty points give 42 edges, and ten uniform points give 15:

```wl
{Length[HGEvolve[triangle, "Sprinkling", 0, "GlobalEdges", "SprinklingDensity" -> 20, "RandomSeed" -> 1]], Length[HGEvolve[triangle, "Uniform", 0, "GlobalEdges", "SprinklingDensity" -> 10, "RandomSeed" -> 1]]}
```

<!-- => {42, 15} -->

### Sprinkling options

The number of edges of a sprinkling of twenty points under each option:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
sprinkle[opts___] := Length[HGEvolve[triangle, "Sprinkling", 0, "GlobalEdges", "SprinklingDensity" -> 20, "RandomSeed" -> 1, opts]];
<|"TimeExtent" -> sprinkle["SprinklingTimeExtent" -> 2.],
  "SpatialExtent" -> sprinkle["SprinklingSpatialExtent" -> 2.],
  "SpatialDim 1" -> sprinkle["SprinklingSpatialDim" -> 1],
  "SpatialDim 3" -> sprinkle["SprinklingSpatialDim" -> 3],
  "LightconeAngle" -> sprinkle["SprinklingLightconeAngle" -> 0.5],
  "AlexandrovCutoff" -> sprinkle["SprinklingAlexandrovCutoff" -> 2.],
  "TransitivityReduction" -> sprinkle["SprinklingTransitivityReduction" -> False],
  "MaxEdgesPerVertex" -> sprinkle["SprinklingMaxEdgesPerVertex" -> 2]|>
```

<!-- => <|"TimeExtent" -> 3, "SpatialExtent" -> 37, "SpatialDim 1" -> 34, "SpatialDim 3" -> 21, "LightconeAngle" -> 18, "AlexandrovCutoff" -> 12, "TransitivityReduction" -> 48, "MaxEdgesPerVertex" -> 23|> -->

### Brill-Lindquist options

The number of edges of twenty Brill-Lindquist points in a 10 by 10 box under each option:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
bl[opts___] := Length[HGEvolve[triangle, "BrillLindquist", 0, "GlobalEdges", "SprinklingDensity" -> 20, "RandomSeed" -> 1, opts, "BrillLindquistBoxX" -> {-5., 5.}, "BrillLindquistBoxY" -> {-5., 5.}]];
<|"Default" -> bl[], "Mass1" -> bl["BrillLindquistMass1" -> 1.], "Mass2" -> bl["BrillLindquistMass2" -> 1.],
  "Separation" -> bl["BrillLindquistSeparation" -> 4.]|>
```

<!-- => <|"Default" -> 26, "Mass1" -> 31, "Mass2" -> 33, "Separation" -> 18|> -->

`"BrillLindquistBoxX"` and `"BrillLindquistBoxY"` give the domain; with one of them at its default, the same points spread over a larger area:

```wl
{Length[HGEvolve[triangle, "BrillLindquist", 0, "GlobalEdges", "SprinklingDensity" -> 20, "RandomSeed" -> 1, "BrillLindquistBoxX" -> {-5., 5.}]], Length[HGEvolve[triangle, "BrillLindquist", 0, "GlobalEdges", "SprinklingDensity" -> 20, "RandomSeed" -> 1, "BrillLindquistBoxY" -> {-5., 5.}]]}
```

<!-- => {12, 7} -->

### "EdgeThreshold"

The largest distance joined by an edge of a Brill-Lindquist, a Poisson or a uniform initial condition. Ten Poisson points with the default threshold:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, "Poisson", 0, "GlobalEdges", "SprinklingDensity" -> 10, "RandomSeed" -> 1]
```

<!-- => {{0, 0, 1}, {1, 2, 3}, {2, 4, 2}, {3, 5, 6}, {4, 0, 3}, {5, 3, 4}} -->

A threshold of 3 on ten Poisson points and on ten uniform points:

```wl
{Length[HGEvolve[triangle, "Poisson", 0, "GlobalEdges", "SprinklingDensity" -> 10, "RandomSeed" -> 1, "EdgeThreshold" -> 3.]], Length[HGEvolve[triangle, "Uniform", 0, "GlobalEdges", "SprinklingDensity" -> 10, "RandomSeed" -> 1, "EdgeThreshold" -> 3.]]}
```

<!-- => {11, 9} -->

### "PoissonMinDistance"

The least separation of the points of a Poisson initial condition:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
Length[HGEvolve[triangle, "Poisson", 0, "GlobalEdges", "SprinklingDensity" -> 10, "RandomSeed" -> 1, "PoissonMinDistance" -> 3.]]
```

<!-- => 20 -->

## Applications

A causal set of 12 points sprinkled into two-dimensional Minkowski space, as the hyperedges of its causal links:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, <|"Type" -> "Sprinkling", "Density" -> 12, "Seed" -> 1|>, 0, "GlobalEdges"]
```

<!-- => {{0, 0, 1}, {1, 0, 2}, {2, 0, 3}, {3, 0, 4}, {4, 0, 5}, {5, 6, 1}, {6, 6, 2}, {7, 6, 3}, {8, 6, 4}, {9, 7, 4}, {10, 7, 5}, {11, 8, 3}, {12, 8, 4}, {13, 8, 5}, {14, 1, 9}, {15, 10, 4}, {16, 10, 5}, {17, 11, 9}} -->

One step of the triangle rule closes a triangle over each pair of links with the same first vertex, and reaches 43 states:

```wl
HGEvolve[triangle, <|"Type" -> "Sprinkling", "Density" -> 12, "Seed" -> 1|>, 1, "NumStates"]
```

<!-- => 43 -->

```wl
HGEvolve[triangle, <|"Type" -> "Sprinkling", "Density" -> 12, "Seed" -> 1|>, 1, "StatesGraphStructure", ImageSize -> 450, AspectRatio -> 1/2]
```

Two steps reach 239 states up to isomorphism:

```wl
HGEvolve[triangle, <|"Type" -> "Sprinkling", "Density" -> 12, "Seed" -> 1|>, 2, "NumStates", "CanonicalizeStates" -> Full]
```

<!-- => 239 -->

## Properties and Relations

The count, the list and the graph of the causal relation agree:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
{HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "NumCausalEdges"], Length[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "CausalEdges"]], EdgeCount[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "CausalGraphStructure"]]}
```

<!-- => {16, 16, 16} -->

So do the count, the records and the graph of the states:

```wl
{HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "NumStates"], Length[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "States"]], VertexCount[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "StatesGraphStructure"]]}
```

<!-- => {19, 19, 19} -->

The branchial graph over every step has one edge per branchial pair:

```wl
{HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "NumBranchialEdges"], Length[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "BranchialEdges"]], EdgeCount[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "BranchialGraphStructure", "BranchialStep" -> All]]}
```

<!-- => {9, 9, 9} -->

---

The number of distinct canonical hashes among the states is the number of states under `Full`:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
{Length @ Counts @ Lookup["CanonicalHash"] @ Values @ HGEvolve[chain, {{1, 2}}, 3, "States", "IncludeCanonicalHashes" -> True], HGEvolve[chain, {{1, 2}}, 3, "NumStates", "CanonicalizeStates" -> Full]}
```

<!-- => {4, 4} -->

---

The states and events counted by `"StepStatistics"` over all steps are the states and events of the evolution:

```wl
wolfram = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}};
stats = HGEvolve[wolfram, {{1, 2}, {1, 3}}, 3, "StepStatistics"];
{Total[Lookup[stats, "RawStates"]], Total[Lookup[stats, "Events"]]} === Values[HGEvolve[wolfram, {{1, 2}, {1, 3}}, 3, {"NumStates", "NumEvents"}]]
```

<!-- => True -->

Under `"ExploreFromCanonicalStatesOnly"` the statistics are the same:

```wl
stats === HGEvolve[wolfram, {{1, 2}, {1, 3}}, 3, "StepStatistics", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]
```

<!-- => True -->

---

Expanding one state per class gives every property of expanding every state:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 4, "Debug", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True] === HGEvolve[triangle, {{1, 2}, {1, 3}}, 4, "Debug", "CanonicalizeStates" -> Full]
```

<!-- => True -->

---

A sampled evolution with a seed gives the same result each time it runs:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 4, "Debug", "TransitionRate" -> 0.25, "RandomSeed" -> 7] === HGEvolve[loops, {{1, 1}, {1, 1}}, 4, "Debug", "TransitionRate" -> 0.25, "RandomSeed" -> 7]
```

<!-- => True -->

---

`"Positional"` and `Automatic` event identities agree on two steps of the chain rule:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
Table[HGEvolve[chain, {{1, 2}}, 2, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> id], {id, {"Positional", Automatic}}]
```

<!-- => {3, 3} -->

They differ on three steps of the triangle rule, where the positions of the edges depend on the vertex labels:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
Table[HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "NumEvents", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> id], {id, {"Positional", Automatic}}]
```

<!-- => {11, 10} -->

---

[HGEvolve]() runs the evolution from the initial state on every call. A session continues one evolution; after *k* steps it gives what [HGEvolve]() gives for *k* steps:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
session = HGSessionOpen[chain, {{1, 2}}, "NumStates"];
{Table[HGSessionStep[session, 1], 3], Table[HGEvolve[chain, {{1, 2}}, k, "NumStates"], {k, 3}]}
```

<!-- => {{2, 4, 10}, {2, 4, 10}} -->

```wl
HGSessionClose[session]
```

<!-- => Null -->

## Possible Issues

The number of states grows exponentially with the number of steps:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
Table[HGEvolve[triangle, {{1, 2}, {1, 3}}, k, "NumStates"], {k, 0, 4}]
```

<!-- => {1, 3, 7, 19, 75} -->

`"CanonicalizeStates" -> Full` counts the states up to isomorphism:

```wl
Table[HGEvolve[triangle, {{1, 2}, {1, 3}}, k, "NumStates", "CanonicalizeStates" -> Full], {k, 0, 4}]
```

<!-- => {1, 2, 4, 7, 13} -->

---

A styled graph draws every state, so a styled picture of a large evolution is slow to render and hard to read. The `Structure` graphs and `"CanonicalizeStates" -> Full` keep a picture small. The Wolfram Physics Project rule reaches 45 states up to isomorphism in four steps:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 4, "StatesGraphStructure", "CanonicalizeStates" -> Full, ImageSize -> 450, AspectRatio -> 1/2]
```

---

A run capped by `"MaxStatesPerStep"` with no `"RandomSeed"` can keep different states each time:

```wl
loops = {{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}};
HGEvolve[loops, {{1, 1}, {1, 1}}, 3, "NumStates", "MaxStatesPerStep" -> 2]
```

<!-- => 7; the message HGEvolve::warn is issued -->

---

`"ExploreFromCanonicalStatesOnly" -> True` without `"CanonicalizeStates" -> Full` expands every state:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
HGEvolve[triangle, {{1, 2}, {1, 3}}, 3, "NumStates", "ExploreFromCanonicalStatesOnly" -> True]
```

<!-- => 19; the message HGEvolve::warn is issued -->

---

A rule whose sides are not lists of hyperedges:

```wl
HGEvolve[{{1, 2}} -> 3, {{1, 2}}, 2, "NumStates"]
```

<!-- => $Failed; the message HGEvolve::badrule is issued -->

A flat list is not a hypergraph:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}};
HGEvolve[chain, {1, 2}, 2, "NumStates"]
```

<!-- => $Failed; the message HGEvolve::badinit is issued -->

A number of steps that is not a non-negative integer:

```wl
HGEvolve[chain, {{1, 2}}, -1, "NumStates"]
```

<!-- => $Failed; the message HGEvolve::steps is issued -->

An unknown property:

```wl
HGEvolve[chain, {{1, 2}}, 2, "Foo"]
```

<!-- => $Failed; the message HGEvolve::unknownprop is issued -->

An unknown initial condition:

```wl
HGEvolve[chain, "Foo", 2, "NumStates"]
```

<!-- => $Failed; the message HGEvolve::unknownic is issued -->
