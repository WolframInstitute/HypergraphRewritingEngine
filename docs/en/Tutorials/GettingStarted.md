---
Template: TechNote
Name: GettingStarted
Title: Getting Started with Hypergraph Rewriting
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/tutorial/GettingStarted
Keywords: [hypergraph, multiway, rewriting, tutorial, states graph, causal graph, branchial graph, canonicalization, Wolfram physics]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [Sessions]
---

A hypergraph is a list of hyperedges. A rewrite rule replaces the hyperedges that match its left side by those of its right side. [HGEvolve]() applies a rule to a hypergraph in every possible way for a number of steps, and gives the states reached, the events between them, and the causal and branchial relations between the events.

## Hypergraphs and rules

A hyperedge is an ordered list of vertices. A hyperedge may have any number of vertices, and the same hyperedge may occur more than once.

A hypergraph with one binary edge from vertex 1 to vertex 2:

```wl
oneEdge = {{1, 2}}
```

<!-- => {{1, 2}} -->

---

A rewrite rule is a [Rule]() whose two sides are hypergraphs. The vertices of a rule are pattern variables. A vertex on both sides is kept by every application, and a vertex only on the right is created by every application.

The chain rule replaces one edge by a path of two edges through a new vertex:

```wl
chain = {{1, 2}} -> {{1, 3}, {3, 2}}
```

<!-- => {{1, 2}} -> {{1, 3}, {3, 2}} -->

---

A binary hyperedge is a directed edge, so [Graph]() can draw each side of the rule:

```wl
Map[Graph[DirectedEdge @@@ #, VertexLabels -> Automatic, ImageSize -> Small] &, {First[chain], Last[chain]}]
```

## One step, then three

<code>[HGEvolve]()[*rules*, *init*, *n*, *prop*]</code> evolves the hypergraph *init* under *rules* for *n* steps and gives the property *prop*. `"StatesGraph"` is the graph whose vertices are the states reached and whose edges are the events between them; each state is drawn as its hypergraph.

One step of the chain rule from one edge has one match, so the states graph has the initial state and one successor:

```wl
HGEvolve[chain, oneEdge, 1, "StatesGraph", ImageSize -> 200]
```

---

The path of two edges has two matches, so the second step adds two states, each a path of three edges. Each of those has three matches, so the third step adds six:

```wl
HGEvolve[chain, oneEdge, 3, "StatesGraph", ImageSize -> 540]
```

All states at one step are paths of the same length. They are separate states because they were reached by different histories, that is, by splitting the edges in a different order.

## The evolution graph

`"EvolutionGraph"` also draws the events. A state is a blue box. An event is a yellow box showing its input state, with the consumed edge dashed, and its output state, with the produced edges highlighted. Gray edges lead from a state to its events and from an event to its output state.

Two steps of the chain rule; the [AspectRatio]() option sets the height of the graph relative to its width:

```wl
HGEvolve[chain, oneEdge, 2, "EvolutionGraph", ImageSize -> 420, AspectRatio -> 1]
```

---

With no *prop*, [HGEvolve]() gives the evolution graph with the causal and branchial edges, `"EvolutionCausalBranchialGraph"`. An orange causal edge runs from an event to a later event that consumed an edge it produced. A pink branchial edge joins two events applied to the same state that consumed a common edge. The chain rule has no branchial edges, since each of its events consumes one edge and no two matches share one:

```wl
HGEvolve[chain, oneEdge, 2, ImageSize -> 420, AspectRatio -> 1]
```

## Choosing what comes back

The fourth argument selects the result. The graph properties are `"StatesGraph"`, `"CausalGraph"`, `"BranchialGraph"`, `"EvolutionGraph"`, `"EvolutionCausalGraph"`, `"EvolutionBranchialGraph"` and the default `"EvolutionCausalBranchialGraph"`.

Each takes the suffix `Structure` for the same graph with plain vertices, which draws faster:

```wl
HGEvolve[chain, oneEdge, 3, "StatesGraphStructure", ImageSize -> 400]
```

---

`"States"` gives the state records, an [Association]() keyed by state id:

```wl
Dataset[states = HGEvolve[chain, oneEdge, 2, "States"]]
```

A record has the keys:

|   |   |
|---|---|
| `"Id"` | the state id |
| `"CanonicalId"` | the id of the state's class under `"CanonicalizeStates"`; its own id under the default `None` |
| `"ContentStateId"` | the lowest id of the states with the same edge list |
| `"Step"` | the step at which the state was reached |
| `"Edges"` | the hyperedges, each as its edge id followed by its vertices |
| `"IsInitial"` | 1 for an initial state, 0 otherwise |

The engine numbers vertices from 0 in order of first appearance and edges from 0 in order of creation, so the initial edge `{1, 2}` is `{0, 0, 1}`, edge 0 from vertex 0 to vertex 1:

```wl
states[0]
```

<!-- => <|"Id" -> 0, "CanonicalId" -> 0, "ContentStateId" -> 0, "Step" -> 0, "Edges" -> {{0, 0, 1}}, "IsInitial" -> 1|> -->

---

`"Events"` gives the event records, keyed by event id:

```wl
Dataset[HGEvolve[chain, oneEdge, 2, "Events"]]
```

An event record has the keys `"Id"`; `"CanonicalId"`, its id under `"CanonicalizeEvents"`; `"RuleIndex"`, the position of the rule that fired, counted from 0; `"InputState"` and `"OutputState"`; `"CanonicalInputState"` and `"CanonicalOutputState"`, the same two under `"CanonicalizeStates"`; and `"ConsumedEdges"` and `"ProducedEdges"`, the ids of the edges it removed and created.

State and event ids are assigned in the order the engine creates them, which can differ between two evolutions of the same input. To use the ids of states and events together, get both from one evolution with the property `{"States", "Events"}`.

---

With a list of rules, `"RuleIndex"` separates the events of each rule. Two rules that extend an edge by one or by two new edges, applied to two loops at vertex 1, fire twice each in the first step:

```wl
KeySort @ Counts @ Lookup["RuleIndex"] @ Values @ HGEvolve[{{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}, {{1, 1}, {1, 1}}, 1, "Events"]
```

<!-- => <|0 -> 2, 1 -> 2|> -->

---

`"NumStates"`, `"NumEvents"`, `"NumCausalEdges"` and `"NumBranchialEdges"` are the counts. A list of properties gives an association:

```wl
HGEvolve[chain, oneEdge, 3, {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}]
```

<!-- => <|"NumStates" -> 10, "NumEvents" -> 9, "NumCausalEdges" -> 8, "NumBranchialEdges" -> 0|> -->

## Arity and vertex names

This rule splits a ternary hyperedge into two that share the new vertex; a ternary hyperedge is drawn as a filled triangle:

```wl
HGEvolve[{{1, 2, 3}} -> {{1, 2, 4}, {2, 4, 3}}, {{1, 2, 3}}, 2, "StatesGraph", ImageSize -> 420, AspectRatio -> 1/2]
```

---

The vertices of a rule may be symbols. The chain rule written with symbols reaches the same ten states:

```wl
HGEvolve[{{x, y}} -> {{x, z}, {z, y}}, oneEdge, 3, "NumStates"]
```

<!-- => 10 -->

## Identifying states

The number of states grows with the number of histories. For the chain rule, the number of states at steps 0 to 5:

```wl
Table[HGEvolve[chain, oneEdge, k, "NumStates"], {k, 0, 5}]
```

<!-- => {1, 2, 4, 10, 34, 154} -->

Step $k$ adds $k!$ states, one for each order in which the edges were split, and each is a path of $k + 1$ edges.

---

Under the default `"CanonicalizeStates" -> None`, two histories that produce the same hypergraph give two states. A rule that duplicates an edge, applied to two equal edges, matches twice and produces the same three edges twice. The two states have the same `"ContentStateId"` and different `"Id"`:

```wl
Dataset[HGEvolve[{{1, 2}} -> {{1, 2}, {1, 2}}, {{1, 2}, {1, 2}}, 1, "States"]]
```

---

`"CanonicalizeStates" -> Full` identifies isomorphic states, states that are the same up to a renaming of their vertices. Every path of $k + 1$ edges is then one state, so the chain rule has one state per step:

```wl
Table[HGEvolve[chain, oneEdge, k, "NumStates", "CanonicalizeStates" -> Full], {k, 0, 5}]
```

<!-- => {1, 2, 3, 4, 5, 6} -->

Every event is still an edge of the states graph, so under `Full` several edges can join the same two states.

---

`"CanonicalizeStates" -> Automatic` identifies states with the same edge list, without renaming vertices; these are the classes `"ContentStateId"` numbers. It never identifies two states that `Full` keeps apart. The two states of the duplicating rule are one state under `Automatic`:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {1, 2}}, {{1, 2}, {1, 2}}, 1, "NumStates", "CanonicalizeStates" -> Automatic]
```

<!-- => 2 -->

## Causal and branchial graphs

The triangle rule matches two edges from a common vertex, and produces them again with the edge that closes the triangle between their other ends:

```wl
triangle = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}
```

<!-- => {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}} -->

---

Its left side is the initial hypergraph:

```wl
twoEdges = {{1, 2}, {1, 3}}
```

<!-- => {{1, 2}, {1, 3}} -->

---

The two initial edges match the rule in either order, so the initial state has two successors:

```wl
HGEvolve[triangle, twoEdges, 2, "StatesGraph", ImageSize -> 420]
```

---

The two states at step 1 are the same triangle with the closing edge in opposite directions:

```wl
Dataset[HGEvolve[triangle, twoEdges, 1, "States"]]
```

---

`"CausalGraph"` has the events as vertices, and a directed edge from an event to each later event that consumed an edge it produced. An edge implied by a longer causal path is dropped, since `"CausalTransitiveReduction"` is on by default.

The two events of step 1 both consume both initial edges, so neither follows the other, and each starts its own causal tree:

```wl
HGEvolve[triangle, twoEdges, 2, "CausalGraph", ImageSize -> 420, AspectRatio -> 1]
```

---

Two events are branchially related when they were applied to the same state and consumed a common edge, so no single history contains both. `"BranchialGraph"` joins the two states such a pair of events produced. By default it shows the pairs of the final step:

```wl
HGEvolve[triangle, twoEdges, 3, "BranchialGraph", ImageSize -> 540, AspectRatio -> 1/4]
```

---

`"BranchialStep" -> All` shows the pairs of every step, nine of them, against six at the final step:

```wl
HGEvolve[triangle, twoEdges, 3, "BranchialGraph", "BranchialStep" -> All, ImageSize -> 640, AspectRatio -> 1/4]
```

---

The four counts of the same evolution; `"NumBranchialEdges"` counts the pairs of every step:

```wl
HGEvolve[triangle, twoEdges, 3, {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}]
```

<!-- => <|"NumStates" -> 19, "NumEvents" -> 18, "NumCausalEdges" -> 16, "NumBranchialEdges" -> 9|> -->

---

Up to isomorphism, 7 of the 19 states are distinct:

```wl
HGEvolve[triangle, twoEdges, 3, "NumStates", "CanonicalizeStates" -> Full]
```

<!-- => 7 -->

```wl
HGEvolve[triangle, twoEdges, 3, "StatesGraph", "CanonicalizeStates" -> Full, ImageSize -> 450, AspectRatio -> 1/2]
```

## A rule from the Wolfram Physics Project

This rule replaces two edges from a common vertex by the first of them and three edges that meet at a new vertex:

```wl
physicsRule = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}
```

<!-- => {{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}} -->

---

The number of states up to isomorphism at steps 0 to 4:

```wl
Table[HGEvolve[physicsRule, twoEdges, k, "NumStates", "CanonicalizeStates" -> Full], {k, 0, 4}]
```

<!-- => {1, 2, 4, 10, 45} -->

---

The states graph at three steps:

```wl
HGEvolve[physicsRule, twoEdges, 3, "StatesGraph", "CanonicalizeStates" -> Full, ImageSize -> 540, AspectRatio -> 1/2]
```

---

The states graph at four steps, as a `Structure` graph:

```wl
HGEvolve[physicsRule, twoEdges, 4, "StatesGraphStructure", "CanonicalizeStates" -> Full, ImageSize -> 500, AspectRatio -> 1/2]
```

## Generated initial conditions and the GPU

A string names a generated initial condition, shaped by options of [HGEvolve](). The triangle rule on a 4 by 4 grid:

```wl
HGEvolve[triangle, "Grid", 1, "NumStates", "GridWidth" -> 4, "GridHeight" -> 4, "RandomSeed" -> 1]
```

The families are `"Grid"`, `"Cylinder"`, `"Torus"`, `"Sphere"`, `"Klein"`, `"Mobius"`, `"Sprinkling"` (also `"Minkowski"`), `"BrillLindquist"`, `"Poisson"` and `"Uniform"`; the [HGEvolve]() reference page lists their options.

---

`"TargetDevice" -> "GPU"` runs the evolution on the GPU engine, which gives the same result as the CPU:

```wl
HGEvolve[triangle, twoEdges, 5, "NumStates", "TargetDevice" -> "GPU"] === HGEvolve[triangle, twoEdges, 5, "NumStates"]
```

<!-- => True -->

## Continuing an evolution

[HGEvolve]() runs the evolution from the initial state on every call. [HGSessionOpen]() opens an evolution that [HGSessionStep]() continues from where it stopped; see [Continuing an Evolution with Sessions](paclet:WolframInstitute/HypergraphRewriteEngine/tutorial/Sessions).
