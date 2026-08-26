---
Template: Symbol
Name: HGEvolve
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGEvolve
Keywords: [hypergraph, multiway, rewriting, Wolfram physics, causal graph, branchial graph, evolution]
SeeAlso: []
RelatedGuides: [Hypergraph Rewriting Engine]
---

## Usage

`HGEvolve[rules, initial, steps]` performs multiway hypergraph rewriting of the hypergraph `initial` under `rules` for `steps` steps, returning the combined evolution/causal/branchial graph.

`HGEvolve[rules, initial, steps, property]` returns the specified `property`.

## Details & Options

- The *rules* argument is a rule or list of rules of the form *lhs* `->` *rhs*, e.g. `{{1, 2}, {2, 3}} -> {{2, 1}, {3, 2}, {1, 4}}`. Each side is a list of hyperedges and each hyperedge is a list of vertices. Vertices are variables that bind to (not necessarily distinct) vertices of the state hypergraph; a vertex appearing in *rhs* but not *lhs* is a freshly created vertex.
- The *initial* argument is a single hypergraph state or a list of states given as numeric vertex labels, e.g. `{{1, 2}, {2, 3}}`. It may also be a `Graph` or a named initial condition (`"Grid"`, `"Sprinkling"`, ...), shaped by the initial-condition options below.
- The *steps* argument is an integer number of evolution steps.
- The engine applies the rules in all possible ways (multiway evolution), optionally deduplicating isomorphic states, and builds the causal and branchial graphs. It runs in a standalone process, so a crash or abort never affects the notebook.
- The accepted values for *property* include:

|   |   |
|---|---|
| `"StatesGraph"` | graph of state vertices |
| `"CausalGraph"` | graph of event vertices with directed edges denoting a causal relation between events |
| `"BranchialGraph"` | graph of state vertices with undirected edges denoting branchial relationships |
| `"EvolutionGraph"` | graph of state and event vertices |
| `"EvolutionCausalGraph"` | state and event vertices with directed causal edges between events |
| `"EvolutionBranchialGraph"` | state and event vertices with undirected branchial edges |
| `"EvolutionCausalBranchialGraph"` | state and event vertices with both directed causal and undirected branchial edges (the default) |
| `"States"`, `"Events"` | the raw state / event objects (associations, described below) |
| `"CausalEdges"`, `"BranchialEdges"` | the causal / branchial edge lists |
| `"NumStates"`, `"NumEvents"`, `"NumCausalEdges"`, `"NumBranchialEdges"` | the corresponding counts |
| `"Debug"` | an association of the four counts |
| `"All"` | an association of the states, the events, both edge lists and all four counts |
| `"GlobalEdges"` | every edge the evolution created, as `{id, v1, v2, ...}` |
| `"StateBitvectors"` | an association from state id to the ids of the edges that state holds |

- `prop` may also be a list of property strings, in which case an association keyed by those strings is returned.
- Any `*Graph` property may take the suffix `Structure` to return the same graph without vertex styling (a lighter-weight rendering), e.g. `"StatesGraphStructure"`.
- A raw `"States"` result is a list of associations, one per state, with keys `"Id"`, `"CanonicalId"`, `"Step"`, `"IsInitial"`, `"Edges"` (the state's hyperedges), and — when `"IncludeCanonicalHashes" -> True` — `"CanonicalHash"`. A raw `"Events"` result is a list of associations with keys `"Id"`, `"CanonicalId"`, `"RuleIndex"`, `"Step"`, `"InputState"`/`"OutputState"` (state ids), `"ConsumedEdges"`/`"ProducedEdges"` (edge ids), and `"InputStateEdges"`/`"OutputStateEdges"` (the full edge lists).

### Edge identity

Every state is a set of edges drawn from one global store, and the two properties below expose
that store rather than the states built out of it. `"GlobalEdges"` returns each edge once with
its id, and `"StateBitvectors"` says which of those ids each state holds — so a state's contents
are the edges its id list names, and two states sharing an edge name the same id.

### Evolution and output options

- State and event deduplication, exploration limits, and the output content are controlled by:

| Option | Default | |
|---|---|---|
| `"CanonicalizeStates"` | `None` | the identity states are merged by: `None` (every provenance distinct), `Automatic` (equal contents), or `Full` (equal up to isomorphism) |
| `"CanonicalizeEvents"` | `None` | merge equivalent events: `None`, `Full`, `Automatic`, `Positional`, or a list of keys |
| `"CausalTransitiveReduction"` | `True` | remove redundant transitive causal edges |
| `"ExploreFromCanonicalStatesOnly"` | `False` | quotient exploration: expand each canonical state once, at its shortest depth (off by default, so every provenance is explored) |
| `"QuotientInitialStates"` | `False` | collapse isomorphic initial states to a single canonical root (requires `"ExploreFromCanonicalStatesOnly"`); off keeps each initial state a distinct entry point |
| `"MaxSuccessorStatesPerParent"` | `0` | cap the successor states generated from each parent (0 = unlimited); the bound holds at any thread count, which successors meet it is not reproducible |
| `"MaxStatesPerStep"` | `0` | cap the states retained per evolution step (0 = unlimited); the bound holds at any thread count, which states meet it is not reproducible |
| `"ExplorationProbability"` | `1.` | probability of exploring each branch; below 1 prunes stochastically |
| `"TransitionRate"` | `1.` | probability of keeping each transition; reproducible from `"RandomSeed"`, and reaches full depth at any rate |
| `"RuleWeights"` | `{}` | per-rule multipliers on `"TransitionRate"`, in rule order |
| `"UniformRandom"` | `False` | with `"MatchesPerStep"`, cap the states kept per step by arrival order; see `"TransitionRate"` for sampling that is actually uniform |
| `"MatchesPerStep"` | `0` | with `"UniformRandom"`, the per-step cap described above (0 = all) |
| `"BranchialStep"` | `Automatic` | step at which branchial edges are computed: `Automatic`, `All`, `-1` (final), or a 1-based step |
| `"EdgeDeduplication"` | `True` | one causal/branchial edge per event pair, rather than one per shared hyperedge |
| `"TargetDevice"` | `"CPU"` | `"CPU"` or `"GPU"`, where a GPU build is bundled (falls back to CPU with a message otherwise) |
| `"IncludeStateContents"` | `False` | attach each state's hyperedge list to the result |
| `"IncludeEventContents"` | `False` | attach each event's matched/produced edges to the result |
| `"IncludeCanonicalHashes"` | `False` | attach a run-stable isomorphism hash to each state, for fusing results across runs |
| `"ShowProgress"` | `False` | print progress during evolution |
| `"ShowGenesisEvents"` | `False` | include the synthetic genesis events that create the initial states |
| `"AspectRatio"` | `None` | aspect ratio for the returned graph |
| `"DebugFFI"` | `False` | print low-level foreign-function-interface diagnostics |
| `"ColorByRule"` | `False` | colour each transition edge by the rule that produced it; applies to the styled graph properties, whose edge payloads carry the rule index (the `Structure` variants carry topology only and are unaffected) |

### Analysis options

Geometry and physics analyses — dimension, curvature, geodesics, entropy, topological
charge, branchial sharpness — are **not** part of this paclet. They live in the companion
`hypergraph_viz` project, which consumes this engine as a dependency. `HGEvolve` accepts no
options for them; use `Options[HGEvolve]` for the list it does accept.

### Initial-condition options

- Instead of an explicit *initial*, an initial condition can be generated in place by naming it and shaping it with these options:

| Option | Default | |
|---|---|---|
| `"InitialCondition"` | `"Edges"` | `"Edges"`, `"Grid"`, `"Sprinkling"`, `"BrillLindquist"`, `"Poisson"`, `"Uniform"` |
| `"Topology"` | `"Flat"` | `"Flat"`, `"Cylinder"`, `"Torus"`, `"Sphere"`, `"Klein"`, `"Mobius"` |
| `"MajorRadius"` | `10.` | major radius for curved topologies |
| `"MinorRadius"` | `3.` | minor radius for the torus |
| `"GridWidth"`, `"GridHeight"` | `10`, `10` | grid dimensions for the `"Grid"` condition |
| `"GridHoles"` | `{}` | list of `{x, y, radius}` holes in the grid |
| `"SprinklingDensity"` | `500` | number of spacetime points sprinkled |
| `"SprinklingTimeExtent"` | `10.` | time-dimension extent |
| `"SprinklingSpatialExtent"` | `10.` | spatial-dimension extent |
| `"SprinklingSpatialDim"` | `2` | 1, 2, or 3 spatial dimensions |
| `"SprinklingLightconeAngle"` | `1.` | speed of light (`c = 1`) |
| `"SprinklingAlexandrovCutoff"` | `5.` | maximum proper-time separation |
| `"SprinklingTransitivityReduction"` | `True` | remove redundant causal edges in the sprinkling |
| `"SprinklingMaxEdgesPerVertex"` | `50` | connectivity limit |
| `"BrillLindquistMass1"`, `"BrillLindquistMass2"` | `3.`, `3.` | black-hole masses |
| `"BrillLindquistSeparation"` | `10.` | separation between the black holes |
| `"BrillLindquistBoxX"`, `"BrillLindquistBoxY"` | `{-15., 15.}` | spatial domain |
| `"EdgeThreshold"` | `Automatic` | maximum distance for edge creation |
| `"PoissonMinDistance"` | `1.` | minimum separation for Poisson-disk sampling |
| `"RandomSeed"` | `Automatic` | random seed for reproducibility |

- Use `Options[HGEvolve]` for the full list in a session.

## Basic Examples

### Simple binary splitting

Define a rule that splits a binary edge into two edges:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
```

Define an initial state:

```wl
initialEdges = {{1, 2}};
```

Evolve for 3 steps, returning the combined evolution/causal/branchial graph:

```wl
HGEvolve[rules, initialEdges, 3]
```

### Higher-arity edges

A rule that splits a ternary edge into two ternary edges:

```wl
rules = {{{1, 2, 3}} -> {{1, 2, 4}, {2, 4, 3}}};
```

```wl
HGEvolve[rules, {{1, 1, 1}}, 3]
```

### Multiple rules and multiple initial states

Rules and initial states may both be lists; each initial state is a distinct entry point of the multiway system. Isomorphic states are merged with `"CanonicalizeStates" -> Full` to keep the graph compact:

```wl
rule1 = {{1, 2}, {2, 1}} -> {{1, 2, 3}};
rule2 = {{1, 2}, {2, 3}} -> {{1, 3}, {2, 3}, {3, 4}};
rule3 = {{1, 1, 2}} -> {{1, 2}, {2, 1}};
```

```wl
HGEvolve[{rule1, rule2, rule3}, {{{1, 2}, {2, 1}}, {{1, 1, 2}}}, 2, "StatesGraph", "CanonicalizeStates" -> Full]
```

## Scope

### Identifying isomorphic states

By default every distinct provenance is a separate state. `"CanonicalizeStates"` merges states that are equal up to isomorphism, collapsing the multiway graph. Compare the raw and canonical state counts:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
{HGEvolve[rules, {{1, 2}, {1, 3}}, 4, "NumStates"],
 HGEvolve[rules, {{1, 2}, {1, 3}}, 4, "NumStates", "CanonicalizeStates" -> Full]}
```

### Generated initial conditions

An initial condition can be generated instead of passing an explicit edge list. Here a small grid is evolved for two steps (canonicalized to stay compact):

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
HGEvolve[rules, "Grid", 2, "NumStates", "GridWidth" -> 3, "GridHeight" -> 3,
         "CanonicalizeStates" -> Full]
```

### Inspecting the raw states

The `"States"` property returns the state objects keyed by id; each is an association carrying its own hyperedges and metadata. Read the id, step, initial flag, and edges of each:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
states = HGEvolve[rules, {{1, 2}, {1, 3}}, 2, "States"];
Column[{#["Id"], #["Step"], #["IsInitial"], #["Edges"]} & /@ Values[states]]
```

The parallel `"Events"` property returns the update events, each recording its rule, the input and output state ids, and the consumed and produced edges.

### Branchial structure

The branchial graph joins states that branch from a common ancestor. Return it for a small canonicalized evolution:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
HGEvolve[rules, {{1, 2}, {1, 3}}, 3, "BranchialGraph", "CanonicalizeStates" -> Full]
```

### Events, edge lists, and counts

`"Events"` returns the update events, `"CausalEdges"` / `"BranchialEdges"` the raw causal / branchial edge lists, and `"Debug"` an association of the four counts:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
{Length[HGEvolve[rules, {{1, 2}, {1, 3}}, 3, "Events"]],
 Length[HGEvolve[rules, {{1, 2}, {1, 3}}, 3, "CausalEdges"]],
 HGEvolve[rules, {{1, 2}, {1, 3}}, 3, "Debug"]}
```

### Edge identity

`"GlobalEdges"` returns every edge the evolution created with its id, and `"StateBitvectors"`
the ids each state holds. Asking for both is what makes either useful: the ids alone do not say
what an edge is, and the edge list alone does not say which state holds it.

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
{Length[HGEvolve[rules, {{1, 2}}, 2, "GlobalEdges"]],
 Length[HGEvolve[rules, {{1, 2}}, 2, "StateBitvectors"]]}
```

## Options

### "CanonicalizeStates"

This chooses WHEN two states are the same state, and the three settings are three different
questions, not three accuracies of one question. Each is applied as the run proceeds, so it
decides what gets explored and not merely how results are grouped.

`None` asks nothing: every provenance is its own state, so a hypergraph reached by two different
histories appears twice.

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 2, "StatesGraph", "CanonicalizeStates" -> None]
```

`Full` asks whether two states are the same up to isomorphism, exactly. Relabelled copies of one
hypergraph become one state:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 2, "StatesGraph", "CanonicalizeStates" -> Full]
```

`Automatic` asks whether their CONTENTS are equal — the same hyperedges over the same vertex
names. That is cheap, and it is FINER than isomorphism, not an approximation of it: two
isomorphic states whose vertices are labelled differently are different states under
`Automatic`, and stay separate. On a chain rule at 5 steps it merges nothing where `Full`
collapses 154 states to 6. Choose it when vertex names are meaningful and you want them
respected; choose `Full` when they are arbitrary and only the shape matters.

### "CanonicalizeEvents"

Event canonicalization merges equivalent update events. `None` (the default) keeps every
application as its own event. `Full` identifies two applications when their canonical input and
output states agree. `Automatic` also distinguishes which edges were consumed and produced and at
which step, resolving edges by their canonical positions — the identity is a property of the
event, not of the schedule that produced it. `Positional` uses the same components but resolves
edge positions in each raw state's own labeling, reproducing the upstream `MultiwaySystem`
convention (reproducible for a fixed input order, not isomorphism-invariant); it is incompatible
with `"ExploreFromCanonicalStatesOnly"`, which it disables with a warning. A list such as
`{"InputState", "OutputState", "Rule"}` selects a custom identity from the components
`"InputState"`, `"OutputState"`, `"Step"`, `"Rule"`, `"ConsumedEdges"`, `"ProducedEdges"`.
Compare the event structure with it off and on:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 2, "EvolutionGraphStructure", "CanonicalizeEvents" -> None]
```

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 2, "EvolutionGraphStructure", "CanonicalizeEvents" -> Full]
```

### "CausalTransitiveReduction"

Transitive reduction removes causal edges implied by longer paths. Off keeps every causal relation:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "CausalGraphStructure", "CausalTransitiveReduction" -> False]
```

On reduces to the transitive skeleton:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "CausalGraphStructure", "CausalTransitiveReduction" -> True]
```

### "ExploreFromCanonicalStatesOnly"

Quotient exploration expands each canonical state once, at its shortest depth, rather than expanding every isomorphic copy — the compact way to explore a symmetric system. It is off by default and pairs with `"CanonicalizeStates" -> Full`:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 4, "StatesGraph", "CanonicalizeStates" -> Full, "ExploreFromCanonicalStatesOnly" -> True]
```

### "MaxSuccessorStatesPerParent"

Limits the successor states generated from each parent state (0 = unlimited), bounding the branching:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "StatesGraphStructure", "MaxSuccessorStatesPerParent" -> 1]
```

The bound holds at any thread count. *Which* successors meet it does not: a parent's successors
are counted as they are produced, and matches forwarded into a state from an ancestor arrive
asynchronously, so a capped run may keep a different subset on a different thread count. Use
`"MatchesPerStateRule"` where the kept set has to be reproducible.

### "MaxStatesPerStep"

Limits the states retained per evolution step (0 = unlimited).

The bound holds at any thread count. *Which* states meet it does not: the cap is applied as
states arrive, so a capped run may return a different subset on different runs or thread counts.
Uncapped runs are unaffected -- their state, event, causal and branchial sets do not depend on
the thread count or the scheduling order. Use `"MatchesPerStateRule"` for a reproducible cap, or
`"TransitionRate"` with a `"RandomSeed"` for a reproducible thinning.

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "StatesGraphStructure", "MaxStatesPerStep" -> 2]
```

### "ExplorationProbability"

Below 1, each branch is explored with the given probability, pruning the multiway system stochastically:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "StatesGraphStructure", "ExplorationProbability" -> 0.5]
```

### "TransitionRate"

Below 1, each transition is kept with the given probability. The draw is taken from the transition's own isomorphism-invariant identity together with `"RandomSeed"`, never from thread state, so the same seed selects the same subgraph at any thread count and on either device.

It also carries a depth guarantee that `"ExplorationProbability"` does not. A fixed rate is a knife edge: below one over the branching factor, a thinned evolution dies out before reaching the requested depth. A state whose every draw failed keeps its lowest-keyed transition, so the sample still reaches full depth.

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 6, "StatesGraphStructure",
  "TransitionRate" -> 0.125, "RandomSeed" -> 7]
```

`"TransitionRate"` applies on BOTH devices, and the same seed selects the same subgraph on either: the keep-or-drop decision is one shared body reached from the host engine and from the device rewrite kernel alike, keyed on the transition's own isomorphism-invariant identity rather than on any scheduling order.

### "RuleWeights"

Per-rule multipliers on `"TransitionRate"`, given in rule order. The rate a rule's transitions are drawn at is the product of the two, so the knobs compose rather than one replacing the other: a rate of 1 with weights `{1, 0}` still samples, dropping the second rule entirely and leaving the first untouched.

A short list is a partial override. Rules past its end take weight 1, so weighting the first of five rules does not require spelling out four ones. `{}` weights every rule equally, which is what a run that sets nothing gets.

```wl
rules = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 4, "StatesGraphStructure",
  "RuleWeights" -> {1., 0.25}, "RandomSeed" -> 7]
```

`"RuleWeights"` applies on both devices; the weights are uploaded with the run and multiply `"TransitionRate"` in the same shared decision.

### "UniformRandom"

With `"MatchesPerStep"`, this stops keeping new states once that many exist for the step, tracing a narrow history rather than the full multiway system:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "StatesGraphStructure", "UniformRandom" -> True, "MatchesPerStep" -> 1]
```

It is a cap by ARRIVAL ORDER, not a uniform draw, and which states arrive first depends on the schedule. Two runs of the same input can therefore keep different states, and the states kept are not a uniform sample of the ones available: capping clips the offspring distribution to a point mass.

Use `"TransitionRate"` for sampling that is uniform and reproducible. A rate is defined per transition and needs no notion of a step, so it needs no barrier, it is drawn from the transition's own identity and the seed rather than from arrival order, and it preserves the branching structure a sample exists to represent.

### "AspectRatio"

Sets the aspect ratio of the returned graph:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 3, "StatesGraphStructure", "AspectRatio" -> 1/2]
```

### "TargetDevice"

Evolution runs on the CPU by default. Where a GPU build is bundled, `"GPU"` runs on the device; otherwise it falls back to the CPU with a message. The GPU engine honors the same `"CanonicalizeStates"` modes (`None`, `Automatic`, `Full`) as the CPU, and its state counts match the CPU's in every mode:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 3, "NumStates", "TargetDevice" -> "GPU"]
```

## Properties and Relations

### Cross-run isomorphism hashes

Run-local state ids differ between runs, but `"IncludeCanonicalHashes"` attaches an isomorphism-stable hash usable to fuse pruned runs by isomorphism class:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
HGEvolve[rules, {{1, 2}, {1, 3}}, 2, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True] // Length
```

### Continuing an evolution instead of re-running it

`HGEvolve` answers one question and discards the exploration that answered it, so asking for three steps and then five re-runs the first three. A **session** keeps the engine, its graph and the frontier the budget stopped at, so each step carries the *same* exploration further and every state, event and relation already built keeps its identity:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
s = HGSessionOpen[rules, {{1, 2}}, {"NumStates", "NumEvents"}];
{HGSessionQuery[s], HGSessionStep[s, 1], HGSessionStep[s, 1], HGSessionStep[s, 1]}
HGSessionClose[s]
```

which gives `{<|NumStates -> 1, NumEvents -> 0|>, <|2, 1|>, <|4, 3|>, <|10, 9|>}` — the same numbers as `HGEvolve[rules, {{1, 2}}, k, ...]` for `k = 0, 1, 2, 3`, reached by continuing rather than restarting. `HGSessionQuery[s]` re-reads the accumulated graph without exploring further. `HGSessionClose[s]` releases the engine.

Three things are fixed when the session opens, and the later verbs refuse to change them:

- **The rules.** A session's rule set was fixed at `HGSessionOpen`; applying different ones would answer about a system the session is not exploring.
- **The identity convention** (`"CanonicalizeStates"`, `"CanonicalizeEvents"`, `"ShowGenesisEvents"`). These decide what a state and an event *are*. The engine reads them back from its own graph rather than from a step's request, so a query cannot report exact canonical forms as tree-mode ones.
- **What the run records.** An artifact the session was not opened for cannot be produced afterwards — the evolution that would have built it has already run. Open with the property you intend to ask for; asking for another returns an empty relation and the engine says so rather than serving the emptiness silently.

A session's handle names a live engine process, so one session at a time is served per worker: a second `HGSessionOpen` while one is live is an error rather than an eviction, because evicting would discard an exploration without being asked.

### Steering: continuing from part of the frontier

An exploration that branches faster than it is worth exploring can be carried forward along a chosen branch. `HGSessionFrontier` reports the states a continuation would resume from, and `HGSessionStep[..., "From" -> {...}]` expands only those:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
s = HGSessionOpen[rules, {{1, 2}}, "NumStates"];
HGSessionStep[s, 2];
f = HGSessionFrontier[s];
HGSessionStep[s, 1, Automatic, "From" -> {First[f]}]
HGSessionClose[s]
```

which reaches 7 states where an unsteered step of the same depth reaches 10.

**The unselected branches are retained, not discarded.** They stay on the frontier and a later step resumes them, so a steered detour costs nothing in what remains reachable: continuing the session above without a selection lands on 34 states, exactly what `HGEvolve[rules, {{1, 2}}, 4, "NumStates"]` gives. Steering narrows what runs *next*, never what is *reachable*.

Naming a state that is not on the frontier is an error rather than a step that quietly does nothing, since a caller steering toward a state the exploration has already passed would otherwise get an unexplained empty result. Both devices serve the same contract: under `TargetDevice -> "GPU"` the worker resolves the selection against the frontier it last reported and puts the unselected entries back after the step, each at the depth it was stranded at.

### Sending only what a step added

A step re-serialises the whole accumulated graph. `"Delivery" -> "Delta"` sends only what the session has not already been sent, and the Wolfram side merges it back into the graph the session holds — so the wire is incremental and the *answer* is not: `HGSessionStep` returns the whole accumulated graph either way.

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
s = HGSessionOpen[rules, {{1, 2}}, "StatesGraphStructure"];
Table[HGSessionStep[s, 1, Automatic, "Delivery" -> "Delta"], {6}];
HGSessionClose[s]
```

Measured on this evolution at depth 7 (5,914 states), `tools/dev/session_step_cost.wls`: the engine leg falls from 381 ms to 269 ms and the reply from 516,535 to 443,660 bytes. That is a 29% cut to the engine call and 8% to the step, because the rest of a step is the Wolfram Graph construction — 1,640 ms of the 2,021 ms total — which a delta cannot reduce, since the graph being built is the whole accumulated one either way.

Ask for a full delivery at any time to resynchronise; doing so resets what the session believes you hold, so the next delta is measured from it. Delta delivery is not served on the quotient reconstruction route (`"ExploreFromCanonicalStatesOnly" -> True` with causal output), where the causal relation is reduced on read and an edge already sent can leave the reduction — a delta has no way to withdraw one, so those replies carry the whole graph and say so.

### Keeping a bounded number of transitions per rule

`"MatchesPerStateRule" -> k` keeps at most `k` of each state's own transitions **per rule**, and the choice is made when that state's matching is complete rather than as matches arrive:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 4, "NumStates", "MatchesPerStateRule" -> 1]
```

The distinction from the other caps is what it is for. `"MaxSuccessorStatesPerParent"` bounds children per parent regardless of which rule produced them. `"MatchesPerStep"` bounds by **arrival order**, which depends on the schedule, so two runs of the same evolution can keep different states. `"MatchesPerStateRule"` ranks a state's transitions by their own isomorphism-invariant identity and the `"RandomSeed"`, so the kept set is the same at any thread count — choosing `k` of `M` needs all `M`, and all `M` exist only once that state's matching has finished.

Applies to a state's **own** matches. A match forwarded into a state from an ancestor arrives asynchronously and is not counted against the cap, for the same reason: its arrival races the point where the count would be taken. Applies on both devices.
