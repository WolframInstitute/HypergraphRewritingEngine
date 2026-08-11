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

- `prop` may also be a list of property strings, in which case an association keyed by those strings is returned.
- Any `*Graph` property may take the suffix `Structure` to return the same graph without vertex styling (a lighter-weight rendering), e.g. `"StatesGraphStructure"`.
- A raw `"States"` result is a list of associations, one per state, with keys `"Id"`, `"CanonicalId"`, `"Step"`, `"IsInitial"`, `"Edges"` (the state's hyperedges), and — when `"IncludeCanonicalHashes" -> True` — `"CanonicalHash"`. A raw `"Events"` result is a list of associations with keys `"Id"`, `"CanonicalId"`, `"RuleIndex"`, `"Step"`, `"InputState"`/`"OutputState"` (state ids), `"ConsumedEdges"`/`"ProducedEdges"` (edge ids), and `"InputStateEdges"`/`"OutputStateEdges"` (the full edge lists).

### Evolution and output options

- State and event deduplication, exploration limits, and the output content are controlled by:

| Option | Default | |
|---|---|---|
| `"CanonicalizeStates"` | `None` | merge states by isomorphism class: `None`, `Automatic` (fast content hash, may merge some non-isomorphic states), or `Full` (exact isomorphism) |
| `"CanonicalizeEvents"` | `None` | merge equivalent events: `None`, `Automatic`, `Full`, or a list of keys |
| `"CausalTransitiveReduction"` | `True` | remove redundant transitive causal edges |
| `"ExploreFromCanonicalStatesOnly"` | `False` | quotient exploration: expand each canonical state once, at its shortest depth (off by default, so every provenance is explored) |
| `"QuotientInitialStates"` | `False` | collapse isomorphic initial states to a single canonical root (requires `"ExploreFromCanonicalStatesOnly"`); off keeps each initial state a distinct entry point |
| `"MaxSuccessorStatesPerParent"` | `0` | cap the successor states generated from each parent (0 = unlimited) |
| `"MaxStatesPerStep"` | `0` | cap the states retained per evolution step (0 = unlimited) |
| `"ExplorationProbability"` | `1.` | probability of exploring each branch; below 1 prunes stochastically |
| `"TransitionRate"` | `1.` | probability of keeping each transition; reproducible from `"RandomSeed"`, and reaches full depth at any rate (CPU only) |
| `"RuleWeights"` | `{}` | per-rule multipliers on `"TransitionRate"`, in rule order (CPU only) |
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

## Options

### "CanonicalizeStates"

State canonicalization equivalences states by isomorphism class. `Automatic` uses a fast content hash; `Full` is exact. With canonicalization off, each provenance is its own state:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 2, "StatesGraph", "CanonicalizeStates" -> None]
```

With exact canonicalization, isomorphic states merge:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 2, "StatesGraph", "CanonicalizeStates" -> Full]
```

### "CanonicalizeEvents"

Event canonicalization merges equivalent update events. Compare the event structure with it off and on:

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
HGEvolve[rules, {{1, 1}, {1, 1}}, 2, "CausalGraphStructure", "CausalTransitiveReduction" -> False]
```

On reduces to the transitive skeleton:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 2, "CausalGraphStructure", "CausalTransitiveReduction" -> True]
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

### "MaxStatesPerStep"

Limits the states retained per evolution step (0 = unlimited):

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

`"TransitionRate"` applies on the CPU. Under `"TargetDevice" -> "GPU"` it is reported in `"Warnings"` and the evolution runs unsampled, rather than silently returning a different answer per device.

### "RuleWeights"

Per-rule multipliers on `"TransitionRate"`, given in rule order. The rate a rule's transitions are drawn at is the product of the two, so the knobs compose rather than one replacing the other: a rate of 1 with weights `{1, 0}` still samples, dropping the second rule entirely and leaving the first untouched.

A short list is a partial override. Rules past its end take weight 1, so weighting the first of five rules does not require spelling out four ones. `{}` weights every rule equally, which is what a run that sets nothing gets.

```wl
rules = {{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 4, "StatesGraphStructure",
  "RuleWeights" -> {1., 0.25}, "RandomSeed" -> 7]
```

`"RuleWeights"` applies on the CPU. Under `"TargetDevice" -> "GPU"` it is reported in `"Warnings"` and every rule is weighted equally.

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
