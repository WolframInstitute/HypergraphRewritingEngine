---
Template: Symbol
Name: HGEvolve
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGEvolve
Keywords: [hypergraph, multiway, rewriting, Wolfram physics, causal graph, branchial graph, evolution]
SeeAlso: [HGToGraph, HGGrid, HGMinkowskiSprinkling, EdgeId]
RelatedGuides: [Hypergraph Rewriting Engine]
---

## Usage

`HGEvolve[rules, initial, steps]` performs multiway hypergraph rewriting of the hypergraph `initial` under `rules` for `steps` steps, returning the combined evolution/causal/branchial graph.

`HGEvolve[rules, initial, steps, property]` returns the specified `property`.

## Details & Options

- The *rules* argument is a rule or list of rules of the form *lhs* `->` *rhs*, e.g. `{{1, 2}, {2, 3}} -> {{2, 1}, {3, 2}, {1, 4}}`. Each side is a list of hyperedges and each hyperedge is a list of vertices. Vertices are variables that bind to (not necessarily distinct) vertices of the state hypergraph; a vertex appearing in *rhs* but not *lhs* is a freshly created vertex.
- The *initial* argument is a single hypergraph state or a list of states given as numeric vertex labels, e.g. `{{1, 2}, {2, 3}}`. It may also be a `Graph`, a named initial condition (`"Grid"`, `"Sprinkling"`, ...), or an initial-condition generator result (`HGGrid`, `HGTorus`, ...).
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
| `"CausalEdges"`, `"BranchialEdges"`, `"GlobalEdges"` | the causal / branchial / global edge lists |
| `"NumStates"`, `"NumEvents"`, `"NumCausalEdges"`, `"NumBranchialEdges"` | the corresponding counts |
| `"DimensionData"`, `"CurvatureData"`, `"GeodesicData"`, `"TopologicalData"`, `"EntropyData"`, `"HilbertSpaceData"`, `"BranchialData"`, `"MultispaceData"`, `"AlignmentData"`, `"StateBitvectors"` | results of the corresponding analysis family (each requires its `"*Analysis"` option; see below) |
| `"All"`, `"Debug"` | an association of, respectively, all graph/list/count properties, or the four counts |

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
| `"UniformRandom"` | `False` | select matches by uniform random reservoir sampling instead of exhaustively |
| `"MatchesPerStep"` | `0` | matches applied per step in uniform-random mode (0 = all) |
| `"BranchialStep"` | `Automatic` | step at which branchial edges are computed: `Automatic`, `All`, `-1` (final), or a 1-based step |
| `"EdgeDeduplication"` | `True` | one causal/branchial edge per event pair, rather than one per shared hyperedge |
| `"TargetDevice"` | `"CPU"` | `"CPU"` or `"GPU"`, where a GPU build is bundled (falls back to CPU with a message otherwise). The GPU engine always computes the canonical (`Full`-quotiented) multiway |
| `"IncludeStateContents"` | `False` | attach each state's hyperedge list to the result |
| `"IncludeEventContents"` | `False` | attach each event's matched/produced edges to the result |
| `"IncludeCanonicalHashes"` | `False` | attach a run-stable isomorphism hash to each state, for fusing results across runs |
| `"ShowProgress"` | `False` | print progress during evolution |
| `"ShowGenesisEvents"` | `False` | include the synthetic genesis events that create the initial states |
| `"AspectRatio"` | `None` | aspect ratio for the returned graph |
| `"DebugFFI"` | `False` | print low-level foreign-function-interface diagnostics |

### Analysis options

- Each analysis family is gated by a boolean (all `False` by default) and, when enabled, attaches its results to the evolution (see also the dedicated `HGHausdorffAnalysis`, `HGGeodesicPlot`, `HGStateDimensionPlot`, and related reference pages).

- *Dimension* — local Hausdorff-dimension estimation per vertex:

| Option | Default | |
|---|---|---|
| `"DimensionAnalysis"` | `False` | compute local dimensions for all states |
| `"DimensionFormula"` | `"LinearRegression"` | estimator: `"LinearRegression"` or `"DiscreteDerivative"` |
| `"DimensionRadius"` | `{1, 5}` | `{minR, maxR}` ball radii used for the fit |
| `"DimensionColorBy"` | `"Mean"` | statistic used for coloring: `"Mean"`, `"Variance"`, `"Min"`, `"Max"` |
| `"DimensionPalette"` | `"TemperatureMap"` | `ColorData` palette for dimension coloring |
| `"DimensionRange"` | `Automatic` | `{min, max}` color-scale range, or `Automatic` |
| `"DimensionPerVertex"` | `False` | include per-vertex dimension data |
| `"DimensionTimestepAggregation"` | `False` | include a per-timestep aggregation section |

- *Geodesic* — test-particle paths traced through a state:

| Option | Default | |
|---|---|---|
| `"GeodesicAnalysis"` | `False` | trace geodesic paths through the graph |
| `"GeodesicSources"` | `Automatic` | source vertex IDs, or `Automatic` (near high-dimension regions) |
| `"GeodesicMaxSteps"` | `50` | maximum path length |
| `"GeodesicBundleWidth"` | `5` | number of paths per bundle |
| `"GeodesicFollowGradient"` | `False` | follow the dimension gradient rather than a random walk |
| `"GeodesicDimensionPercentile"` | `0.9` | percentile used when auto-selecting sources |

- *Topological / particle* — Robertson–Seymour defect detection:

| Option | Default | |
|---|---|---|
| `"TopologicalAnalysis"` | `False` | detect topological defects (K5 / K3,3 minors) |
| `"TopologicalCharge"` | `False` | compute per-vertex topological charge |
| `"DetectK5Minors"` | `True` | look for K5 minors (non-planarity) |
| `"DetectK33Minors"` | `True` | look for K3,3 minors (non-planarity) |
| `"DetectDimensionSpikes"` | `True` | detect defects via dimension anomalies |
| `"DetectHighDegree"` | `True` | detect high-degree vertices |
| `"DimensionSpikeThreshold"` | `1.5` | multiplier above the mean to flag a spike |
| `"DegreePercentile"` | `0.95` | degree percentile treated as "high" |
| `"ChargeRadius"` | `3.` | radius for local charge computation |
| `"ChargePerVertex"` | `False` | include per-vertex charge data |
| `"ChargeTimestepAggregation"` | `False` | include a per-timestep aggregation section |

- *Curvature* — Ollivier–Ricci, Wolfram–Ricci, and dimension-gradient curvature:

| Option | Default | |
|---|---|---|
| `"CurvatureAnalysis"` | `False` | compute per-vertex curvature |
| `"CurvatureMethod"` | `"All"` | `"OllivierRicci"`, `"WolframRicci"`, `"DimensionGradient"`, `"Both"`, or `"All"` |
| `"CurvaturePerVertex"` | `False` | include per-vertex curvature data |
| `"CurvatureTimestepAggregation"` | `False` | include a per-timestep aggregation section |

- *Branch alignment* — curvature shape-space via PCA (requires `"CurvatureAnalysis"`):

| Option | Default | |
|---|---|---|
| `"BranchAlignment"` | `False` | compute branch alignment |
| `"BranchAlignmentMethod"` | `"WolframRicci"` | curvature used: `"WolframRicci"` or `"OllivierRicci"` |

- *Entropy*, *Hilbert space*, *branchial*, and *multispace* measures:

| Option | Default | |
|---|---|---|
| `"EntropyAnalysis"` | `False` | compute graph-entropy and information measures |
| `"EntropyTimestepAggregation"` | `False` | include a per-timestep aggregation section |
| `"HilbertSpaceAnalysis"` | `False` | compute state-bitvector inner products and vertex probabilities |
| `"HilbertStep"` | `-1` | step to analyze (`-1` = final) |
| `"HilbertScope"` | `"Global"` | `"Global"`, `"PerTimestep"`, or `"Both"` |
| `"BranchialAnalysis"` | `False` | compute branchial distribution sharpness and branch entropy |
| `"BranchialScope"` | `"Global"` | `"Global"`, `"PerTimestep"`, or `"Both"` |
| `"BranchialPerVertex"` | `False` | include per-vertex sharpness data |
| `"MultispaceAnalysis"` | `False` | compute vertex/edge probabilities across branches |
| `"MultispaceScope"` | `"Global"` | `"Global"`, `"PerTimestep"`, or `"Both"` |

### Initial-condition options

- Instead of an explicit *initial*, an initial condition can be generated in place. The standalone generators (`HGGrid`, `HGMinkowskiSprinkling`, `HGBrillLindquist`, ...) are usually clearer, but the same controls are available as options:

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
HGEvolve[rules, HGGrid[3, 3], 2, "NumStates", "CanonicalizeStates" -> Full]
```

### Inspecting the raw states

The `"States"` property returns the state objects keyed by id; each is an association carrying its own hyperedges and metadata. Read the id, step, initial flag, and edges of each:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
states = HGEvolve[rules, {{1, 2}, {1, 3}}, 2, "States"];
Column[{#["Id"], #["Step"], #["IsInitial"], #["Edges"]} & /@ Values[states]]
```

The parallel `"Events"` property returns the update events, each recording its rule, the input and output state ids, and the consumed and produced edges.

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

### "UniformRandom"

Uniform-random mode applies a random subset of matches per step via reservoir sampling, tracing a single stochastic history rather than the full multiway system:

```wl
rules = {{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}};
HGEvolve[rules, {{1, 1}, {1, 1}}, 3, "StatesGraphStructure", "UniformRandom" -> True, "MatchesPerStep" -> 1]
```

### "AspectRatio"

Sets the aspect ratio of the returned graph:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 3, "StatesGraphStructure", "AspectRatio" -> 1/2]
```

### "TargetDevice"

Evolution runs on the CPU by default. Where a GPU build is bundled, `"GPU"` runs on the device; otherwise it falls back to the CPU with a message. The GPU engine computes the canonical (`Full`-quotiented) multiway on the device — it always deduplicates isomorphic states — so its state counts match `"CanonicalizeStates" -> Full`:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
HGEvolve[rules, {{1, 2}}, 3, "NumStates", "TargetDevice" -> "GPU", "CanonicalizeStates" -> Full]
```

## Applications

### Dimension analysis

Enabling an analysis family attaches its results to the evolution. Here local Hausdorff dimensions are computed for a small grid evolution (see `HGStateDimensionPlot` and `HGHausdorffAnalysis` for visualization and detail):

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
res = HGEvolve[rules, HGGrid[4, 4], 1, "States", "DimensionAnalysis" -> True, "IncludeStateContents" -> True];
Length[res]
```

## Properties and Relations

### Cross-run isomorphism hashes

Run-local state ids differ between runs, but `"IncludeCanonicalHashes"` attaches an isomorphism-stable hash usable to fuse pruned runs by isomorphism class:

```wl
rules = {{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}};
HGEvolve[rules, {{1, 2}, {1, 3}}, 2, "States", "CanonicalizeStates" -> Full, "IncludeCanonicalHashes" -> True] // Length
```
