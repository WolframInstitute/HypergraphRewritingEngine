---
Template: Symbol
Name: HGEvolve
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGEvolve
Keywords: [hypergraph, multiway, rewriting, Wolfram physics, causal graph]
---

## Usage

`HGEvolve[rules, init, steps]` evolves the hypergraph `init` under `rules` for `steps` steps, returning the combined evolution/causal/branchial graph.

`HGEvolve[rules, init, steps, prop]` returns the property `prop`.

## Details

- Each rule is written `lhs -> rhs`, where each side is a list of hyperedges and each hyperedge is a list of vertices. `rules` may be a single rule or a list of rules.
- `init` is a list of hyperedges, a `Graph`, a named initial condition (`"Grid"`, `"Sprinkling"`, ...), or an initial-condition generator result (`HGGrid`, `HGTorus`, ...).
- The engine applies the rules in all possible ways (multiway evolution), deduplicating isomorphic states, and builds the causal and branchial graphs.
- The property `prop` is a string or list of strings. Common properties: `"States"`, `"Events"`, `"StatesGraph"`, `"CausalGraph"`, `"BranchialGraph"`, `"EvolutionGraph"`, `"EvolutionCausalBranchialGraph"` (the default), `"NumStates"`, `"NumEvents"`, `"CausalEdges"`, `"BranchialEdges"`.
- `HGEvolve` computes in a standalone engine process, so a crash or abort never affects the notebook.

## Options

- `"CanonicalizeStates"` (default `None`) — state deduplication: `None`, `Automatic` (fast content hash), or `Full` (exact isomorphism).
- `"TargetDevice"` (default `"CPU"`) — `"CPU"` or `"GPU"`, where a GPU build is bundled.
- `"CausalTransitiveReduction"` (default `True`) — reduce the causal graph to its transitive reduction.
- `"ExploreFromCanonicalStatesOnly"` (default `False`) — quotient exploration: expand each canonical state once at its shortest depth.
- `"IncludeCanonicalHashes"` (default `False`) — attach a run-stable isomorphism hash to each state, for fusing results across runs.

Use `Options[HGEvolve]` for the full list.

## Basic Examples

Evolve a single edge-splitting rule for four steps:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4]
```

Get just the states graph:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "StatesGraph"]
```

The canonical Wolfram Physics rule from two initial edges:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 5]
```

Deduplicate isomorphic states exactly, and count them:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "NumStates", "CanonicalizeStates" -> Full]
```

## See Also

HGGrid, HGToGraph
