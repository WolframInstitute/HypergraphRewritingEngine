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

This rule matches two edges that share a vertex and adds the edge closing the triangle between their other two endpoints. By default `HGEvolve` returns the combined evolution/causal/branchial graph:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 3]
```

The multiway states graph, with isomorphic states merged via `"CanonicalizeStates" -> Full`:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 4, "StatesGraph", "CanonicalizeStates" -> Full]
```

Higher-arity hyperedges work identically — here a rule on ternary (3-vertex) edges:

```wl
HGEvolve[{{1, 2, 3}, {3, 4, 5}} -> {{1, 2, 3}, {3, 4, 5}, {5, 6, 7}}, {{1, 2, 3}, {3, 4, 5}}, 4, "StatesGraph", "CanonicalizeStates" -> Full]
```

Count the distinct states up to isomorphism:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 5, "NumStates", "CanonicalizeStates" -> Full]
```

## See Also

HGGrid, HGToGraph
