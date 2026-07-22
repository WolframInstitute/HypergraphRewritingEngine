---
Template: TechNote
Name: Getting Started with Hypergraph Rewriting
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/tutorial/GettingStarted
Keywords: [hypergraph, multiway, rewriting, tutorial, Wolfram physics]
---

# Getting Started with Hypergraph Rewriting

This tutorial walks through the core workflow: define a rule, evolve a hypergraph in
all possible ways, and inspect the resulting states, causal, and branchial structure.

## A first evolution

A rewrite rule is written `lhs -> rhs`. Each side is a list of hyperedges, and each
hyperedge is a list of vertices. This rule matches two edges that share a vertex and adds
the edge closing the triangle between their other two endpoints:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 3]
```

The second argument is the initial state (a list of hyperedges); the third is the
number of steps. By default `HGEvolve` returns the combined evolution/causal/branchial
graph.

## Choosing what to return

Pass a property string as the fourth argument to select the output:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 3, "StatesGraph"]
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 3, "CausalGraph"]
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 3, "NumStates"]
```

Common properties include `"States"`, `"Events"`, `"StatesGraph"`, `"CausalGraph"`,
`"BranchialGraph"`, and the counts `"NumStates"`, `"NumEvents"`.

## Identifying isomorphic states

By default every distinct provenance is a separate state. Set `"CanonicalizeStates"`
to `Full` to identify states that are the same up to isomorphism, collapsing the
multiway graph:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 5, "NumStates"]
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 5, "NumStates", "CanonicalizeStates" -> Full]
```

`Automatic` uses a faster content hash (which may merge some non-isomorphic states);
`Full` is exact.

## The canonical Wolfram Physics rule

This is the rule from the Wolfram Physics Project. Its uncanonicalized multiway blows up
quickly, so show the isomorphism-merged states graph:

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 4, "StatesGraph", "CanonicalizeStates" -> Full]
```

## Higher-arity hyperedges

Edges may have any arity. This rule acts on ternary (3-vertex) hyperedges, extending a chain:

```wl
HGEvolve[{{1, 2, 3}, {3, 4, 5}} -> {{1, 2, 3}, {3, 4, 5}, {5, 6, 7}}, {{1, 2, 3}, {3, 4, 5}}, 4, "StatesGraph", "CanonicalizeStates" -> Full]
```

## Structured initial conditions

Instead of an explicit edge list, use an initial-condition generator or a named
condition. Visualize one with `HGToGraph`:

```wl
HGToGraph[HGGrid[8, 8]]
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, HGGrid[4, 4], 2, "NumStates"]
```

See also `HGTorus`, `HGSphere`, `HGCylinder`, `HGMinkowskiSprinkling`, and the other
generators.

## Running on the GPU

Where a GPU build is bundled, evolve on the GPU with `"TargetDevice" -> "GPU"` (it
falls back to the CPU with a message if no GPU binary is present):

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}, {{1, 2}, {1, 3}}, 5, "NumStates", "TargetDevice" -> "GPU"]
```

## Next steps

See the [HGEvolve](paclet:WolframInstitute/HypergraphRewriteEngine/ref/HGEvolve)
reference page for the full option list, and `Options[HGEvolve]` in a session.
