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
hyperedge is a list of vertices. This rule rewrites a single edge into two:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4]
```

The second argument is the initial state (a list of hyperedges); the third is the
number of steps. By default `HGEvolve` returns the combined evolution/causal/branchial
graph.

## Choosing what to return

Pass a property string as the fourth argument to select the output:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "StatesGraph"]
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "CausalGraph"]
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "NumStates"]
```

Common properties include `"States"`, `"Events"`, `"StatesGraph"`, `"CausalGraph"`,
`"BranchialGraph"`, and the counts `"NumStates"`, `"NumEvents"`.

## Identifying isomorphic states

By default every distinct provenance is a separate state. Set `"CanonicalizeStates"`
to `Full` to identify states that are the same up to isomorphism, collapsing the
multiway graph:

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "NumStates", "CanonicalizeStates" -> Full]
```

`Automatic` uses a faster content hash (which may merge some non-isomorphic states);
`Full` is exact.

## The canonical Wolfram Physics rule

```wl
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 5, "StatesGraph"]
```

## Structured initial conditions

Instead of an explicit edge list, use an initial-condition generator or a named
condition. Visualize one with `HGToGraph`:

```wl
HGToGraph[HGGrid[8, 8]]
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, HGGrid[8, 8], 2, "NumStates"]
```

See also `HGTorus`, `HGSphere`, `HGCylinder`, `HGMinkowskiSprinkling`, and the other
generators.

## Running on the GPU

Where a GPU build is bundled, evolve on the GPU with `"TargetDevice" -> "GPU"` (it
falls back to the CPU with a message if no GPU binary is present):

```wl
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 5, "NumStates", "TargetDevice" -> "GPU"]
```

## Next steps

See the [HGEvolve](paclet:WolframInstitute/HypergraphRewriteEngine/ref/HGEvolve)
reference page for the full option list, and `Options[HGEvolve]` in a session.
