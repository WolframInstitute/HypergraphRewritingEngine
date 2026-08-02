---
Template: Guide
Name: Hypergraph Rewriting Engine
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/guide/HypergraphRewriting
Keywords: [hypergraph, multiway, rewriting, Wolfram physics]
SeeAlso: [HGEvolve]
---

# Hypergraph Rewriting Engine

A high-performance multiway hypergraph rewriting engine. It applies rewrite rules to a hypergraph in all possible ways, building the multiway states graph together with its causal and branchial structure, and can canonicalize states so isomorphic ones are identified.

## Functions

- `HGEvolve` — multiway hypergraph rewriting; returns states, events, and the causal/branchial graphs

`HGEvolve` is the paclet's entire public surface. Named initial conditions (grids, tori, spheres, sprinklings) are selected through its own options rather than by separate functions; see the `HGEvolve` page for the list.

Geometry and physics analyses live in the companion `hypergraph_viz` project, which consumes this engine as a dependency.
