---
Template: Symbol
Name: HGToGraph
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGToGraph
Keywords: [hypergraph, graph, visualization, conversion]
SeeAlso: [HGEvolve, HGGrid, HGMinkowskiSprinkling]
---

## Usage

`HGToGraph[icResult]` converts an initial-condition result (from `HGGrid`, `HGTorus`, `HGMinkowskiSprinkling`, ...) into a `Graph`.

`HGToGraph[edges]` converts a list of hyperedges into a `Graph`.

`HGToGraph[edges, coords]` uses the given vertex coordinates for the layout.

## Details

- The initial-condition generators return an initial-condition object; `HGToGraph` renders it as a `Graph` for visualization or further processing.
- `edges` is a list of hyperedges (each a list of vertices) — the same form `HGEvolve` takes as its initial state.
- An initial-condition object can also be passed directly as the second argument of `HGEvolve`, so no explicit conversion is needed to evolve from it.

## Basic Examples

Convert a generated grid initial condition to a `Graph`:

```wl
HGToGraph[HGGrid[8, 8]]
```

Convert an explicit edge list:

```wl
HGToGraph[{{1, 2}, {2, 3}, {3, 1}, {1, 4}}]
```
