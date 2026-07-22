---
Template: Symbol
Name: HGGrid
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGGrid
SeeAlso: [HGGridWithHoles, HGToGraph, HGEvolve]
---

## Usage

`HGGrid[w, h]` generates a regular $w$x$h$ grid hypergraph.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "Diagonals" adds diagonal edges; "RandomizeDirections" randomizes edge orientation.

## Basic Examples

```wl
HGToGraph[HGGrid[8, 8]]
```
