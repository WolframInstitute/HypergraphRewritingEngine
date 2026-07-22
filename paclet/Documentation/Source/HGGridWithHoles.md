---
Template: Symbol
Name: HGGridWithHoles
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGGridWithHoles
SeeAlso: [HGGrid, HGToGraph, HGEvolve]
---

## Usage

`HGGridWithHoles[w, h, holes]` generates a $w$x$h$ grid with circular holes, each hole `{cx, cy, r}`.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "Diagonals", "RandomizeDirections" as for HGGrid.

## Basic Examples

```wl
HGToGraph[HGGridWithHoles[16, 16, {{8, 8, 3}}]]
```
