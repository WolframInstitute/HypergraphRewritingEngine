---
Template: Symbol
Name: HGKleinBottle
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGKleinBottle
SeeAlso: [HGMobiusStrip, HGSphere, HGToGraph, HGEvolve]
---

## Usage

`HGKleinBottle[resolution, height]` generates a Klein bottle topology hypergraph.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "Radius" sets the scale.

## Basic Examples

```wl
HGToGraph[HGKleinBottle[16, 6]]
```
