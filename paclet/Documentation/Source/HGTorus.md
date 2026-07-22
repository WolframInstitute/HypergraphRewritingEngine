---
Template: Symbol
Name: HGTorus
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGTorus
SeeAlso: [HGCylinder, HGSphere, HGToGraph, HGEvolve]
---

## Usage

`HGTorus[resolution]` generates a toroidal topology hypergraph.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "MajorRadius", "MinorRadius" set the torus radii.

## Basic Examples

```wl
HGToGraph[HGTorus[16]]
```
