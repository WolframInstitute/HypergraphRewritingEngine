---
Template: Symbol
Name: HGSphere
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSphere
SeeAlso: [HGTorus, HGKleinBottle, HGToGraph, HGEvolve]
---

## Usage

`HGSphere[resolution]` generates a spherical topology hypergraph by UV sampling.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "Radius" sets the sphere radius.

## Basic Examples

```wl
HGToGraph[HGSphere[16]]
```
