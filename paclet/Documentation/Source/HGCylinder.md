---
Template: Symbol
Name: HGCylinder
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGCylinder
SeeAlso: [HGTorus, HGSphere, HGToGraph, HGEvolve]
---

## Usage

`HGCylinder[resolution, height]` generates a cylindrical topology hypergraph.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "Radius" sets the cylinder radius; "RandomizeDirections" randomizes orientation.

## Basic Examples

```wl
HGToGraph[HGCylinder[16, 6]]
```
