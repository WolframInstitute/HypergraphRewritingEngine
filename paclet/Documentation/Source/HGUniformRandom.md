---
Template: Symbol
Name: HGUniformRandom
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGUniformRandom
SeeAlso: [HGPoissonDisk, HGToGraph, HGEvolve]
---

## Usage

`HGUniformRandom[n]` generates an $n$-point uniformly random point-cloud hypergraph.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.


## Basic Examples

```wl
HGToGraph[HGUniformRandom[120]]
```
