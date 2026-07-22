---
Template: Symbol
Name: HGPoissonDisk
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGPoissonDisk
SeeAlso: [HGUniformRandom, HGToGraph, HGEvolve]
---

## Usage

`HGPoissonDisk[n, minDistance]` generates an $n$-point Poisson-disk-sampled hypergraph with minimum spacing `minDistance`.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.


## Basic Examples

```wl
HGToGraph[HGPoissonDisk[120, 0.08]]
```
