---
Template: Symbol
Name: HGMinkowskiSprinkling
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGMinkowskiSprinkling
SeeAlso: [HGBrillLindquist, HGToGraph, HGEvolve]
---

## Usage

`HGMinkowskiSprinkling[n]` generates an $n$-point causal set by Minkowski sprinkling.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.


## Basic Examples

```wl
HGToGraph[HGMinkowskiSprinkling[60]]
```
