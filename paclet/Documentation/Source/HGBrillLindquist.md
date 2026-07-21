---
Template: Symbol
Name: HGBrillLindquist
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGBrillLindquist
---

## Usage

`HGBrillLindquist[n, {mass1, mass2}, separation]` generates an $n$-point Brill-Lindquist two-black-hole initial condition.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.


## Basic Examples

```wl
HGToGraph[HGBrillLindquist[120, {1, 1}, 2]]
```

## See Also

HGEvolve, HGToGraph
