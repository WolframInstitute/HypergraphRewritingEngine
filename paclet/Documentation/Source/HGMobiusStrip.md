---
Template: Symbol
Name: HGMobiusStrip
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGMobiusStrip
---

## Usage

`HGMobiusStrip[resolution, width]` generates a Mobius strip topology hypergraph.

## Details


- The result is an initial-condition object: pass it as the second argument of `HGEvolve`, or convert it to a `Graph` with `HGToGraph`.
- "Radius" sets the scale.

## Basic Examples

```wl
HGToGraph[HGMobiusStrip[24, 4]]
```

## See Also

HGEvolve, HGToGraph
