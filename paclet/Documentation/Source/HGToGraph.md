---
Template: Symbol
Name: HGToGraph
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGToGraph
---

## Usage

`HGToGraph[icResult]` converts an initial-condition result (from `HGGrid`, `HGTorus`, `HGMinkowskiSprinkling`, ...) into a `Graph` for visualization or further processing.

## Details

- The initial-condition generators return an initial-condition object; `HGToGraph` renders it as a `Graph`.
- Pass an initial-condition object directly as the second argument of `HGEvolve` to evolve from it.

## Basic Examples

```wl
HGToGraph[HGGrid[8, 8]]
```

## See Also

HGGrid, HGTorus, HGEvolve
