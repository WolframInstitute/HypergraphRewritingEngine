---
Template: Symbol
Name: HGHausdorffAnalysis
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGHausdorffAnalysis
---

## Usage

`HGHausdorffAnalysis[edges]` computes the local Hausdorff dimension at each vertex of the hypergraph `edges`.

## Details

- `edges` is a list of hyperedges. The result gives a dimension estimate per vertex from ball-growth around it.
- See `Options[HGHausdorffAnalysis]` for the analysis and styling options.

## Basic Examples

```wl
HGHausdorffAnalysis[{{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 3}}]
```

## See Also

HGEvolve
