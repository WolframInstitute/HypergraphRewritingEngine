---
Template: Guide
Name: Hypergraph Rewriting Engine
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/guide/HypergraphRewriting
Keywords: [hypergraph, multiway, rewriting, Wolfram physics]
SeeAlso: [HGEvolve, HGToGraph, HGGrid]
---

# Hypergraph Rewriting Engine

A high-performance multiway hypergraph rewriting engine. It applies rewrite rules to a hypergraph in all possible ways, building the multiway states graph together with its causal and branchial structure, and can canonicalize states so isomorphic ones are identified.

## Functions

### Evolution

- `HGEvolve` — multiway hypergraph rewriting; returns states, events, and the causal/branchial graphs

### Initial conditions

- `HGGrid` — a regular grid graph
- `HGGridWithHoles` — a grid with circular holes
- `HGCylinder` — a cylindrical topology
- `HGTorus` — a toroidal topology
- `HGSphere` — a spherical topology
- `HGKleinBottle` — a Klein bottle topology
- `HGMobiusStrip` — a Mobius strip topology
- `HGMinkowskiSprinkling` — a causal set by Minkowski sprinkling
- `HGBrillLindquist` — a Brill-Lindquist initial condition
- `HGPoissonDisk` — a Poisson-disk sampled graph
- `HGUniformRandom` — a uniformly random point cloud
- `HGToGraph` — convert an initial-condition result to a Graph

### Geometry and physics analysis

- `HGHausdorffAnalysis` — local Hausdorff dimension per vertex
- `HGStateDimensionPlot` — a state coloured by local dimension
- `HGTimestepUnionPlot` — the union graph at a timestep, dimension-coloured
- `HGDimensionFilmstrip` — a grid of timestep union graphs
- `HGGeodesicPlot` — geodesic paths over a state graph
- `HGGeodesicFilmstrip` — geodesic plots per timestep
- `HGLensingPlot` — gravitational lensing deflection vs impact parameter
- `HGBranchAlignmentBatch` — curvature-weighted PCA alignment for all states
