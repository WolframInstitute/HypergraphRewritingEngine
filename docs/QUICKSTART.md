# HypergraphRewritingEngine — Quickstart

A high-performance multiway hypergraph rewriting engine (Wolfram Physics model) for
Wolfram Language. It applies rewrite rules to a hypergraph in all possible ways,
building the multiway states graph together with its causal and branchial structure,
and it canonicalizes states so isomorphic ones are identified.

## Install

The paclet bundles its own engine binary for each platform, so installation is
self-contained — nothing else to build or configure.

```wolfram
PacletInstall["/path/to/WolframInstitute__HypergraphRewriteEngine-0.0.1.paclet"]
Needs["HypergraphRewriting`"]
```

(To produce that `.paclet` from a source checkout, see [Building from source](#building-from-source).)

## First evolution

Rules are written `lhs -> rhs`, where each side is a list of hyperedges (each
hyperedge a list of vertices). The initial state is a list of hyperedges. The third
argument is the number of steps.

```wolfram
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4]
```

This rewrites a single edge into two, in every possible way, for 4 steps. By default
`HGEvolve` returns the combined evolution/causal/branchial graph. To get a specific
piece, pass a **property** as the fourth argument:

```wolfram
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "StatesGraph"]
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "CausalGraph"]
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "States"]      (* the raw state list *)
HGEvolve[{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}}, 4, "NumStates"]   (* just a count *)
```

The canonical Wolfram Physics rule, with two initial edges:

```wolfram
HGEvolve[{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, {{1, 2}, {1, 3}}, 5]
```

Multiple rules: pass a list of rules, `{lhs1 -> rhs1, lhs2 -> rhs2, ...}`.

## Properties

The fourth argument selects what to return (a string, or a list of strings). Common
values:

- `"States"`, `"Events"` — the raw state and event lists.
- `"StatesGraph"`, `"CausalGraph"`, `"BranchialGraph"` — the individual graphs.
- `"EvolutionGraph"`, `"EvolutionCausalGraph"`, `"EvolutionBranchialGraph"`,
  `"EvolutionCausalBranchialGraph"` (the default) — combined graphs.
- `"NumStates"`, `"NumEvents"`, `"NumCausalEdges"`, `"NumBranchialEdges"` — counts.
- `"CausalEdges"`, `"BranchialEdges"` — edge lists.

The editor autocompletes these: type `HGEvolve[rules, init, steps, ` and the property
strings appear as a dropdown; typing an option name inside the call offers the option
list.

## Key options

- `"CanonicalizeStates" -> None | Automatic | Full` — how states are deduplicated.
  `None` keeps every state distinct; `Automatic` uses a fast content hash; `Full` is
  exact isomorphism (identifies all isomorphic states). Default `None`.
- `"TargetDevice" -> "CPU" | "GPU"` — run on the CPU (default) or, where a GPU build
  is bundled, the GPU (like `NetTrain`). Falls back to CPU with a message if no GPU
  binary is present.
- `"CausalTransitiveReduction" -> True | False` — reduce the causal graph to its
  transitive reduction (default `True`).
- `"ExploreFromCanonicalStatesOnly" -> True | False` — quotient exploration: expand
  each canonical state once, so the run costs the canonical closure rather than the
  (exponentially larger) provenance count.
- `"IncludeCanonicalHashes" -> True` — attach a run-stable isomorphism hash to each
  state, for fusing results across separate runs.

`Options[HGEvolve]` lists the full set.

## Initial-condition generators

Instead of an explicit edge list, the second argument can be a named initial
condition or a generator result — grids, tori, spheres, Klein bottles, Minkowski
sprinklings, Brill-Lindquist data, and more (`HGGrid`, `HGCylinder`, `HGTorus`,
`HGSphere`, `HGMinkowskiSprinkling`, `HGBrillLindquist`, …). For example:

```wolfram
HGEvolve[rules, "Grid", steps]
HGEvolve[rules, HGGrid[8, 8], steps]
```

## Building from source

```bash
cmake -B build -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
cmake --build build -j
wolframscript -file tools/build_paclet.wls   # -> paclet_archive/*.paclet
```

See [CROSS_COMPILATION.md](../CROSS_COMPILATION.md) for the Windows / macOS / Linux /
GPU cross-compile targets.
