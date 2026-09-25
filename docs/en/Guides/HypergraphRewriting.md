---
Template: Guide
Name: HypergraphRewriting
Title: Hypergraph Rewriting
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/guide/HypergraphRewriting
Description: Multiway hypergraph rewriting: states, events, causal and branchial graphs, state and event identity, statistics of each step, sampling, generated initial conditions, sessions and the GPU
Keywords: [hypergraph, multiway, rewriting, causal graph, branchial graph, Wolfram physics, isomorphism, canonicalization, statistics, sampling, session, GPU]
RelatedTutorials: [GettingStarted, AdvancedMultiwayEvolution, SamplingAndPruning, InitialConditions, Sessions, GPUEvolution]
Links: ["[HypergraphRewritingEngine on GitHub](https://github.com/WolframInstitute/HypergraphRewritingEngine)", "[Wolfram Physics Project technical introduction](https://www.wolframphysics.org/technical-introduction/)"]
---

## Abstract

Multiway hypergraph rewriting applies a set of rules to a hypergraph in every possible way. The result is the set of states reached, the events between them, and the causal and branchial relations between the events. The paclet computes these in a separate engine process, on the CPU or the GPU, and gives them as graphs, records or counts.

## Functions

### Multiway Evolution

- `HGEvolve` evolves a hypergraph under a list of rules for a number of steps and gives a graph, the records or the counts
- `Rule` (WL) a rewrite rule, each side a list of hyperedges and each hyperedge a list of vertices
- `Graph` (WL) the form of every graph property, and an initial hypergraph given as a graph

### Results of an Evolution

- `HGEvolve` the property selects the result: `"StatesGraph"`, `"CausalGraph"`, `"BranchialGraph"`, the evolution graphs, `"States"`, `"Events"`, `"CausalEdges"`, `"BranchialEdges"` and the counts
- `Dataset` (WL) the `"States"` and `"Events"` records as a table
- `VertexCount` (WL), `EdgeCount` (WL), `EdgeTags` (WL), `Lookup` (WL) …

### State and Event Identity

- `HGEvolve` `"CanonicalizeStates"` selects when two states are one state, and `"CanonicalizeEvents"` when two applications are one event
- `HGEvolve` `"IncludeCanonicalHashes"` adds a hash that is equal for isomorphic states in every run
- `IsomorphicGraphQ` (WL), `CanonicalGraph` (WL) …

### Causal and Branchial Structure

- `HGEvolve` the causal graph over events and the branchial graph over states, and the pairs behind them as `"CausalEdges"` and `"BranchialEdges"`
- `TransitiveReductionGraph` (WL) the reduction `"CausalTransitiveReduction"` applies
- `TopologicalSort` (WL), `ConnectedComponents` (WL) …

### Statistics of a Step

- `HGEvolve` `"StepStatistics"` gives, for each step, the number of states, the isomorphism classes, the entropy of the states over the classes and summaries of per-state invariants
- `Counts` (WL), `Entropy` (WL), `Histogram` (WL) …

### Sampling and Caps

- `HGEvolve` `"TransitionRate"`, `"RuleWeights"`, `"ExplorationProbability"` and `"MatchesPerStateRule"` select a reproducible subsystem with `"RandomSeed"`; `"MaxStatesPerStep"` and `"MaxSuccessorStatesPerParent"` cap the work in arrival order

### Generated Initial Conditions

- `HGEvolve` a name such as `"Grid"`, `"Torus"` or `"Sprinkling"` in place of the initial hypergraph, shaped by options such as `"GridWidth"`
- `Association` (WL) a generated initial condition given by its `"Type"` and parameters

### Continuing an Evolution

- `HGSessionOpen` opens an evolution that can be continued
- `HGSessionObject` an open session
- `HGSessionStep` evolves a session further from where it stopped
- `HGSessionFrontier` the states the next step expands
- `HGSessionQuery` reads a property of a session without evolving it
- `HGSessionClose` closes a session

### Evolution on the GPU

- `HGEvolve` `"TargetDevice" -> "GPU"` runs the evolution on the GPU engine
- `HGSessionOpen` `"TargetDevice" -> "GPU"` opens a session on the GPU engine
