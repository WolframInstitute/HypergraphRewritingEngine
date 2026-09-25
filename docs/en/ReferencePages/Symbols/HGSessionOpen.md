---
Template: Symbol
Name: HGSessionOpen
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSessionOpen
Keywords: [session, continuation, multiway, hypergraph rewriting, evolution, frontier]
SeeAlso: [HGSessionObject, HGSessionStep, HGSessionQuery, HGSessionFrontier, HGSessionClose, HGEvolve]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [GettingStarted]
---

## Usage

<code>[HGSessionOpen]()[*rules*, *init*, *prop*]</code> opens a multiway evolution of *init* under *rules* that can be continued, and gives an [HGSessionObject]() whose steps report the property *prop*.

<code>[HGSessionOpen]()[*rules*, *init*]</code> opens a session whose steps report `"EvolutionCausalBranchialGraph"`.

## Details & Options

- A session is an evolution that is continued rather than run again. [HGSessionStep]() evolves it further from where the last step stopped, [HGSessionQuery]() reads it, [HGSessionFrontier]() lists the states the next step expands, and [HGSessionClose]() releases it.
- *rules* and *init* take the forms [HGEvolve]() takes: a rule or a list of rules; a hypergraph, a list of hypergraphs (one initial state each), a [Graph](), or a named initial condition such as `"Grid"`.
- *prop* is a property [HGEvolve]() accepts, or a list of properties. It sets what a step reports by default; [HGSessionStep]() and [HGSessionQuery]() can ask for any other property of the same evolution.
- The session starts at step 0: [HGSessionOpen]() applies no rule. After steps totalling *k*, the session reports what <code>[HGEvolve]()[*rules*, *init*, *k*, *prop*]</code> gives.
- [HGSessionOpen]() takes the options of [HGEvolve](), with the same defaults, and every option of [Graph](). They hold for the whole session: a step or a query cannot change the rules, how states and events are identified, the device, or the options of the graphs it gives.
- `"BranchialStep"` set to `Automatic` is resolved for each step's or query's own properties, so a graph asked for later has the branchial edges [HGEvolve]() gives it.
- `"TargetDevice" -> "GPU"` opens the session on the GPU engine. Where no GPU engine is available, it issues the message `HGEvolve::gpudev` and opens on the CPU engine.
- One session is open at a time for each device. While one is open, [HGSessionOpen]() issues the message `HGSessionOpen::refused` and gives <code>[$Failed]()</code>; the open session is unaffected.
- The session is held by a persistent engine worker process. Where none can be started, [HGSessionOpen]() issues the message `HGSessionOpen::noworker` and gives <code>[$Failed]()</code>; [HGEvolve]() still works.
- A property [HGEvolve]() does not accept issues the message `HGEvolve::unknownprop` and gives <code>[$Failed]()</code>.

## Basic Examples

Open a session on a rule that splits an edge into two, and evolve it:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionQuery[s]
```

<!-- => 1 -->

```wl
HGSessionStep[s, 3]
```

<!-- => 10 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Scope

A list of properties; each step reports an association:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, {"NumStates", "NumEvents"}];
HGSessionStep[s, 2]
```

<!-- => <|"NumStates" -> 4, "NumEvents" -> 3|> -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

A single rule, several initial states, and a named initial condition:

```wl
s = HGSessionOpen[{{1, 2}} -> {{1, 3}, {3, 2}}, {{{1, 2}}, {{1, 2}, {2, 3}}}, "NumStates"];
HGSessionQuery[s]
```

<!-- => 2 -->

```wl
HGSessionClose[s];
s = HGSessionOpen[{{1, 2}} -> {{1, 3}, {3, 2}}, "Grid", "NumStates", "GridWidth" -> 2, "GridHeight" -> 2];
HGSessionQuery[s]
```

<!-- => 1 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Options

### "CanonicalizeStates"

With `Full`, isomorphic states are one state:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates", "CanonicalizeStates" -> Full];
HGSessionStep[s, 3]
```

<!-- => 4 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

### "TransitionRate"

A rate below 1 keeps each transition with that probability, drawn from `"RandomSeed"`, so a seeded session gives the same sample as a seeded [HGEvolve]():

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates", "TransitionRate" -> 0.5, "RandomSeed" -> 7];
HGSessionStep[s, 4] ===
 HGEvolve[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, 4, "NumStates", "TransitionRate" -> 0.5, "RandomSeed" -> 7]
```

<!-- => True -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Properties and Relations

A session stepped to *k* reports what [HGEvolve]() gives for *k* steps:

```wl
s = HGSessionOpen[{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}}, {{1, 2}, {1, 3}}, "NumStates"];
HGSessionStep[s, 3];
HGSessionQuery[s, {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}] ===
 HGEvolve[{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}}, {{1, 2}, {1, 3}}, 3,
  {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}]
```

<!-- => True -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Possible Issues

A second session for the same device is refused while one is open:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

A property [HGEvolve]() does not accept:

```wl
HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumState"]
```

<!-- => $Failed; the message HGEvolve::unknownprop is issued -->
