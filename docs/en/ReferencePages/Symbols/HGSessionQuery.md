---
Template: Symbol
Name: HGSessionQuery
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSessionQuery
Keywords: [session, query, read, multiway, hypergraph rewriting, delta delivery]
SeeAlso: [HGSessionObject, HGSessionOpen, HGSessionStep, HGSessionFrontier, HGSessionClose, HGEvolve]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [Sessions, GettingStarted]
---

## Usage

<code>[HGSessionQuery]()[*session*]</code> gives the property *session* was opened for, over the evolution so far, without evolving further.

<code>[HGSessionQuery]()[*session*, *prop*]</code> gives the property *prop* of the evolution so far.

## Details & Options

- *session* is an [HGSessionObject]() given by [HGSessionOpen]().
- [HGSessionQuery]() does not evolve. It gives the same result as <code>[HGSessionStep]()[*session*, 0]</code>, and after steps totalling *k* the result is the one <code>[HGEvolve]()[*rules*, *init*, *k*, *prop*]</code> gives with the session's rules, initial states and options.
- A new session is at step 0, so a query gives the property of the initial states.
- *prop* is any property [HGEvolve]() accepts, or a list of them, whether or not it was named at [HGSessionOpen](). `Automatic` stands for the property the session was opened for.
- Graphs take the [Graph]() options the session was opened with.
- The following option can be given:

|   |   |   |
|---|---|---|
| `"Delivery"` | `"Full"` | `"Full"` sends the whole result from the engine; `"Delta"` sends only what this session has not sent before |

- With `"Delivery" -> "Delta"` the result is still the whole evolution; only the transfer from the engine is smaller.
- A property [HGEvolve]() does not accept issues the message `HGEvolve::unknownprop` and gives <code>[$Failed]()</code>.
- On a closed session it issues the message `HGSessionOpen::refused` and gives <code>[$Failed]()</code>; for an expression that is not an [HGSessionObject]() it issues `HGSessionStep::badsession` and gives <code>[$Failed]()</code>.

## Basic Examples

A new session holds the initial state; after three steps a query reads the evolution without extending it:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionQuery[s]
```

<!-- => 1 -->

```wl
HGSessionStep[s, 3];
HGSessionQuery[s]
```

<!-- => 10 -->

Another property of the same evolution:

```wl
HGSessionQuery[s, "NumEvents"]
```

<!-- => 9 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Scope

A list of properties gives an association:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 2];
HGSessionQuery[s, {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}]
```

<!-- => <|"NumStates" -> 4, "NumEvents" -> 3, "NumCausalEdges" -> 2, "NumBranchialEdges" -> 0|> -->

A graph, from a session opened for a count; it takes the session's [ImageSize]():

```wl
HGSessionClose[s];
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates", ImageSize -> 300];
HGSessionStep[s, 3];
HGSessionQuery[s, "StatesGraphStructure"]
```

The state and event records:

```wl
First[HGSessionQuery[s, "States"]]
```

<!-- => <|"Id" -> 0, "CanonicalId" -> 0, "ContentStateId" -> 0, "Step" -> 0, "Edges" -> {{0, 0, 1}}, "IsInitial" -> 1|> -->

```wl
Keys[First[HGSessionQuery[s, "Events"]]]
```

<!-- => {"Id", "CanonicalId", "RuleIndex", "InputState", "OutputState", "CanonicalInputState", "CanonicalOutputState", "ConsumedEdges", "ProducedEdges"} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Options

### "Delivery"

After the graph has been sent once, a delta query sends nothing new and gives the same graph:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "StatesGraphStructure"];
HGSessionStep[s, 3];
EdgeCount[HGSessionQuery[s, "Delivery" -> "Delta"]]
```

<!-- => 9 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Properties and Relations

A query and a step of zero give the same result, and after three steps the counts are those of a three-step [HGEvolve]():

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 3];
{HGSessionQuery[s] === HGSessionStep[s, 0],
 HGSessionQuery[s, {"NumStates", "NumEvents"}] ===
  HGEvolve[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, 3, {"NumStates", "NumEvents"}]}
```

<!-- => {True, True} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Possible Issues

A property [HGEvolve]() does not accept:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionQuery[s, "NumState"]
```

<!-- => $Failed; the message HGEvolve::unknownprop is issued -->

A closed session, and an expression that is not a session:

```wl
HGSessionClose[s];
HGSessionQuery[s]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

```wl
HGSessionQuery["x"]
```

<!-- => $Failed; the message HGSessionStep::badsession is issued -->
