---
Template: Symbol
Name: HGSessionStep
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSessionStep
Keywords: [session, step, continuation, steering, frontier, multiway, hypergraph rewriting]
SeeAlso: [HGSessionObject, HGSessionOpen, HGSessionQuery, HGSessionFrontier, HGSessionClose, HGEvolve]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [Sessions, GettingStarted]
---

## Usage

<code>[HGSessionStep]()[*session*, *n*]</code> evolves *session* *n* steps further and gives the property the session was opened for, over the whole evolution so far.

<code>[HGSessionStep]()[*session*, *n*, *prop*]</code> gives the property *prop* instead.

## Details & Options

- *session* is an [HGSessionObject]() given by [HGSessionOpen](); *n* is a non-negative integer.
- A step of *n* expands the states on the session's frontier (the states [HGSessionFrontier]() lists) through *n* more levels. After a step without `"From"`, the frontier is the set of states at the new depth.
- Steps add up: after steps totalling *k*, counts and graph sizes are those of <code>[HGEvolve]()[*rules*, *init*, *k*, *prop*]</code> with the session's rules, initial states and options, and graphs and states agree with it up to isomorphism.
- State, event and edge ids, and the names of vertices created by the rules, are assigned in the order the engine creates them, so a session's raw records can carry different ids from [HGEvolve]()'s.
- <code>[HGSessionStep]()[*session*, 0]</code> does not evolve and gives the same result as [HGSessionQuery]().
- *prop* is any property [HGEvolve]() accepts, or a list of them, whether or not it was named at [HGSessionOpen](). `Automatic` stands for the property the session was opened for.
- The following options can be given:

|   |   |   |
|---|---|---|
| `"From"` | `All` | the frontier states to expand: `All`, or a list of ids from [HGSessionFrontier]() (an empty list means `All`) |
| `"Delivery"` | `"Full"` | `"Full"` sends the whole result from the engine; `"Delta"` sends only what this session has not sent before |

- With `"From"`, only the named frontier states are expanded; the others stay on the frontier, and a later step without `"From"` expands them up to the depth that step reaches.
- A `"From"` naming a state that is not on the frontier issues the message `HGSessionOpen::refused` and gives <code>[$Failed]()</code>; the session is unchanged.
- With `"Delivery" -> "Delta"` the result is still the whole evolution; only the transfer from the engine is smaller.
- A negative or non-integer *n* issues the message `HGSessionStep::negsteps` and gives <code>[$Failed]()</code>. A property [HGEvolve]() does not accept issues `HGEvolve::unknownprop` and gives <code>[$Failed]()</code>.
- On a closed session it issues the message `HGSessionOpen::refused` and gives <code>[$Failed]()</code>; for an expression that is not an [HGSessionObject]() it issues `HGSessionStep::badsession` and gives <code>[$Failed]()</code>.

## Basic Examples

Each step continues from where the last one stopped:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
{HGSessionStep[s, 1], HGSessionStep[s, 1], HGSessionStep[s, 2]}
```

<!-- => {2, 4, 34} -->

A step can give another property:

```wl
HGSessionStep[s, 0, "NumEvents"]
```

<!-- => 33 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Scope

Four steps at once reach the same evolution as four single steps:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 4]
```

<!-- => 34 -->

A list of properties gives an association:

```wl
HGSessionStep[s, 1, {"NumStates", "NumEvents", "NumCausalEdges"}]
```

<!-- => <|"NumStates" -> 154, "NumEvents" -> 153, "NumCausalEdges" -> 152|> -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

A graph, which takes the [Graph]() options the session was opened with:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates", ImageSize -> 300];
HGSessionStep[s, 3, "StatesGraphStructure"]
```

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Options

### "From"

Expand only one of the two frontier states:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 2];
HGSessionStep[s, 1, "From" -> {3}]
```

<!-- => 7 -->

State 2 stays on the frontier beside the three states the step created:

```wl
Sort[HGSessionFrontier[s]]
```

<!-- => {2, 4, 5, 6} -->

A step without `"From"` expands every frontier state up to depth 4, so the session again holds the depth-4 evolution:

```wl
{HGSessionStep[s, 1],
 HGEvolve[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, 4, "NumStates"]}
```

<!-- => {34, 34} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

### "Delivery"

A delta step sends only the states and edges the session has not sent, and gives the whole graph:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "StatesGraphStructure"];
Table[VertexCount[HGSessionStep[s, 1, "Delivery" -> "Delta"]], 3]
```

<!-- => {2, 4, 10} -->

A full delivery gives the same graph:

```wl
VertexCount[HGSessionQuery[s]]
```

<!-- => 10 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Properties and Relations

After each step the count is [HGEvolve]()'s for the same number of steps:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
{Table[HGSessionStep[s, 1], 5],
 Table[HGEvolve[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, k, "NumStates"], {k, 5}]}
```

<!-- => {{2, 4, 10, 34, 154}, {2, 4, 10, 34, 154}} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Possible Issues

`"From"` accepts only ids on the frontier; the session is unchanged:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 1];
HGSessionStep[s, 1, "From" -> {0}]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

A number of steps that is not a non-negative integer:

```wl
HGSessionStep[s, 1.5]
```

<!-- => $Failed; the message HGSessionStep::negsteps is issued -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

A closed session, and an expression that is not a session:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionClose[s];
HGSessionStep[s, 1]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

```wl
HGSessionStep[5, 1]
```

<!-- => $Failed; the message HGSessionStep::badsession is issued -->
