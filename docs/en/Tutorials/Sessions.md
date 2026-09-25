---
Template: TechNote
Name: Sessions
Title: Continuing an Evolution with Sessions
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/tutorial/Sessions
Keywords: [hypergraph, multiway, rewriting, session, frontier, steering, delta delivery, continuation]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [GettingStarted]
---

[HGEvolve]() runs an evolution from the initial state every time it is called, so asking for one more step repeats every step before it. A session keeps the evolution in the engine: [HGSessionOpen]() opens it, [HGSessionStep]() evolves it further from where it stopped, [HGSessionQuery]() reads it, [HGSessionFrontier]() lists the states the next step expands, and [HGSessionClose]() releases it.

## Evolving step by step

This rule splits an edge in two. A state with *k* edges has *k* successors, so the number of states grows quickly with the depth:

```wl
rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
init = {{1, 2}};
Table[HGEvolve[rules, init, k, "NumStates"], {k, 0, 5}]
```

<!-- => {1, 2, 4, 10, 34, 154} -->

Each of those runs started again from the initial state. A session starts at step 0 and keeps what it has explored:

```wl
session = HGSessionOpen[rules, init, {"NumStates", "NumEvents"}];
HGSessionQuery[session]
```

<!-- => <|"NumStates" -> 1, "NumEvents" -> 0|> -->

Each step continues from the last, and after *k* steps the session holds what [HGEvolve]() gives for *k* steps:

```wl
{HGSessionStep[session, 1], HGSessionStep[session, 2]}
```

<!-- => {<|"NumStates" -> 2, "NumEvents" -> 1|>, <|"NumStates" -> 10, "NumEvents" -> 9|>} -->

A step can give a property the session was not opened for, and a query reads any property without evolving:

```wl
{HGSessionStep[session, 1, "NumCausalEdges"], HGSessionQuery[session, "NumCausalEdges"],
 HGEvolve[rules, init, 4, "NumCausalEdges"]}
```

<!-- => {32, 32, 32} -->

## One session at a time

The engine worker holds one session for each device, so a second session is refused until the first is closed:

```wl
HGSessionOpen[rules, init, "NumStates"]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

```wl
HGSessionClose[session]
```

<!-- => Null -->

A closed session cannot be read or stepped:

```wl
HGSessionStep[session, 1]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

## Expanding part of the frontier

The frontier is the set of states the next step expands. After two steps it holds the two states at depth 2:

```wl
session = HGSessionOpen[rules, init, "NumStates"];
HGSessionStep[session, 2];
frontier = Sort[HGSessionFrontier[session]]
```

<!-- => {2, 3} -->

`"From"` expands only the states it names. Expanding one of the two adds its three successors:

```wl
HGSessionStep[session, 1, "From" -> {Last[frontier]}]
```

<!-- => 7 -->

The state left out stays on the frontier:

```wl
Sort[HGSessionFrontier[session]]
```

<!-- => {2, 4, 5, 6} -->

A step without `"From"` expands every frontier state up to the new depth, including the state left out, so the session again holds the depth-4 evolution:

```wl
{HGSessionStep[session, 1], HGEvolve[rules, init, 4, "NumStates"], Length[HGSessionFrontier[session]]}
```

<!-- => {34, 34, 24} -->

```wl
HGSessionClose[session]
```

<!-- => Null -->

## Sending only what a step added

A graph property sends every state and edge again at each step. With `"Delivery" -> "Delta"`, the engine sends only what it has not sent before, and the graph returned is still the whole evolution:

```wl
session = HGSessionOpen[rules, init, "StatesGraphStructure"];
HGSessionStep[session, 3];
{VertexCount[HGSessionStep[session, 1, "Delivery" -> "Delta"]],
 VertexCount[HGEvolve[rules, init, 4, "StatesGraphStructure"]]}
```

<!-- => {34, 34} -->

```wl
HGSessionClose[session]
```

<!-- => Null -->

## Identifying isomorphic states

Options are fixed when the session opens. With `"CanonicalizeStates" -> Full`, states that differ only in the names of their vertices are one state, and the 154 states of depth 5 are 6:

```wl
session = HGSessionOpen[rules, init, "NumStates", "CanonicalizeStates" -> Full];
{HGSessionStep[session, 5], HGEvolve[rules, init, 5, "NumStates", "CanonicalizeStates" -> Full]}
```

<!-- => {6, 6} -->

```wl
HGSessionClose[session]
```

<!-- => Null -->
