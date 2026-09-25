---
Template: Symbol
Name: HGSessionFrontier
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSessionFrontier
Keywords: [session, frontier, steering, continuation, multiway, hypergraph rewriting]
SeeAlso: [HGSessionObject, HGSessionOpen, HGSessionStep, HGSessionQuery, HGSessionClose, HGEvolve]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [Sessions, GettingStarted]
---

## Usage

<code>[HGSessionFrontier]()[*session*]</code> gives the ids of the states the next [HGSessionStep]() of *session* expands.

## Details & Options

- *session* is an [HGSessionObject]() given by [HGSessionOpen]().
- The frontier holds the states the session has created and not yet expanded: after a step, the states that step created, and any state a step with `"From"` left out.
- The ids are the state ids of the session's results: the keys of `"States"` and the vertices of the states graph. They are not sorted.
- The frontier of a new session is its initial states. An empty frontier means every state has been expanded, so a further step adds nothing.
- Some of the frontier's ids can be given to [HGSessionStep]() as `"From" -> {`*id*, ...`}` to expand only those states; the others stay on the frontier.
- [HGSessionFrontier]() does not explore.
- On a closed session it issues the message `HGSessionOpen::refused` and gives <code>[$Failed]()</code>; for an expression that is not an [HGSessionObject]() it issues `HGSessionStep::badsession` and gives <code>[$Failed]()</code>.

## Basic Examples

A new session's frontier is its initial state; after two steps it holds the two states the second step created:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionFrontier[s]
```

<!-- => {0} -->

```wl
HGSessionStep[s, 2];
Sort[HGSessionFrontier[s]]
```

<!-- => {2, 3} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Scope

With several initial states, a new session's frontier holds all of them:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{{1, 2}}, {{1, 2}, {2, 3}}}, "NumStates"];
Sort[HGSessionFrontier[s]]
```

<!-- => {0, 1} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

A rule that merges two consecutive edges ends after three steps from a path of three edges, and the frontier is then empty:

```wl
s = HGSessionOpen[{{{1, 2}, {2, 3}} -> {{1, 3}}}, {{1, 2}, {2, 3}, {3, 4}}, "NumStates"];
HGSessionStep[s, 3];
HGSessionFrontier[s]
```

<!-- => {} -->

A further step adds no state:

```wl
HGSessionStep[s, 1]
```

<!-- => 5 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

With `"CanonicalizeStates" -> Full`, the splitting rule reaches one new class of states per step, so one state is on the frontier:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates", "CanonicalizeStates" -> Full];
HGSessionStep[s, 3];
Length[HGSessionFrontier[s]]
```

<!-- => 1 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Properties and Relations

The frontier's ids are the keys of the states the last step created, and vertices of the states graph:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 2];
{Sort[HGSessionFrontier[s]], Sort[Keys[Select[HGSessionQuery[s, "States"], #Step == 2 &]]],
 Sort[VertexList[HGSessionQuery[s, "StatesGraphStructure"]]]}
```

<!-- => {{2, 3}, {2, 3}, {0, 1, 2, 3}} -->

A step with `"From"` expands the states named; the other state stays on the frontier beside the three states the step created:

```wl
HGSessionStep[s, 1, "From" -> {3}];
Sort[HGSessionFrontier[s]]
```

<!-- => {2, 4, 5, 6} -->

State 2 keeps the step at which it was created:

```wl
HGSessionQuery[s, "States"][2, "Step"]
```

<!-- => 2 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Possible Issues

`"From"` accepts only ids on the frontier. State 0 was expanded by the first step, so naming it is refused and the session is unchanged:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 2];
HGSessionStep[s, 1, "From" -> {0}]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

```wl
Sort[HGSessionFrontier[s]]
```

<!-- => {2, 3} -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

A closed session, and an expression that is not a session:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionClose[s];
HGSessionFrontier[s]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

```wl
HGSessionFrontier[7]
```

<!-- => $Failed; the message HGSessionStep::badsession is issued -->
