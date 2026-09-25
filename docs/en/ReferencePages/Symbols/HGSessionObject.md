---
Template: Symbol
Name: HGSessionObject
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSessionObject
Keywords: [session, continuation, multiway, hypergraph rewriting, engine worker]
SeeAlso: [HGSessionOpen, HGSessionStep, HGSessionQuery, HGSessionFrontier, HGSessionClose, HGEvolve]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [Sessions, GettingStarted]
---

## Usage

<code>[HGSessionObject]()[…]</code> represents a multiway evolution held by an engine worker, as given by [HGSessionOpen]().

## Details & Options

- [HGSessionOpen]() gives an [HGSessionObject](). [HGSessionStep]() evolves it further, [HGSessionQuery]() reads it, [HGSessionFrontier]() lists the states the next step expands, and [HGSessionClose]() releases it.
- The evolution (its states, events and frontier) is held by the engine worker process. The object holds what addresses it: a handle, the device, the properties named at [HGSessionOpen]() and the options that decide how replies are read.
- It is displayed as a summary box showing the device and the properties the session reports, with the state canonicalization under the expander.
- The device is the one serving the session. A session opened with `"TargetDevice" -> "GPU"` where no GPU engine is available issues the message `HGEvolve::gpudev` and is served, and shown, as `CPU`.
- An [HGSessionObject]() is made only by [HGSessionOpen]().
- No verb changes the expression: it addresses the same session before and after a step, and a copy addresses the same session.
- After [HGSessionClose](), every verb on the object issues the message `HGSessionOpen::refused` and gives <code>[$Failed]()</code>. A handle is not reused.
- One session is open at a time for each device.
- A verb given an expression that is not an [HGSessionObject]() issues the message `HGSessionStep::badsession` and gives <code>[$Failed]()</code>.

## Basic Examples

Open a session; the object shows its device and the property it reports:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"]
```

```wl
HGSessionStep[s, 2]
```

<!-- => 4 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Scope

A session reporting two properties gives an association of them at each step:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, {"NumStates", "NumEvents"}];
HGSessionStep[s, 1]
```

<!-- => <|"NumStates" -> 2, "NumEvents" -> 1|> -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Properties and Relations

A copy of the object addresses the same session, so a step through the copy is seen through the original:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
t = s;
HGSessionStep[t, 3];
HGSessionQuery[s]
```

<!-- => 10 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

---

Closing leaves the expression an [HGSessionObject](), and the session it named is gone:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionClose[s];
Head[s]
```

<!-- => HGSessionObject -->

```wl
HGSessionQuery[s]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

## Possible Issues

While a session is open, a second [HGSessionOpen]() for the same device is refused:

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

An expression that is not a session:

```wl
HGSessionStep[5, 1]
```

<!-- => $Failed; the message HGSessionStep::badsession is issued -->
