---
Template: Symbol
Name: HGSessionClose
Context: HypergraphRewriting`
Paclet: WolframInstitute/HypergraphRewriteEngine
URI: WolframInstitute/HypergraphRewriteEngine/ref/HGSessionClose
Keywords: [session, close, release, multiway, hypergraph rewriting]
SeeAlso: [HGSessionObject, HGSessionOpen, HGSessionStep, HGSessionQuery, HGSessionFrontier, HGEvolve]
RelatedGuides: [HypergraphRewriting]
RelatedTutorials: [Sessions, GettingStarted]
---

## Usage

<code>[HGSessionClose]()[*session*]</code> releases the engine holding *session*.

## Details & Options

- *session* is an [HGSessionObject]() given by [HGSessionOpen]().
- [HGSessionClose]() gives [Null]().
- Closing discards the session's evolution. [HGSessionStep](), [HGSessionQuery](), [HGSessionFrontier]() and [HGSessionClose]() on a closed session issue the message `HGSessionOpen::refused` and give <code>[$Failed]()</code>.
- A session's handle is not reused, so a verb on a closed session is refused rather than answered from a session opened later.
- One session is open at a time for each device. [HGSessionOpen]() is refused while another session is open, so a session that is no longer needed should be closed.
- An expression that is not an [HGSessionObject]() issues the message `HGSessionStep::badsession` and gives <code>[$Failed]()</code>.

## Basic Examples

Open a session on a rule that splits an edge into two, step it twice, and close it:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 2]
```

<!-- => 4 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Scope

Read what is needed before closing; nothing can be read afterwards:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionStep[s, 3];
HGSessionQuery[s, "NumEvents"]
```

<!-- => 9 -->

```wl
HGSessionClose[s]
```

<!-- => Null -->

## Properties and Relations

While a session is open, a second one cannot be opened:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

After closing the first, a new session opens:

```wl
HGSessionClose[s];
t = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
Head[t]
```

<!-- => HGSessionObject -->

```wl
HGSessionClose[t]
```

<!-- => Null -->

## Possible Issues

A closed session cannot be read, stepped or closed again:

```wl
s = HGSessionOpen[{{{1, 2}} -> {{1, 3}, {3, 2}}}, {{1, 2}}, "NumStates"];
HGSessionClose[s];
HGSessionQuery[s, "NumEvents"]
```

<!-- => $Failed; the message HGSessionOpen::refused is issued -->

An expression that is not a session:

```wl
HGSessionClose[7]
```

<!-- => $Failed; the message HGSessionStep::badsession is issued -->
