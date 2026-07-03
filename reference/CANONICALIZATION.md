# Canonicalization & equivalencing — options, naming, and cross-mapping

State and event canonicalization are **independent axes**. This document fixes the
naming used by the reference (`MultiwayReference.wl`), maps it to the C++ engine's
internal names, to the Wolfram Language paclet's user options, and to the
authoritative `Wolfram/Multicomputation` `MultiwaySystem` paclet we validate against.

## Axis 1 — State canonicalization

Whether two states are treated as the same node when they are isomorphic as ordered
hypergraphs (vertex relabeling; within-edge order preserved; multiplicity preserved).

| Reference option `"StateCanonicalization"` | Meaning | Engine `StateCanonicalizationMode` | Paclet option `"CanonicalizeStates"` | MultiwaySystem `"CanonicalStateFunction"` |
|---|---|---|---|---|
| `"Canonical"` (default) | merge isomorphic states (exact, IR) | `Full` | `Full` / `True` | `"CanonicalHypergraph"` |
| `"None"` | keep all states distinct | `None` | `None` / `False` | `Identity` |

The exact canonicalizer is McKay-style individualization–refinement (IR) in the engine;
the reference uses color-refinement + exhaustive within-cell lex-min relabeling (same
result, simpler). The Weisfeiler–Leman hash is a fast **approximation** of `Canonical`,
never the reference.

`"FullCapture"` is a separate flag: when on, every raw successor state is its own node
(the multiway *forest*) and `states` reports the count of distinct canonical classes —
this matches the engine's full-capture evolution. When off, isomorphic states are merged
before expansion (the deduplicated multiway *graph*).

## Axis 2 — Event canonicalization

Whether two rewrite events are treated as the same. **Three conventions**, surfaced
side-by-side in every per-step result so the distinction is visible:

| Reference per-step field / option value | Meaning | Engine `EventSignatureKeys` | Paclet | MultiwaySystem `"CanonicalEventFunction"` |
|---|---|---|---|---|
| `eventsNone` / `"None"` | every rewrite application distinct | `EVENT_SIG_NONE` | `"CanonicalizeEvents" -> False` | `None` |
| `eventsStates` / `"States"` | identity = (canon input state, canon output state) | `EVENT_SIG_FULL` = `InputState\|OutputState` | `"CanonicalizeEvents" -> True, "FullEventCanonicalization" -> True` | `Full` |
| `eventsAutomaticPositional` / `"Automatic"`+`"Positional"` | States + step + consumed/produced edges by **canonical edge rank, ordered** | `EVENT_SIG_AUTOMATIC` | `"CanonicalizeEvents" -> True, "FullEventCanonicalization" -> False` | `Automatic` |
| `eventsAutomaticCanonical` / `"Automatic"`+`"Canonical"` | States + step + consumed/produced edges **marked as edge colors and canonicalized** | (new; principled) | (new option) | — (no paclet equivalent) |

### The Positional vs Canonical distinction (the key choice)

`Automatic` is strictly finer than `States`: it also distinguishes events by *which*
edges were consumed/produced. The two sub-conventions differ only in how symmetric
edges are treated:

- **Positional** (`eventsAutomaticPositional`) — matches MultiwaySystem's
  `CanonicalEventFunction -> Automatic`. It identifies edges by their position in one
  canonical labeling and keeps them ordered by match/rhs position, **without quotienting
  state automorphisms**. So two symmetric edge-role assignments stay distinct — e.g. the
  two `{{1,1},{1,1}}` self-loop matches count as **2** events.
- **Canonical** (`eventsAutomaticCanonical`) — the principled IR merge. Consumed/produced
  edges are marked as edge colors and the marked hypergraph is canonicalized, so it
  **does quotient automorphisms**: genuinely isomorphic events merge — the two self-loop
  matches count as **1**.

Given IR is the project's reference canonicalization, `Canonical` is the more principled
definition; `Positional` exists for bit-parity with the MultiwaySystem paclet.

## Independence

The two axes are orthogonal: any state mode combines with any event mode. Causal and
branchial edges are computed on the raw event structure and are independent of the event
canonicalization choice; transitive reduction (default on) applies only to the causal
graph.

## Validation

`reference/validate.wls` checks the reference against the 12-case golden corpus
(`eventsAutomaticPositional` vs the corpus `Automatic` column).
`reference/compare_multiwaysystem.wls` checks it live against the
`Wolfram/Multicomputation` `MultiwaySystem` paclet. States, `eventsNone`, `eventsStates`,
causal, and branchial match exactly; `eventsAutomaticPositional` matches up to
MultiwaySystem's internal `CanonicalLinkedHypergraph` tie-break.
