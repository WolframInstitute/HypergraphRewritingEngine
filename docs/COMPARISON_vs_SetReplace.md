# HypergraphRewritingEngine compared with SetReplace and MultiwaySystem

This document compares this project's paclet (`HGEvolve`) with two other Wolfram Language
implementations of hypergraph rewriting:

- SetReplace 0.3.196 (Maksim Piskunov, MIT licence, <https://github.com/maxitg/SetReplace>), whose
  `WolframModel` is the Wolfram Physics Project's evolution function.
- `MultiwaySystem` from the Wolfram/Multicomputation paclet 0.1.8, which enumerates the states of
  a multiway system as `HGEvolve` does. This project's validation checks `HGEvolve` against it
  (reference/compare_multiwaysystem.wls).

## What each computes

| | SetReplace `WolframModel` | `MultiwaySystem` | `HGEvolve` |
| --- | --- | --- | --- |
| Multiway system | a token-event graph: tokens are edges, and an edge may be consumed by several events ("EventSelectionFunction" -> "MultiwaySpacelike"); states are not enumerated | every state reached, with its events | every state reached, with its events |
| Identifying isomorphic states | not during evolution | `"CanonicalStateFunction"` | `"CanonicalizeStates"`, applied while the evolution runs |
| Causal and branchial relations | causal graph of the token-event graph | causal and branchial graphs | both, as counts, pair lists and graphs; the causal relation transitively reduced |
| Implementation | a C++ library (libSetReplace) under a Wolfram Language layer; one thread per rule while finding matches | Wolfram Language | a C++ engine in its own process, parallel in every phase; a CUDA engine with the same results |
| Other | single-history evolution with event ordering; many analysis and plotting functions | properties as Wolfram Language graphs | quotient exploration, reproducible sampling, sessions that continue an evolution |

`MultiwaySystem` and `HGEvolve` compute the same object, so their times compare like for like. SetReplace's multiway is a different object with its own event count; its times are listed beside that count, not as a ratio.

## Speed

Measured at commit e6f26a9d (engine as of 48a32a35) with `reference/bench_authority.wls`, on an
Intel Core i9-14900K (24 cores, 32 threads) under Windows, Wolfram Language 14.3, paclets
Wolfram/Multicomputation 0.1.8, SetReplace 0.3.196 and this paclet 1.0.0. Each entry is the median
of three runs; a run longer than 120 s ends that row.

- `MultiwaySystem`: the states graph and the causal graph, with `"CanonicalStateFunction" -> "CanonicalHypergraph"`.
- Reference: `reference/MultiwayReference.wl`, this project's Wolfram Language implementation of
  the same definition, which returns lists rather than graphs.
- `HGEvolve`: `{"StatesGraph", "CausalEdges"}` with `"CanonicalizeStates" -> Full` and
  `"CanonicalizeEvents" -> Full`, through the paclet, graph construction included.
- SetReplace: `WolframModel[rules, init, <|"MaxGenerations" -> d|>, "EventSelectionFunction" ->
  "MultiwaySpacelike"]` and its `"EventsCount"`.

At every depth that more than one of the first three completes, they report the same numbers of
states and causal edges.

At the deepest depth `MultiwaySystem` completes:

| Rule | Depth | MultiwaySystem ms | HGEvolve ms | Ratio |
| --- | --- | --- | --- | --- |
| BinarySplit | 6 | 23963.5 | 30.4 | 788 |
| TriangleClose | 5 | 8718.4 | 35.8 | 244 |
| WPP | 4 | 20408.2 | 46.0 | 444 |
| PathSelfLoop | 4 | 10401.2 | 9.5 | 1095 |
| Ternary | 6 | 10112.8 | 167.8 | 60 |
| TwoRule | 3 | 2406.0 | 45.6 | 53 |
| Chain3 | 5 | 3801.5 | 18.3 | 208 |

Every rule and depth:

### BinarySplit

`{{{1, 2}} -> {{1, 3}, {3, 2}}}` from `{{1, 2}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 46.5 | 2.4 | 20.3 | 1 | 2.1 |
| 2 | 3 | 2 | 81.9 | 8.2 | 16.2 | 3 | 1.9 |
| 3 | 4 | 8 | 154.4 | 29.8 | 26.5 | 7 | 2.1 |
| 4 | 5 | 32 | 420.3 | 128.3 | 10.8 | 15 | 2.4 |
| 5 | 6 | 152 | 2358.5 | 673.3 | 19.7 | 31 | 2.8 |
| 6 | 7 | 872 | 23963.5 | 4651.8 | 30.4 | 63 | 2.2 |
| 7 | 8 | 5912 | > 120 s | 38125.1 | 214.1 | 127 | 5.4 |
| 8 | 9 | 46232 | not run | > 120 s | 1595.3 | 255 | 7.7 |

### TriangleClose

`{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}}}` from `{{1, 2}, {1, 3}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 44.6 | 4.8 | 20.6 | 2 | 2.1 |
| 2 | 4 | 4 | 106.1 | 18.2 | 15.6 | 6 | 2.3 |
| 3 | 7 | 16 | 242.1 | 65.5 | 23.0 | 18 | 1.8 |
| 4 | 13 | 72 | 912.0 | 254.1 | 38.9 | 82 | 5.3 |
| 5 | 22 | 520 | 8718.4 | 1710.8 | 35.8 | 994 | 42.1 |
| 6 | 37 | 5416 | > 120 s | 18521.7 | 221.2 | 62410 | 7101.6 |
| 7 | 58 | 81256 | not run | > 120 s | 2667.9 |  | > 120 s |

### WPP

`{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}` from `{{1, 2}, {1, 3}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 53.0 | 5.2 | 21.9 | 2 | 2.1 |
| 2 | 4 | 4 | 138.9 | 22.5 | 26.4 | 6 | 1.6 |
| 3 | 10 | 20 | 984.4 | 97.5 | 22.9 | 22 | 2.1 |
| 4 | 45 | 124 | 20408.2 | 599.1 | 46.0 | 174 | 7.1 |
| 5 | 302 | 1332 | > 120 s | 6421.4 | 234.8 | 4278 | 146.5 |
| 6 | 2677 | 19772 | not run | 109083.0 | 3101.2 |  | not run |

### PathSelfLoop

`{{{1, 2}, {2, 3}} -> {{1, 3}, {3, 4}, {1, 4}, {2, 4}}}` from `{{1, 1}, {1, 1}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 53.6 | 4.7 | 22.4 | 2 | 2.1 |
| 2 | 3 | 6 | 154.8 | 25.3 | 18.5 | 8 | 2.8 |
| 3 | 4 | 24 | 798.9 | 77.6 | 11.0 | 26 | 3.3 |
| 4 | 6 | 114 | 10401.2 | 546.0 | 9.5 | 116 | 5.1 |
| 5 | 13 | 744 | > 120 s | 3906.7 | 29.8 | 746 | 48.9 |
| 6 | 46 | 7062 | not run | 41284.0 | 279.4 | 6092 | 830.8 |

### Ternary

`{{{1, 2, 3}} -> {{1, 2, 4}, {2, 4, 3}}}` from `{{1, 1, 1}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 44.0 | 2.5 | 12.3 | 1 | 2.2 |
| 2 | 4 | 2 | 77.8 | 9.0 | 19.1 | 3 | 1.6 |
| 3 | 9 | 8 | 143.8 | 32.5 | 30.5 | 7 | 2.2 |
| 4 | 23 | 32 | 347.7 | 135.9 | 30.1 | 15 | 2.5 |
| 5 | 65 | 152 | 1395.8 | 696.3 | 67.7 | 31 | 3.2 |
| 6 | 197 | 872 | 10112.8 | 4392.3 | 167.8 | 63 | 3.7 |

### TwoRule

`{{{1, 2}} -> {{1, 2}, {2, 3}}, {{1, 2}} -> {{1, 2}, {2, 3}, {3, 4}}}` from `{{1, 2}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0 | 54.0 | 4.6 | 8.2 | 2 | 5.0 |
| 2 | 10 | 10 | 182.0 | 31.7 | 19.6 | 12 | 8.2 |
| 3 | 38 | 92 | 2406.0 | 280.9 | 45.6 | 62 | 35.1 |
| 4 | 166 | 1030 | > 120 s | 4932.9 | 213.7 | 312 | 163.8 |
| 5 | 789 | 14808 | not run | 96145.9 | 1442.3 | 1562 | 787.2 |

### Chain3

`{{{1, 2}, {2, 3}} -> {{1, 2}, {2, 4}, {4, 3}}}` from `{{1, 2}, {2, 3}}`

| Depth | States | Causal edges | MultiwaySystem ms | Reference ms | HGEvolve ms | SetReplace events | SetReplace ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 43.6 | 2.7 | 17.9 | 1 | 1.4 |
| 2 | 3 | 2 | 83.7 | 10.8 | 10.9 | 3 | 2.3 |
| 3 | 4 | 8 | 152.1 | 38.2 | 15.2 | 9 | 1.6 |
| 4 | 5 | 32 | 573.1 | 153.7 | 18.1 | 31 | 3.1 |
| 5 | 6 | 156 | 3801.5 | 834.0 | 18.3 | 129 | 6.6 |

## Reproduction

    wolframscript -file reference/bench_authority.wls <maxDepth> <runs> <minDepth> <rule | All> <limit>

The rule names are those of the tables above; the runs above used 3 runs from depth 1 with a
limit of 120 seconds.
