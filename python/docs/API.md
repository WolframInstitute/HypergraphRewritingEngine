# hgevolve API reference

`import hgevolve as hg`. The package runs the CPU engine of the HypergraphRewriting paclet in the Python process. Each function and option corresponds to an HGEvolve function or option, and the names below map one to the other. The meaning of every option is the one docs/en/ReferencePages/Symbols/HGEvolve.md gives.

## Functions

### `hg.evolve(rules, initial, steps, *, data=None, **options)`

`HGEvolve[rules, initial, steps]`. Returns a `Result`.

- `rules`: one `(lhs, rhs)` pair or a list of them. A side is a list of edges, and an edge is a list of vertex labels. Labels may be integers or any hashable values. Each rule's labels are numbered from 0 in order of first appearance in the left side, then the right side, as HGEvolve numbers them.
- `initial`: one hypergraph (a list of edges) or a list of hypergraphs, which may also be NumPy arrays. Integer labels are sent as given. Other labels are numbered from 0 in order of first appearance across the states. A generated initial condition is a type name ("Grid", "Torus", "BrillLindquist", ...) or a dict with `"Type"` and the generator's keys, as HGEvolve takes it (`initial_condition`).
- `steps`: a non-negative integer.
- `data`: what the reply holds, as a list of names from the data table below. The default is `hg.DEFAULT_DATA`: states, events, causal and branchial edges, and the four counts.
- `options`: the options table below.

### `hg.Session(rules, initial, *, data=None, **options)`

`HGSessionOpen`. A process has at most one open session. A second open session raises `EngineError`. The identity options (`canonicalize_states`, `canonicalize_events`, `show_genesis_events`, `explore_from_canonical_states_only`) are fixed when the session opens.

- `s.step(steps=1, *, start=None, data=None)`: `HGSessionStep`. `start` is a list of frontier state ids to expand. The other frontier states stay on the frontier.
- `s.query(*, data=None)`: `HGSessionQuery`.
- `s.frontier`: `HGSessionFrontier`, as of the last reply.
- `s.close()`: `HGSessionClose`. A `with` block closes the session, and so does deleting the object.

### `hg.encode_job(rules, initial, steps, *, data=None, graph_properties=None, **options)`

The WXF bytes of the job `evolve` runs, which `hg_evolve` reads on stdin. `graph_properties` takes HGEvolve graph property names and adds the "GraphData" the paclet builds graphs from.

### `hg.initial_condition(spec)`

The initial condition HGEvolve generates for `spec`: a type name or a dict with `"Type"` and the generator's keys ("Width", "Height", "Holes", "Resolution", "Radius", "Density", "MinDistance", "EdgeThreshold", "BoxX", "Mass1", "Seed", ...; paclet_source/initial_condition_job.hpp lists them). A key not given takes the default of the HGEvolve option it stands for, and "Seed" defaults to 0. Returns a dict: "Edges", "VertexCount", "VertexCoordinates" and, where the generator gives them, "VertexCoordinates3D", "SpacetimePoints", "EdgeThreshold" and "DimensionEstimate". An unknown type or key raises `ValueError`.

### `hg.build_stamp()`

The stamp of the engine build: commit, variant (`python`) and configuration.

## Results

`Result` attributes are `None` when the job did not ask for them.

| Attribute | Type | HGEvolve key |
|---|---|---|
| `states` | `Table` | `"States"` |
| `events` | `Table` | `"Events"` |
| `causal_edges` | `Relation` (`From`, `To`, `RawFrom`, `RawTo`) | `"CausalEdges"` |
| `branchial_edges` | `Relation` (`From`, `To`) | `"BranchialEdges"` |
| `branchial_state_edges` | `Relation` (`From`, `To`) | `"BranchialStateEdges"` |
| `branchial_state_vertices` | 1-D array | `"BranchialStateVertices"` |
| `global_edges` | `Table` (`Id`, `Vertices`) | `"GlobalEdges"` |
| `state_bitvectors` | `Table` (`State`, `Edges`) | `"StateBitvectors"` |
| `num_states`, `num_events`, `num_causal_edges`, `num_branchial_edges` | `int` | `"NumStates"`, ... |
| `step_statistics` | list of dicts, one per step | `"StepStatistics"` |
| `device` | dict | `"Device"` |
| `warnings` | list of dicts (`Kind`, `Count`, `Context`, `Partial`) | `"Warnings"` |
| `session`, `frontier` | session handle, frontier state ids | `"Session"`, `"Frontier"` |
| `path_costs` | dict of state id to path cost, under `search="BestFirst"` or `beam_cost="Path"` | `"PathCosts"` |
| `reduced` | `True` when the evolution ran out of memory and the reply holds the counts only | `"Reduced"` |
| `values` | dict of every key that is not a table or relation | |

A `Table` holds records as columns. Its records are HGEvolve's: a "States" record has `Id`, `CanonicalId`, `ContentStateId`, `Step`, `Edges` and `IsInitial`, plus `CanonicalHash` and `VertexCurvature` when asked for. An "Events" record has `Id`, `CanonicalId`, `RuleIndex`, `InputState`, `OutputState`, `CanonicalInputState`, `CanonicalOutputState`, `ConsumedEdges` and `ProducedEdges`. `t["CanonicalId"]` and `t.canonical_id` are the same column. Column types:

- An integer field is a 1-D `int64` array with one entry per record.
- A list field (`ConsumedEdges`) is a `Ragged`: `values` and `offsets`, where record i's list is `values[offsets[i]:offsets[i+1]]`.
- `Edges` is an `EdgeLists`: record i's edges are `offsets[i]` to `offsets[i+1] - 1`. Edge j has id `ids[j]` and vertices `vertices[j]`, a `Ragged` over `vertices.values` and `vertices.offsets`.
- `VertexCurvature` is a `VertexReals`: `vertices`, `values` and per-record `offsets`.

`t.records()` gives the records as dicts in HGEvolve's form. A `Relation` is an n x k `int64` array (`r.array`) whose rows are in ascending order. `r["From"]` and `r.from_` are column views of it.

### Memory

Every array is a read-only NumPy view of the reply the engine built; nothing is copied. Each array holds a reference to the reply, so the reply lives as long as its last array, after the `Result` and the `Session` are gone. Values in `step_statistics`, `device`, `warnings` and `values` are Python objects. "StateMatrices" are NumPy arrays.

## Options

| Python | HGEvolve | Value |
|---|---|---|
| `canonicalize_states` | `"CanonicalizeStates"` | `None`, `"Automatic"`, `"Full"` |
| `canonicalize_events` | `"CanonicalizeEvents"` | `None`, `"Full"`, `"Automatic"`, `"Positional"` or a list of components |
| `causal_transitive_reduction` | `"CausalTransitiveReduction"` | bool |
| `show_genesis_events` | `"ShowGenesisEvents"` | bool |
| `explore_from_canonical_states_only` | `"ExploreFromCanonicalStatesOnly"` | bool |
| `undirected_edges` | `"UndirectedEdges"` | bool |
| `generational` | `"Generational"` | bool |
| `event_ordering_function` | `"EventOrderingFunction"` | `None`, `"Automatic"`, an ordering name or a list of them |
| `max_successor_states_per_parent` | `"MaxSuccessorStatesPerParent"` | int, 0 is no cap |
| `max_states_per_step` | `"MaxStatesPerStep"` | int, 0 is no cap |
| `matches_per_state_rule` | `"MatchesPerStateRule"` | int, 0 is no cap |
| `uniform_random` | `"UniformRandom"` | bool |
| `matches_per_step` | `"MatchesPerStep"` | int |
| `exploration_probability` | `"ExplorationProbability"` | float |
| `transition_rate` | `"TransitionRate"` | float |
| `rule_weights` | `"RuleWeights"` | list of floats |
| `state_cost` | `"StateCost"` | `None`, an expression string, or `{"Code": [...], "Constants": [...]}` |
| `transition_cost` | `"TransitionCost"` | `None`, `{"Field": name, "TimeStep": dt}`, or a callable `f(parent_edges, child_edges) -> float` |
| `search` | `"Search"` | `"Automatic"`, `"BestFirst"` |
| `search_budget` | `"SearchBudget"` | int; the states best-first search creates (default 1000) |
| `beam_cost` | `"BeamCost"` | `"Transition"` (default), `"Path"`: with `max_states_per_step` and a cost, rank by the child's path cost and keep one transition per successor |
| `random_seed` | `"RandomSeed"` | int; `None` (the default) draws one from Python's `random` |
| `branchial_step` | `"BranchialStep"` | int (1-based step, negative from the end) or `"all"`; default -1 |
| `include_canonical_hashes` | `"IncludeCanonicalHashes"` | bool |
| `include_vertex_curvature` | `"IncludeVertexCurvature"` | bool |
| `step_statistics_weighting` | `"StepStatisticsWeighting"` | `"States"`, `"Classes"` |
| `statistics_graph` | `"StatisticsGraph"` | `"Consecutive"`, `"AllPairs"` |
| `step_statistics_measures` | `"StepStatisticsMeasures"` | `None`, a name, a list of names, or `"all"` |
| `step_statistics_branchial` | `"StepStatisticsBranchial"` | `None`, `"Graph"`, `"Overlap"`, a list, or `"all"` |
| `step_statistics_matrices` | `"StepStatisticsMatrices"` | `None`, a matrix name, a list, or `"all"` |
| `step_statistics_matrix_limit` | `"StepStatisticsMatrixLimit"` | bytes; `None` is Automatic |
| `threads` | `"Threads"` | int; `None` is Automatic (`HG_TUNE_THREADS`, else one per logical CPU) |
| `tuning` | `"Tuning"` | dict of tuning names to integers (docs/TUNING.md) |
| `show_progress` | `"ShowProgress"` | bool; progress lines go to stderr |

`state_cost` as a string is a Python expression over "Invariants" statistic names. It may use `+ - * /`, `**` with a real exponent, unary `-`, `abs`, `sqrt`, `exp`, `log`, `min` and `max`: `state_cost="abs(BallDimension - 1.5)"`. It compiles to the engine's postfix program in the shapes HGEvolve uses.

`transition_cost` scores a transition from its parent and child states (docs/TRANSITION_COST.md). The built-in cost is the heat residual of a per-vertex field (`"Degree"`, `"Curvature"`, `"LoopCount"` or `"EdgeMultiplicity"`) with time step `dt`. A callable receives the two states' edges as lists of vertex ids, in which a vertex the rewrite keeps has one id in both, and returns the cost. The engine calls it from its workers, one call at a time under the GIL. An exception it raises is raised by `evolve`. `hg.heat_residual(parent, child, field="Degree", time_step=0.0, statistics_graph="Consecutive")` is the engine's residual, so a callable can combine it with other terms. `search="BestFirst"` takes the transition of lowest path cost next and returns each state's path cost in `r.path_costs`.

The HGEvolve options for graphs and display (`"EdgeDeduplication"`, `"ColorByRule"`, `"MultiedgeStyle"`, Graph options), `"TargetDevice"` and the initial-condition generator options have no Python counterpart. The package runs the CPU engine.

## Data names

| Python | HGEvolve |
|---|---|
| `states` | `"States"` |
| `events` | `"Events"` |
| `events_minimal` | `"EventsMinimal"` |
| `causal_edges` | `"CausalEdges"` |
| `branchial_edges` | `"BranchialEdges"` |
| `branchial_state_edges` | `"BranchialStateEdges"` |
| `branchial_state_edges_all_siblings` | `"BranchialStateEdgesAllSiblings"` |
| `num_states` | `"NumStates"` |
| `num_events` | `"NumEvents"` |
| `num_causal_edges` | `"NumCausalEdges"` |
| `num_branchial_edges` | `"NumBranchialEdges"` |
| `step_statistics` | `"StepStatistics"` |
| `global_edges` | `"GlobalEdges"` |
| `state_bitvectors` | `"StateBitvectors"` |
| `device` | `"Device"` |

## Errors and crashes

- A job the engine refuses raises `hg.EngineError` with the engine's message. A wrong argument raises `TypeError` or `ValueError`.
- Each entry of the reply's "Warnings" is also issued as an `hg.EngineWarning` through Python's `warnings` module.
- The engine runs inside the Python process. If the engine crashes (a segmentation fault or an abort), the Python process ends with it, and no exception is raised. Evolutions that may crash, or that should be stopped from outside, belong in a child process (`examples/parallel_sweep.py`). An evolution whose allocation fails is truncated, with an `EngineWarning`, and a reply that does not fit either has `reduced=True` and the counts only. An address-space cap (`resource.setrlimit(resource.RLIMIT_AS, ...)` or `ulimit -v`) makes allocations fail at the cap instead of filling memory and swap.
- The engine runs one job at a time per process, with the GIL released. Calls from several threads wait for each other.
