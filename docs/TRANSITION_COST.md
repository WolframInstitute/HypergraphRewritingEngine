# Transition cost and best-first search

"StateCost" scores one state from its global "Invariants". A PDE residual needs more: the parent state, the child state, a field on the vertices of both, and the vertex correspondence of the transition. A rewrite keeps the ids of the vertices it does not delete, so the correspondence is the vertex id. "TransitionCost" scores a transition from the parent and the child, and "Search" -> "BestFirst" finds a history of lowest total cost.

## Options

| Option | Default | Value |
|---|---|---|
| `"TransitionCost"` | `None` | `<\|"Field" -> f, "TimeStep" -> dt\|>`: the heat residual of field f with time step dt |
| `"Search"` | `Automatic` | `"BestFirst"`: expand the transition of lowest path cost next |
| `"SearchBudget"` | `1000` | the number of states a best-first search creates after the initial states |
| `"BeamCost"` | `"Transition"` | `"Path"`: the step cap ranks by path cost and makes each successor once |

The cost of a transition is its "TransitionCost" plus the "StateCost" of its child, each when given. The hgevolve Python client also takes a callable as `transition_cost`: `f(parent_edges, child_edges) -> float`, called for every costed transition with the two states' edges as lists of vertex ids.

## Fields

The fields are per-vertex values of a state. G is the state's graph under "StatisticsGraph", the graph the geometry statistics read.

| Field | Value at vertex v |
|---|---|
| `"Degree"` | the degree of v in G |
| `"Curvature"` | the Ollivier-Ricci curvature k_v, as "VertexCurvature" reports it; NaN for a vertex with no neighbour |
| `"LoopCount"` | the number of hyperedges whose every vertex is v |
| `"EdgeMultiplicity"` | the number of hyperedges that contain v and are not loops, each copy counted once |

## The heat residual

For parent P with field f_P and graph G_P, child C with field f_C, and time step dt, the residual at a vertex v of both P and C is

    r_v = f_C(v) - f_P(v) - dt * sum over the neighbours w of v in G_P of (f_P(w) - f_P(v))

The sum is the graph Laplacian A - D of G_P applied to f_P, so r_v is the error of one forward Euler step of the heat equation du/dt = (A - D) u. The cost is the sum of r_v^2 over the vertices of both states with a finite r_v. Vertices that the rewrite deletes or creates have no residual.

One function computes it, `hgcommon::tc_child_cost` in `common/include/hgcommon/transition_cost_core.hpp`, one body for host and device. It reads the parent's part (`tc_parent_build`: the field, e_v = -dt ((A - D) f_P)(v) at each vertex, their sum of squares B and each vertex's hyperedges, built once per parent) and the hyperedges the transition consumes and produces. A consumed hyperedge that a produced one equals is dropped from both. The cost is B - (sum over T of e_v^2) + (sum over T of r_v^2), where T holds the parent's vertices whose field can change. The degree, loop count and edge multiplicity of v read only the hyperedges that contain v, so T is the vertices of the consumed and produced hyperedges, and a transition costs time in the size of the rewrite and the degree of its vertices after the parent's part is built. The curvature reads a wider neighbourhood, so under "Curvature" T is every vertex and the child's field is computed in full.

The squares are exact (double-double) and each of the three sums adds its terms in ascending order of value. The cost is therefore a function of the multisets of per-vertex values and does not depend on the vertex ids, so the CPU engine and the device, which number new vertices differently, compute the same double. The CPU engine calls `tc_child_cost` for each costed transition, the device calls it in `transition_cost_device`, and `hgevolve.heat_residual` calls it from Python with two whole states (`tc_transition_cost` finds the hyperedges that differ), so a Python callback can use the engine's residual directly.

## The step cap with a transition cost

Under "MaxStatesPerStep" a step keeps the transitions into its N lowest-cost successor states. A successor's cost is the lowest cost of a transition into it (`hgcommon::successor_lowest`, on both engines), and its successor rank breaks ties, as for "StateCost". Each step chooses by the cost of the step's transitions alone. The device computes each candidate's cost in the cost log that idle blocks run before the step's selection (`step_cost_one`, `successor_cost_device`), as it does for "StateCost".

## The step cap ranked by path cost

`"BeamCost" -> "Path"` keeps the K = "MaxStatesPerStep" successor states of lowest path cost at each step, where a state's path cost is its parent's path cost plus the cost of the transition that made it, 0 for an initial state. `"BeamCost" -> "Transition"`, the default, ranks by the transition's cost alone. Under "Path", of the transitions into a kept successor only the ones of lowest (path cost, rank) are taken (`hgcommon::successor_cheapest`), so a step makes each successor once. With "CanonicalizeStates" -> Automatic two transitions that make the same edge list make one successor, so duplicate states are dropped by their edge lists without canonicalisation; the vertex ids stay the coordinates of the field. The rewrite that makes a state records its path cost (`beam_path_cost_`), and the state's drain adds it to the costs of its transitions. The reply's "PathCosts" gives every state's path cost.

This is the selection experiment 015 runs in Python: a beam of K paths ranked by summed cost, one engine call for the whole history instead of one call per beam state per step. A beam of K over T steps creates at most K T states, so a history of 6,000 events at K = 8 creates 48,000 states. Best-first search with the same costs creates every state cheaper than the cheapest state at the final depth first; in experiment 015 it created 6,571 states without passing depth 3.

## Planned for experiment 015

These are not implemented. Each is a function of the path and the step that the drain can compute, so each fits the step cap of both engines.

- A residual against the state W events back on the same path: each state keeps the edge list of its ancestor W steps back (a ring of W per kept path), and the cost reads that ancestor's field and the heat flow exp(dt L) of it on a grid of dt, the minimum over the grid.
- A target field per step on the initial vertex ids: a list of vectors g_k, and the cost |u_child - g_(k+1)| over the vertices that carry an initial id.
- The Lean hypergraph Laplacian as an option of the residual: every hyperedge adds f(w) - f(v) for each other vertex w of the edge as a set, with multiplicity, where the residual here reads the simple graph of "StatisticsGraph".
- The chosen path in the reply: the states of the lowest-cost final state's history at a stride, and the cost terms of each of its transitions.
## Best-first search

Each state has a path cost: 0 for an initial state, and the parent's path cost plus the transition's cost for a child. The frontier holds transitions, ordered by the child's path cost, then the transition's rank, then a key of the parent's history (each initial state's position in the list, mixed with the rank of each transition along the path). The search takes the first transition, rewrites it, matches the child and adds the child's transitions to the frontier. Under "CanonicalizeStates" Full or Automatic a transition into a state the search already created, an initial state included, is dropped. The search stops when it has created "SearchBudget" states or the frontier is empty. A child at step "Steps" is not matched.

With non-negative costs this is Dijkstra's algorithm on the multiway system: states are created in order of path cost. Under "CanonicalizeStates" None every state has one history and its path cost is the sum of the costs along it. Under Full a class is created once, at the least path cost over the histories through the states the search created, and when "Steps" does not cut a history short this is the least path cost over all histories.

One transition is taken at a time, and the next is taken after the child's matching completes, so a search with a given "RandomSeed" creates the same states at any thread count. The rank of a transition reads the seed, as it does for the step caps. The rewrite and the matching of each taken transition run on the thread that drives the search, and the costs of the transitions it adds run on all workers.

The reply has `"PathCosts"`: each state's path cost, by state id.

Best-first search does not combine with the step caps, "TransitionRate", "RuleWeights", "ExplorationProbability", "ExploreFromCanonicalStatesOnly", "Generational", "EventOrderingFunction" or sessions; the job drops them with a warning.

## Devices

"TransitionCost" runs on both devices. The device computes each candidate's cost with the same `tc_child_cost`, in the work log of costs that idle blocks run before a step's selection, and keeps a successor at its cheapest transition with the same `hgcommon::successor_lowest`. The two devices keep the same states and events (`TransitionCost.StepCapKeepsTheSameStatesOnBothEngines`), and `tc_child_cost` gives the same double on both (`TransitionCost.ChildCostIsTheSameDoubleOnHostAndDevice`).

"BeamCost" -> "Path" runs on both devices. The device adds the parent's path cost in `successor_cost_device`, marks the cheapest transitions with the same `hgcommon::successor_cheapest`, and its rewrite records the child's path cost from the selected record (`record_path_cost`). The two devices keep the same states with the same path costs, bit for bit (`BeamPath.TheSameStatesAndPathCostsOnBothEngines`).

Best-first search runs on the CPU engine, and the GPU binary runs it there with an "Engine" warning. The search takes one transition at a time, and the device engine creates states only in steps of its persistent kernel: it has no operation that applies one chosen match and returns the child's candidates with their costs. A device search keeps the host's heap and order and needs two device operations, the candidates of one state with their ranks, identities and costs, and the rewrite of one chosen match.

A Python callable cost is host code. The Python client runs every job on the CPU engine.

## Complexity

A state P with m hyperedges, n vertices and k costed transitions:

- Host, "Degree", "LoopCount" and "EdgeMultiplicity": the parent's part costs O(m log m) once per drain, and each transition O(s log s), where s is the number of incidences of the hyperedges at the rewritten vertices. A drain costs O(m log m + k s log s). Reading the transitions is Omega(k) and reading the parent is Omega(m), so the bound is a log factor above the minimum. The log comes from sorting the vertex ids and the terms.
- Host, "Curvature": each transition computes the child's curvature in full, one Ollivier transport problem per edge of G. The curvature of a vertex far from the rewrite does not change, so a transition could compute only the vertices near the rewrite. That bound is not implemented.
- Device: each candidate builds the parent's part itself, so a transition costs O(m log m) on one thread. The candidates of a step run in parallel. Sharing the parent's part between the candidates of a state would bring the device to the host's bound.
- Best-first search with budget B and frontier F: B heap pops and pushes, O((B + F) log F) on the driver thread, plus the matching and the costs of the B expanded states. Every pop waits for the previous child's matching, so the search is sequential in B; the costs of each child's transitions run on all workers.

## Measurements

Callgrind instructions (Ir), one process, clang++ -O3. The job is a cycle of n vertices with a loop at each vertex and the loop-move rule `{{x,x},{x,y}} -> {{x,y},{y,y}}`: the parent's part is built once (`tc_parent_of`) and each of the n transitions is costed (`tc_child_cost`). The table gives the whole process's Ir divided by n, so the parent's part and the start-up are spread over the transitions.

| Field | n | Ir per transition |
|---|---|---|
| Degree | 100 | 28,000 |
| Degree | 400 | 14,233 |
| Degree | 1600 | 11,253 |
| LoopCount | 100 | 28,353 |
| LoopCount | 400 | 14,615 |
| LoopCount | 1600 | 11,654 |

A transition's own work is about 5,600 Ir at every n (the child's local field over the hyperedges at the rewritten vertices). The host builds the parent's part in one pass: its arena starts at 80 bytes per incidence (`kTcBytesPerIncidence`). `sg_sort` sorts up to 64 values by insertion, which is most of a child's sorts.

Best-first search, the growth rules from a 4-cycle with the degree residual, 10 steps, one worker: the whole run is 39.7, 67.5, 121.9, 220.3 and 424.2 million Ir for budgets of 250, 500, 1000, 2000 and 4000 states, linear in the budget, 103,000 Ir per state created by `run_best_first` with the rewrite, the matching and the costs it runs.

Wall time of the same search with budget 4000, three runs each, box not quiet (load 5): 1 worker 0.068 to 0.072 s, 2 workers 0.066 to 0.069 s, 4 workers 0.067 to 0.069 s, 8 workers 0.067 to 0.071 s, 16 workers 0.070 to 0.071 s, with byte-identical replies. A wide state (a 2000-vertex cycle with a loop at each vertex, budget 40): 1 worker 0.135 to 0.152 s, 4 workers 0.118 to 0.121 s, 16 workers 0.134 to 0.143 s. The rewrite, the matching and the drain of the popped transition run on the driver thread; the costs of the drained transitions are computed across the workers in chunks of 64 (`cost_best_first`).

"BeamCost": the loop move on a 400-cycle with 40 loops at vertex 0, Automatic states, a beam of 8 for 40 steps under the loop count's residual, one worker: ranked by transition the run is 324.8 billion Ir, because the ties at each cut keep every tied transition and its state; ranked by path it is 46.0 billion Ir.

Device against host, whole `hg_evolve_gpu` and `hg_evolve` calls on the same job, RTX 4090 at 4% use before the run, box at load 9: the beam job above takes 0.45 s on the CPU engine at 16 workers and 51 to 60 s on the device; one costed step of the loop move on a 1600-cycle takes 0.05 s on the CPU and 51 s on the device, on a 400-cycle 0.06 s and 0.46 s.
