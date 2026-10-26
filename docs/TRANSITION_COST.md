# Transition cost and best-first search

"StateCost" scores one state from its global "Invariants". A PDE residual needs more: the parent state, the child state, a field on the vertices of both, and the vertex correspondence of the transition. A rewrite keeps the ids of the vertices it does not delete, so the correspondence is the vertex id. "TransitionCost" scores a transition from the parent and the child, and "Search" -> "BestFirst" finds a history of lowest total cost.

## Options

| Option | Default | Value |
|---|---|---|
| `"TransitionCost"` | `None` | `<\|"Field" -> f, "TimeStep" -> dt\|>`: the heat residual of field f with time step dt |
| `"Search"` | `Automatic` | `"BestFirst"`: expand the transition of lowest path cost next |
| `"SearchBudget"` | `1000` | the number of states a best-first search creates after the initial states |

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

## Best-first search

Each state has a path cost: 0 for an initial state, and the parent's path cost plus the transition's cost for a child. The frontier holds transitions, ordered by the child's path cost, then the transition's rank, then a key of the parent's history (each initial state's position in the list, mixed with the rank of each transition along the path). The search takes the first transition, rewrites it, matches the child and adds the child's transitions to the frontier. Under "CanonicalizeStates" Full or Automatic a transition into a state the search already created, an initial state included, is dropped. The search stops when it has created "SearchBudget" states or the frontier is empty. A child at step "Steps" is not matched.

With non-negative costs this is Dijkstra's algorithm on the multiway system: states are created in order of path cost. Under "CanonicalizeStates" None every state has one history and its path cost is the sum of the costs along it. Under Full a class is created once, at the least path cost over the histories through the states the search created, and when "Steps" does not cut a history short this is the least path cost over all histories.

One transition is taken at a time, and the next is taken after the child's matching completes, so a search with a given "RandomSeed" creates the same states at any thread count. The rank of a transition reads the seed, as it does for the step caps. The matching of each state runs on all workers.

The reply has `"PathCosts"`: each state's path cost, by state id.

Best-first search does not combine with the step caps, "TransitionRate", "RuleWeights", "ExplorationProbability", "ExploreFromCanonicalStatesOnly", "Generational", "EventOrderingFunction" or sessions; the job drops them with a warning.

## Devices

"TransitionCost" runs on both devices. The device computes each candidate's cost with the same `tc_child_cost`, in the work log of costs that idle blocks run before a step's selection, and keeps a successor at its cheapest transition with the same `hgcommon::successor_lowest`. The two devices keep the same states and events (`TransitionCost.StepCapKeepsTheSameStatesOnBothEngines`), and `tc_child_cost` gives the same double on both (`TransitionCost.ChildCostIsTheSameDoubleOnHostAndDevice`).

Best-first search runs on the CPU engine, and the GPU binary runs it there with an "Engine" warning. The search takes one transition at a time, and the device engine creates states only in steps of its persistent kernel: it has no operation that applies one chosen match and returns the child's candidates with their costs. A device search keeps the host's heap and order and needs two device operations, the candidates of one state with their ranks, identities and costs, and the rewrite of one chosen match.

A Python callable cost is host code. The Python client runs every job on the CPU engine.

## Complexity

A state P with m hyperedges, n vertices and k costed transitions:

- Host, "Degree", "LoopCount" and "EdgeMultiplicity": the parent's part costs O(m log m) once per drain, and each transition O(s log s), where s is the number of incidences of the hyperedges at the rewritten vertices. A drain costs O(m log m + k s log s). Reading the transitions is Omega(k) and reading the parent is Omega(m), so the bound is a log factor above the minimum. The log comes from sorting the vertex ids and the terms.
- Host, "Curvature": each transition computes the child's curvature in full, one Ollivier transport problem per edge of G. The curvature of a vertex far from the rewrite does not change, so a transition could compute only the vertices near the rewrite. That bound is not implemented.
- Device: each candidate builds the parent's part itself, so a transition costs O(m log m) on one thread. The candidates of a step run in parallel. Sharing the parent's part between the candidates of a state would bring the device to the host's bound.
- Best-first search with budget B and frontier F: B heap pops and pushes, O((B + F) log F) on the host thread, plus the matching and the costs of the B expanded states. Every pop waits for the previous child's matching, so the search is sequential in B. The matching of each state runs on all workers.

## Measurements

Callgrind instructions (Ir), one worker, `hg_evolve` built with the same flags at each commit. The job is a cycle of n vertices with a loop at each vertex and the loop-move rule `{{x,x},{x,y}} -> {{x,y},{y,y}}`, one step under "MaxStatesPerStep" 1, so the initial state's n transitions are each costed once. The table gives the Ir of `transition_cost_of` per transition.

| Field | n | efed30ff | this change |
|---|---|---|---|
| Degree | 100 | 9,602 | 12,477 |
| Degree | 400 | 10,276 | 16,548 |
| Degree | 1600 | 11,022 | 18,330 |
| LoopCount | 1600 | 11,400 | 18,883 |
| Curvature | 100 | 2,438,425 | 2,378,929 |
| Curvature | 400 | 30,926,544 | 30,016,530 |

For the local fields the cost per transition grows by 47% from n = 100 to n = 1600 while n grows 16-fold, and is 30% to 66% above efed30ff, whose sums ran in vertex order over std::vector storage. For the curvature the cost per transition grows with n, as the bound above says, and is 3% lower.

Best-first search, the growth rules from a 4-cycle with the degree residual: the whole run is 40.3, 68.5, 123.7, 223.6 and 431.3 million Ir for budgets of 250, 500, 1000, 2000 and 4000 states, linear in the budget. The driver's own work (`run_best_first`) is 2,600 Ir per state created.

Wall time of the same search with budget 4000, three runs each, box not quiet (load 12): 0.33 s at 1 worker, 0.32 to 0.35 s at 2, 0.38 to 0.44 s at 4, 0.46 to 0.50 s at 8 and 0.54 to 0.62 s at 16, with byte-identical replies.

