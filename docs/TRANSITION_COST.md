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

One function computes it, `hgcommon::tc_child_cost` in `common/include/hgcommon/transition_cost_core.hpp`, from the parent's part (`tc_parent_of`, once per parent) and the hyperedges the transition consumes and produces. Only the vertices of those hyperedges can change their degree, loop count or edge multiplicity, so a transition costs time in the size of the rewrite after the parent's part is built. The curvature reads the whole child. The CPU engine calls it for each costed transition, and `hgevolve.heat_residual` calls it from Python with two whole states (`tc_transition_cost` finds the hyperedges that differ), so a Python callback can use the engine's residual directly.

## The step cap with a transition cost

Under "MaxStatesPerStep" a step keeps the transitions into its N lowest-cost successor states. A successor's cost is the lowest cost of a transition into it, and its successor rank breaks ties, as for "StateCost". Each step chooses by the cost of the step's transitions alone.

## Best-first search

Each state has a path cost: 0 for an initial state, and the parent's path cost plus the transition's cost for a child. The frontier holds transitions, ordered by the child's path cost, then the transition's rank, then a key of the parent's history (each initial state's position in the list, mixed with the rank of each transition along the path). The search takes the first transition, rewrites it, matches the child and adds the child's transitions to the frontier. Under "CanonicalizeStates" Full or Automatic a transition into a state the search already created, an initial state included, is dropped. The search stops when it has created "SearchBudget" states or the frontier is empty. A child at step "Steps" is not matched.

With non-negative costs this is Dijkstra's algorithm on the multiway system: states are created in order of path cost. Under "CanonicalizeStates" None every state has one history and its path cost is the sum of the costs along it. Under Full a class is created once, at the least path cost over the histories through the states the search created, and when "Steps" does not cut a history short this is the least path cost over all histories.

One transition is taken at a time, and the next is taken after the child's matching completes, so a search with a given "RandomSeed" creates the same states at any thread count. The rank of a transition reads the seed, as it does for the step caps. The matching of each state runs on all workers.

The reply has `"PathCosts"`: each state's path cost, by state id.

Best-first search does not combine with the step caps, "TransitionRate", "RuleWeights", "ExplorationProbability", "ExploreFromCanonicalStatesOnly", "Generational", "EventOrderingFunction" or sessions; the job drops them with a warning.

## Devices

The device has no transition cost and no best-first search. The GPU binary runs a job with either on the CPU engine with an "Engine" warning.
