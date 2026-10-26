"""transition_cost, search="BestFirst" and heat_residual (docs/TRANSITION_COST.md)."""

import pytest

import hgevolve as hg

CYCLE = [[1, 2], [2, 3], [3, 4], [4, 1]]
LOOP_MOVE = ([["x", "x"], ["x", "y"]], [["x", "y"], ["y", "y"]])
# From {{x}}: one cheap first step (cost 1) whose continuation costs 10, and one dearer first step
# (cost 2) whose continuation costs 0. The cost is the child's largest arity through ARITY_COST.
TRAP = [([["x"]], [["x", "x"]]), ([["x"]], [["x", "x", "x"]]),
        ([["x", "x"]], [["x", "x", "x", "x"]]), ([["x", "x", "x"]], [["x", "x", "x", "x", "x"]])]
ARITY_COST = {2: 1.0, 3: 2.0, 4: 10.0, 5: 0.0}


def trap_cost(parent, child):
    return ARITY_COST[max(len(e) for e in child)]


def by_step(r):
    """{step: sorted path costs} of a best-first reply."""
    step = dict(zip(r.states.id.tolist(), r.states.step.tolist()))
    out = {}
    for sid, cost in r.path_costs.items():
        out.setdefault(step[sid], []).append(cost)
    return {s: sorted(c) for s, c in out.items()}


def test_heat_residual_on_the_four_cycle():
    # f_P = (2, 0, 0, 0) loops, (A - D) f_P = (-4, 2, 0, 2), dt = 1/4 predicts (1, 1/2, 0, 1/2);
    # the child's loops (1, 1, 0, 0) leave residuals (0, 1/2, 0, -1/2): 1/2.
    parent = CYCLE + [[1, 1], [1, 1]]
    child = CYCLE + [[1, 1], [2, 2]]
    assert hg.heat_residual(parent, child, "LoopCount", 0.25) == 0.5
    assert hg.heat_residual(CYCLE, CYCLE + [[1, 3]], "Degree", 0.7) == 2.0
    with pytest.raises(ValueError):
        hg.heat_residual(CYCLE, CYCLE, "Mass", 0.1)


def test_built_in_cost_through_the_engine():
    r = hg.evolve(LOOP_MOVE, CYCLE + [[1, 1], [1, 1]], 1, search="BestFirst",
                  transition_cost={"Field": "LoopCount", "TimeStep": 0.25},
                  data=["states", "num_states"])
    # The two loops at 1 with the edge 1-2 make two transitions costing 1/2. The loops also match
    # {x,y} as {1,1}: the child equals the parent, f_C = (2, 0, 0, 0), residuals (1, -1/2, 0, -1/2).
    assert by_step(r) == {0: [0.0], 1: [0.5, 0.5, 1.5, 1.5]}


def test_best_first_finds_a_lower_total_cost_than_the_step_cap():
    beam = hg.evolve(TRAP, [[1]], 2, max_states_per_step=1, transition_cost=trap_cost,
                     data=["states"])
    two = [i for i, s in enumerate(beam.states.step.tolist()) if s == 2]
    assert len(two) == 1
    assert max(len(v) for _, v in beam.states.edges[two[0]]) == 4   # 1 + 10 = 11
    best = hg.evolve(TRAP, [[1]], 2, search="BestFirst", search_budget=3,
                     transition_cost=trap_cost, data=["states"])
    assert by_step(best) == {0: [0.0], 1: [1.0, 2.0], 2: [2.0]}


RULES = [([["x", "y"], ["y", "z"]], [["x", "y"], ["y", "w"], ["w", "z"]]),
         ([["x", "y"]], [["x", "y"], ["y", "y"]])]


def test_a_callback_using_the_engine_residual_matches_the_built_in():
    # The transitions' ranks, which break ties of cost, read the seed: both runs use one.
    built_in = hg.evolve(RULES, CYCLE, 5, search="BestFirst", search_budget=40, random_seed=7,
                         transition_cost={"Field": "Degree", "TimeStep": 0.3}, data=["states"])
    calls = []

    def f(parent, child):
        calls.append(1)
        return hg.heat_residual(parent, child, "Degree", 0.3)

    hook = hg.evolve(RULES, CYCLE, 5, search="BestFirst", search_budget=40, random_seed=7,
                     transition_cost=f, data=["states"])
    assert calls
    assert by_step(hook) == by_step(built_in)
    assert len(built_in.path_costs) == 41


def test_result_does_not_depend_on_threads():
    runs = [by_step(hg.evolve(RULES, CYCLE, 6, search="BestFirst", search_budget=60,
                              random_seed=7, canonicalize_states="Full", threads=t,
                              transition_cost={"Field": "Curvature", "TimeStep": 0.5},
                              data=["states"]))
            for t in (1, 4)]
    assert runs[0] == runs[1]


def test_a_raising_callback_raises_from_evolve():
    def f(parent, child):
        raise KeyError("no cost")

    with pytest.raises(KeyError):
        hg.evolve(TRAP, [[1]], 2, search="BestFirst", transition_cost=f, data=["num_states"])


def test_refusals_and_warnings():
    with pytest.raises(hg.EngineError):
        hg.evolve(TRAP, [[1]], 2, transition_cost={"Field": "Mass", "TimeStep": 0.1},
                  max_states_per_step=1)
    with pytest.raises(hg.EngineError):
        hg.evolve(TRAP, [[1]], 2, transition_cost={"Field": "Degree"}, max_states_per_step=1)
    with pytest.warns(hg.EngineWarning, match="TransitionCostNeedsSelection"):
        hg.evolve(TRAP, [[1]], 2, transition_cost={"Field": "Degree", "TimeStep": 0.1},
                  data=["num_states"])
    with pytest.warns(hg.EngineWarning, match="BestFirstOptions"):
        hg.evolve(TRAP, [[1]], 2, search="BestFirst", max_states_per_step=1,
                  data=["num_states"])
