"""The hgevolve API: inputs, options, errors, warnings, sessions and the job encoding."""

import os
import re
import warnings

import numpy as np
import pytest

import hgevolve as hg
from hgevolve._cost import state_cost_program
from hgevolve._options import initial_states, rules_association

CHAIN = ([[1, 2]], [[1, 2], [2, 3]])
LOOPS = ([[1, 1]], [[1, 1], [1, 1]])
HERE = os.path.dirname(os.path.abspath(__file__))
API_DOC = os.path.join(HERE, "..", "docs", "API.md")


def test_rule_labels_are_numbered_by_first_appearance():
    r = rules_association([([["x", "y"]], [["x", "z"], ["z", "y"]])])
    assert r["Rule1"].head == "Rule"
    assert r["Rule1"].args == [[[0, 1]], [[0, 2], [2, 1]]]
    # Integer labels are numbered too, as HGEvolve numbers them.
    assert rules_association(([[1, 2]], [[1, 4], [4, 3], [3, 2]]))["Rule1"].args == \
        [[[0, 1]], [[0, 2], [2, 3], [3, 1]]]


def test_initial_states_forms():
    assert initial_states([[1, 2]]) == [[[1, 2]]]
    assert initial_states([[[1, 2]], [[5, 6]]]) == [[[1, 2]], [[5, 6]]]
    assert initial_states([["a", "b"], ["b", "c"]]) == [[[0, 1], [1, 2]]]
    assert initial_states(np.array([[1, 2], [2, 3]])) == [[[1, 2], [2, 3]]]
    with pytest.raises(ValueError):
        initial_states([])
    for generated in ({"Type": "Grid", "Width": 3}, "Grid"):
        with pytest.raises(ValueError, match="Wolfram Language paclet"):
            initial_states(generated)


def test_symbolic_and_integer_rules_agree():
    a = hg.evolve(([["x", "y"]], [["x", "z"], ["z", "y"]]), [[1, 2]], 3, data=["num_states"])
    b = hg.evolve(CHAIN, [[1, 2]], 3, data=["num_states"])
    assert a.num_states == b.num_states == 10  # HGEvolve.md: <!-- => 10 -->


def test_counts_of_the_documentation():
    # HGEvolve.md: Table[HGEvolve[chain, {{1, 2}}, k, "NumStates"], {k, 0, 5}]
    assert [hg.evolve(CHAIN, [[1, 2]], k, data=["num_states"]).num_states
            for k in range(6)] == [1, 2, 4, 10, 34, 154]


def test_unknown_option_and_data_name():
    with pytest.raises(TypeError, match="unknown option"):
        hg.evolve(CHAIN, [[1, 2]], 1, canonicalise_states="Full")
    with pytest.raises(ValueError, match="is not one of"):
        hg.evolve(CHAIN, [[1, 2]], 1, data=["NumStates"])


def test_engine_refusal_is_an_engine_error():
    with pytest.raises(hg.EngineError, match="arity"):
        hg.evolve(([[1, 2]], [[1, 2], [2, 3]]), [[1] * 17], 1)
    with pytest.raises(hg.EngineError, match="EventOrderingFunction"):
        hg.evolve(CHAIN, [[1, 2]], 1, event_ordering_function="Random")


def test_engine_warnings_are_python_warnings():
    with pytest.warns(hg.EngineWarning, match="QuotientNeedsFull"):
        hg.evolve(CHAIN, [[1, 2]], 1, explore_from_canonical_states_only=True)


def test_generational_and_event_ordering():
    rule = ([["x", "y"], ["y", "z"]], [["x", "z"]])
    init = [[1, 2], [2, 3], [3, 4], [4, 5]]
    # HGEvolve.md: Length /@ Lookup[Values @ HGEvolve[..., 5, "States", "Generational" -> True], "Edges"]
    r = hg.evolve(rule, init, 5, data=["states"], generational=True)
    assert [len(e) for e in r.states.edges] == [4, 2, 1]
    r = hg.evolve(rule, init, 3, data=["events"], event_ordering_function="Automatic")
    assert len(r.events) == 3
    with pytest.raises(hg.EngineError):
        hg.evolve(rule, init, 1, generational=True, event_ordering_function="Automatic")


def test_undirected_edges():
    # HGEvolve.md: HGEvolve[{{x, y}} -> {{x, y}, {y, z}}, {{1, 2}}, 1, "NumEvents", "UndirectedEdges" -> #]
    rule = ([["x", "y"]], [["x", "y"], ["y", "z"]])
    counts = [hg.evolve(rule, [[1, 2]], 1, data=["num_events"], undirected_edges=u).num_events
              for u in (False, True)]
    assert counts == [1, 2]


def test_step_statistics_measure_names():
    r = hg.evolve(LOOPS, [[1, 1], [1, 1]], 2, data=["step_statistics"],
                  step_statistics_measures="all", canonicalize_states="Full")
    assert len(r.step_statistics) == 3
    assert "Invariants" in r.step_statistics[0]


def test_state_cost_compiles_to_the_engine_program():
    p = state_cost_program("abs(BallDimension - 1.5)")
    assert p == {"Code": ["BallDimension", "Const", "Minus", "Plus", "Abs"], "Constants": [1.5]}
    p = state_cost_program("VertexCount / 2 + max(EdgeCount, 3) ** 2")
    assert p["Code"] == ["VertexCount", "Const", "Power", "Times", "EdgeCount", "Const", "Max",
                         "Power", "Plus"]
    assert p["Constants"] == [2.0, -1.0, 3.0, 2.0]
    with pytest.raises(ValueError):
        state_cost_program("sin(VertexCount)")


def test_session_steps_match_evolve():
    # HGEvolve.md: {Table[HGSessionStep[session, 1], 3], Table[HGEvolve[chain, {{1, 2}}, k, "NumStates"], {k, 3}]}
    with hg.Session(CHAIN, [[1, 2]]) as s:
        stepped = [s.step(1, data=["num_states"]).num_states for _ in range(3)]
    assert stepped == [hg.evolve(CHAIN, [[1, 2]], k, data=["num_states"]).num_states
                       for k in range(1, 4)]


def test_one_session_per_process():
    with hg.Session(CHAIN, [[1, 2]]):
        with pytest.raises(hg.EngineError, match="Open refused"):
            hg.Session(CHAIN, [[1, 2]])
    with hg.Session(CHAIN, [[1, 2]]) as s:
        assert s.query(data=["num_states"]).num_states == 1


def test_steered_step():
    with hg.Session(CHAIN, [[1, 2]], canonicalize_states="Full") as s:
        r = s.step(2)
        before = list(r.frontier)
        r2 = s.step(1, start=before[:1])
        assert set(before[1:]) <= set(r2.frontier)


def test_threads_do_not_change_the_reply_content():
    a = hg.evolve(LOOPS, [[1, 1], [1, 1]], 3, threads=1, include_canonical_hashes=True)
    b = hg.evolve(LOOPS, [[1, 1], [1, 1]], 3, threads=4, include_canonical_hashes=True)
    assert sorted(a.states.canonical_hash.tolist()) == sorted(b.states.canonical_hash.tolist())
    assert (a.num_events, a.num_causal_edges, a.num_branchial_edges) == \
        (b.num_events, b.num_causal_edges, b.num_branchial_edges)


def test_ragged_columns():
    r = hg.evolve(CHAIN, [[1, 2]], 2)
    edges = r.states.edges
    assert len(edges) == len(r.states)
    assert edges.offsets[0] == 0 and edges.offsets[-1] == len(edges.ids)
    assert [len(x) for x in edges] == np.diff(edges.offsets).tolist()
    consumed = r.events.consumed_edges
    assert consumed.lengths().tolist() == [1] * len(r.events)
    assert r.states["CanonicalId"] is r.states.canonical_id


def test_encode_job_is_the_job_hg_evolve_reads():
    b = hg.encode_job(CHAIN, [[1, 2]], 2, data=["num_states"], random_seed=5)
    assert b[:2] == b"8:"
    assert b"RequestedData" in b and b"NumStates" in b and b"Rule1" in b


def test_every_option_and_data_name_is_documented():
    with open(API_DOC) as f:
        doc = f.read()
    for name, (wire, _) in hg.OPTIONS.items():
        assert re.search(rf"`{name}`.*`\"{wire}\"`", doc), name
    for name, wire in hg.DATA.items():
        assert re.search(rf"`{name}`.*`\"{wire}\"`", doc), name


def test_build_stamp_names_the_module():
    assert "variant=python" in hg.build_stamp()
