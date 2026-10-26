"""The arrays of a Result are views of the engine's reply: no copy, read-only, and valid for as
long as an array is alive, after its Result and its Session are gone."""

import gc

import numpy as np
import pytest

import hgevolve as hg

CHAIN = ([[1, 2]], [[1, 2], [2, 3]])


def data_pointer(a):
    return a.__array_interface__["data"][0]


def test_columns_are_views_of_the_reply():
    r = hg.evolve(CHAIN, [[1, 2]], 4)
    reply = r._reply
    for column, part, array in [
        ("Id", "values", r.states.id),
        ("Step", "values", r.states.step),
        ("Edges", "values", r.states.edges.ids),
        ("Edges", "offsets", r.states.edges.offsets),
        ("Edges", "vertices", r.states.edges.vertices.values),
        ("Edges", "vertex_offsets", r.states.edges.vertices.offsets),
    ]:
        assert data_pointer(array) == reply.address("States", column, part)
        assert not array.flags.owndata
    assert data_pointer(r.events.consumed_edges.values) == \
        reply.address("Events", "ConsumedEdges", "values")
    # A second read of the reply gives arrays over the same memory.
    again = reply.tables()["States"][1][0][2]["values"]
    assert data_pointer(again) == data_pointer(r.states.id)


def test_relations_are_views():
    r = hg.evolve(CHAIN, [[1, 2]], 4)
    a = r.causal_edges.array
    assert a.shape == (r.num_causal_edges, 4)
    assert not a.flags.owndata
    # A column of a relation is a strided view of the same array.
    assert np.shares_memory(r.causal_edges["From"], a)


def test_arrays_are_read_only():
    r = hg.evolve(CHAIN, [[1, 2]], 3)
    with pytest.raises(ValueError):
        r.states.id[0] = 7
    with pytest.raises(ValueError):
        r.causal_edges.array[0, 0] = 7


def test_arrays_outlive_their_result():
    r = hg.evolve(CHAIN, [[1, 2]], 4)
    ids = r.states.id
    edges = r.states.edges.vertices.values
    expected_ids, expected_edges = ids.copy(), edges.copy()
    del r
    gc.collect()
    # New allocations would reuse freed memory; the views must still read the reply.
    _ = [np.ones(1 << 16) for _ in range(64)]
    assert np.array_equal(ids, expected_ids)
    assert np.array_equal(edges, expected_edges)


def test_arrays_outlive_their_session():
    s = hg.Session(CHAIN, [[1, 2]], canonicalize_states="Full")
    r = s.step(3)
    ids = r.states.id
    hashes_before = hg.evolve(CHAIN, [[1, 2]], 3, canonicalize_states="Full").num_states
    expected = ids.copy()
    s.close()
    del s, r
    gc.collect()
    _ = hg.evolve(CHAIN, [[1, 2]], 5)
    _ = [np.ones(1 << 16) for _ in range(64)]
    assert np.array_equal(ids, expected)
    assert len(ids) == hashes_before


def test_array_outlives_a_deleted_open_session():
    # The session object is deleted without close(); its finaliser closes it.
    s = hg.Session(CHAIN, [[1, 2]])
    r = s.step(2)
    steps = r.states.step
    expected = steps.copy()
    del s, r
    gc.collect()
    # The slot is free: a new session opens.
    with hg.Session(CHAIN, [[1, 2]]) as again:
        again.step(1)
    assert np.array_equal(steps, expected)
