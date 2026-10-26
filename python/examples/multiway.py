"""A multiway evolution: every state reachable in each number of steps, every event between them,
and the causal and branchial relations, as NumPy arrays.

HGEvolve[{{x, y}, {x, z}} -> {{x, y}, {x, w}, {y, w}, {z, w}}, {{1, 2}, {1, 3}}, 3]"""
import numpy as np

import hgevolve as hg

rule = ([["x", "y"], ["x", "z"]], [["x", "y"], ["x", "w"], ["y", "w"], ["z", "w"]])
r = hg.evolve(rule, [[1, 2], [1, 3]], 3, canonicalize_states="Full")

print(r)
print("states per step:", np.bincount(r.states.step).tolist())
print("edges per state:", r.states.edges.lengths().tolist())
first = r.states.edges[0]
print("state", r.states.id[0], "edges:", [(eid, v.tolist()) for eid, v in first])

# Events: the rule applied, the states it joins, and the edges it consumes and produces.
e = r.events
print("event 0:", e.rule_index[0], e.input_state[0], "->", e.output_state[0],
      "consumes", e.consumed_edges[0].tolist(), "produces", e.produced_edges[0].tolist())

# Relations are n x k arrays; a column is a view.
print("causal edges:", r.causal_edges.array.shape, "branchial edges:", len(r.branchial_edges))
out_degree = np.bincount(r.causal_edges["From"], minlength=int(e.id.max()) + 1)
print("largest causal out-degree:", int(out_degree.max()))
