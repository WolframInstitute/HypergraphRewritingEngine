"""A sampled history from a Brill-Lindquist initial condition.

The engine generates the initial condition (hg.initial_condition), as HGEvolve does for
HGEvolve[rule, "BrillLindquist", 10, "SprinklingDensity" -> 1600, "RandomSeed" -> 1,
"UniformRandom" -> True, "MatchesPerStep" -> 1]:

    python3 brill_lindquist_history.py [DENSITY]

DENSITY is the number of points, 400 by default."""
import sys

import hgevolve as hg

density = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ic = hg.initial_condition({"Type": "BrillLindquist", "Density": density, "Seed": 1})
rule = ([[1, 2], [1, 3]], [[1, 2], [1, 4], [2, 4], [3, 4]])
r = hg.evolve(rule, ic["Edges"], 10, uniform_random=True, matches_per_step=1, random_seed=1,
              data=["states", "events", "causal_edges", "num_states", "num_events",
                    "num_causal_edges"])
print(ic["VertexCount"], "points,", len(ic["Edges"]), "initial edges;", r.num_states, "states,",
      r.num_events, "events,", r.num_causal_edges, "causal edges")
print("edges per step:", r.states.edges.lengths().tolist())
