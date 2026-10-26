"""A sampled history from a Brill-Lindquist initial condition.

The paclet generates its initial conditions in the Wolfram Language with the kernel's random
generator, so the edges come from there, exported once:

    edges = HypergraphRewriting`Private`hgInitialEdges["BrillLindquist",
      {"SprinklingDensity" -> 1600, "RandomSeed" -> 1}];
    Export["bl1600.json", edges, "RawJSON"]

    python3 brill_lindquist_history.py bl1600.json

which runs HGEvolve[rule, "BrillLindquist", 10, "SprinklingDensity" -> 1600, "RandomSeed" -> 1,
"UniformRandom" -> True, "MatchesPerStep" -> 1]. Without an argument a small two-dimensional
point set made here stands in."""
import json
import random
import sys

import hgevolve as hg


def stand_in(n=200, threshold=0.12, seed=1):
    rng = random.Random(seed)
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    return [[i, j] for i in range(n) for j in range(i + 1, n)
            if (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 < threshold ** 2]


edges = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else stand_in()
rule = ([[1, 2], [1, 3]], [[1, 2], [1, 4], [2, 4], [3, 4]])
r = hg.evolve(rule, edges, 10, uniform_random=True, matches_per_step=1, random_seed=1,
              data=["states", "events", "causal_edges", "num_states", "num_events",
                    "num_causal_edges"])
print(len(edges), "initial edges;", r.num_states, "states,", r.num_events, "events,",
      r.num_causal_edges, "causal edges")
print("edges per step:", r.states.edges.lengths().tolist())
