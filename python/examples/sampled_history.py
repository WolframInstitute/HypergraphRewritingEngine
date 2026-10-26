"""One sampled history from a large initial state: each step keeps one transition, drawn from the
seed ("UniformRandom" -> True, "MatchesPerStep" -> 1).

HGEvolve[rule, grid, 20, "UniformRandom" -> True, "MatchesPerStep" -> 1, "RandomSeed" -> 7]"""
import hgevolve as hg

# A 30 x 30 grid, built in Python: vertex (i, j) is i * 30 + j.
n = 30
grid = [[i * n + j, i * n + j + 1] for i in range(n) for j in range(n - 1)] + \
       [[i * n + j, (i + 1) * n + j] for i in range(n - 1) for j in range(n)]

rule = ([["x", "y"], ["x", "z"]], [["x", "y"], ["x", "w"], ["y", "w"], ["z", "w"]])
r = hg.evolve(rule, grid, 20, uniform_random=True, matches_per_step=1, random_seed=7,
              data=["states", "events", "num_states", "num_events"])

print(r.num_states, "states,", r.num_events, "events")
sizes = r.states.edges.lengths()
for step, size in zip(r.states.step.tolist(), sizes.tolist()):
    print("step", step, "edges", size)
