"""Step statistics: per step, the population of states summarised, with the geometry measures and
the branchial graph metrics."""
import hgevolve as hg

rule = ([[1, 2], [1, 3]], [[1, 2], [1, 4], [2, 4], [3, 4]])
r = hg.evolve(rule, [[1, 2], [1, 3]], 4, data=["step_statistics"],
              step_statistics_measures=["VolumeGrowth"], step_statistics_branchial="all")

for s in r.step_statistics:
    inv = s["Invariants"]
    print("step", s["Step"], "raw states", s["RawStates"], "classes", s["Classes"],
          "mean vertex count", inv["VertexCount"]["Mean"],
          "branchial components", s.get("BranchialComponents"))

# The same statistics, each class of a step counted once.
by_class = hg.evolve(rule, [[1, 2], [1, 3]], 4, data=["step_statistics"],
                     step_statistics_weighting="Classes", canonicalize_states="Full")
print("last step, by class:", by_class.step_statistics[-1]["Invariants"]["VertexCount"])
