"""Single histories in the Wolfram model's update orders.

"Generational" -> True applies, at each step, every match that shares no edge with an earlier
match in the standard order. "EventOrderingFunction" applies one match per step, the first in
the order its names give (SetReplace's names)."""
import hgevolve as hg

rule = ([["x", "y"], ["y", "z"]], [["x", "z"]])
path = [[1, 2], [2, 3], [3, 4], [4, 5]]

r = hg.evolve(rule, path, 5, data=["states"], generational=True)
print("Generational: edges per state", [len(e) for e in r.states.edges])

for order in ("Automatic", ["NewestEdge", "RuleOrdering"]):
    r = hg.evolve(rule, path, 3, data=["events"], event_ordering_function=order)
    print("EventOrderingFunction", order, "consumed:", r.events.consumed_edges.tolist())
