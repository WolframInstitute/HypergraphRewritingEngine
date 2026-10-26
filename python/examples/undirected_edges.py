"""Undirected edges: with "UndirectedEdges" -> True a two-vertex edge is an unordered pair, so a
rule's edge {x, y} matches it in either order."""
import hgevolve as hg

grow = ([["x", "y"]], [["x", "y"], ["y", "z"]])
for undirected in (False, True):
    r = hg.evolve(grow, [[1, 2]], 1, data=["num_events"], undirected_edges=undirected)
    print("UndirectedEdges", undirected, "events:", r.num_events)

flip = ([["x", "y"]], [["y", "x"]])
for undirected in (False, True):
    r = hg.evolve(flip, [[1, 2], [2, 3]], 1, data=["num_states"], canonicalize_states="Full",
                  undirected_edges=undirected)
    print("UndirectedEdges", undirected, "states up to isomorphism:", r.num_states)
