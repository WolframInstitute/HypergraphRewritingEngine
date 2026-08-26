#!/usr/bin/env python3
"""Turn the rich_sweep data set into paper figures.

WHAT THIS EXISTS TO FIX. The rule-shape evidence was one table and one figure carrying two
curves, a one-edge and a two-edge left-hand side, and the figure plotted parallel efficiency
alone. Three things were wrong with that and none of them is a drawing problem:

  * two points do not establish a trend in left-hand side SIZE, and sizes three and four were
    never measured;
  * size was the only axis measured at all -- edge ARITY and left-hand side CONNECTIVITY were
    held fixed at "binary" and "chain" without being named as choices;
  * the run produced state, event, causal-edge and branchial-edge counts at every depth and the
    paper reported none of them, so a reader could not tell a workload with 200k states and 40k
    causal edges from one with 200k states and 4M.

SPEEDUP AGAINST WHAT, STATED IN THE FIGURE. A speedup curve is meaningless without its
denominator, so the scaling figures carry the ideal line y=x and the efficiency figures carry
y=1, both drawn and both in the legend. The baseline is this same binary on this same workload
at one thread, and the caption says so; no number here is relative to another system.

Reads the two files rich_sweep.sh writes and emits pgfplots \\addplot bodies, in the same style
as the existing fragments: the axis environment lives in main.tex, this supplies the plots.

Usage: rich_plots.py <rich-dir> [--out paper/tables]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt   # noqa: E402  (path set above)


def parse(path):
    """One dict per RICH line. Values stay strings; callers convert what they need."""
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if not line.startswith("RICH "):
            continue
        d = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        rows.append(d)
    return rows


def num(d, k, default=0.0):
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return default


# Distinguishable without colour, because the paper is read printed as often as on screen.
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o", "square", "triangle"]
DASHES = ["solid", "dashed", "dotted", "densely dashed", "loosely dashed",
          "densely dotted", "dashdotted", "solid"]


def curve(i, coords, label):
    return (r"\addplot[mark=%s, black, %s, mark size=1.4pt] coordinates {%s};"
            % (MARKS[i % len(MARKS)], DASHES[i % len(DASHES)], coords)
            + "\n" + r"\addlegendentry{%s}" % label)


def scaling_figure(rows, rules, labels, fname, tool, out, metric="eff"):
    """Efficiency or speedup against thread count, one curve per rule.

    The baseline is the SAME rule at one thread, so each curve is normalised by its own serial
    point and the curves are comparable to each other rather than to a shared constant.
    """
    body = [pt.provenance(tool)]
    drawn = 0
    for rule in rules:
        pts = {}
        for r in rows:
            if r.get("rule") != rule:
                continue
            th = int(num(r, "threads", 0))
            ms = num(r, "ms", 0.0)
            if th <= 0 or ms <= 0:
                continue
            # MIN over repeats: contention can only make a run slower, so the minimum is the
            # estimate interference cannot inflate.
            if th not in pts or ms < pts[th]:
                pts[th] = ms
        if 1 not in pts or len(pts) < 3:
            continue
        base = pts[1]
        coords = []
        for th in sorted(pts):
            sp = base / pts[th]
            coords.append("(%d,%.3f)" % (th, sp if metric == "speedup" else sp / th))
        body.append(curve(drawn, "".join(coords), labels.get(rule, rule)))
        drawn += 1
    if not drawn:
        return 0
    # The reference the curves are read against, drawn rather than described.
    if metric == "speedup":
        mx = max(int(num(r, "threads", 1)) for r in rows) or 32
        body.append(r"\addplot[gray, thick, no marks, forget plot] coordinates {(1,1)(%d,%d)};"
                    % (mx, mx))
    else:
        body.append(r"\addplot[gray, thick, no marks, forget plot] coordinates {(1,1)(32,1)};")
    pt.write_raw(fname, "\n".join(body) + "\n")
    return drawn


def depth_figure(rows, rules, labels, fname, tool, out, ykey):
    """A count against evolution depth, one curve per shape, for whatever count is asked for."""
    body = [pt.provenance(tool)]
    drawn = 0
    for rule in rules:
        pts = {}
        for r in rows:
            if r.get("rule") != rule:
                continue
            d = int(num(r, "steps", 0))
            y = num(r, ykey, 0.0)
            if d <= 0 or y <= 0:
                continue
            pts[d] = max(pts.get(d, 0.0), y)
        if len(pts) < 2:
            continue
        coords = "".join("(%d,%d)" % (d, int(pts[d])) for d in sorted(pts))
        body.append(curve(drawn, coords, labels.get(rule, rule)))
        drawn += 1
    if not drawn:
        return 0
    pt.write_raw(fname, "\n".join(body) + "\n")
    return drawn


def relation_figure(rows, rules, labels, fname, tool, out):
    """Branchial edges against causal edges, one curve per shape.

    Both axes are sizes of relations over the SAME state set, so a shape's position on this plot
    is a property of the rule rather than of how long it was run: a shape that generates many
    branchial pairs per causal edge sits high whatever depth it reached.
    """
    body = [pt.provenance(tool)]
    drawn = 0
    for rule in rules:
        pts = []
        for r in rows:
            if r.get("rule") != rule:
                continue
            c, b = num(r, "causal_edges"), num(r, "branchial_edges")
            if c > 0 and b > 0:
                pts.append((c, b))
        if len(pts) < 2:
            continue
        pts.sort()
        coords = "".join("(%d,%d)" % (int(c), int(b)) for c, b in pts)
        body.append(curve(drawn, coords, labels.get(rule, rule)))
        drawn += 1
    if not drawn:
        return 0
    pt.write_raw(fname, "\n".join(body) + "\n")
    return drawn


def shape_table(rows, out, tool):
    """One row per shape: how deep it reached and how large its relations became there.

    The deepest row per shape, because that is the point where the shape is most itself -- the
    relation sizes at depth one are dominated by the seed for every shape alike.
    """
    best = {}
    for r in rows:
        rule = r.get("rule")
        if not rule:
            continue
        d = int(num(r, "steps", 0))
        if rule not in best or d > int(num(best[rule], "steps", 0)):
            best[rule] = r
    if not best:
        return 0
    b = [pt.provenance(tool),
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Rule & Shape & Arity & LHS & Depth & States & Causal & Branchial \\", r"\midrule"]
    for rule in sorted(best, key=lambda k: (best[k].get("shape", ""),
                                            int(num(best[k], "lhs_edges", 0)),
                                            int(num(best[k], "arity", 0)))):
        r = best[rule]
        b.append(r"\texttt{%s} & %s & %d & %d & %d & %s & %s & %s \\" % (
            pt.tex_escape(rule), r.get("shape", "?"), int(num(r, "arity")),
            int(num(r, "lhs_edges")), int(num(r, "steps")),
            "{:,}".format(int(num(r, "states"))),
            "{:,}".format(int(num(r, "causal_edges"))),
            "{:,}".format(int(num(r, "branchial_edges")))))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t14_shape_space.tex", "\n".join(b) + "\n")
    return len(best)


SIZE_RULES = ["chain1a2", "chain2a2", "chain3a2", "chain4a2"]
SIZE_LABELS = {"chain1a2": "1-edge LHS", "chain2a2": "2-edge LHS",
               "chain3a2": "3-edge LHS", "chain4a2": "4-edge LHS"}

SHAPE_RULES = ["chain3a2", "star3a2", "tree4a2", "cycle4a2", "disc2a2"]
SHAPE_LABELS = {"chain3a2": "chain (path)", "star3a2": "star (hub)",
                "tree4a2": "tree (branching)", "cycle4a2": "cycle (ring)",
                "disc2a2": "disconnected"}

# THE ARITY PANEL COMPARES AT FIXED SIZE, and its rules have to be ones the TIMED phase actually
# runs -- the first list named two-edge shapes that only the depth phase sweeps, so the panel
# could draw exactly one curve however much data was collected. Three edges is the size the timed
# phase carries at arity 2 and 3 and mixed, so that is where arity is compared.
ARITY_RULES = ["chain3a2", "chain3a3", "mixed3"]
ARITY_LABELS = {"chain3a2": "arity 2", "chain3a3": "arity 3",
                "mixed3": "mixed arity 2/3"}

# THE DEPTH PANELS GET THEIR OWN LIST, NOT THE UNION OF THE SCALING LISTS. Concatenating those
# lists put chain3a2 in twice -- it is the 3-edge point of the size axis AND the path point of
# the connectivity axis -- so the figure drew it twice under two different names, and it mixed
# two naming schemes ("3-edge LHS" beside "chain (path)") inside one legend. Six shapes, each
# named the same way: connectivity first, then size, so one legend reads consistently.
DEPTH_RULES = ["chain1a2", "chain2a2", "chain4a2", "star3a2", "cycle4a2", "disc2a2"]
DEPTH_LABELS = {"chain1a2": "chain, 1 edge", "chain2a2": "chain, 2 edges",
                "chain4a2": "chain, 4 edges", "star3a2": "star, 3 edges",
                "cycle4a2": "ring, 4 edges", "disc2a2": "disconnected, 2 parts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rich_dir")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()

    pt.OUT = a.out
    depth = parse(os.path.join(a.rich_dir, "rich_depth.txt"))
    scale = parse(os.path.join(a.rich_dir, "rich_scaling.txt"))
    print("depth rows %d, scaling rows %d" % (len(depth), len(scale)))

    tool = "tools/dev/rich_sweep.sh over tools/sampling_cost_smoke.cpp"
    n = 0
    n += scaling_figure(scale, SIZE_RULES, SIZE_LABELS, "f_eff_size.tex", tool, a.out)
    n += scaling_figure(scale, SIZE_RULES, SIZE_LABELS, "f_speedup_size.tex", tool, a.out,
                        metric="speedup")
    n += scaling_figure(scale, SHAPE_RULES, SHAPE_LABELS, "f_eff_shape.tex", tool, a.out)
    n += scaling_figure(scale, ARITY_RULES, ARITY_LABELS, "f_eff_arity.tex", tool, a.out)
    n += depth_figure(depth, DEPTH_RULES, DEPTH_LABELS,
                      "f_states_depth.tex", tool, a.out, "states")
    n += depth_figure(depth, DEPTH_RULES, DEPTH_LABELS,
                      "f_branchial_depth.tex", tool, a.out, "branchial_edges")
    n += relation_figure(depth, DEPTH_RULES, DEPTH_LABELS,
                         "f_relations.tex", tool, a.out)
    rows = shape_table(depth, a.out, tool)
    print("wrote %d curves across the figures, %d rows in t14_shape_space" % (n, rows))


if __name__ == "__main__":
    main()
