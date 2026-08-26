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
import collections
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



# SATURATED RUNS ARE NOT DATA ABOUT THE STATE SPACE, and they have to be removed before anything
# is plotted against depth.
#
# The engine catches a container ceiling, records a warning and returns a TRUNCATED graph rather
# than throwing (parallel_evolution.cpp, the CapacityExhausted case). sampling_cost_smoke did not
# read those warnings, so the sweep collected saturated runs as ordinary rows. Their signature is
# unmistakable once looked for: chain1a2 totals 2,097,149 states whether asked for depth 9, 10 or
# 11, and the width at depth 9 comes out 1,965,054 / 116,128 / 6,528 in those three runs, because
# the ceiling is redistributed over however many levels were requested. Plotted, that reads as a
# state space that stopped growing.
#
# The tool now emits truncated=1, so newly collected rows say so directly. Rows collected before
# that are classified here by the PLATEAU they form: within one shape, if two or more depths
# report essentially the same state count as the shape's maximum, every one of them is at the
# ceiling. A shape whose deepest run merely happens to be its largest forms no plateau and is
# kept -- which is why the test is "two or more at the maximum" and not "equal to the maximum".
#
# VALIDATED against the flag on 13 re-run rows: chain1a2 depths 5 and 6 (103,761 and 1,339,281
# states, still growing) report truncated=0, depths 7 through 14 (all 2,097,148-149) report
# truncated=1, and this rule agrees on every one.
def saturated_depths(rows):
    """Return {(rule, steps)} for rows sitting at a shape's capacity ceiling."""
    by_rule = collections.defaultdict(list)
    for r in rows:
        rule = r.get("rule")
        if rule:
            by_rule[rule].append(r)
    out = set()
    for rule, rs in by_rule.items():
        if any(r.get("truncated") == "1" for r in rs):
            out |= {(rule, int(num(r, "steps"))) for r in rs if r.get("truncated") == "1"}
            continue
        peak = max((num(r, "states") for r in rs), default=0.0)
        if peak <= 0:
            continue
        at_peak = [r for r in rs if num(r, "states") >= 0.999 * peak]
        if len(at_peak) >= 2:
            out |= {(rule, int(num(r, "steps"))) for r in at_peak}
    return out


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


def depth_figure(rows, rules, labels, fname, tool, out, ykey, sat=frozenset()):
    """A count against evolution depth, one curve per shape, for whatever count is asked for.

    Points at a shape's capacity ceiling are DROPPED, not drawn: a truncated run's counts say
    where the container filled up, not how large the state space is at that depth.
    """
    body = [pt.provenance(tool)]
    drawn = 0
    for rule in rules:
        pts = {}
        for r in rows:
            if r.get("rule") != rule:
                continue
            d = int(num(r, "steps", 0))
            if (rule, d) in sat:
                continue
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


def relation_figure(rows, rules, labels, fname, tool, out, sat=frozenset()):
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
            if (rule, int(num(r, "steps"))) in sat:
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


def shape_table(rows, out, tool, sat=frozenset()):
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
        if (rule, d) in sat:
            continue
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
    ap.add_argument("--measured-on", default="",
                    help="machine string for the provenance line; defaults to measured_on.txt "
                         "in the data directory. Required: a figure must not claim it was "
                         "measured where it was drawn.")
    a = ap.parse_args()

    pt.OUT = a.out

    # THE FRAGMENT MUST NAME THE MACHINE THAT MEASURED, NOT THE ONE THAT DREW. paper_tables
    # stamps pt.machine() -- the host running the generator -- and these rows are produced on a
    # remote box and plotted here. Left alone, every figure in this set claimed it was measured
    # on the developer's desktop. rich_sweep.sh writes measured_on.txt beside the data; --measured-on
    # overrides it for a data set collected before that file existed.
    measured = a.measured_on
    if not measured:
        mf = os.path.join(a.rich_dir, "measured_on.txt")
        if os.path.exists(mf):
            measured = open(mf).read().strip()
    if measured:
        pt._MEASURED_ON = measured
        print("measured on: %s" % measured)
    else:
        raise SystemExit("no measured_on.txt in %s and no --measured-on given: refusing to stamp "
                         "these fragments with this machine's name" % a.rich_dir)

    depth = parse(os.path.join(a.rich_dir, "rich_depth.txt"))
    scale = parse(os.path.join(a.rich_dir, "rich_scaling.txt"))
    print("depth rows %d, scaling rows %d" % (len(depth), len(scale)))

    tool = "tools/dev/rich_sweep.sh over tools/sampling_cost_smoke.cpp"
    sat = saturated_depths(depth)
    if sat:
        print("dropping %d saturated (shape, depth) points: %s"
              % (len(sat), ", ".join("%s@%d" % t for t in sorted(sat))))
    n = 0
    n += scaling_figure(scale, SIZE_RULES, SIZE_LABELS, "f_eff_size.tex", tool, a.out)
    n += scaling_figure(scale, SIZE_RULES, SIZE_LABELS, "f_speedup_size.tex", tool, a.out,
                        metric="speedup")
    n += scaling_figure(scale, SHAPE_RULES, SHAPE_LABELS, "f_eff_shape.tex", tool, a.out)
    n += scaling_figure(scale, ARITY_RULES, ARITY_LABELS, "f_eff_arity.tex", tool, a.out)
    n += depth_figure(depth, DEPTH_RULES, DEPTH_LABELS,
                      "f_states_depth.tex", tool, a.out, "states", sat)
    n += depth_figure(depth, DEPTH_RULES, DEPTH_LABELS,
                      "f_branchial_depth.tex", tool, a.out, "branchial_edges", sat)
    n += relation_figure(depth, DEPTH_RULES, DEPTH_LABELS,
                         "f_relations.tex", tool, a.out, sat)
    rows = shape_table(depth, a.out, tool, sat)
    print("wrote %d curves across the figures, %d rows in t14_shape_space" % (n, rows))


if __name__ == "__main__":
    main()
