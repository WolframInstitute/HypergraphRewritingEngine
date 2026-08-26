#!/usr/bin/env python3
"""Emit the persistent-evolver depth table from a bench_gpu_evolve sweep.

WHY THIS IS SEPARATE FROM T7. T7 compares the device against the CPU, so every row needs a CPU
arm, and that arm is what makes deep rows unaffordable: at depth eight the CPU side is 262,144
states and a fifteen-iteration thread sweep there had not finished after forty minutes. The
question this table answers is GPU-against-GPU -- a resident kernel against a conventional
per-step evolve() -- so it needs no CPU arm and reaches depth nine in minutes.

WHAT IT SHOWS, and why the range matters more than the ratio. The persistent evolver's advantage
is the elimination of a fixed per-call cost, so it is largest where the evolution is smallest and
disappears once real compute dominates that constant. A speedup quoted without the depth it was
measured at is therefore a statement about small problems. The measured range runs from 16.5x at
depth four to parity at depth eight, and the evolve() column shows the constant directly: it is
about 50 ms at depths four, five and six while the state count grows from 53 to 3,867.

Reads the `steps=... | evolve()_median_ms=... | PersistentEvolver_median_ms=...` lines that
bench_gpu_evolve prints in mode 0, one per depth.

Usage: persistent_depth_table.py <bench-gpu-evolve-output> [--out paper/tables] [--measured-on M]
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt   # noqa: E402

ROW = re.compile(r"steps=(\d+) states=(\d+) events=(\d+) \| evolve\(\)_median_ms=([\d.]+) \| "
                 r"PersistentEvolver_median_ms=([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--measured-on", default="")
    a = ap.parse_args()

    pt.OUT = a.out
    if a.measured_on:
        pt._MEASURED_ON = a.measured_on

    rows = []
    for line in open(a.sweep):
        m = ROW.search(line)
        if m:
            d, states, events, call, persist = m.groups()
            rows.append((int(d), int(states), int(events), float(call), float(persist)))
    if not rows:
        raise SystemExit("no bench_gpu_evolve rows in %s" % a.sweep)
    rows.sort()

    b = [pt.provenance("tools/bench_gpu_evolve.cpp (mode 0)"),
         r"\begin{tabular}{rrrrr}", r"\toprule",
         r"Depth & States & \texttt{evolve()} ms & Persistent ms & Speedup \\", r"\midrule"]
    for d, states, _events, call, persist in rows:
        b.append(r"%d & %s & %.1f & %.1f & $%.2f\times$ \\"
                 % (d, "{:,}".format(states), call, persist, call / persist))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t16_persistent_depth.tex", "\n".join(b) + "\n")
    for d, states, _e, call, persist in rows:
        print("  depth %d  %9s states  %8.1f -> %8.1f ms  %5.2fx"
              % (d, "{:,}".format(states), call, persist, call / persist))


if __name__ == "__main__":
    main()
