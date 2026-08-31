#!/usr/bin/env python3
"""Emit the resident-kernel occupancy table from Nsight Compute reports, one per workload.

The table answers whether any hardware unit is near its limit on any corpus shape, and what
the lane average is: achieved occupancy, active threads per warp, SM / memory / DRAM
throughput, and the profiled duration of k_persistent_evolve, per workload. Nsight Compute
reports are binary, so this reads each one back through `ncu --import --page details` and
takes the mean over the profiled launches of the persistent kernel; the profiler serialises
and fixes clocks, so the duration column is a profiler figure, not a wall-clock one, and the
caption says so.

Usage: ncu_occupancy_table.py <dir-with-ncu_*_<workload>.ncu-rep> [--out paper/tables]
       [--measured-on M] [--ncu /path/to/ncu]
"""

import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt   # noqa: E402

METRICS = [
    ("Duration", "dur"),
    ("Compute (SM) Throughput", "sm"),
    ("Memory Throughput", "mem"),
    ("DRAM Throughput", "dram"),
    ("Avg. Active Threads Per Warp", "lanes"),
    ("Achieved Occupancy", "occ"),
]


def read_report(ncu, path):
    out = subprocess.run([ncu, "--import", path, "--page", "details"],
                         capture_output=True, text=True).stdout
    # One block per profiled launch; keep the persistent kernel's.
    blocks = re.split(r"\n(?=  \S)", out)
    rows = []
    for b in blocks:
        head = b.splitlines()[0] if b.splitlines() else ""
        if "k_persistent_evolve" not in head:
            continue
        vals = {}
        for label, key in METRICS:
            m = re.search(re.escape(label) + r"\s+(\S*)\s+([\d.]+)", b)
            if m:
                unit, v = m.group(1), float(m.group(2))
                if key == "dur":
                    v = v * (1000.0 if unit == "ms" else 1.0)   # microseconds
                vals[key] = v
        if len(vals) == len(METRICS):
            rows.append(vals)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--measured-on", default="")
    ap.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    a = ap.parse_args()
    pt.OUT = a.out
    if a.measured_on:
        pt._MEASURED_ON = a.measured_on

    table = []
    for path in sorted(glob.glob(os.path.join(a.dir, "ncu_*.ncu-rep"))):
        name = os.path.basename(path)[len("ncu_"):-len(".ncu-rep")]
        depth, workload = name.split("_", 1) if name.startswith("d") else ("", name)
        rows = read_report(a.ncu, path)
        if not rows:
            raise SystemExit("no k_persistent_evolve launch in %s" % path)
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        table.append((workload, depth.lstrip("d"), len(rows), mean))
    if not table:
        raise SystemExit("no ncu_*.ncu-rep under %s" % a.dir)
    table.sort(key=lambda t: -t[3]["dur"])

    # Launches profiled per report: a capture cut by its time cap carries fewer than the run's
    # launches, and the column says so rather than presenting a partial mean as a full one.
    b = [pt.provenance("Nsight Compute over tools/bench_gpu_evolve.cpp (mode 2)"),
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Workload & Depth & Launches & Duration (\textmu s) & SM \% & Mem.\ \% & DRAM \% & "
         r"Lanes/warp & Occ.\ \% \\",
         r"\midrule"]
    for workload, depth, n, m in table:
        b.append(r"\texttt{%s} & %s & %d & %.1f & %.2f & %.2f & %.2f & %.2f & %.2f \\" % (
            pt.tex_escape(workload), depth, n, m["dur"], m["sm"], m["mem"], m["dram"],
            m["lanes"], m["occ"]))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t_device_occupancy.tex", "\n".join(b) + "\n")
    for workload, depth, n, m in table:
        print("  %-12s d%s  %5d launch(es)  %8.1f us  lanes %.2f  occ %.2f%%  sm %.2f%%"
              % (workload, depth, n, m["dur"], m["lanes"], m["occ"], m["sm"]))


if __name__ == "__main__":
    main()
