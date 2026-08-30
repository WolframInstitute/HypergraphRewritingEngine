#!/usr/bin/env python3
"""Emit the host instruction profile table from a callgrind_annotate output.

WHY THIS EXISTS. The paper made three forward references to an instruction profile in Section 8
-- "23% of executed instructions to the join's expansion step", "the majority of instructions",
"approximately half" -- and Section 8 contained no such profile. Two of the three were also wrong.
A claim about where the instructions go needs the instrument that counts them.

WHAT IT READS. `callgrind_annotate` output, which the attrib phase of tools/dev/remote_session.sh
already produces as cg_wpp.txt from a single-threaded depth-6 run under callgrind. Instruction
counts are deterministic, so this needs no quiet machine and no repeats -- which is the whole
reason to attribute with callgrind rather than with wall time.

THE FLAT PROFILE ONLY. callgrind_annotate prints a flat per-function block and then per-file
source annotations for the same functions. Summing both double-counts: a first attempt at this
script attributed 886% of the program total. The flat block ends at the first separator line
after its header, and that is where parsing stops.

Grouping is by the file a function is defined in, which is what makes the buckets mean something
in this codebase: canonicalization is ir_core.hpp, the join is the expansion task, dedup is the
two concurrent containers. Anything unmatched is reported as Other rather than silently dropped,
so the buckets and the residue always sum to the covered fraction.

Usage: instruction_profile.py <callgrind_annotate-output> [--out paper/tables] [--measured-on M]
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt   # noqa: E402

# Ordered: the first bucket whose key appears in the entry wins, so more specific keys go first.
BUCKETS = [
    ("Match expansion and join",      ("submit_expand_task", "expand_task", "bitset.hpp")),
    ("Canonicalization (IR)",         ("ir_core.hpp", "ir_refine", "ir_canonical", "irpartition")),
    ("Dedup sets and maps",           ("concurrent_key_set", "concurrent_map")),
    ("Arena allocation",              ("arena", "operator new")),
    ("Hypergraph storage",            ("hypergraph.hpp", "hypergraph.cpp")),
    ("Causal and branchial",          ("causal_graph", "branchial")),
    ("Job system and work stealing",  ("job_system", "work_stealing", "deque")),
]


def parse(path):
    lines = open(path).read().split("\n")
    total = None
    for line in lines:
        m = re.match(r"^\s*([\d,]+) \(100\.0%\)\s+PROGRAM TOTALS", line)
        if m:
            total = int(m.group(1).replace(",", ""))
            break
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Ir") and "file:function" in line:
            start = i + 2
            break
    if total is None or start is None:
        raise SystemExit("%s is not callgrind_annotate output" % path)
    end = start
    while end < len(lines) and not lines[end].startswith("----"):
        end += 1

    buckets, covered = collections.Counter(), 0
    for line in lines[start:end]:
        m = re.match(r"^\s*([\d,]+) \(\s*[\d.]+%\)\s+(.*)$", line)
        if not m:
            continue
        n, what = int(m.group(1).replace(",", "")), m.group(2).lower()
        covered += n
        for name, keys in BUCKETS:
            if any(k in what for k in keys):
                buckets[name] += n
                break
        else:
            buckets["Other"] += n
    return total, covered, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("callgrind")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--measured-on", default="")
    a = ap.parse_args()

    pt.OUT = a.out
    if a.measured_on:
        pt._MEASURED_ON = a.measured_on

    total, covered, buckets = parse(a.callgrind)
    b = [pt.provenance("tools/dev/remote_session.sh attrib phase, callgrind over "
                       "tools/bench_cpu_evolve"),
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Component & Instructions & Share \\", r"\midrule"]
    for name, n in buckets.most_common():
        if name == "Other":
            continue
        b.append(r"%s & %s & %.1f\%% \\" % (name, "{:,}".format(n), 100.0 * n / total))
    b.append(r"\midrule")
    b.append(r"Unattributed & %s & %.1f\%% \\"
             % ("{:,}".format(buckets["Other"]), 100.0 * buckets["Other"] / total))
    b.append(r"\midrule")
    b.append(r"\textbf{Total} & %s & %.1f\%% \\"
             % ("{:,}".format(total), 100.0 * covered / total))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t15_instruction_profile.tex", "\n".join(b) + "\n")
    # The coverage figure the caption cites, from the same parse, so the sentence cannot
    # drift from the table it describes.
    pt.value("InstrFlatCoverage", "%.1f" % (100.0 * covered / total))
    pt.write_values("values_instr.tex")
    print("total %s Ir, flat profile covers %.1f%%" % ("{:,}".format(total), 100.0 * covered / total))
    for name, n in buckets.most_common():
        print("  %-32s %5.2f%%" % (name, 100.0 * n / total))


if __name__ == "__main__":
    main()
