#!/usr/bin/env python3
"""Every matcher candidate branch is exercised by the bench corpus. Checked, not assumed.

The join reaches candidates three ways and they are not interchangeable:

  0  arity_scan          an unbound seed edge with no repeated variable, scanned by arity
  1  repeated_var_seed   an unbound seed edge WITH a repeated variable ({{x,x}}), which cannot
                         be matched by scanning distinct positions and takes the signature
                         partition instead
  2  bound_intersect     a seed with bound variables, intersecting the inverted index

Branch 1 was reached by NOTHING in the generated corpus until the Repeated shape was added:
every CPU bench and every GPU sweep left it dark, and the only workload touching it anywhere was
one hand-written self-loop in the oracle corpus. A gate that says so mechanically is the point --
the previous state of this was an assumption.

Counters live behind HG_MATCH_BRANCH_STATS (pattern_matcher.hpp), which is off in every shipping
build, and bench_cpu_evolve prints them. This configures its own build directory so the gate
cannot report on a stale binary.

Usage: tools/dev/match_branch_coverage.py [--build DIR] [--depth N]
Exit 0 when every branch fired, 1 naming the branches that did not.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRANCHES = ["arity_scan", "repeated_var_seed", "bound_intersect"]

# One workload per branch the corpus is claimed to cover, plus a hand-picked one. Named
# explicitly rather than sweeping the whole corpus: a sweep that happens to cover a branch tells
# you less than a named workload that must.
WORKLOADS = ["wolfram24", "wpp", "rep-l1a2g1r1", "rep-l1a3g1r1", "disc-l2a2g1r1", "cycle4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(ROOT, "build_branch"))
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()

    bench = os.path.join(args.build, "bench_cpu_evolve")
    if not os.path.exists(bench):
        subprocess.run(["cmake", "-S", ROOT, "-B", args.build, "-DCMAKE_BUILD_TYPE=Release",
                        "-DHG_MATCH_BRANCH_STATS=ON"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run(["cmake", "--build", args.build, "--target", "bench_cpu_evolve", "-j4"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Each run reports its own process's counters, so the union over runs is the coverage.
    total = {b: 0 for b in BRANCHES}
    per_workload = {}
    for w in WORKLOADS:
        out = subprocess.run([bench, str(args.depth), "1", "1", w],
                             capture_output=True, text=True, timeout=900)
        m = re.search(r"\[matchbranch:[^\]]*\] " +
                      " ".join(r"%s=(\d+)" % b for b in BRANCHES), out.stdout + out.stderr)
        if not m:
            print("FAIL  %s produced no counters -- was the build configured with "
                  "HG_MATCH_BRANCH_STATS?" % w)
            return 1
        counts = [int(m.group(i + 1)) for i in range(len(BRANCHES))]
        per_workload[w] = counts
        for b, c in zip(BRANCHES, counts):
            total[b] += c

    width = max(len(w) for w in WORKLOADS)
    print("%-*s  %s" % (width, "workload", "  ".join(BRANCHES)))
    for w in WORKLOADS:
        print("%-*s  %s" % (width, w, "  ".join(
            "%*d" % (len(b), c) for b, c in zip(BRANCHES, per_workload[w]))))

    missing = [b for b in BRANCHES if total[b] == 0]
    if missing:
        print("\nFAIL  no corpus workload reaches: %s" % ", ".join(missing))
        print("      A branch nothing exercises is a branch no bench and no sweep can regress.")
        return 1
    print("\nPASS  every candidate branch is reached: %s" %
          ", ".join("%s=%d" % (b, total[b]) for b in BRANCHES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
