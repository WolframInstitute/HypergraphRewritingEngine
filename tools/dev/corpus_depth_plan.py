#!/usr/bin/env python3
"""Choose a benchmark depth per corpus workload from a corpusgrow run.

A THREAD-SCALING RATIO IS ONLY ABOUT THE ENGINE IF THE RUN IS LONGER THAN THE ENGINE'S PER-CALL
FLOOR. The generated corpus spans four orders of magnitude in work at a fixed depth, so one depth
for all of it measures the large workloads and measures process startup on the small ones. This
reads `bench_cpu_evolve corpusgrow` output -- raw state counts at depths two and three -- and picks
the shallowest depth expected to reach TARGET raw states, from each workload's own growth rate.

The estimate does not have to be right. It sets the depth; the sweep then MEASURES the work that
depth produced and prints it, so a workload that came in under target is visible in its own row
rather than hidden in an average.

Usage: corpus_depth_plan.py <corpusgrow.txt> [target] [max_depth]
"""
import math
import sys

TARGET_DEFAULT = 5000
MAX_DEPTH_DEFAULT = 7


def plan(path, target, max_depth):
    """Pick, per workload, the depth whose work lands closest to `target` raw states.

    CLOSEST, NOT "FIRST AT OR ABOVE". The corpus spans four orders of magnitude and its growth
    rates differ by workload, so "shallowest depth reaching target" lands some workloads a factor
    of twenty past it -- star-l3a2g2r2 goes 2821 raw states at depth two to 381301 at depth three,
    and the deeper run costs minutes to say what the shallower one says in seconds. Depth two is
    a candidate for exactly that reason. The distance is measured in log space because the
    quantity is multiplicative: half the target and twice the target are equally wrong.
    """
    rows = []
    for line in open(path):
        f = line.split()
        # "<name> d2=<canon>/<raw> d3=<canon>/<raw> <verdict>"
        if len(f) < 6 or not f[1].startswith("d2=") or not f[2].startswith("/"):
            continue
        name = f[0]
        try:
            raw2 = int(f[2][1:])
            raw3 = int(f[4][1:])
        except ValueError:
            continue
        if raw3 <= raw2 or raw2 < 1:
            # Not growing: no depth makes it a scaling measurement. Reported, not silently kept.
            rows.append((name, 3, raw3, "DEAD"))
            continue
        ratio = raw3 / raw2
        best, best_dist = 2, abs(math.log(raw2 / target))
        for d in range(3, max_depth + 1):
            est = raw3 * (ratio ** (d - 3))
            dist = abs(math.log(est / target))
            if dist < best_dist:
                best, best_dist = d, dist
            if est > target:
                break
        rows.append((name, best, raw3, "ok"))
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else TARGET_DEFAULT
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_DEPTH_DEFAULT
    rows = plan(path, target, max_depth)
    dead = [r for r in rows if r[3] == "DEAD"]
    for name, depth, _raw3, status in rows:
        if status == "DEAD":
            continue
        print(f"{name} {depth}")
    if dead:
        print(f"# {len(dead)} workload(s) not growing, excluded: "
              f"{' '.join(r[0] for r in dead)}", file=sys.stderr)
    print(f"# planned {len(rows) - len(dead)} workloads, target {target} raw states, "
          f"max depth {max_depth}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
