#!/usr/bin/env python3
"""More threads must mean faster: a gate over an every-thread-count sweep, per corpus case.

Reads bench_cpu_evolve sweep output (one `threads=N ... median_ms=M` line per thread count)
from one or more log files and fails when a row's median wall time RISES between two
consecutive thread counts by more than the tolerance. The tolerance exists for measurement
noise on sub-ten-millisecond rows, not for regressions: on this box interleaved medians swing
a few percent, so a rise above the tolerance is a defect, and a rise below it on a row that
takes seconds is reported too.

Usage: mono_check.py <mono_*.log ...> [--tolerance 0.05]
Exit 0 when every row is monotone within tolerance; 1 otherwise, listing every violation.
"""

import argparse
import os
import re
import sys

ROW = re.compile(r"threads=(\d+) steps=(\d+) canonical=(\d+) raw=(\d+) median_ms=([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="fractional rise tolerated between consecutive counts (noise band)")
    a = ap.parse_args()

    violations = []
    checked = 0
    for path in a.logs:
        rows = []
        with open(path) as f:
            for line in f:
                m = ROW.search(line)
                if m:
                    rows.append((int(m.group(1)), float(m.group(5)), m.group(3), m.group(4)))
        if not rows:
            violations.append("%s: no sweep rows" % path)
            continue
        rows.sort()
        counts = {(r[2], r[3]) for r in rows}
        if len(counts) != 1:
            violations.append("%s: output differs across thread counts %s -- the determinism "
                              "contract, not monotonicity, is what failed" % (path, sorted(counts)))
        name = os.path.basename(path)
        for (t0, m0, _, _), (t1, m1, _, _) in zip(rows, rows[1:]):
            checked += 1
            rise = (m1 - m0) / m0 if m0 else 0.0
            if rise > a.tolerance:
                violations.append("%s: %d -> %d threads: %.3f -> %.3f ms (+%.1f%%)"
                                  % (name, t0, t1, m0, m1, 100 * rise))
        best = min(rows, key=lambda r: r[1])
        print("%-28s %2d counts  1t %.1f ms  best %.1f ms at %d threads  last %.1f ms at %d"
              % (name, len(rows), rows[0][1], best[1], best[0], rows[-1][1], rows[-1][0]))

    for v in violations:
        print("VIOLATION  " + v)
    print("%d transitions checked, %d violation(s)" % (checked, len(violations)))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
