#!/usr/bin/env python3
"""The SHAPE of what a de-header would move, per header, from SOURCE_MAP.md.

WHY THIS AND NOT A LINE COUNT. "68% of the engine lives in headers" is true and decides
nothing: a template body cannot move at all, and a one-line accessor moved into a
translation unit costs its inlining and buys back one line of header. The question #20
actually turns on is the SIZE DISTRIBUTION of the movable bodies -- whether a header is
big because of a few large functions (move those, the rest stays) or because of hundreds
of tiny ones (moving them is a large diff that buys a small number and risks the hot
path).

Reads the report `source_map.py` already writes rather than re-parsing the tree: one
libclang pass costs about four minutes, and a second body asking the same question of the
same sources would be a second implementation of the rule.
"""
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, "SOURCE_MAP.md")

FILE_RE = re.compile(r"^## `([^`]+)` — \d+ definitions")
DEF_RE = re.compile(r"^- \*\*(.+?)\*\* \*\((\w[\w ]*), (\d+) lines(, pinned: (\w+))?\)\* `:(\d+)`")

BODY_KINDS = {"function", "method", "constructor", "destructor"}

# The buckets a decision is actually made in. A body under ~5 lines is an accessor: moving
# it is all cost. Over ~40 it is a routine whose call overhead is already noise.
BUCKETS = [(1, 2), (3, 5), (6, 15), (16, 40), (41, 10 ** 9)]


def bucket_label(lo, hi):
    return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"


def main():
    if not os.path.exists(REPORT):
        sys.exit(f"{REPORT} not found. Run tools/dev/source_map.py first.")

    want = sys.argv[1] if len(sys.argv) > 1 else "hypergraph/include"

    cur = None
    per_file = defaultdict(list)          # path -> [(size, name, line)]
    for raw in open(REPORT):
        m = FILE_RE.match(raw)
        if m:
            cur = m.group(1)
            continue
        if cur is None or not cur.startswith(want):
            continue
        d = DEF_RE.match(raw)
        if not d:
            continue
        name, kind, size, _, pin, line = d.groups()
        if kind not in BODY_KINDS or pin:
            continue
        per_file[cur].append((int(size), name, int(line)))

    if not per_file:
        sys.exit(f"no movable bodies found under {want!r} — check the path prefix")

    totals = defaultdict(lambda: [0, 0])
    print(f"Movable (non-template, non-device, non-constexpr) bodies under {want}\n")
    print(f"{'header':52s} {'defs':>5s} {'lines':>6s} {'median':>7s} {'max':>5s}")
    grand_defs = grand_lines = 0
    for path in sorted(per_file, key=lambda p: -sum(s for s, _, _ in per_file[p])):
        sizes = sorted(s for s, _, _ in per_file[path])
        n, tot = len(sizes), sum(sizes)
        grand_defs += n
        grand_lines += tot
        print(f"{path[-52:]:52s} {n:5d} {tot:6d} {sizes[n // 2]:7d} {sizes[-1]:5d}")
        for lo, hi in BUCKETS:
            k = bucket_label(lo, hi)
            for s in sizes:
                if lo <= s <= hi:
                    totals[k][0] += 1
                    totals[k][1] += s

    print(f"\n{'TOTAL':52s} {grand_defs:5d} {grand_lines:6d}")

    print("\nsize distribution of those bodies:")
    print(f"  {'lines':>8s} {'defs':>6s} {'lines':>7s}  {'share of movable lines':s}")
    for lo, hi in BUCKETS:
        k = bucket_label(lo, hi)
        n, tot = totals[k]
        share = (100.0 * tot / grand_lines) if grand_lines else 0.0
        bar = "#" * int(share / 2)
        print(f"  {k:>8s} {n:6d} {tot:7d}  {share:5.1f}% {bar}")

    print("\nlargest movable bodies:")
    flat = sorted((s, p, ln, nm) for p, v in per_file.items() for s, nm, ln in v)
    for s, p, ln, nm in flat[::-1][:15]:
        print(f"  {s:4d} lines  {p}:{ln}  {nm}")


if __name__ == "__main__":
    main()
