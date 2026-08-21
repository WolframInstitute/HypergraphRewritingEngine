#!/usr/bin/env python3
"""The SHAPE of what a de-header would move, per header, from SOURCE_MAP.md.

WHY THIS AND NOT A LINE COUNT. "68% of the engine lives in headers" is true and decides
nothing: a template body cannot move at all, and a body that must stay for a stated reason is
not work outstanding. What this prints is the SIZE DISTRIBUTION of the movable bodies, so a
header's remaining cost can be read rather than guessed.

READS SOURCE_MAP.md, NOT THE TREE. One libclang pass costs about four minutes, and a second
body asking the same question of the same sources would be a second implementation of the
rule. THE CONSEQUENCE IS THAT THIS REPORT IS AS OLD AS THAT FILE: after a de-header pass its
counts describe the tree as it was when SOURCE_MAP.md was generated, not as it is. Regenerate
the map before trusting a number here, and check header line counts against git as the
independent reading.

THE STATE OF THE WORK. The project is de-headered: every non-template body in the host tree
and in the GPU port has moved to a translation unit except where a stated reason keeps it in
place, and each of those reasons is recorded at the definition itself --

  hypergraph.hpp    edge_accessor returns a lambda's closure type, deduced from the body
  bitset.hpp        contains and find_chunk are HG_INLINE with the measurement that pins them
  arena.hpp         ArenaWorkerRegistry is compiled by a GenMC harness that links no library
  signature.hpp     a defaulted default constructor kept in-class stays trivial
  cuda_check.hpp    cuda_check_at is one comparison at every CUDA call in the port
  portable_intrinsics.hpp  every body is a single instruction, and some are device code

WHAT MOVING A BODY COSTS, MEASURED. An earlier experiment moved hypergraph.hpp's small member
bodies and measured +0.21% (callgrind, two-edge rule, depth 4, one thread), and that number
was the reason the accessors stayed. IT NO LONGER HOLDS, because the project now builds with
link-time optimisation (CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE), which restores the
cross-translation-unit inlining that moving a body out of a header removes. Re-measured across
the whole de-header, one header at a time, on
`sampling_cost_smoke off wolfram24 2 5 1 4 full`:

  baseline before any move   36,846,454 instructions
  host side complete         36,331,780 instructions   (-1.40%)

with no single header costing more than +0.26%, and that one attributed by callgrind_annotate
to LTO re-deciding hgcommon::ir_refine rather than to any moved body. The instrument is
deterministic to 36 instructions on a fixed binary, so those deltas are codegen and not noise.
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
