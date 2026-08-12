#!/usr/bin/env python3
"""What a header change costs: which CUDA translation units it rebuilds, and how expensive they are.

WHY THIS EXISTS. `gpu/src/persistent.cu` needs about 12.5 GB and half an hour to compile at -j1
(board #123), and this box builds CUDA one TU at a time so nothing else starves. So the cost of
editing a header under `gpu/` or `common/` is not "a rebuild" -- it is the SUM of the expensive
TUs that include it, and that number is knowable before the edit rather than after the wait.
`persistent.cu` was rebuilt four times in one day because the pattern was edit-build-discover-edit.

It resolves includes TRANSITIVELY, which is the whole point: `hgcommon/core.hpp` appears in
almost no `.cu` file directly and reaches every one of them through two or three hops. A direct
grep answers a question nobody asked.

WHAT THE BUCKETS MEAN. A TU's cost is read from `tools/dev/cuda_tu_cost.tsv`, which records
measured peak RSS and wall time per `.cu` file. A TU with no row is reported as UNMEASURED and
counted at the default, so a new `.cu` file shows up as an unknown rather than as free.

Usage:
    python3 tools/dev/include_cost.py gpu/include/hg_gpu/evolve.hpp [more files...]
    python3 tools/dev/include_cost.py --all        # every header, ranked by what it costs

Exit code is 0 unless a named file does not exist.
"""

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COSTS = os.path.join(REPO, "tools", "dev", "cuda_tu_cost.tsv")

# Where an #include may resolve from. These are the -I roots the CUDA targets are built with;
# a path that resolves under none of them is a system header and is not followed.
INCLUDE_ROOTS = [
    os.path.join(REPO, "gpu", "include"),
    os.path.join(REPO, "common", "include"),
    os.path.join(REPO, "hypergraph", "include"),
    os.path.join(REPO, "job_system", "include"),
    os.path.join(REPO, "lockfree_deque", "include"),
    os.path.join(REPO, "wxf"),
    os.path.join(REPO, "paclet_source"),
    REPO,
]

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)

# A TU with no measured row costs this. Deliberately not zero: an unmeasured TU is an unknown,
# and reporting an unknown as free is how a 30-minute build gets started by accident.
DEFAULT_MB, DEFAULT_MIN = 2000, 5.0


def read_costs():
    costs = {}
    if not os.path.exists(COSTS):
        return costs
    with open(COSTS) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            costs[parts[0]] = (float(parts[1]), float(parts[2]))
    return costs


def resolve(inc, from_file):
    """An include as written, resolved against the roots the CUDA build uses. Relative-to-the-
    including-file is tried FIRST, which is what the compiler does for a quoted include."""
    cand = os.path.normpath(os.path.join(os.path.dirname(from_file), inc))
    if os.path.exists(cand):
        return cand
    for root in INCLUDE_ROOTS:
        cand = os.path.normpath(os.path.join(root, inc))
        if os.path.exists(cand):
            return cand
    return None


def direct_includes(path, cache):
    if path in cache:
        return cache[path]
    out = []
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        cache[path] = out
        return out
    for inc in INCLUDE_RE.findall(text):
        r = resolve(inc, path)
        if r:
            out.append(r)
    cache[path] = out
    return out


def reaches(tu, target, cache, seen=None):
    """Does `tu` include `target`, transitively? Cycles are impossible to loop on because a
    file already on the stack is not re-entered -- which also makes `#pragma once` irrelevant
    here, since we are asking about the include GRAPH and not about expansion."""
    if seen is None:
        seen = set()
    if tu in seen:
        return False
    seen.add(tu)
    if os.path.samefile(tu, target) if os.path.exists(tu) else False:
        return True
    for inc in direct_includes(tu, cache):
        if os.path.exists(inc) and os.path.exists(target) and os.path.samefile(inc, target):
            return True
        if reaches(inc, target, cache, seen):
            return True
    return False


def cuda_tus():
    tus = []
    for base in ("gpu/src", "gpu/tests"):
        d = os.path.join(REPO, base)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".cu"):
                tus.append(os.path.join(d, name))
    return tus


def report(target, tus, costs, cache):
    rel = os.path.relpath(target, REPO)
    hit = [tu for tu in tus if reaches(tu, target, cache)]
    total_min, peak_mb, unmeasured = 0.0, 0.0, []
    for tu in hit:
        key = os.path.relpath(tu, REPO)
        if key in costs:
            mb, mins = costs[key]
        else:
            mb, mins = DEFAULT_MB, DEFAULT_MIN
            unmeasured.append(key)
        total_min += mins
        peak_mb = max(peak_mb, mb)
    bucket = ("CHEAP" if total_min < 5 else "MODERATE" if total_min < 20 else "EXPENSIVE")
    print(f"{rel}")
    print(f"  {len(hit)} CUDA TU(s), ~{total_min:.0f} min at -j1, peak ~{peak_mb:.0f} MB  [{bucket}]")
    for tu in hit:
        key = os.path.relpath(tu, REPO)
        mb, mins = costs.get(key, (DEFAULT_MB, DEFAULT_MIN))
        mark = "  (UNMEASURED)" if key in unmeasured else ""
        print(f"    {mins:6.1f} min  {mb:7.0f} MB  {key}{mark}")
    if not hit:
        print("    (no CUDA translation unit includes it: a change here needs no device build)")
    return total_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="rank every header under gpu/ and common/ by what changing it costs")
    ap.add_argument("--check", action="store_true",
                    help="the cost table names only translation units that exist, and every "
                         "device TU has a row. A stale row makes this tool quote a cost for a "
                         "file that is gone; a missing one makes a new TU read as the default "
                         "rather than as measured, and both are silent.")
    args = ap.parse_args()

    costs = read_costs()
    tus = cuda_tus()
    cache = {}

    if args.check:
        # A STALE ROW IS A FINDING; AN ABSENT ONE IS NOT. A row naming a .cu file that no longer
        # exists makes this tool quote a cost for something that is gone, silently. A TU with no
        # row is reported as UNMEASURED at a visible default every time it appears, which says so
        # on its own -- and requiring a row for every TU would push someone to invent one, which
        # is the failure this check found on its first run.
        have = {os.path.relpath(tu, REPO) for tu in tus if "/src/" in tu}
        named = set(costs)
        stale = sorted(named - have)
        for k in stale:
            print(f"  STALE  cost table names {k}, which is not a device translation unit")
        print(f"{len(have)} device TU(s), {len(named & have)} measured, "
              f"{len(have - named)} unmeasured, {len(stale)} finding(s)")
        return 1 if stale else 0

    if args.all:
        headers = []
        for base in ("gpu/include", "common/include"):
            for root, _, names in os.walk(os.path.join(REPO, base)):
                for n in names:
                    if n.endswith((".hpp", ".cuh", ".h")):
                        headers.append(os.path.join(root, n))
        ranked = sorted(((report(h, tus, costs, cache), h) for h in sorted(headers)),
                        reverse=True)
        print("\nranked by cost:")
        for mins, h in ranked:
            print(f"  {mins:6.1f} min  {os.path.relpath(h, REPO)}")
        return 0

    if not args.files:
        ap.print_help()
        return 0

    rc = 0
    for f in args.files:
        p = f if os.path.isabs(f) else os.path.join(REPO, f)
        if not os.path.exists(p):
            print(f"FAIL: {f} does not exist")
            rc = 1
            continue
        report(p, tus, costs, cache)
    return rc


if __name__ == "__main__":
    sys.exit(main())
