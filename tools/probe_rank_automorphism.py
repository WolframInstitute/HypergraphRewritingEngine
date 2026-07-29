#!/usr/bin/env python3
"""Swap the block-count invariance test's initial state between a graph with a nontrivial
automorphism group (directed 4-cycle, rotation group of order 4) and a rigid one (directed
path, trivial automorphism group).

The question it settles: are the produced-edge ranks schedule-dependent in general, or only on
states whose canonical labeling is a coset rather than a single labeling? Run with `cycle` and
with `path` and compare which key components survive the 3-blocks vs 17-blocks comparison.
"""
import sys, pathlib

TEST = pathlib.Path(__file__).resolve().parents[1] / "gpu/tests/test_rewrite.cu"
MARKER = "AutomaticEventIdentityIsTheSameAtEveryBlockCount"
CYCLE = "{{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}}"
PATH  = "{{0u, 1u}, {1u, 2u}, {2u, 3u}}"

want = sys.argv[1] if len(sys.argv) > 1 else "path"
new, old = (PATH, CYCLE) if want == "path" else (CYCLE, PATH)

lines = TEST.read_text().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if MARKER in l and l.startswith("TEST("))
for i in range(start, start + 15):
    if "const std::vector<std::vector<VertexId>> init =" in lines[i]:
        if old not in lines[i]:
            print(f"already {want}")
            sys.exit(0)
        lines[i] = lines[i].replace(old, new)
        TEST.write_text("".join(lines))
        print(f"set to {want}: {lines[i].strip()}")
        sys.exit(0)
print("init line not found", file=sys.stderr)
sys.exit(1)
