#!/usr/bin/env python3
"""Per-page column fill of a two-column PDF, from `pdftotext -layout` output.

A page whose two columns differ greatly in filled lines is where LaTeX cut a column short
(a pending double float plus a following heading is the usual cause). `-layout` renders both
columns on one text line, so each line is split at the gutter position and a column counts
as filled on that line when its half holds text. Pages whose smaller column holds under 60%
of the larger one are printed; the last page is exempt.

Usage: python3 tools/dev/paper_column_fill.py paper/main.pdf [gutter_col=58]
"""
import subprocess
import sys


def main():
    pdf = sys.argv[1]
    gutter = int(sys.argv[2]) if len(sys.argv) > 2 else 58
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True,
                          text=True, check=True).stdout
    pages = text.split("\f")
    flagged = 0
    for i, page in enumerate(pages, 1):
        lines = [l for l in page.splitlines() if l.strip()]
        if len(lines) < 10:
            continue
        left = sum(1 for l in lines if l[:gutter].strip())
        right = sum(1 for l in lines if l[gutter:].strip())
        lo, hi = min(left, right), max(left, right)
        if hi and lo / hi < 0.6 and i < len(pages) - 1:
            flagged += 1
            print(f"page {i}: left={left} right={right}")
    print(f"{flagged} page(s) flagged over {len(pages) - 1}")


if __name__ == "__main__":
    main()
