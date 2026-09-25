#!/usr/bin/env python3
"""Markdown lint for the documentation sources and the tracked top-level documents.

Outside fenced ``` blocks it checks that
  - every inline code span closes on the line it opens, with a run of backticks of the same
    length. The converter follows CommonMark, where a span may run across a line break, so an
    unclosed span silently takes the text after it into code; a Wolfram context such as
    `HypergraphRewriting`` ends in a backtick and closes a single-backtick span early, and has to
    be written in a double-backtick span;
  - every paragraph (and every list item) has an even number of $ outside code spans and
    <code>...</code> spans, since each $...$ formula the converter typesets must close.

Files: the tracked docs/en/**/*.md, docs/research/**/*.md, docs/registry/**/*.md and *.md at the
repository root.

Usage:  tools/dev/lint_docs.py [--selftest] [file ...]
Exit:   0 clean, 1 findings (or a failed self-test)
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = ["docs/en", "docs/research", "docs/registry"]
FENCE = re.compile(r"^\s*(```|~~~)")
HTML_CODE = re.compile(r"<code>.*?</code>")
LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s")


def files():
    """The tracked markdown files in scope; untracked notes beside them are not documents."""
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.md"], capture_output=True,
                            text=True, check=True).stdout.split()
    return [ROOT / p for p in sorted(listed)
            if "/" not in p or any(p.startswith(t + "/") for t in TREES)]


def strip_spans(line):
    """The line with its code spans removed, and the column of an unclosed span or None."""
    out, i = [], 0
    while i < len(line):
        if line[i] == "`":
            j = i
            while j < len(line) and line[j] == "`":
                j += 1
            run = line[i:j]
            k = line.find(run, j)
            # A closing run must be exactly as long as the opening one.
            while k != -1 and (k + len(run) < len(line) and line[k + len(run)] == "`"
                               or k > 0 and line[k - 1] == "`"):
                k = line.find(run, k + len(run))
            if k == -1:
                return "".join(out), i
            i = k + len(run)
            continue
        out.append(line[i])
        i += 1
    return "".join(out), None


def lint_text(text, name):
    findings = []
    inside = False
    para, para_start = [], 0
    in_front = False

    def end_paragraph():
        body = "".join(para)
        dollars = len(re.findall(r"(?<!\\)\$", body))
        if dollars % 2:
            findings.append(f"{name}:{para_start}: odd number of $ in the paragraph")

    lines = text.split("\n")
    for n, line in enumerate(lines, 1):
        if n == 1 and line.strip() == "---":
            in_front = True
            continue
        if in_front:
            if line.strip() == "---":
                in_front = False
            continue
        if FENCE.match(line):
            if para:
                end_paragraph()
                para = []
            inside = not inside
            continue
        if inside:
            continue
        if not line.strip() or line.lstrip().startswith("<!--"):
            if para:
                end_paragraph()
                para = []
            continue
        if LIST_ITEM.match(line) and para:
            end_paragraph()
            para = []
        plain, unclosed = strip_spans(HTML_CODE.sub("", line))
        if unclosed is not None:
            findings.append(f"{name}:{n}: a code span opened at column {unclosed + 1} does not "
                            f"close on this line")
        if not para:
            para_start = n
        para.append(plain + "\n")
    if para:
        end_paragraph()
    return findings


def selftest():
    bad = lint_text("A span `open and\nnever closed.\n\nA $x formula.\n", "fixture")
    good = lint_text("`HGEvolve` and ``HypergraphRewriting` `` and $x$.\n\n```wl\n`a\n```\n"
                     "\n- gives <code>[$Failed]()</code>.\n- a $y$ item.\n", "fixture")
    ok = len(bad) == 2 and good == []
    print("self-test:", "PASS" if ok else f"FAIL (bad {bad}, good {good})")
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    targets = [Path(a).resolve() for a in argv] or files()
    findings = []
    for f in targets:
        findings += lint_text(f.read_text(encoding="utf-8", errors="replace"),
                              str(f.relative_to(ROOT)))
    for x in findings:
        print(x)
    print(f"{len(findings)} finding(s) over {len(targets)} file(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
