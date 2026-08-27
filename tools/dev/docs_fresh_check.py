#!/usr/bin/env python3
"""Fail if a documentation notebook is older than the markdown it is generated from.

WHY THIS EXISTS. The notebooks under paclet/Documentation/English are GENERATED from
paclet/Documentation/Source/*.md by tools/build_docs.wls, and they are committed, because that is
what the paclet ships. A commit that edits the markdown and does not rerun the generator leaves
the shipped documentation saying something the project no longer does -- and nothing notices,
because both files are present and both are valid.

It has happened twice. 12aac993 is titled "docs: the built notebooks match their markdown again",
and 033289e7 then changed HGEvolve.md without rebuilding, so the shipped page told users the
sampling options were CPU only for a day after they worked on both devices.

MTIME IS NOT THE INSTRUMENT. A fresh clone gives every file the same checkout time, so a
timestamp comparison passes on any machine that has just cloned and fails on any machine that has
just touched a file. The question is about COMMITS: was the source last changed in a commit newer
than the one that last changed its notebook? git answers that identically everywhere.

MAPPING SOURCE TO NOTEBOOK. The generator picks a target directory from each source's Template
frontmatter -- Symbol, Guide or TechNote -- and the notebook's basename is the document title
rather than the file's, so the mapping is not a path substitution. This reads the frontmatter for
the kind and then matches within that kind's directory: when a kind holds exactly one notebook,
that is the one; otherwise the source's own stem must appear in the notebook name. A source whose
notebook cannot be identified is REPORTED rather than skipped, because a silent skip is how a
check stops checking.

Usage:  tools/dev/docs_fresh_check.py
Exit:   0 clean, 1 stale or unmappable, 2 could not run git
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "paclet" / "Documentation" / "Source"
OUT_DIR = ROOT / "paclet" / "Documentation" / "English"

# Template frontmatter value -> directory the generator writes into.
KIND_DIR = {
    "Symbol": OUT_DIR / "ReferencePages" / "Symbols",
    "Guide": OUT_DIR / "Guides",
    "TechNote": OUT_DIR / "Tutorials",
}


def last_commit_epoch(path: Path):
    """Committer epoch of the last commit touching this path, or None if never committed."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%ct", "--", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("docs_fresh_check: git log failed: " + out.stderr.strip())
    s = out.stdout.strip()
    return int(s) if s else None


def template_of(md: Path):
    """The Template: value from the frontmatter, or None."""
    text = md.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^Template:\s*(\S+)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


def notebook_for(md: Path, kind: str):
    d = KIND_DIR.get(kind)
    if d is None or not d.is_dir():
        return None
    books = sorted(d.glob("*.nb"))
    if len(books) == 1:
        return books[0]
    stem = md.stem.lower()
    for b in books:
        if stem in b.stem.lower().replace(" ", ""):
            return b
    return None


def main():
    if not SRC_DIR.is_dir():
        print(f"docs_fresh_check: no source directory at {SRC_DIR}", file=sys.stderr)
        return 2

    findings = []
    checked = 0
    for md in sorted(SRC_DIR.glob("*.md")):
        kind = template_of(md)
        if kind is None:
            findings.append(f"{md.relative_to(ROOT)}: no Template frontmatter, so no notebook "
                            f"can be identified for it")
            continue
        nb = notebook_for(md, kind)
        if nb is None:
            findings.append(f"{md.relative_to(ROOT)}: Template {kind}, but no notebook in "
                            f"{KIND_DIR.get(kind)} matches it")
            continue

        src_t = last_commit_epoch(md)
        nb_t = last_commit_epoch(nb)
        checked += 1
        if src_t is None:
            continue                      # uncommitted source; nothing to compare against
        if nb_t is None:
            findings.append(f"{nb.relative_to(ROOT)}: never committed, but its source has been")
            continue
        if src_t > nb_t:
            findings.append(
                f"{nb.relative_to(ROOT)}: STALE -- {md.relative_to(ROOT)} was last changed in a "
                f"newer commit. Run ./build_docs.sh and commit the result.")

    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) over {checked} source/notebook pair(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
