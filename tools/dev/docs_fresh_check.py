#!/usr/bin/env python3
"""Fail if a documentation notebook is older than the markdown it is generated from.

WHY THIS EXISTS. The notebooks under paclet/Documentation/English are GENERATED from
docs/en/**/*.md by tools/build_docs.wls, and they are committed, because that is
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
frontmatter -- Symbol, Guide or TechNote -- and names the notebook after the page's Name, which
must equal the source's file name. A source whose Name differs, or whose notebook is missing, is
REPORTED rather than skipped, because a silent skip is how a check stops checking; so is a
notebook no source maps to.

A SHALLOW CLONE is refused (exit 2): its one grafted commit touches every path, so every source
and notebook compare equal and the check would pass without comparing anything.

AN UNEVALUATED NOTEBOOK IS REPORTED. A notebook whose source has a ```wl fence must carry at
least one Output, Print or Message cell. A notebook built without evaluation has the right path
and a newer commit than its source, so the commit comparison alone passes it.

Usage:  tools/dev/docs_fresh_check.py
Exit:   0 clean, 1 stale, unmappable or unevaluated, 2 could not run git or shallow clone
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "docs" / "en"
OUT_DIR = ROOT / "paclet" / "Documentation" / "English"

# Template frontmatter value -> directory the generator writes into.
KIND_DIR = {
    "Symbol": OUT_DIR / "ReferencePages" / "Symbols",
    "Guide": OUT_DIR / "Guides",
    "TechNote": OUT_DIR / "Tutorials",
}


# An evaluating fence in a source page.
WL_FENCE_RE = re.compile(r"^```[ \t]*(?:wl|wolfram|mathematica)\b", re.MULTILINE | re.IGNORECASE)
# Cell styles only an evaluation produces. Inside a notebook string a quote is escaped (\"), so an
# unescaped "Output" is a cell style, not text.
EVALUATED_CELL_RE = re.compile(r'(?<!\\)"(?:Output|Print|Message)"')


def shallow_clone():
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"],
                         capture_output=True, text=True)
    return out.returncode != 0 or out.stdout.strip() != "false"


def last_commit_epoch(path: Path):
    """Committer epoch of the last commit touching this path, or None if never committed."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%ct", "--", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("docs_fresh_check: git log failed: " + out.stderr.strip())
    s = out.stdout.strip()
    return int(s) if s else None


def frontmatter(md: Path, key: str):
    """The value of `key:` in the frontmatter, or None."""
    text = md.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^" + key + r":\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


def sources():
    return sorted(p for p in SRC_DIR.rglob("*.md") if ".generated" not in p.parts)


def main():
    if not SRC_DIR.is_dir():
        print(f"docs_fresh_check: no source directory at {SRC_DIR}", file=sys.stderr)
        return 2
    if shallow_clone():
        print("docs_fresh_check: shallow clone (or git failed); commit times cannot be compared. "
              "Fetch the full history.", file=sys.stderr)
        return 2

    findings = []
    checked = 0
    mapped = set()
    for md in sources():
        kind = frontmatter(md, "Template")
        name = frontmatter(md, "Name") or md.stem
        if kind not in KIND_DIR:
            findings.append(f"{md.relative_to(ROOT)}: Template {kind}, so no notebook can be "
                            f"identified for it")
            continue
        if name != md.stem:
            findings.append(f"{md.relative_to(ROOT)}: Name {name!r} differs from the file name")
            continue
        nb = KIND_DIR[kind] / f"{name}.nb"
        mapped.add(nb)
        if not nb.is_file():
            findings.append(f"{md.relative_to(ROOT)}: no notebook at {nb.relative_to(ROOT)}. "
                            f"Run ./build_docs.sh and commit the result.")
            continue

        if WL_FENCE_RE.search(md.read_text(errors="replace")) and \
                not EVALUATED_CELL_RE.search(nb.read_text(errors="replace")):
            findings.append(f"{nb.relative_to(ROOT)}: UNEVALUATED -- its source has wl examples "
                            f"and it has no Output, Print or Message cell. Build with evaluation.")

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

    for d in KIND_DIR.values():
        for nb in sorted(d.glob("*.nb")) if d.is_dir() else []:
            if nb not in mapped:
                findings.append(f"{nb.relative_to(ROOT)}: no source maps to it")

    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) over {checked} source/notebook pair(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
