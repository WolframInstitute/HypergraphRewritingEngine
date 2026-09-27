#!/usr/bin/env python3
"""Check the shipped documentation against the symbols the paclet exports.

WHY THIS EXISTS. `paclet/Kernel/HypergraphRewriting.wl` declares its public surface with
the declaration list at the top of the kernel. The documentation under `paclet/Documentation/English/` is a set of BUILT
notebooks, tracked in git and generated from the markdown under `docs/en/`, so a page can
outlive the symbol it documents and nothing regenerates or removes it. That is what
happened: the visualisation split deleted 21 functions and their reference pages stayed,
inside the shipped archive, describing calls a user cannot make.

TWO CHECKS, both mechanical, over the built notebooks (what ships) and over their markdown
sources (so a defect is reported before anything is built):

  ORPHAN PAGE   ReferencePages/Symbols/<Name>.nb or .md where <Name> is not exported
  DEAD LINK     a page links to .../ref/<Name> for a <Name> not exported or with no reference
                page, or to paclet:<this paclet>/guide/<Name> or /tutorial/<Name> for a <Name>
                with no such page. A source page's SeeAlso, RelatedGuides and RelatedTutorials
                lists are links of kind ref, guide and tutorial.

The second matters on its own: removing a page while leaving the guide entry turns a
wrong page into a broken link, which is not an improvement.

WHAT THIS CANNOT CONCLUDE. It does not check that a page's CONTENT is accurate, only that
its subject exists. A page documenting an exported symbol incorrectly reads as fine here.

Exit code is the number of findings, so CI can gate on it.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KERNEL = os.path.join(ROOT, "paclet", "Kernel", "HypergraphRewriting.wl")
PACLET_INFO = os.path.join(ROOT, "paclet", "PacletInfo.wl")
DOCS = os.path.join(ROOT, "paclet", "Documentation", "English")
SRC = os.path.join(ROOT, "docs", "en")
# The kind in a documentation URI -> the directory holding pages of that kind, the same under
# DOCS (.nb) and SRC (.md).
KIND_DIR = {"ref": os.path.join("ReferencePages", "Symbols"), "guide": "Guides",
            "tutorial": "Tutorials"}
FRONTMATTER_KIND = {"SeeAlso": "ref", "RelatedGuides": "guide", "RelatedTutorials": "tutorial"}
FRONTMATTER_RE = re.compile(r'\ufeff?---\r?\n(.*?)\r?\n---', re.DOTALL)

# The public symbols are the ones the kernel names in a list right after BeginPackage:
#   {HGEvolve, HGSessionObject, ...};
EXPORT_LIST_RE = re.compile(r'BeginPackage\[[^\]]*\].*?^\{([A-Za-z0-9$,\s]+)\};', re.DOTALL | re.MULTILINE)
NAME_RE = re.compile(r'[A-Za-z$][A-Za-z0-9$]*')
# A documentation link is "paclet:<publisher>/<paclet>/ref/<Symbol>".
REF_RE = re.compile(r'/ref/([A-Za-z$][A-Za-z0-9$]*)')
# A notebook wraps long strings with a backslash-newline, and it does so mid-name: a link
# stored as "ref/\<newline>HGMinkowskiSprinkling" reads as a link to nothing unless the
# wrap is undone first. Missing those made this report 3 dead links where there are 4.
CONTINUATION_RE = re.compile(r'\\\r?\n')


def paclet_name():
    with open(PACLET_INFO, errors="replace") as f:
        m = re.search(r'"Name"\s*->\s*"([^"]+)"', f.read())
    if not m:
        sys.exit(f'{PACLET_INFO} has no "Name" entry; the link scan needs the paclet name')
    return m.group(1)


def links_in(path, paclet):
    """(kind, name) for every link in a page to this paclet's pages."""
    with open(path, errors="replace") as f:
        text = CONTINUATION_RE.sub("", f.read())
    out = {("ref", n) for n in REF_RE.findall(text)}
    # A notebook link runs to its closing quote, so a tail with spaces (a title, not a page
    # name) is captured whole and reported; in markdown it ends at whitespace or a delimiter.
    tail = r'([^"]*)' if path.endswith(".nb") else r'([^"\s)\]>]*)'
    for kind, name in re.findall(r'paclet:' + re.escape(paclet) + r'/(guide|tutorial)/' + tail,
                                 text):
        out.add((kind, name))
    if path.endswith(".md"):
        fm = FRONTMATTER_RE.match(text)
        if fm:
            for key, kind in FRONTMATTER_KIND.items():
                km = re.search(r'^' + key + r':\s*\[(.*?)\]\s*$', fm.group(1), re.M)
                if km:
                    out |= {(kind, n.strip()) for n in km.group(1).split(",") if n.strip()}
    return out


def pages_of(kind):
    """Page names of one kind, built or source."""
    names = set()
    for root, ext in ((DOCS, ".nb"), (SRC, ".md")):
        d = os.path.join(root, KIND_DIR[kind])
        if os.path.isdir(d):
            names |= {n[:-len(ext)] for n in os.listdir(d) if n.endswith(ext)}
    return names


def main():
    if not os.path.exists(KERNEL):
        sys.exit(f"{KERNEL} does not exist; run this inside the repository")

    with open(KERNEL, errors="replace") as f:
        m = EXPORT_LIST_RE.search(f.read())
        exported = set(NAME_RE.findall(m.group(1))) if m else set()
    if not exported:
        sys.exit("found no public-symbol list in the Kernel source; refusing to report every "
                 "page as an orphan on what is more likely a parse failure here")

    findings = []
    paclet = paclet_name()
    pages = {kind: pages_of(kind) for kind in KIND_DIR}

    for root, ext in ((DOCS, ".nb"), (SRC, ".md")):
        d = os.path.join(root, KIND_DIR["ref"])
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(ext) and name[:-len(ext)] not in exported:
                findings.append(
                    f"ORPHAN   {os.path.relpath(os.path.join(d, name), ROOT)} documents "
                    f"`{name[:-len(ext)]}`, which the Kernel does not export.")

    for root, ext in ((DOCS, ".nb"), (SRC, ".md")):
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not name.endswith(ext):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, ROOT)
                for kind, target in sorted(links_in(path, paclet)):
                    if kind == "ref" and target not in exported:
                        findings.append(f"DEADLINK {rel} links to ref `{target}`, which the "
                                        f"Kernel does not export.")
                    elif target not in pages[kind]:
                        findings.append(f"DEADLINK {rel} links to {kind} `{target}`, which "
                                        f"has no page.")

    for f_ in findings:
        print(f_)
    print(f"\n{len(findings)} findings over {len(exported)} exported symbol(s): "
          f"{', '.join(sorted(exported))}")
    # 1, not the count. sys.exit() takes the low 8 bits of what it is given, so a run with
    # exactly 256 findings -- or 512 -- exits 0 and the gate reports success on its worst
    # result. The count is already printed above; the STATUS only has to say whether the
    # check passed.
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
